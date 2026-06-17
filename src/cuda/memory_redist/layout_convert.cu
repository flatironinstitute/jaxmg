// Copyright 2026 JAXMg contributors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Local row-major -> column-major layout conversion kernels.
//
// cuSOLVERMp assumes each local distributed matrix shard is physically stored
// in column-major order. JAX normally materializes rank-2 buffers in row-major
// physical order. The public cuSOLVERMp path deliberately keeps the FFI
// boundary row-major and performs the local layout conversion here, inside the
// fused native call, so XLA does not need to materialize a second full local
// matrix shard just to satisfy cuSOLVERMp's local-storage convention:
//
//   input physical address:  data[row * cols + col]
//   output physical address: data[col * rows + row]
//
// The logical matrix is not transposed. Only the local buffer's physical
// storage order changes. The implementation follows the
// Catanzaro/Keller/Garland decomposition shape used by the standalone
// experiments in experiments/inplace_layout_convert.
//
// Integration constraints:
//   * The input and output pointer are the same donated local shard.
//   * Scratch is O(max(rows, cols)) elements and is supplied by the fused FFI
//     handler, sharing the same XLA scratch allocation as redistribution.
//   * The same exported launcher is used for the inverse conversion. A
//     column-major (rows, cols) address grid is a row-major (cols, rows)
//     address grid, so calling this decomposition with swapped dimensions
//     restores the row-major JAX-facing layout.
//   * The operation is byte-preserving. Dtype semantics stay with JAX and
//     cuSOLVERMp; this file only moves 4-, 8-, or 16-byte payloads.

#include <cuda_runtime.h>

#include <cstdint>

namespace {

struct Bytes4 {
  std::uint32_t x;
};

struct Bytes8 {
  std::uint64_t x;
};

struct Bytes16 {
  std::uint64_t lo;
  std::uint64_t hi;
};

struct GcdResult {
  std::uint64_t gcd;
  std::uint64_t inverse;
};

GcdResult ExtendedGcd(std::uint64_t a, std::uint64_t b) {
  std::int64_t x = 0;
  std::int64_t last_x = 1;
  std::int64_t y = 1;
  std::int64_t last_y = 0;
  const std::uint64_t original_b = b;
  while (b != 0) {
    const std::uint64_t quotient = a / b;
    const std::uint64_t new_b = a % b;
    a = b;
    b = new_b;

    const std::int64_t new_x =
        last_x - static_cast<std::int64_t>(quotient) * x;
    last_x = x;
    x = new_x;

    const std::int64_t new_y =
        last_y - static_cast<std::int64_t>(quotient) * y;
    last_y = y;
    y = new_y;
  }

  std::uint64_t inverse = 0;
  if (a == 1) {
    inverse = last_x < 0 ? static_cast<std::uint64_t>(
                               last_x + static_cast<std::int64_t>(original_b))
                         : static_cast<std::uint64_t>(last_x);
  }
  return {a, inverse};
}

__device__ __forceinline__ std::uint64_t ShuffleColumn(
    std::uint64_t row, std::uint64_t col, std::uint64_t rows,
    std::uint64_t cols, std::uint64_t gcd, std::uint64_t inverse) {
  const std::uint64_t b = cols / gcd;
  std::uint64_t r = col + row * (cols - 1);
  const std::int64_t condition =
      static_cast<std::int64_t>(row) - static_cast<std::int64_t>(col % gcd);
  if (condition > static_cast<std::int64_t>(rows - gcd)) {
    r += rows;
  }
  const std::uint64_t div = r / gcd;
  const std::uint64_t mod = r - div * gcd;
  return ((inverse * (div % b)) % b) + mod * b;
}

template <typename T>
__global__ void ColumnPreShuffleKernel(T* data, T* scratch, std::uint64_t rows,
                                       std::uint64_t cols, std::uint64_t col,
                                       std::uint64_t b) {
  for (std::uint64_t row = threadIdx.x; row < rows; row += blockDim.x) {
    const std::uint64_t src_row = (row + (col / b)) % rows;
    scratch[row] = data[src_row * cols + col];
  }
  __syncthreads();
  for (std::uint64_t row = threadIdx.x; row < rows; row += blockDim.x) {
    data[row * cols + col] = scratch[row];
  }
}

template <typename T>
__global__ void RowShuffleKernel(T* data, T* scratch, std::uint64_t rows,
                                 std::uint64_t cols, std::uint64_t row,
                                 std::uint64_t gcd,
                                 std::uint64_t inverse) {
  for (std::uint64_t col = threadIdx.x; col < cols; col += blockDim.x) {
    const std::uint64_t src_col =
        ShuffleColumn(row, col, rows, cols, gcd, inverse);
    scratch[col] = data[row * cols + src_col];
  }
  __syncthreads();
  for (std::uint64_t col = threadIdx.x; col < cols; col += blockDim.x) {
    data[row * cols + col] = scratch[col];
  }
}

template <typename T>
__global__ void ColumnPostShuffleKernel(T* data, T* scratch, std::uint64_t rows,
                                        std::uint64_t cols, std::uint64_t col,
                                        std::uint64_t a) {
  for (std::uint64_t row = threadIdx.x; row < rows; row += blockDim.x) {
    const std::uint64_t src_row = (row * cols + col - (row / a)) % rows;
    scratch[row] = data[src_row * cols + col];
  }
  __syncthreads();
  for (std::uint64_t row = threadIdx.x; row < rows; row += blockDim.x) {
    data[row * cols + col] = scratch[row];
  }
}

template <typename T>
cudaError_t LaunchTyped(cudaStream_t stream, void* data, void* scratch,
                        std::int64_t rows, std::int64_t cols) {
  if (rows < 0 || cols < 0 || data == nullptr || scratch == nullptr) {
    return cudaErrorInvalidValue;
  }
  const std::uint64_t urows = static_cast<std::uint64_t>(rows);
  const std::uint64_t ucols = static_cast<std::uint64_t>(cols);
  const std::uint64_t elements = urows * ucols;
  if (urows == 0 || ucols == 0 || elements <= 1 || urows == 1 ||
      ucols == 1) {
    return cudaSuccess;
  }

  GcdResult gcd_result = ExtendedGcd(urows, ucols);
  std::uint64_t gcd = gcd_result.gcd;
  std::uint64_t inverse = gcd_result.inverse;
  if (gcd > 1) {
    inverse = ExtendedGcd(urows / gcd, ucols / gcd).inverse;
  }

  constexpr int kBlockSize = 256;
  T* typed_data = static_cast<T*>(data);
  T* typed_scratch = static_cast<T*>(scratch);

  if (gcd > 1) {
    const std::uint64_t b = ucols / gcd;
    for (std::uint64_t col = 0; col < ucols; ++col) {
      ColumnPreShuffleKernel<T><<<1, kBlockSize, 0, stream>>>(
          typed_data, typed_scratch, urows, ucols, col, b);
    }
  }

  for (std::uint64_t row = 0; row < urows; ++row) {
    RowShuffleKernel<T><<<1, kBlockSize, 0, stream>>>(
        typed_data, typed_scratch, urows, ucols, row, gcd, inverse);
  }

  const std::uint64_t a = urows / gcd;
  for (std::uint64_t col = 0; col < ucols; ++col) {
    ColumnPostShuffleKernel<T><<<1, kBlockSize, 0, stream>>>(
        typed_data, typed_scratch, urows, ucols, col, a);
  }

  return cudaGetLastError();
}

}  // namespace

extern "C" cudaError_t JaxmgLaunchRowMajorToColumnMajorDecomposition(
    cudaStream_t stream, void* data, void* scratch, std::int64_t rows,
    std::int64_t cols, std::int64_t element_bytes) {
  switch (element_bytes) {
    case 4:
      return LaunchTyped<Bytes4>(stream, data, scratch, rows, cols);
    case 8:
      return LaunchTyped<Bytes8>(stream, data, scratch, rows, cols);
    case 16:
      return LaunchTyped<Bytes16>(stream, data, scratch, rows, cols);
    default:
      return cudaErrorInvalidValue;
  }
}
