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
// Catanzaro/Keller/Garland in-place transpose decomposition adapted for the
// bounded scratch allocation used by this backend.
//
// Integration constraints:
//   * The input and output pointer are the same donated local shard.
//   * Scratch is supplied by the fused FFI handler, sharing the same XLA
//     scratch allocation as redistribution.  The minimum requirement is
//     O(max(rows, cols)) elements, but production solver calls usually have a
//     much larger redistribution scratch buffer available.  The kernels below
//     use the full scratch span to process many independent rows/columns per
//     launch, which preserves the low-memory decomposition while avoiding one
//     kernel launch per row or column.
//   * The same exported launcher is used for the inverse conversion. A
//     column-major (rows, cols) address grid is a row-major (cols, rows)
//     address grid, so calling this decomposition with swapped dimensions
//     restores the row-major JAX-facing layout.
//   * The operation is byte-preserving. Dtype semantics stay with JAX and
//     cuSOLVERMp; this file only moves 4-, 8-, or 16-byte payloads.

#include "layout_convert.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <limits>

namespace {

// Trivially copyable 4-byte payload used for dtype-agnostic byte movement.
struct Bytes4 {
  std::uint32_t x;
};

// Trivially copyable 8-byte payload used for float64 and complex64 storage.
struct Bytes8 {
  std::uint64_t x;
};

// Trivially copyable 16-byte payload used for complex128 storage.
struct Bytes16 {
  std::uint64_t lo;
  std::uint64_t hi;
};

// Result of the extended-GCD calculation used by the Catanzaro decomposition.
struct GcdResult {
  std::uint64_t gcd;
  std::uint64_t inverse;
};

// Computes gcd(a, b) and the modular inverse needed by the row-shuffle phase
// when the relevant reduced dimensions are coprime.
GcdResult ExtendedGcd(std::uint64_t a, std::uint64_t b) {
  // The FFI entry point accepts signed int64 dimensions and rejects negative
  // values before reaching this helper. That keeps quotients inside int64_t,
  // which is required by the classic extended-GCD recurrence below.
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

// Maps one logical row/column coordinate to the source column needed by the
// Catanzaro row-shuffle phase.
__device__ __forceinline__ std::uint64_t ShuffleColumn(
    std::uint64_t row, std::uint64_t col, std::uint64_t rows,
    std::uint64_t cols, std::uint64_t gcd, std::uint64_t inverse) {
  // Catanzaro/Keller/Garland column permutation.  For a logical element
  // (row, col), this returns the source column that should be read so the row
  // shuffle phase realizes the row-major -> column-major address permutation
  // without an out-of-place matrix-sized buffer.
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

// Applies the optional non-coprime pre-shuffle to a batch of columns.
template <typename T>
__global__ void ColumnPreShuffleBatchKernel(
    T* data, T* scratch, std::uint64_t rows, std::uint64_t cols,
    std::uint64_t first_col, std::uint64_t col_count, std::uint64_t b) {
  // Optional pre-shuffle for non-coprime rectangular shapes. Each block owns
  // one column and uses one column-length scratch slice, so multiple columns
  // can run in one launch when the caller provides a larger scratch window.
  const std::uint64_t local_col = blockIdx.x;
  if (local_col >= col_count) {
    return;
  }
  const std::uint64_t col = first_col + local_col;
  T* column_scratch = scratch + local_col * rows;
  for (std::uint64_t row = threadIdx.x; row < rows; row += blockDim.x) {
    const std::uint64_t src_row = (row + (col / b)) % rows;
    column_scratch[row] = data[src_row * cols + col];
  }
  __syncthreads();
  for (std::uint64_t row = threadIdx.x; row < rows; row += blockDim.x) {
    data[row * cols + col] = column_scratch[row];
  }
}

// Applies the main row-wise permutation to a batch of rows using one scratch
// slice per row.
template <typename T>
__global__ void RowShuffleBatchKernel(
    T* data, T* scratch, std::uint64_t rows, std::uint64_t cols,
    std::uint64_t first_row, std::uint64_t row_count, std::uint64_t gcd,
    std::uint64_t inverse) {
  // Main permutation phase. Each block owns one row and gathers the permuted
  // source columns into row scratch before writing the row back.  This keeps
  // global writes coalesced and bounds scratch to row_count * cols elements.
  const std::uint64_t local_row = blockIdx.x;
  if (local_row >= row_count) {
    return;
  }
  const std::uint64_t row = first_row + local_row;
  T* row_scratch = scratch + local_row * cols;
  for (std::uint64_t col = threadIdx.x; col < cols; col += blockDim.x) {
    const std::uint64_t src_col =
        ShuffleColumn(row, col, rows, cols, gcd, inverse);
    row_scratch[col] = data[row * cols + src_col];
  }
  __syncthreads();
  for (std::uint64_t col = threadIdx.x; col < cols; col += blockDim.x) {
    data[row * cols + col] = row_scratch[col];
  }
}

// Applies the final non-coprime/coprime column correction to a batch of
// columns.
template <typename T>
__global__ void ColumnPostShuffleBatchKernel(
    T* data, T* scratch, std::uint64_t rows, std::uint64_t cols,
    std::uint64_t first_col, std::uint64_t col_count, std::uint64_t a) {
  // Final column correction. Like the pre-shuffle, it is batched over as many
  // columns as fit in the scratch allocation.
  const std::uint64_t local_col = blockIdx.x;
  if (local_col >= col_count) {
    return;
  }
  const std::uint64_t col = first_col + local_col;
  T* column_scratch = scratch + local_col * rows;
  for (std::uint64_t row = threadIdx.x; row < rows; row += blockDim.x) {
    const std::uint64_t src_row = (row * cols + col - (row / a)) % rows;
    column_scratch[row] = data[src_row * cols + col];
  }
  __syncthreads();
  for (std::uint64_t row = threadIdx.x; row < rows; row += blockDim.x) {
    data[row * cols + col] = column_scratch[row];
  }
}

// Small constexpr-friendly minimum helper for unsigned 64-bit launch geometry.
std::uint64_t MinU64(std::uint64_t a, std::uint64_t b) {
  return a < b ? a : b;
}

// Small constexpr-friendly maximum helper for unsigned 64-bit launch geometry.
std::uint64_t MaxU64(std::uint64_t a, std::uint64_t b) {
  return a > b ? a : b;
}

constexpr std::uint64_t kMaxGridX = 2147483647ULL;

// Dispatches the in-place decomposition for one concrete byte payload type,
// batching as many rows/columns per kernel launch as the scratch window allows.
template <typename T>
cudaError_t LaunchTyped(cudaStream_t stream, void* data, void* scratch,
                        std::int64_t rows, std::int64_t cols,
                        std::int64_t scratch_elements) {
  if (rows < 0 || cols < 0 || scratch_elements < 0 || data == nullptr ||
      scratch == nullptr) {
    return cudaErrorInvalidValue;
  }
  const std::uint64_t urows = static_cast<std::uint64_t>(rows);
  const std::uint64_t ucols = static_cast<std::uint64_t>(cols);
  const std::uint64_t uscratch_elements =
      static_cast<std::uint64_t>(scratch_elements);
  if (ucols != 0 &&
      urows > std::numeric_limits<std::uint64_t>::max() / ucols) {
    return cudaErrorInvalidValue;
  }
  const std::uint64_t elements = urows * ucols;
  if (urows == 0 || ucols == 0 || elements <= 1 || urows == 1 ||
      ucols == 1) {
    return cudaSuccess;
  }
  if (uscratch_elements < MaxU64(urows, ucols)) {
    return cudaErrorInvalidValue;
  }

  GcdResult gcd_result = ExtendedGcd(urows, ucols);
  std::uint64_t gcd = gcd_result.gcd;
  std::uint64_t inverse = gcd_result.inverse;
  if (gcd > 1) {
    // Non-coprime shapes use the reduced dimensions for the modular inverse
    // in the row permutation. Coprime shapes keep the direct inverse.
    inverse = ExtendedGcd(urows / gcd, ucols / gcd).inverse;
  }

  constexpr int kBlockSize = 256;
  T* typed_data = static_cast<T*>(data);
  T* typed_scratch = static_cast<T*>(scratch);
  const std::uint64_t row_batch =
      MaxU64(1, MinU64(urows, uscratch_elements / ucols));
  const std::uint64_t col_batch =
      MaxU64(1, MinU64(ucols, uscratch_elements / urows));

  if (gcd > 1) {
    // The decomposition has three phases for non-coprime rectangles.  Each
    // loop launches the largest batch that fits the caller's scratch window
    // and CUDA's grid-x limit.
    const std::uint64_t b = ucols / gcd;
    for (std::uint64_t col = 0; col < ucols;) {
      const std::uint64_t batch =
          MinU64(kMaxGridX, MinU64(col_batch, ucols - col));
      ColumnPreShuffleBatchKernel<T>
          <<<static_cast<unsigned int>(batch), kBlockSize, 0, stream>>>(
              typed_data, typed_scratch, urows, ucols, col, batch, b);
      col += batch;
    }
  }

  for (std::uint64_t row = 0; row < urows;) {
    // Coprime rectangles skip the pre-shuffle but still use the batched row
    // shuffle and post-shuffle.
    const std::uint64_t batch =
        MinU64(kMaxGridX, MinU64(row_batch, urows - row));
    RowShuffleBatchKernel<T>
        <<<static_cast<unsigned int>(batch), kBlockSize, 0, stream>>>(
            typed_data, typed_scratch, urows, ucols, row, batch, gcd,
            inverse);
    row += batch;
  }

  const std::uint64_t a = urows / gcd;
  for (std::uint64_t col = 0; col < ucols;) {
    const std::uint64_t batch =
        MinU64(kMaxGridX, MinU64(col_batch, ucols - col));
    ColumnPostShuffleBatchKernel<T>
        <<<static_cast<unsigned int>(batch), kBlockSize, 0, stream>>>(
            typed_data, typed_scratch, urows, ucols, col, batch, a);
    col += batch;
  }

  return cudaGetLastError();
}

}  // namespace

// C ABI launcher called by the ordinary C++ FFI backend. It chooses the typed
// byte payload by element size and leaves dtype semantics to JAX/cuSOLVERMp.
extern "C" cudaError_t JaxmgLaunchRowMajorToColumnMajorDecomposition(
    cudaStream_t stream, void* data, void* scratch, std::int64_t rows,
    std::int64_t cols, std::int64_t scratch_elements,
    std::int64_t element_bytes) {
  // The C++ FFI backend dispatches by element byte width rather than by dtype.
  // The permutation is byte-preserving, so 8-byte float64 and complex64 payloads
  // use the same movement kernel.
  switch (element_bytes) {
    case 4:
      return LaunchTyped<Bytes4>(stream, data, scratch, rows, cols,
                                 scratch_elements);
    case 8:
      return LaunchTyped<Bytes8>(stream, data, scratch, rows, cols,
                                 scratch_elements);
    case 16:
      return LaunchTyped<Bytes16>(stream, data, scratch, rows, cols,
                                  scratch_elements);
    default:
      return cudaErrorInvalidValue;
  }
}
