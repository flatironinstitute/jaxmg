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
// Device-side Cholesky log-determinant extraction for cuSOLVERMp POTRS.
//
// The factor entering this file is already in cuSOLVERMp's local
// column-major, 2D block-cyclic layout. For a global diagonal index g, tile
// q=g/T_A belongs to process coordinate (q mod process_rows,
// q mod process_cols). The owning rank maps that element to its local tile row
// and tile column and reads it with the matrix descriptor's physical leading
// dimension. Padding is excluded by iterating only over 0 <= g < n.
//
// File workflow:
//   1. Initialize the FFI scalar to NaN before the solver runs.
//   2. Reset it to zero after a successful factorization and solve.
//   3. Let every thread inspect a grid-stride subset of the global diagonal.
//   4. Reduce contributions within each CUDA block in shared memory.
//   5. Atomically add one partial sum per block to the rank-local real output.
//      The caller subsequently all-reduces that scalar with NCCL.
//
// Thread-local and block-local accumulation uses float64 to limit summation
// error. The public result follows JAX's real-component convention: float32
// for float32/complex64 matrices and float64 for float64/complex128 matrices.
// The implementation allocates no matrix-, tile-, or vector-sized workspace.

#include <algorithm>
#include <cmath>

#include <cuComplex.h>

#include "potrs_logdet.h"

namespace xla::gpu {
namespace {

constexpr int kLogdetBlockSize = 256;
constexpr int kMaxLogdetBlocks = 256;

// Returns the magnitude used by the Cholesky log-determinant identity for each
// supported real or complex matrix dtype.
template <typename DataType>
__device__ double DiagonalMagnitude(DataType value);

template <>
__device__ double DiagonalMagnitude<float>(float value) {
  return fabs(static_cast<double>(value));
}

template <>
__device__ double DiagonalMagnitude<double>(double value) {
  return fabs(value);
}

template <>
__device__ double DiagonalMagnitude<cuFloatComplex>(cuFloatComplex value) {
  return hypot(static_cast<double>(cuCrealf(value)),
               static_cast<double>(cuCimagf(value)));
}

template <>
__device__ double DiagonalMagnitude<cuDoubleComplex>(
    cuDoubleComplex value) {
  return hypot(cuCreal(value), cuCimag(value));
}

// Initializes the single device scalar without a host-to-device staging copy.
template <typename ResultType>
__global__ void SetLogdetValueKernel(ResultType* output, ResultType value) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *output = value;
  }
}

// Accumulates this process coordinate's contribution to log(det(A)).
//
// Each global diagonal element is inspected exactly once on every rank, but
// only its owning process coordinate performs a memory load. This simple
// mapping keeps the kernel independent of row-major versus column-major rank
// numbering: the caller has already converted the communicator rank into the
// corresponding (process_row, process_col) coordinate.
template <typename DataType, typename ResultType>
__global__ void AccumulateLocalLogdetKernel(
    const DataType* factor, int64_t n, int64_t tile_size,
    int64_t process_rows, int64_t process_cols, int32_t process_row,
    int32_t process_col, int64_t local_physical_rows, ResultType* output) {
  extern __shared__ double block_sums[];

  double thread_sum = 0.0;
  const int64_t first =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride =
      static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t global_index = first; global_index < n;
       global_index += stride) {
    const int64_t global_tile = global_index / tile_size;
    if (global_tile % process_rows != process_row ||
        global_tile % process_cols != process_col) {
      continue;
    }

    const int64_t tile_offset = global_index % tile_size;
    const int64_t local_tile_row = global_tile / process_rows;
    const int64_t local_tile_col = global_tile / process_cols;
    const int64_t local_row = local_tile_row * tile_size + tile_offset;
    const int64_t local_col = local_tile_col * tile_size + tile_offset;
    const int64_t local_offset =
        local_row + local_col * local_physical_rows;
    const double magnitude = DiagonalMagnitude(factor[local_offset]);
    thread_sum += 2.0 * log(magnitude);
  }

  block_sums[threadIdx.x] = thread_sum;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset) {
      block_sums[threadIdx.x] += block_sums[threadIdx.x + offset];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    atomicAdd(output, static_cast<ResultType>(block_sums[0]));
  }
}

// Launches the dtype-specialized reduction after resetting the output scalar.
template <typename DataType, typename ResultType>
cudaError_t LaunchLocalLogdet(cudaStream_t cuda_stream, const void* factor,
                              int64_t n, int64_t tile_size,
                              int64_t process_rows, int64_t process_cols,
                              int32_t process_row, int32_t process_col,
                              int64_t local_physical_rows,
                              void* logdet_out) {
  auto* typed_output = static_cast<ResultType*>(logdet_out);
  cudaError_t status =
      cudaMemsetAsync(typed_output, 0, sizeof(ResultType), cuda_stream);
  if (status != cudaSuccess) {
    return status;
  }
  const int block_count = static_cast<int>(std::min<int64_t>(
      kMaxLogdetBlocks, (n + kLogdetBlockSize - 1) / kLogdetBlockSize));
  AccumulateLocalLogdetKernel<DataType, ResultType>
      <<<block_count, kLogdetBlockSize,
         kLogdetBlockSize * static_cast<int>(sizeof(double)), cuda_stream>>>(
          static_cast<const DataType*>(factor), n, tile_size, process_rows,
          process_cols, process_row, process_col, local_physical_rows,
          typed_output);
  return cudaGetLastError();
}

}  // namespace

cudaError_t InitializeCholeskyLogdet(cudaStream_t cuda_stream,
                                     cudaDataType_t output_dtype,
                                     void* logdet_out) {
  if (cuda_stream == nullptr || logdet_out == nullptr) {
    return cudaErrorInvalidValue;
  }
  switch (output_dtype) {
    case CUDA_R_32F:
      SetLogdetValueKernel<float><<<1, 1, 0, cuda_stream>>>(
          static_cast<float*>(logdet_out), NAN);
      break;
    case CUDA_R_64F:
      SetLogdetValueKernel<double><<<1, 1, 0, cuda_stream>>>(
          static_cast<double*>(logdet_out), NAN);
      break;
    default:
      return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

cudaError_t AccumulateLocalCholeskyLogdet(
    cudaStream_t cuda_stream, cudaDataType_t dtype, const void* factor,
    int64_t n, int64_t tile_size, int64_t process_rows,
    int64_t process_cols, int32_t process_row, int32_t process_col,
    int64_t local_physical_rows, void* logdet_out) {
  if (cuda_stream == nullptr || factor == nullptr || logdet_out == nullptr) {
    return cudaErrorInvalidValue;
  }
  if (n <= 0 || tile_size <= 0 || process_rows <= 0 || process_cols <= 0 ||
      process_row < 0 || process_row >= process_rows || process_col < 0 ||
      process_col >= process_cols || local_physical_rows <= 0) {
    return cudaErrorInvalidValue;
  }

  switch (dtype) {
    case CUDA_R_32F:
      return LaunchLocalLogdet<float, float>(
          cuda_stream, factor, n, tile_size, process_rows, process_cols,
          process_row, process_col, local_physical_rows, logdet_out);
    case CUDA_R_64F:
      return LaunchLocalLogdet<double, double>(
          cuda_stream, factor, n, tile_size, process_rows, process_cols,
          process_row, process_col, local_physical_rows, logdet_out);
    case CUDA_C_32F:
      return LaunchLocalLogdet<cuFloatComplex, float>(
          cuda_stream, factor, n, tile_size, process_rows, process_cols,
          process_row, process_col, local_physical_rows, logdet_out);
    case CUDA_C_64F:
      return LaunchLocalLogdet<cuDoubleComplex, double>(
          cuda_stream, factor, n, tile_size, process_rows, process_cols,
          process_row, process_col, local_physical_rows, logdet_out);
    default:
      return cudaErrorInvalidValue;
  }
}

}  // namespace xla::gpu
