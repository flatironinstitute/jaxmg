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
// Device kernels that turn cuSOLVERMp factorization outputs into the public
// factor-only results while they are still in 2D block-cyclic storage.

#include <cuComplex.h>

#include "factor_extract.h"

namespace xla::gpu {
namespace {

constexpr int kFactorExtractBlockSize = 256;

// Maps a local block-cyclic index to its global index along one grid axis.
__device__ int64_t GlobalIndex(int64_t local, int64_t tile_size,
                               int64_t process_count, int32_t process) {
  return ((local / tile_size) * process_count + process) * tile_size +
         local % tile_size;
}

template <typename DataType>
__global__ void ZeroDistributedStrictUpperKernel(
    DataType* a, int64_t tile_size, int64_t process_rows, int64_t process_cols,
    int32_t process_row, int32_t process_col, int64_t leading_dimension,
    int64_t numroc_rows, int64_t numroc_cols) {
  const int64_t element =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= numroc_rows * numroc_cols) return;

  const int64_t local_row = element % numroc_rows;
  const int64_t local_col = element / numroc_rows;
  const int64_t global_row =
      GlobalIndex(local_row, tile_size, process_rows, process_row);
  const int64_t global_col =
      GlobalIndex(local_col, tile_size, process_cols, process_col);
  if (global_row < global_col) {
    a[local_row + local_col * leading_dimension] = DataType{};
  }
}

template <typename DataType>
cudaError_t LaunchZeroDistributedStrictUpper(
    cudaStream_t cuda_stream, void* a, int64_t tile_size, int64_t process_rows,
    int64_t process_cols, int32_t process_row, int32_t process_col,
    int64_t leading_dimension, int64_t numroc_rows, int64_t numroc_cols) {
  const int64_t element_count = numroc_rows * numroc_cols;
  if (element_count == 0) return cudaSuccess;
  const int block_count = static_cast<int>(
      (element_count + kFactorExtractBlockSize - 1) / kFactorExtractBlockSize);
  ZeroDistributedStrictUpperKernel<DataType>
      <<<block_count, kFactorExtractBlockSize, 0, cuda_stream>>>(
          static_cast<DataType*>(a), tile_size, process_rows, process_cols,
          process_row, process_col, leading_dimension, numroc_rows,
          numroc_cols);
  return cudaGetLastError();
}

__global__ void GatherDistributedPivotsKernel(
    const int64_t* local_ipiv, int32_t* piv, int64_t local_len, int64_t n,
    int64_t tile_size, int64_t process_count, int32_t process,
    int64_t index_base) {
  const int64_t local =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (local >= local_len) return;
  const int64_t value = local_ipiv[local];
  if (value < index_base) return;
  const int64_t global = GlobalIndex(local, tile_size, process_count, process);
  if (global < n) piv[global] = static_cast<int32_t>(value - index_base);
}

}  // namespace

cudaError_t ZeroDistributedStrictUpper(
    cudaStream_t cuda_stream, cudaDataType_t dtype, void* a, int64_t tile_size,
    int64_t process_rows, int64_t process_cols, int32_t process_row,
    int32_t process_col, int64_t leading_dimension, int64_t numroc_rows,
    int64_t numroc_cols) {
  if (cuda_stream == nullptr || a == nullptr || tile_size <= 0 ||
      process_rows <= 0 || process_cols <= 0 || process_row < 0 ||
      process_col < 0 || numroc_rows < 0 || numroc_cols < 0 ||
      leading_dimension < numroc_rows) {
    return cudaErrorInvalidValue;
  }
  switch (dtype) {
    case CUDA_R_32F:
      return LaunchZeroDistributedStrictUpper<float>(
          cuda_stream, a, tile_size, process_rows, process_cols, process_row,
          process_col, leading_dimension, numroc_rows, numroc_cols);
    case CUDA_R_64F:
      return LaunchZeroDistributedStrictUpper<double>(
          cuda_stream, a, tile_size, process_rows, process_cols, process_row,
          process_col, leading_dimension, numroc_rows, numroc_cols);
    case CUDA_C_32F:
      return LaunchZeroDistributedStrictUpper<cuFloatComplex>(
          cuda_stream, a, tile_size, process_rows, process_cols, process_row,
          process_col, leading_dimension, numroc_rows, numroc_cols);
    case CUDA_C_64F:
      return LaunchZeroDistributedStrictUpper<cuDoubleComplex>(
          cuda_stream, a, tile_size, process_rows, process_cols, process_row,
          process_col, leading_dimension, numroc_rows, numroc_cols);
    default:
      return cudaErrorInvalidValue;
  }
}

cudaError_t GatherDistributedPivots(cudaStream_t cuda_stream,
                                    const int64_t* local_ipiv, int32_t* piv,
                                    int64_t local_len, int64_t n,
                                    int64_t tile_size, int64_t process_count,
                                    int32_t process, int64_t index_base) {
  if (cuda_stream == nullptr || local_ipiv == nullptr || piv == nullptr ||
      local_len < 0 || n < 0 || tile_size <= 0 || process_count <= 0 ||
      process < 0) {
    return cudaErrorInvalidValue;
  }
  if (local_len == 0) return cudaSuccess;
  const int block_count = static_cast<int>(
      (local_len + kFactorExtractBlockSize - 1) / kFactorExtractBlockSize);
  GatherDistributedPivotsKernel<<<block_count, kFactorExtractBlockSize, 0,
                                  cuda_stream>>>(
      local_ipiv, piv, local_len, n, tile_size, process_count, process,
      index_base);
  return cudaGetLastError();
}

}  // namespace xla::gpu
