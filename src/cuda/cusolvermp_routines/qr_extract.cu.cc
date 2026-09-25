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

#include <cuComplex.h>

#include "qr_extract.h"

namespace xla::gpu {
namespace {

constexpr int kQrExtractBlockSize = 256;

// Local block-cyclic indices remain ordered by their global coordinate. The
// kernel therefore maps each local R entry back to its global row and column,
// copies GEQRF's upper triangle, and explicitly zeros the lower triangle.
template <typename DataType>
__global__ void ExtractDistributedQrRKernel(
    const DataType* packed_qr, DataType* r, int64_t tile_size,
    int64_t process_rows, int64_t process_cols, int32_t process_row,
    int32_t process_col, int64_t qr_leading_dimension,
    int64_t r_leading_dimension, int64_t r_numroc_rows,
    int64_t r_numroc_cols) {
  const int64_t element =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t element_count = r_numroc_rows * r_numroc_cols;
  if (element >= element_count) return;

  const int64_t local_row = element % r_numroc_rows;
  const int64_t local_col = element / r_numroc_rows;
  const int64_t global_row =
      ((local_row / tile_size) * process_rows + process_row) * tile_size +
      local_row % tile_size;
  const int64_t global_col =
      ((local_col / tile_size) * process_cols + process_col) * tile_size +
      local_col % tile_size;

  r[local_row + local_col * r_leading_dimension] =
      global_row <= global_col
          ? packed_qr[local_row + local_col * qr_leading_dimension]
          : DataType{};
}

template <typename DataType>
cudaError_t LaunchExtractDistributedQrR(
    cudaStream_t cuda_stream, const void* packed_qr, void* r,
    int64_t tile_size, int64_t process_rows, int64_t process_cols,
    int32_t process_row, int32_t process_col, int64_t qr_leading_dimension,
    int64_t r_leading_dimension, int64_t r_numroc_rows,
    int64_t r_numroc_cols) {
  const int64_t element_count = r_numroc_rows * r_numroc_cols;
  if (element_count == 0) return cudaSuccess;
  const int block_count = static_cast<int>(
      (element_count + kQrExtractBlockSize - 1) / kQrExtractBlockSize);
  ExtractDistributedQrRKernel<DataType>
      <<<block_count, kQrExtractBlockSize, 0, cuda_stream>>>(
          static_cast<const DataType*>(packed_qr), static_cast<DataType*>(r),
          tile_size, process_rows, process_cols, process_row, process_col,
          qr_leading_dimension, r_leading_dimension, r_numroc_rows,
          r_numroc_cols);
  return cudaGetLastError();
}

}  // namespace

cudaError_t ExtractDistributedQrR(
    cudaStream_t cuda_stream, cudaDataType_t dtype, const void* packed_qr,
    void* r, int64_t n, int64_t tile_size, int64_t process_rows,
    int64_t process_cols, int32_t process_row, int32_t process_col,
    int64_t qr_leading_dimension, int64_t r_leading_dimension,
    int64_t r_numroc_rows, int64_t r_numroc_cols) {
  if (cuda_stream == nullptr || packed_qr == nullptr || r == nullptr || n < 0 ||
      tile_size <= 0 || process_rows <= 0 || process_cols <= 0 ||
      process_row < 0 || process_col < 0 ||
      qr_leading_dimension < r_numroc_rows ||
      r_leading_dimension < r_numroc_rows || r_numroc_rows < 0 ||
      r_numroc_cols < 0) {
    return cudaErrorInvalidValue;
  }
  switch (dtype) {
    case CUDA_R_32F:
      return LaunchExtractDistributedQrR<float>(
          cuda_stream, packed_qr, r, tile_size, process_rows, process_cols,
          process_row, process_col, qr_leading_dimension, r_leading_dimension,
          r_numroc_rows, r_numroc_cols);
    case CUDA_R_64F:
      return LaunchExtractDistributedQrR<double>(
          cuda_stream, packed_qr, r, tile_size, process_rows, process_cols,
          process_row, process_col, qr_leading_dimension, r_leading_dimension,
          r_numroc_rows, r_numroc_cols);
    case CUDA_C_32F:
      return LaunchExtractDistributedQrR<cuFloatComplex>(
          cuda_stream, packed_qr, r, tile_size, process_rows, process_cols,
          process_row, process_col, qr_leading_dimension, r_leading_dimension,
          r_numroc_rows, r_numroc_cols);
    case CUDA_C_64F:
      return LaunchExtractDistributedQrR<cuDoubleComplex>(
          cuda_stream, packed_qr, r, tile_size, process_rows, process_cols,
          process_row, process_col, qr_leading_dimension, r_leading_dimension,
          r_numroc_rows, r_numroc_cols);
    default:
      return cudaErrorInvalidValue;
  }
}

}  // namespace xla::gpu
