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
// Local rectangle pack/unpack diagnostics for future cuSOLVERMp redistribution.
//
// The 2D block-cyclic planner shows that the column-owner phase needs to move
// rectangular regions that are strided under the current row-major native offset
// model. Before combining that with the XLA communicator, this file validates
// the local addressing primitive:
//
//   matrix[row_start:row_start + row_count,
//          col_start:col_start + col_count] -> contiguous scratch
//   contiguous scratch -> matrix_out[target_row:target_row + row_count,
//                                   target_col:target_col + col_count]
//
// cudaMemcpy2DAsync provides the pack/unpack behavior without introducing a
// custom CUDA kernel yet. The future communicator path can use the same packed
// scratch representation as the send/receive payload.

#include "include/xla_comm_backend.h"

namespace xla::gpu {

absl::Status XlaRectPackUnpackProbePrepare() { return absl::OkStatus(); }

absl::Status XlaRectPackUnpackProbeDispatch(
    cudaStream_t cuda_stream, int64_t row_start, int64_t col_start,
    int64_t row_count, int64_t col_count, int64_t target_row,
    int64_t target_col, ffi::AnyBuffer matrix, ffi::AnyBuffer scratch,
    ffi::Result<ffi::AnyBuffer> matrix_out,
    ffi::Result<ffi::AnyBuffer> scratch_out) {
  if (cuda_stream == nullptr) {
    return absl::InvalidArgumentError(
        "xla_rect_pack_unpack_probe requires a CUDA stream context");
  }
  if (matrix.dimensions().size() != 2 ||
      matrix_out->dimensions().size() != 2 ||
      matrix.dimensions()[0] != matrix_out->dimensions()[0] ||
      matrix.dimensions()[1] != matrix_out->dimensions()[1]) {
    return absl::InvalidArgumentError(
        "xla_rect_pack_unpack_probe expects matching rank-2 matrix input/output");
  }
  if (scratch.dimensions().size() != 1 ||
      scratch_out->dimensions().size() != 1 ||
      scratch.dimensions()[0] != scratch_out->dimensions()[0]) {
    return absl::InvalidArgumentError(
        "xla_rect_pack_unpack_probe expects matching rank-1 scratch input/output");
  }
  if (matrix.element_type() != matrix_out->element_type() ||
      matrix.element_type() != scratch.element_type() ||
      scratch.element_type() != scratch_out->element_type()) {
    return absl::InvalidArgumentError(
        "xla_rect_pack_unpack_probe requires matrix and scratch dtypes to match");
  }
  if (matrix.element_count() <= 0) {
    return absl::InvalidArgumentError(
        "xla_rect_pack_unpack_probe requires a non-empty matrix");
  }

  const int64_t local_rows = matrix.dimensions()[0];
  const int64_t local_cols = matrix.dimensions()[1];
  if (row_count <= 0 || col_count <= 0) {
    return absl::InvalidArgumentError(
        "xla_rect_pack_unpack_probe requires positive row_count and col_count");
  }
  if (row_start < 0 || col_start < 0 || target_row < 0 || target_col < 0 ||
      row_start + row_count > local_rows || col_start + col_count > local_cols ||
      target_row + row_count > local_rows ||
      target_col + col_count > local_cols) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "xla_rect_pack_unpack_probe rectangle out of bounds: source=(%d:%d, "
        "%d:%d), target=(%d:%d, %d:%d), matrix=(%d, %d)",
        row_start, row_start + row_count, col_start, col_start + col_count,
        target_row, target_row + row_count, target_col,
        target_col + col_count, local_rows, local_cols));
  }
  if (scratch.dimensions()[0] < row_count * col_count) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "xla_rect_pack_unpack_probe scratch length %d is smaller than "
        "row_count * col_count = %d",
        scratch.dimensions()[0], row_count * col_count));
  }

  const size_t element_bytes =
      matrix.size_bytes() / static_cast<size_t>(matrix.element_count());
  const size_t matrix_pitch = static_cast<size_t>(local_cols) * element_bytes;
  const size_t packed_pitch = static_cast<size_t>(col_count) * element_bytes;
  const size_t row_bytes = static_cast<size_t>(col_count) * element_bytes;
  const uint64_t source_offset =
      static_cast<uint64_t>(row_start * local_cols + col_start) * element_bytes;
  const uint64_t target_offset =
      static_cast<uint64_t>(target_row * local_cols + target_col) *
      element_bytes;

  se::DeviceAddressBase matrix_base = matrix.device_memory();
  se::DeviceAddressBase matrix_out_base = matrix_out->device_memory();
  se::DeviceAddressBase scratch_out_base = scratch_out->device_memory();

  if (matrix_base.opaque() != matrix_out_base.opaque()) {
    JAXMG_RETURN_IF_CUDA_ERROR(cudaMemcpyAsync(
        matrix_out_base.opaque(), matrix_base.opaque(), matrix.size_bytes(),
        cudaMemcpyDeviceToDevice, cuda_stream));
  }

  void* packed = scratch_out_base.opaque();
  const void* source =
      matrix_base.GetByteSlice(source_offset, row_bytes).opaque();
  void* target =
      matrix_out_base.GetByteSlice(target_offset, row_bytes).opaque();

  JAXMG_RETURN_IF_CUDA_ERROR(cudaMemcpy2DAsync(
      packed, packed_pitch, source, matrix_pitch, row_bytes,
      static_cast<size_t>(row_count), cudaMemcpyDeviceToDevice, cuda_stream));
  JAXMG_RETURN_IF_CUDA_ERROR(cudaMemcpy2DAsync(
      target, matrix_pitch, packed, packed_pitch, row_bytes,
      static_cast<size_t>(row_count), cudaMemcpyDeviceToDevice, cuda_stream));

  return absl::OkStatus();
}

}  // namespace xla::gpu
