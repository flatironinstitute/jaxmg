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
// The 2D block-cyclic planner expresses movement in logical matrix coordinates:
//
//   matrix[row_start:row_start + row_count,
//          col_start:col_start + col_count] -> contiguous scratch
//   contiguous scratch -> matrix_out[target_row:target_row + row_count,
//                                   target_col:target_col + col_count]
//
// The physical addressing depends on the local matrix layout. Current JAXMg
// cuSolverMg diagnostics use row-major local buffers. cuSOLVERMp expects local
// matrices to be column-major, so this probe deliberately supports both:
//
//   layout == 0: scratch is packed row-major, one source row at a time.
//   layout == 1: scratch is packed column-major, one source column at a time.
//
// cudaMemcpy2DAsync provides this pack/unpack behavior without introducing a
// custom CUDA kernel yet. The future communicator path can use the same packed
// scratch representation as the send/receive payload.

#include "include/xla_comm_backend.h"

namespace xla::gpu {

namespace {

enum class RectLayout : int64_t {
  kRowMajor = 0,
  kColumnMajor = 1,
};

absl::StatusOr<RectLayout> DecodeRectLayout(int64_t layout) {
  switch (layout) {
    case static_cast<int64_t>(RectLayout::kRowMajor):
      return RectLayout::kRowMajor;
    case static_cast<int64_t>(RectLayout::kColumnMajor):
      return RectLayout::kColumnMajor;
    default:
      return absl::InvalidArgumentError(absl::StrFormat(
          "xla_rect_pack_unpack_probe received unknown layout %d; expected 0 "
          "(row-major) or 1 (column-major)",
          layout));
  }
}

struct RectCopySpec {
  size_t matrix_pitch;
  size_t packed_pitch;
  size_t copy_bytes;
  size_t copy_height;
  uint64_t matrix_offset;
};

absl::Status ValidateRect(const char* caller, int64_t row_start,
                          int64_t col_start, int64_t row_count,
                          int64_t col_count, int64_t local_rows,
                          int64_t local_cols) {
  if (row_count <= 0 || col_count <= 0) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s requires positive row_count and col_count", caller));
  }
  if (row_start < 0 || col_start < 0 ||
      row_start + row_count > local_rows ||
      col_start + col_count > local_cols) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s rectangle out of bounds: (%d:%d, %d:%d), matrix=(%d, %d)",
        caller, row_start, row_start + row_count, col_start,
        col_start + col_count, local_rows, local_cols));
  }
  return absl::OkStatus();
}

RectCopySpec BuildRectCopySpec(RectLayout layout, int64_t local_rows,
                               int64_t local_cols, int64_t row_start,
                               int64_t col_start, int64_t row_count,
                               int64_t col_count, size_t element_bytes) {
  if (layout == RectLayout::kRowMajor) {
    return RectCopySpec{
        static_cast<size_t>(local_cols) * element_bytes,
        static_cast<size_t>(col_count) * element_bytes,
        static_cast<size_t>(col_count) * element_bytes,
        static_cast<size_t>(row_count),
        static_cast<uint64_t>(row_start * local_cols + col_start) *
            element_bytes,
    };
  }

  return RectCopySpec{
      static_cast<size_t>(local_rows) * element_bytes,
      static_cast<size_t>(row_count) * element_bytes,
      static_cast<size_t>(row_count) * element_bytes,
      static_cast<size_t>(col_count),
      static_cast<uint64_t>(col_start * local_rows + row_start) *
          element_bytes,
  };
}

absl::Status CopyMatrixIfNeeded(cudaStream_t cuda_stream, ffi::AnyBuffer matrix,
                                ffi::Result<ffi::AnyBuffer> matrix_out) {
  se::DeviceAddressBase matrix_base = matrix.device_memory();
  se::DeviceAddressBase matrix_out_base = matrix_out->device_memory();
  if (matrix_base.opaque() == matrix_out_base.opaque()) {
    return absl::OkStatus();
  }
  JAXMG_RETURN_IF_CUDA_ERROR(cudaMemcpyAsync(
      matrix_out_base.opaque(), matrix_base.opaque(), matrix.size_bytes(),
      cudaMemcpyDeviceToDevice, cuda_stream));
  return absl::OkStatus();
}

absl::Status CopyScratchIfNeeded(cudaStream_t cuda_stream,
                                 ffi::AnyBuffer scratch,
                                 ffi::Result<ffi::AnyBuffer> scratch_out) {
  se::DeviceAddressBase scratch_base = scratch.device_memory();
  se::DeviceAddressBase scratch_out_base = scratch_out->device_memory();
  if (scratch_base.opaque() == scratch_out_base.opaque()) {
    return absl::OkStatus();
  }
  JAXMG_RETURN_IF_CUDA_ERROR(cudaMemcpyAsync(
      scratch_out_base.opaque(), scratch_base.opaque(), scratch.size_bytes(),
      cudaMemcpyDeviceToDevice, cuda_stream));
  return absl::OkStatus();
}

absl::Status PackRect(cudaStream_t cuda_stream, RectLayout layout,
                      int64_t local_rows, int64_t local_cols,
                      int64_t row_start, int64_t col_start,
                      int64_t row_count, int64_t col_count,
                      size_t element_bytes, se::DeviceAddressBase matrix_base,
                      se::DeviceAddressBase packed_base) {
  RectCopySpec spec =
      BuildRectCopySpec(layout, local_rows, local_cols, row_start, col_start,
                        row_count, col_count, element_bytes);
  const void* source =
      matrix_base.GetByteSlice(spec.matrix_offset, spec.copy_bytes).opaque();
  void* packed = packed_base.opaque();
  JAXMG_RETURN_IF_CUDA_ERROR(cudaMemcpy2DAsync(
      packed, spec.packed_pitch, source, spec.matrix_pitch, spec.copy_bytes,
      spec.copy_height, cudaMemcpyDeviceToDevice, cuda_stream));
  return absl::OkStatus();
}

absl::Status UnpackRect(cudaStream_t cuda_stream, RectLayout layout,
                        int64_t local_rows, int64_t local_cols,
                        int64_t row_start, int64_t col_start,
                        int64_t row_count, int64_t col_count,
                        size_t element_bytes, se::DeviceAddressBase packed_base,
                        se::DeviceAddressBase matrix_base) {
  RectCopySpec spec =
      BuildRectCopySpec(layout, local_rows, local_cols, row_start, col_start,
                        row_count, col_count, element_bytes);
  const void* packed = packed_base.opaque();
  void* target =
      matrix_base.GetByteSlice(spec.matrix_offset, spec.copy_bytes).opaque();
  JAXMG_RETURN_IF_CUDA_ERROR(cudaMemcpy2DAsync(
      target, spec.matrix_pitch, packed, spec.packed_pitch, spec.copy_bytes,
      spec.copy_height, cudaMemcpyDeviceToDevice, cuda_stream));
  return absl::OkStatus();
}

}  // namespace

absl::Status XlaRectPackUnpackProbePrepare() { return absl::OkStatus(); }

absl::Status XlaRectPackUnpackProbeDispatch(
    cudaStream_t cuda_stream, int64_t layout, int64_t row_start,
    int64_t col_start, int64_t row_count, int64_t col_count,
    int64_t target_row, int64_t target_col, ffi::AnyBuffer matrix,
    ffi::AnyBuffer scratch,
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
  absl::StatusOr<RectLayout> decoded_layout = DecodeRectLayout(layout);
  JAXMG_RETURN_IF_ERROR(decoded_layout.status());

  const int64_t local_rows = matrix.dimensions()[0];
  const int64_t local_cols = matrix.dimensions()[1];
  JAXMG_RETURN_IF_ERROR(ValidateRect("xla_rect_pack_unpack_probe source",
                                     row_start, col_start, row_count,
                                     col_count, local_rows, local_cols));
  JAXMG_RETURN_IF_ERROR(ValidateRect("xla_rect_pack_unpack_probe target",
                                     target_row, target_col, row_count,
                                     col_count, local_rows, local_cols));
  if (scratch.dimensions()[0] < row_count * col_count) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "xla_rect_pack_unpack_probe scratch length %d is smaller than "
        "row_count * col_count = %d",
        scratch.dimensions()[0], row_count * col_count));
  }

  const size_t element_bytes =
      matrix.size_bytes() / static_cast<size_t>(matrix.element_count());

  se::DeviceAddressBase matrix_base = matrix.device_memory();
  se::DeviceAddressBase matrix_out_base = matrix_out->device_memory();
  se::DeviceAddressBase scratch_out_base = scratch_out->device_memory();

  JAXMG_RETURN_IF_ERROR(CopyMatrixIfNeeded(cuda_stream, matrix, matrix_out));
  JAXMG_RETURN_IF_ERROR(PackRect(
      cuda_stream, *decoded_layout, local_rows, local_cols, row_start,
      col_start, row_count, col_count, element_bytes, matrix_base,
      scratch_out_base));
  JAXMG_RETURN_IF_ERROR(UnpackRect(
      cuda_stream, *decoded_layout, local_rows, local_cols, target_row,
      target_col, row_count, col_count, element_bytes, scratch_out_base,
      matrix_out_base));

  return absl::OkStatus();
}

absl::Status XlaRectTransferProbePrepare(
    const CollectiveParams* collective_params,
    CollectiveCliqueRequests* clique_requests) {
  if (collective_params == nullptr || clique_requests == nullptr) {
    return absl::InvalidArgumentError(
        "xla_rect_transfer_probe requires XLA collective prepare contexts");
  }

  absl::StatusOr<GpuCliqueKey> clique_key =
      NodeScopedP2PCliqueKey(*collective_params);
  if (!clique_key.ok()) {
    return clique_key.status();
  }

  absl::StatusOr<std::vector<GlobalDeviceId>> device_group =
      NodeScopedGlobalDeviceGroup(*collective_params);
  if (!device_group.ok()) {
    return device_group.status();
  }
  return clique_requests->RequestClique(*clique_key, {*device_group});
}

absl::Status XlaRectTransferProbeDispatch(
    se::Stream* stream, se::Stream* comm_stream, cudaStream_t cuda_stream,
    int64_t layout, absl::Span<const int64_t> targets,
    absl::Span<const int64_t> src_row_starts,
    absl::Span<const int64_t> src_col_starts,
    absl::Span<const int64_t> dst_row_starts,
    absl::Span<const int64_t> dst_col_starts, int64_t row_count,
    int64_t col_count, ffi::AnyBuffer matrix, ffi::AnyBuffer scratch,
    ffi::Result<ffi::AnyBuffer> matrix_out,
    ffi::Result<ffi::AnyBuffer> scratch_out,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques) {
  if (stream == nullptr || comm_stream == nullptr || cuda_stream == nullptr) {
    return absl::InvalidArgumentError(
        "xla_rect_transfer_probe requires XLA and CUDA stream contexts");
  }
  if (collective_params == nullptr || collective_cliques == nullptr) {
    return absl::InvalidArgumentError(
        "xla_rect_transfer_probe requires XLA collective contexts");
  }
  if (matrix.dimensions().size() != 2 ||
      matrix_out->dimensions().size() != 2 ||
      matrix.dimensions()[0] != matrix_out->dimensions()[0] ||
      matrix.dimensions()[1] != matrix_out->dimensions()[1]) {
    return absl::InvalidArgumentError(
        "xla_rect_transfer_probe expects matching rank-2 matrix input/output");
  }
  if (scratch.dimensions().size() != 1 ||
      scratch_out->dimensions().size() != 1 ||
      scratch.dimensions()[0] != scratch_out->dimensions()[0]) {
    return absl::InvalidArgumentError(
        "xla_rect_transfer_probe expects matching rank-1 scratch input/output");
  }
  if (matrix.element_type() != matrix_out->element_type() ||
      matrix.element_type() != scratch.element_type() ||
      scratch.element_type() != scratch_out->element_type()) {
    return absl::InvalidArgumentError(
        "xla_rect_transfer_probe requires matrix and scratch dtypes to match");
  }
  if (matrix.element_count() <= 0) {
    return absl::InvalidArgumentError(
        "xla_rect_transfer_probe requires a non-empty matrix");
  }
  absl::StatusOr<RectLayout> decoded_layout = DecodeRectLayout(layout);
  JAXMG_RETURN_IF_ERROR(decoded_layout.status());

  absl::StatusOr<GpuCliqueKey> clique_key =
      NodeScopedP2PCliqueKey(*collective_params);
  if (!clique_key.ok()) {
    return clique_key.status();
  }

  const int64_t num_ranks = static_cast<int64_t>(clique_key->num_devices());
  if (num_ranks <= 0) {
    return absl::InvalidArgumentError(
        "xla_rect_transfer_probe requires a non-empty clique");
  }
  const size_t expected_size = static_cast<size_t>(num_ranks);
  if (targets.size() != expected_size ||
      src_row_starts.size() != expected_size ||
      src_col_starts.size() != expected_size ||
      dst_row_starts.size() != expected_size ||
      dst_col_starts.size() != expected_size) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "xla_rect_transfer_probe expected %d entries for targets and all "
        "rectangle offset arrays, got %d/%d/%d/%d/%d",
        num_ranks, targets.size(), src_row_starts.size(),
        src_col_starts.size(), dst_row_starts.size(), dst_col_starts.size()));
  }

  const int64_t local_rows = matrix.dimensions()[0];
  const int64_t local_cols = matrix.dimensions()[1];
  const int64_t rect_elements = row_count * col_count;
  if (rect_elements <= 0) {
    return absl::InvalidArgumentError(
        "xla_rect_transfer_probe requires positive row_count and col_count");
  }
  if (scratch.dimensions()[0] < 2 * rect_elements) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "xla_rect_transfer_probe scratch length %d is smaller than "
        "2 * row_count * col_count = %d",
        scratch.dimensions()[0], 2 * rect_elements));
  }

  std::vector<bool> seen(num_ranks, false);
  for (int64_t source = 0; source < num_ranks; ++source) {
    const int64_t target = targets[source];
    if (target < -1 || target >= num_ranks) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "xla_rect_transfer_probe target[%d]=%d is outside [-1, %d)",
          source, target, num_ranks));
    }
    if (target < 0) {
      continue;
    }
    if (seen[target]) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "xla_rect_transfer_probe target rank %d appears more than once",
          target));
    }
    seen[target] = true;
    JAXMG_RETURN_IF_ERROR(ValidateRect(
        "xla_rect_transfer_probe source", src_row_starts[source],
        src_col_starts[source], row_count, col_count, local_rows, local_cols));
    JAXMG_RETURN_IF_ERROR(ValidateRect(
        "xla_rect_transfer_probe target", dst_row_starts[source],
        dst_col_starts[source], row_count, col_count, local_rows, local_cols));
  }

  std::optional<RankId> rank =
      clique_key->rank(collective_params->global_device_id);
  if (!rank.has_value()) {
    return absl::InvalidArgumentError(
        "xla_rect_transfer_probe could not resolve this device rank");
  }
  const int64_t rank_value = static_cast<int64_t>(rank->value());

  absl::StatusOr<Communicator*> comm = collective_cliques->GetComm(
      *clique_key, collective_params->global_device_id);
  if (!comm.ok()) {
    return comm.status();
  }

  const size_t element_bytes =
      matrix.size_bytes() / static_cast<size_t>(matrix.element_count());
  const uint64_t rect_bytes =
      static_cast<uint64_t>(rect_elements) * element_bytes;
  se::DeviceAddressBase matrix_base = matrix.device_memory();
  se::DeviceAddressBase matrix_out_base = matrix_out->device_memory();
  se::DeviceAddressBase scratch_out_base = scratch_out->device_memory();
  se::DeviceAddressBase send_slot =
      scratch_out_base.GetByteSlice(0, rect_bytes);
  se::DeviceAddressBase recv_slot =
      scratch_out_base.GetByteSlice(rect_bytes, rect_bytes);

  JAXMG_RETURN_IF_ERROR(CopyMatrixIfNeeded(cuda_stream, matrix, matrix_out));
  JAXMG_RETURN_IF_ERROR(CopyScratchIfNeeded(cuda_stream, scratch, scratch_out));

  const int64_t target_rank_value = targets[rank_value];
  if (target_rank_value >= 0) {
    JAXMG_RETURN_IF_ERROR(PackRect(
        cuda_stream, *decoded_layout, local_rows, local_cols,
        src_row_starts[rank_value], src_col_starts[rank_value], row_count,
        col_count, element_bytes, matrix_base, send_slot));
  }

  std::optional<RankId> source_rank;
  int64_t source_for_this_rank = -1;
  for (int64_t source = 0; source < num_ranks; ++source) {
    if (targets[source] == rank_value) {
      source_rank = RankId(source);
      source_for_this_rank = source;
      break;
    }
  }

  if (!source_rank.has_value() && target_rank_value < 0) {
    return absl::OkStatus();
  }

  if (source_rank.has_value() && source_rank->value() == rank_value &&
      target_rank_value == rank_value) {
    JAXMG_RETURN_IF_ERROR(UnpackRect(
        cuda_stream, *decoded_layout, local_rows, local_cols,
        dst_row_starts[rank_value], dst_col_starts[rank_value], row_count,
        col_count, element_bytes, send_slot, matrix_out_base));
    return absl::OkStatus();
  }

  std::vector<RankId> target_ranks;
  if (target_rank_value >= 0) {
    target_ranks.push_back(RankId(target_rank_value));
  }

  absl::Status status = comm_stream->WaitFor(stream);
  if (!status.ok()) {
    return status;
  }

  Future<> future = (*comm)->CollectivePermute(
      send_slot, recv_slot, matrix.element_type(),
      static_cast<size_t>(rect_elements), source_rank, target_ranks,
      GpuCollectives::On(*comm_stream));
  status = future.Await();
  if (!status.ok()) {
    return status;
  }
  status = stream->WaitFor(comm_stream);
  if (!status.ok()) {
    return status;
  }

  if (source_rank.has_value()) {
    JAXMG_RETURN_IF_ERROR(UnpackRect(
        cuda_stream, *decoded_layout, local_rows, local_cols,
        dst_row_starts[source_for_this_rank],
        dst_col_starts[source_for_this_rank], row_count, col_count,
        element_bytes, recv_slot, matrix_out_base));
  }

  return absl::OkStatus();
}

}  // namespace xla::gpu
