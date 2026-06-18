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
// Diagnostic FFI handlers for rectangle packing and transfer.
//
// These probes intentionally exercise the same low-level pack/unpack and raw
// NCCL movement helpers used by the production 2D redistribution path. Keeping
// them in a separate file makes rectangle_pack.cc read as the production
// transport layer while preserving focused diagnostics for layout and
// communicator debugging.

#include "memory_redist.h"

namespace xla::gpu {
namespace {

absl::Status RectTransferProbeDispatchImpl(
    se::Stream* stream, se::Stream* comm_stream, cudaStream_t cuda_stream,
    absl::Span<const int64_t> targets, absl::Span<const int64_t> src_row_starts,
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

  absl::StatusOr<GpuCommunicator*> comm = collective_cliques->GetComm(
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
        cuda_stream, local_rows, src_row_starts[rank_value],
        src_col_starts[rank_value], row_count, col_count, element_bytes,
        matrix_base, send_slot));
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
        cuda_stream, local_rows, dst_row_starts[rank_value],
        dst_col_starts[rank_value], row_count, col_count, element_bytes,
        send_slot, matrix_out_base));
    return absl::OkStatus();
  }

  std::vector<RankId> target_ranks;
  if (target_rank_value >= 0) {
    target_ranks.push_back(RankId(target_rank_value));
  }

  JAXMG_RETURN_IF_ERROR(RunRawNcclSendRecv(
      "xla_rect_transfer_probe", stream, comm_stream, cuda_stream, *comm,
      rank_value, num_ranks, send_slot, recv_slot, rect_bytes, source_rank,
      absl::MakeConstSpan(target_ranks)));

  if (source_rank.has_value()) {
    JAXMG_RETURN_IF_ERROR(UnpackRect(
        cuda_stream, local_rows, dst_row_starts[source_for_this_rank],
        dst_col_starts[source_for_this_rank], row_count, col_count,
        element_bytes, recv_slot, matrix_out_base));
  }

  return absl::OkStatus();
}

}  // namespace

absl::Status XlaRectPackUnpackProbePrepare() { return absl::OkStatus(); }

absl::Status XlaRectPackUnpackProbeDispatch(
    cudaStream_t cuda_stream, int64_t row_start, int64_t col_start,
    int64_t row_count, int64_t col_count,
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
      cuda_stream, local_rows, row_start, col_start, row_count,
      col_count, element_bytes, matrix_base, scratch_out_base));
  JAXMG_RETURN_IF_ERROR(UnpackRect(
      cuda_stream, local_rows, target_row, target_col, row_count,
      col_count, element_bytes, scratch_out_base,
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
    absl::Span<const int64_t> targets, absl::Span<const int64_t> src_row_starts,
    absl::Span<const int64_t> src_col_starts,
    absl::Span<const int64_t> dst_row_starts,
    absl::Span<const int64_t> dst_col_starts, int64_t row_count,
    int64_t col_count, ffi::AnyBuffer matrix, ffi::AnyBuffer scratch,
    ffi::Result<ffi::AnyBuffer> matrix_out,
    ffi::Result<ffi::AnyBuffer> scratch_out,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques) {
  return RectTransferProbeDispatchImpl(
      stream, comm_stream, cuda_stream, targets, src_row_starts, src_col_starts,
      dst_row_starts, dst_col_starts, row_count, col_count, matrix, scratch,
      matrix_out, scratch_out, collective_params, collective_cliques);
}

}  // namespace xla::gpu
