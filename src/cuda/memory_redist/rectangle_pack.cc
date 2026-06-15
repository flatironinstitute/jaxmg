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
// Low-level rectangle packing and NCCL movement for 2D redistribution.
//
// This file is the transport layer shared by the cuSOLVERMp 2D redistribution
// code. It deliberately does not decide which rectangles need to move. That
// planning lives in edge_padding_2d.cc and block_cyclic_2d.cc. This file only
// knows how to copy rectangular column-major regions into bounded scratch, move
// those packed payloads through the borrowed XLA/NCCL communicator, and unpack
// the payloads at their destination.
//
// File workflow:
//   1. Borrow the raw NCCL communicator from XLA's GpuCommunicator and validate
//      that the NCCL rank/count match the XLA collective rank/count.
//   2. Pack rectangular column-major local matrix regions into bounded scratch
//      using cudaMemcpy2DAsync.
//   3. Move packed payloads with grouped NCCL send/recv calls on an XLA-ordered
//      stream.
//   4. Unpack received payloads into their destination rectangles.
//   5. Execute an already-planned sequence of Native2DStepBatch rounds with one
//      saved slab, one send slab, and one receive slab per rank.
//
// The public cuSOLVERMp path should call this file through the higher-level
// phase files rather than constructing rectangle moves directly. The probe
// handlers at the bottom remain as diagnostics for pack/unpack and one-hop
// rectangle transfer debugging.

#include <algorithm>
#include <limits>
#include <vector>

#include "../include/xla_comm_backend.h"
#include "third_party/nccl/nccl.h"

namespace xla::gpu {


absl::Status NcclToStatus(ncclResult_t result, const char* file, int line) {
  if (result == ncclSuccess) {
    return absl::OkStatus();
  }
  return absl::InternalError(absl::StrFormat(
      "NCCL error %d (%s) at %s:%d", static_cast<int>(result),
      ncclGetErrorString(result), file, line));
}

#define JAXMG_RETURN_IF_NCCL_ERROR(expr)                         \
  do {                                                           \
    absl::Status _jaxmg_nccl_status =                            \
        NcclToStatus((expr), __FILE__, __LINE__);                \
    if (!_jaxmg_nccl_status.ok()) return _jaxmg_nccl_status;     \
  } while (0)

absl::StatusOr<ncclComm_t> BorrowNcclComm(const char* caller,
                                          GpuCommunicator* comm) {
  if (comm == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s requires an XLA GPU communicator", caller));
  }
  void* handle = comm->platform_comm().handle;
  if (handle == nullptr) {
    return absl::FailedPreconditionError(absl::StrFormat(
        "%s requires an XLA communicator with a platform NCCL handle", caller));
  }
  return reinterpret_cast<ncclComm_t>(handle);
}

absl::Status ValidateBorrowedNcclComm(const char* caller, ncclComm_t comm,
                                      int64_t expected_rank,
                                      int64_t expected_count) {
  int comm_rank = -1;
  int comm_count = -1;
  JAXMG_RETURN_IF_NCCL_ERROR(ncclCommUserRank(comm, &comm_rank));
  JAXMG_RETURN_IF_NCCL_ERROR(ncclCommCount(comm, &comm_count));
  if (comm_rank != expected_rank || comm_count != expected_count) {
    return absl::FailedPreconditionError(absl::StrFormat(
        "%s borrowed NCCL communicator mismatch: NCCL rank/count=(%d, %d), "
        "XLA rank/count=(%d, %d)",
        caller, comm_rank, comm_count, expected_rank, expected_count));
  }
  return absl::OkStatus();
}

struct NcclStreamChoice {
  cudaStream_t stream;
  bool uses_comm_stream;
};

absl::StatusOr<NcclStreamChoice> ChooseNcclStream(const char* caller,
                                                  se::Stream* comm_stream,
                                                  cudaStream_t cuda_stream) {
  if (comm_stream != nullptr) {
    void* handle = comm_stream->platform_specific_handle().stream;
    if (handle != nullptr) {
      return NcclStreamChoice{/*stream=*/reinterpret_cast<cudaStream_t>(handle),
                              /*uses_comm_stream=*/true};
    }
  }
  if (cuda_stream == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s requires a CUDA stream for raw NCCL", caller));
  }
  return NcclStreamChoice{/*stream=*/cuda_stream,
                          /*uses_comm_stream=*/false};
}

absl::Status RunRawNcclSendRecv(
    const char* caller, se::Stream* stream, se::Stream* comm_stream,
    cudaStream_t cuda_stream, GpuCommunicator* comm, int64_t rank_value,
    int64_t num_ranks, se::DeviceAddressBase send_buffer,
    se::DeviceAddressBase recv_buffer, uint64_t byte_count,
    std::optional<RankId> source_rank, absl::Span<const RankId> target_ranks) {
  if (!source_rank.has_value() && target_ranks.empty()) {
    return absl::OkStatus();
  }
  if (target_ranks.size() > 1) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s raw NCCL path expects at most one target rank per batch", caller));
  }
  if (byte_count > std::numeric_limits<size_t>::max()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s raw NCCL byte count %llu exceeds size_t",
        caller, static_cast<unsigned long long>(byte_count)));
  }

  absl::StatusOr<ncclComm_t> nccl_comm = BorrowNcclComm(caller, comm);
  if (!nccl_comm.ok()) {
    return nccl_comm.status();
  }
  JAXMG_RETURN_IF_ERROR(ValidateBorrowedNcclComm(
      caller, *nccl_comm, rank_value, num_ranks));

  absl::StatusOr<NcclStreamChoice> nccl_stream =
      ChooseNcclStream(caller, comm_stream, cuda_stream);
  if (!nccl_stream.ok()) {
    return nccl_stream.status();
  }

  if (nccl_stream->uses_comm_stream) {
    absl::Status status = comm_stream->WaitFor(stream);
    if (!status.ok()) {
      return status;
    }
  }

  JAXMG_RETURN_IF_NCCL_ERROR(ncclGroupStart());
  if (source_rank.has_value()) {
    JAXMG_RETURN_IF_NCCL_ERROR(ncclRecv(
        recv_buffer.opaque(), static_cast<size_t>(byte_count), ncclUint8,
        source_rank->value(), *nccl_comm, nccl_stream->stream));
  }
  if (!target_ranks.empty()) {
    JAXMG_RETURN_IF_NCCL_ERROR(ncclSend(
        send_buffer.opaque(), static_cast<size_t>(byte_count), ncclUint8,
        target_ranks.front().value(), *nccl_comm, nccl_stream->stream));
  }
  JAXMG_RETURN_IF_NCCL_ERROR(ncclGroupEnd());

  if (nccl_stream->uses_comm_stream) {
    absl::Status status = stream->WaitFor(comm_stream);
    if (!status.ok()) {
      return status;
    }
  }
  return absl::OkStatus();
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

RectCopySpec BuildRectCopySpec(int64_t local_rows, int64_t row_start,
                               int64_t col_start,
                               int64_t row_count, int64_t col_count,
                               size_t element_bytes) {
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

absl::Status PackRect(cudaStream_t cuda_stream, int64_t local_rows,
                      int64_t row_start, int64_t col_start,
                      int64_t row_count, int64_t col_count,
                      size_t element_bytes, se::DeviceAddressBase matrix_base,
                      se::DeviceAddressBase packed_base) {
  RectCopySpec spec = BuildRectCopySpec(
      local_rows, row_start, col_start, row_count, col_count, element_bytes);
  const void* source =
      matrix_base.GetByteSlice(spec.matrix_offset, spec.copy_bytes).opaque();
  void* packed = packed_base.opaque();
  JAXMG_RETURN_IF_CUDA_ERROR(cudaMemcpy2DAsync(
      packed, spec.packed_pitch, source, spec.matrix_pitch, spec.copy_bytes,
      spec.copy_height, cudaMemcpyDeviceToDevice, cuda_stream));
  return absl::OkStatus();
}

absl::Status UnpackRect(cudaStream_t cuda_stream, int64_t local_rows,
                        int64_t row_start, int64_t col_start,
                        int64_t row_count, int64_t col_count,
                        size_t element_bytes,
                        se::DeviceAddressBase packed_base,
                        se::DeviceAddressBase matrix_base) {
  RectCopySpec spec = BuildRectCopySpec(
      local_rows, row_start, col_start, row_count, col_count, element_bytes);
  const void* packed = packed_base.opaque();
  void* target =
      matrix_base.GetByteSlice(spec.matrix_offset, spec.copy_bytes).opaque();
  JAXMG_RETURN_IF_CUDA_ERROR(cudaMemcpy2DAsync(
      target, spec.matrix_pitch, packed, spec.packed_pitch, spec.copy_bytes,
      spec.copy_height, cudaMemcpyDeviceToDevice, cuda_stream));
  return absl::OkStatus();
}

int64_t StepElementCount(const Native2DStep& step) {
  if (step.kind == Native2DStepKind::kRestoreScratch) {
    return step.target.row_count * step.target.col_count;
  }
  return step.source.row_count * step.source.col_count;
}

int64_t MaxStepElementCount(const std::vector<Native2DStep>& steps) {
  int64_t max_elements = 0;
  for (const Native2DStep& step : steps) {
    max_elements = std::max(max_elements, StepElementCount(step));
  }
  return max_elements;
}

absl::Status ExecuteNative2DStepBatches(
    se::Stream* stream, se::Stream* comm_stream, cudaStream_t cuda_stream,
    const std::vector<Native2DStepBatch>& batches, int64_t local_rows,
    int64_t local_cols, int64_t rank_value, int64_t num_ranks,
    size_t element_bytes, int64_t slot_elements, ffi::AnyBuffer matrix,
    se::DeviceAddressBase matrix_out_base,
    se::DeviceAddressBase scratch_out_base, GpuCommunicator* comm) {
  const uint64_t slot_bytes =
      static_cast<uint64_t>(slot_elements) * element_bytes;
  se::DeviceAddressBase saved_slot =
      scratch_out_base.GetByteSlice(0, slot_bytes);
  se::DeviceAddressBase send_slot =
      scratch_out_base.GetByteSlice(slot_bytes, slot_bytes);
  se::DeviceAddressBase recv_slot =
      scratch_out_base.GetByteSlice(2 * slot_bytes, slot_bytes);

  // Execute the schedule in conflict-free batches. Sources are read from the
  // mutable output buffer because later waves/phases consume the layout written
  // by earlier waves/phases.
  for (const Native2DStepBatch& batch : batches) {
    const Native2DStep* send_step = nullptr;
    const Native2DStep* recv_step = nullptr;
    for (const Native2DStep& step : batch.steps) {
      if (step.source_rank == rank_value) {
        send_step = &step;
      }
      if (step.target_rank == rank_value) {
        recv_step = &step;
      }
    }

    if (send_step == nullptr && recv_step == nullptr) {
      continue;
    }

    if (batch.kind == Native2DStepKind::kSaveScratch) {
      if (send_step == nullptr) {
        continue;
      }
      JAXMG_RETURN_IF_ERROR(PackRect(
          cuda_stream, local_rows, send_step->source.row_start,
          send_step->source.col_start,
          send_step->source.row_count, send_step->source.col_count,
          element_bytes, matrix_out_base, saved_slot));
      continue;
    }

    if (batch.kind == Native2DStepKind::kMove && send_step != nullptr) {
      JAXMG_RETURN_IF_ERROR(PackRect(
          cuda_stream, local_rows, send_step->source.row_start,
          send_step->source.col_start,
          send_step->source.row_count, send_step->source.col_count,
          element_bytes, matrix_out_base, send_slot));
    }

    std::optional<RankId> source_rank;
    if (recv_step != nullptr) {
      source_rank = RankId(recv_step->source_rank);
    }
    std::vector<RankId> target_ranks;
    if (send_step != nullptr && send_step->target_rank >= 0) {
      target_ranks.push_back(RankId(send_step->target_rank));
    }

    const Native2DStep* shape_step =
        send_step != nullptr ? send_step : recv_step;
    const int64_t collective_elements =
        batch.kind == Native2DStepKind::kRestoreScratch
            ? shape_step->target.row_count * shape_step->target.col_count
            : (send_step != nullptr
                   ? send_step->source.row_count * send_step->source.col_count
                   : recv_step->target.row_count *
                         recv_step->target.col_count);
    se::DeviceAddressBase collective_send_slot =
        batch.kind == Native2DStepKind::kRestoreScratch ? saved_slot
                                                        : send_slot;

    if (send_step != nullptr && recv_step != nullptr &&
        send_step->source_rank == send_step->target_rank) {
      // Local movement still goes through packed scratch. This keeps overlap
      // behavior identical for local and remote steps and avoids a separate
      // in-place rectangle-copy path.
      se::DeviceAddressBase local_source =
          batch.kind == Native2DStepKind::kRestoreScratch ? saved_slot
                                                          : send_slot;
      JAXMG_RETURN_IF_ERROR(UnpackRect(
          cuda_stream, local_rows, recv_step->target.row_start,
          recv_step->target.col_start,
          recv_step->target.row_count, recv_step->target.col_count,
          element_bytes, local_source, matrix_out_base));
      continue;
    }

    const uint64_t collective_bytes =
        static_cast<uint64_t>(collective_elements) * element_bytes;
    JAXMG_RETURN_IF_ERROR(RunRawNcclSendRecv(
        "xla_rect_native_2d", stream, comm_stream, cuda_stream, comm,
        rank_value, num_ranks, collective_send_slot, recv_slot,
        collective_bytes, source_rank, absl::MakeConstSpan(target_ranks)));

    if (recv_step != nullptr) {
      JAXMG_RETURN_IF_ERROR(UnpackRect(
          cuda_stream, local_rows, recv_step->target.row_start,
          recv_step->target.col_start,
          recv_step->target.row_count, recv_step->target.col_count,
          element_bytes, recv_slot, matrix_out_base));
    }
  }

  return absl::OkStatus();
}

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
