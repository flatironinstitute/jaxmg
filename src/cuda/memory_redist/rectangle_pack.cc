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
// The public cuSOLVERMp wrappers enter native code with ordinary row-major JAX
// local shards. Before those shards reach the rectangle packer, the fused
// solver handlers call the local layout-conversion helpers in this file. Those
// helpers call the CUDA layout-conversion launcher linked into the same backend
// shared library and convert the donated matrix buffers in-place using the same
// bounded scratch allocation used for redistribution. After the solver and
// reverse redistribution, the returned user-visible matrix buffers are
// converted back to row-major JAX storage.
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
// The public cuSOLVERMp path calls this file through the higher-level phase
// files rather than constructing rectangle moves directly. Keeping that
// boundary explicit makes the transport code reusable without exposing
// standalone validation FFI targets in the production backend.

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include "../include/xla_comm_backend.h"
#include "layout_convert.h"
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
  // XLA owns communicator creation and lifetime.  This backend only borrows the
  // CUDA platform handle for the duration of the FFI call and validates it with
  // NCCL metadata before issuing raw sends/receives.
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
  // Prefer XLA's communication stream when it is materialized, because that is
  // the stream XLA expects collective work to use. Some contexts expose only
  // the platform CUDA stream; the fallback keeps validation paths usable there.
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
    // Preserve stream order without a host-wide device synchronization:
    // producer work on the main stream completes before NCCL starts.
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
    // Make later main-stream CUDA work wait for the NCCL send/recv completion.
    absl::Status status = stream->WaitFor(comm_stream);
    if (!status.ok()) {
      return status;
    }
  }
  return absl::OkStatus();
}

struct RectCopySpec {
  // cudaMemcpy2DAsync arguments for a column-major rectangle.  `matrix_pitch`
  // is the byte stride between adjacent local columns in the full matrix;
  // `packed_pitch` is the byte width of the compact scratch rectangle.
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
  // Local matrices are column-major at this point.  A logical rectangle
  // therefore starts at col_start * local_rows + row_start and can be packed as
  // `col_count` 2D rows, each containing `row_count` contiguous elements.
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
  // Production solvers ask XLA to alias work/output buffers where possible.
  // This helper makes the fallback explicit: if XLA had to allocate a distinct
  // output, copy the input once before the in-place native pipeline begins.
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

absl::Status ConvertRowMajorToColumnMajorInPlace(
    cudaStream_t cuda_stream, const char* caller, ffi::AnyBuffer matrix,
    se::DeviceAddressBase scratch_base, int64_t scratch_elements) {
  if (cuda_stream == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s requires a CUDA stream", caller));
  }
  if (matrix.dimensions().size() != 2) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s expects a rank-2 matrix buffer", caller));
  }
  if (matrix.element_count() == 0) {
    return absl::OkStatus();
  }

  const int64_t rows = matrix.dimensions()[0];
  const int64_t cols = matrix.dimensions()[1];
  const int64_t required_scratch_elements = std::max(rows, cols);
  if (scratch_elements < required_scratch_elements) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s scratch length %d is smaller than the row-major -> column-major "
        "conversion requirement max(rows, cols) = %d",
        caller, scratch_elements, required_scratch_elements));
  }

  const size_t element_bytes =
      matrix.size_bytes() / static_cast<size_t>(matrix.element_count());
  if (element_bytes != 4 && element_bytes != 8 && element_bytes != 16) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s supports only 4, 8, and 16 byte matrix elements for native layout "
        "conversion; got %d bytes",
        caller, element_bytes));
  }

  // JAX-facing FFI buffers arrive in row-major local storage, while
  // cuSOLVERMp's local descriptors interpret the same logical matrix in
  // column-major storage. The helper changes only the physical address order
  // inside the donated local shard. It is byte-preserving: float64 and
  // complex64 are both 8-byte payloads, and complex128 is a 16-byte payload.
  // All dtype semantics remain with JAX/cuSOLVERMp.
  JAXMG_RETURN_IF_CUDA_ERROR(::JaxmgLaunchRowMajorToColumnMajorDecomposition(
      cuda_stream, matrix.untyped_data(), scratch_base.opaque(), rows, cols,
      scratch_elements, static_cast<int64_t>(element_bytes)));
  return absl::OkStatus();
}

absl::Status ConvertColumnMajorToRowMajorInPlace(
    cudaStream_t cuda_stream, const char* caller, ffi::AnyBuffer matrix,
    se::DeviceAddressBase scratch_base, int64_t scratch_elements) {
  if (cuda_stream == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s requires a CUDA stream", caller));
  }
  if (matrix.dimensions().size() != 2) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s expects a rank-2 matrix buffer", caller));
  }
  if (matrix.element_count() == 0) {
    return absl::OkStatus();
  }

  const int64_t rows = matrix.dimensions()[0];
  const int64_t cols = matrix.dimensions()[1];
  const int64_t required_scratch_elements = std::max(rows, cols);
  if (scratch_elements < required_scratch_elements) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s scratch length %d is smaller than the column-major -> row-major "
        "conversion requirement max(rows, cols) = %d",
        caller, scratch_elements, required_scratch_elements));
  }

  const size_t element_bytes =
      matrix.size_bytes() / static_cast<size_t>(matrix.element_count());
  if (element_bytes != 4 && element_bytes != 8 && element_bytes != 16) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s supports only 4, 8, and 16 byte matrix elements for native layout "
        "conversion; got %d bytes",
        caller, element_bytes));
  }

  // Column-major storage for an (rows, cols) logical matrix is exactly
  // row-major storage for the transposed (cols, rows) address grid. Applying
  // the row-major -> column-major decomposition to that transposed address
  // grid is therefore the inverse permutation and restores JAX's row-major
  // physical order without changing the logical matrix.
  JAXMG_RETURN_IF_CUDA_ERROR(::JaxmgLaunchRowMajorToColumnMajorDecomposition(
      cuda_stream, matrix.untyped_data(), scratch_base.opaque(), cols, rows,
      scratch_elements, static_cast<int64_t>(element_bytes)));
  return absl::OkStatus();
}

absl::Status PackRect(cudaStream_t cuda_stream, int64_t local_rows,
                      int64_t row_start, int64_t col_start,
                      int64_t row_count, int64_t col_count,
                      size_t element_bytes, se::DeviceAddressBase matrix_base,
                      se::DeviceAddressBase packed_base) {
  // Pack a logical column-major rectangle into contiguous scratch so NCCL sees
  // one linear payload even when the source rectangle is strided in the local
  // matrix.
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
  // Inverse of PackRect: write a contiguous scratch payload into a strided
  // column-major destination rectangle.
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
  // The cyclic scheduler reserves three equal-size slots:
  //   saved_slot: protects one live slab while rotating a closed cycle;
  //   send_slot:  packed outgoing payload for this rank;
  //   recv_slot:  incoming payload for this rank.
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
      // Always pack before moving, even for local moves.  That keeps the local
      // and remote code paths identical and avoids in-place overlap hazards.
      JAXMG_RETURN_IF_ERROR(PackRect(
          cuda_stream, local_rows, send_step->source.row_start,
          send_step->source.col_start,
          send_step->source.row_count, send_step->source.col_count,
          element_bytes, matrix_out_base, send_slot));
    }

    // A batch is conflict-free, so each rank participates in at most one send
    // and one receive.  Raw NCCL send/recv covers remote traffic; the special
    // same-rank case above handles local copies without entering NCCL.
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
        "jaxmg_native_2d_redistribution", stream, comm_stream, cuda_stream, comm,
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

absl::Status ExecuteEdgePaddingBatches(
    se::Stream* stream, se::Stream* comm_stream, cudaStream_t cuda_stream,
    const std::vector<Native2DStepBatch>& batches, int64_t local_rows,
    int64_t local_cols, int64_t rank_value, int64_t num_ranks,
    size_t element_bytes, int64_t scratch_elements, ffi::AnyBuffer /*matrix*/,
    se::DeviceAddressBase matrix_out_base,
    se::DeviceAddressBase scratch_base, GpuCommunicator* comm) {
  // Edge-padding compaction is an open-chain shift, not a closed cycle.  The
  // edge-padding planner emits one dependency wave at a time across independent
  // process rows/columns; within such a wave, a rank must be either a source, a
  // target, or idle.  That invariant lets the whole scratch allocation be used
  // as one temporary payload instead of the saved/send/recv slot layout used by
  // the cyclic block redistribution.
  for (const Native2DStepBatch& batch : batches) {
    if (batch.kind != Native2DStepKind::kMove) {
      return absl::InvalidArgumentError(
          "edge-padding executor expects only open-chain move batches");
    }

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

    const int64_t payload_elements =
        send_step != nullptr
            ? send_step->source.row_count * send_step->source.col_count
            : recv_step->target.row_count * recv_step->target.col_count;
    if (payload_elements > scratch_elements) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "edge-padding payload %d elements exceeds scratch budget %d",
          payload_elements, scratch_elements));
    }

    if (send_step != nullptr) {
      // Edge-padding uses the entire scratch allocation as one payload, so a
      // large open-chain shift can be chunked by the planner without the
      // saved/send/recv subdivision needed by cyclic movement.
      JAXMG_RETURN_IF_ERROR(PackRect(
          cuda_stream, local_rows, send_step->source.row_start,
          send_step->source.col_start, send_step->source.row_count,
          send_step->source.col_count, element_bytes, matrix_out_base,
          scratch_base));
    }

    if (send_step != nullptr && recv_step != nullptr &&
        send_step->source_rank == send_step->target_rank) {
      JAXMG_RETURN_IF_ERROR(UnpackRect(
          cuda_stream, local_rows, recv_step->target.row_start,
          recv_step->target.col_start, recv_step->target.row_count,
          recv_step->target.col_count, element_bytes, scratch_base,
          matrix_out_base));
      continue;
    }

    if (send_step != nullptr && recv_step != nullptr) {
      return absl::InternalError(
          "edge-padding batch scheduled one rank as both remote source and "
          "remote target; padding batches must be split before execution");
    }

    std::optional<RankId> source_rank;
    if (recv_step != nullptr) {
      source_rank = RankId(recv_step->source_rank);
    }
    std::vector<RankId> target_ranks;
    if (send_step != nullptr && send_step->target_rank >= 0) {
      target_ranks.push_back(RankId(send_step->target_rank));
    }

    const uint64_t collective_bytes =
        static_cast<uint64_t>(payload_elements) * element_bytes;
    JAXMG_RETURN_IF_ERROR(RunRawNcclSendRecv(
        "jaxmg_edge_padding_alignment", stream, comm_stream, cuda_stream, comm,
        rank_value, num_ranks, scratch_base, scratch_base, collective_bytes,
        source_rank, absl::MakeConstSpan(target_ranks)));

    if (recv_step != nullptr) {
      JAXMG_RETURN_IF_ERROR(UnpackRect(
          cuda_stream, local_rows, recv_step->target.row_start,
          recv_step->target.col_start, recv_step->target.row_count,
          recv_step->target.col_count, element_bytes, scratch_base,
          matrix_out_base));
    }
  }

  return absl::OkStatus();
}

}  // namespace xla::gpu
