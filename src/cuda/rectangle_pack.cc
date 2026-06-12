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
// Rectangle movement and native 2D block-cyclic redistribution.
//
// This file owns the GPU data motion needed before and after cuSOLVERMp.  It is
// intentionally separate from cusolvermp_probe.cc: rectangle_pack.cc moves bytes
// between JAX-facing block-sharded buffers and cuSOLVERMp-facing local
// 2D-block-cyclic buffers, while cusolvermp_probe.cc creates descriptors and
// calls cuSOLVERMp routines on the resulting layout.
//
// File workflow:
//   1. Borrow the raw NCCL communicator from XLA's GpuCommunicator and validate
//      that the NCCL rank/count match the XLA collective rank/count.
//   2. Pack rectangular column-major local matrix regions into bounded scratch
//      using cudaMemcpy2DAsync.
//   3. Move packed payloads with grouped NCCL send/recv calls on an XLA-ordered
//      stream.
//   4. Unpack received payloads into their destination rectangles.
//   5. Build native edge-padding compaction schedules so local shard padding is
//      converted into global right/bottom edge padding.
//   6. Build native column-owner then row-owner slab cycles for the cuSOLVERMp
//      2D block-cyclic ownership rule.
//   7. Execute the forward and reverse padded redistribution in one FFI call
//      per matrix, using one saved slab, one send slab, and one receive slab per
//      rank.
//
// The lower-level pack/unpack and rectangle-transfer handlers remain in this
// file as diagnostics.  The public cuSOLVERMp path uses the fused padded native
// plan handlers, which plan and execute the complete schedule in C++.

#include <algorithm>
#include <functional>
#include <limits>
#include <map>
#include <tuple>
#include <vector>

#include "include/xla_comm_backend.h"
#include "third_party/nccl/nccl.h"

namespace xla::gpu {

namespace {

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

absl::Status ValidateRankMap(const char* caller,
                             absl::Span<const int64_t> rank_map,
                             int64_t num_ranks) {
  if (rank_map.size() != static_cast<size_t>(num_ranks)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s expected rank_map length %d, got %d", caller, num_ranks,
        rank_map.size()));
  }
  std::vector<bool> seen(num_ranks, false);
  for (int64_t process_rank = 0; process_rank < num_ranks; ++process_rank) {
    const int64_t communicator_rank = rank_map[process_rank];
    if (communicator_rank < 0 || communicator_rank >= num_ranks) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "%s rank_map[%d]=%d is outside [0, %d)", caller, process_rank,
          communicator_rank, num_ranks));
    }
    if (seen[communicator_rank]) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "%s rank_map contains duplicate communicator rank %d", caller,
          communicator_rank));
    }
    seen[communicator_rank] = true;
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

struct NativeLocalRect {
  int64_t row_start;
  int64_t col_start;
  int64_t row_count;
  int64_t col_count;
};

enum class Native2DStepKind : int64_t {
  kMove = 0,
  kSaveScratch = 1,
  kRestoreScratch = 2,
};

struct Native2DStep {
  int64_t phase;
  int64_t sequence;
  Native2DStepKind kind;
  int64_t source_rank;
  int64_t target_rank;
  NativeLocalRect source;
  NativeLocalRect target;
};

struct Native2DStepBatch {
  int64_t phase;
  Native2DStepKind kind;
  std::vector<Native2DStep> steps;
};

absl::Status CheckNativeLocalRect(const NativeLocalRect& rect,
                                  int64_t local_rows, int64_t local_cols) {
  if (rect.row_start < 0 || rect.col_start < 0 || rect.row_count <= 0 ||
      rect.col_count <= 0 || rect.row_start + rect.row_count > local_rows ||
      rect.col_start + rect.col_count > local_cols) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "native 2D redistribution generated out-of-bounds local rectangle "
        "(%d:%d, %d:%d) for local shape (%d, %d)",
        rect.row_start, rect.row_start + rect.row_count, rect.col_start,
        rect.col_start + rect.col_count, local_rows, local_cols));
  }
  return absl::OkStatus();
}

// The optimized native redistribution is the 2D analogue of JAXMg's 1D
// cyclic reshuffler. It does not move individual MB_A x NB_A tiles. Instead it
// performs two coarser, separable permutations:
//
//   phase 0: within every process row, cyclically permute column slabs of
//            shape local_rows x tile_cols into their target process column.
//   phase 1: within every process column, cyclically permute row slabs of
//            shape tile_rows x local_cols into their target process row.
//
// Each phase is still an in-place permutation, so we decompose it into the same
// closed cycles used by the 1D code. Closed cycles save one live slab in
// scratch, rotate the remaining slabs tail-to-head, then restore the saved
// slab. The important low-memory invariant is the same as the original 1D
// JAXMg reshuffler: within one process row/column group, only one tile slab is
// in flight at a time. Different cycles in that same group are therefore
// serialized by assigning increasing dependency sequence numbers. Parallelism
// comes only from applying the same sequence step across independent process
// rows or columns. For example, one global tile-column move is represented as
// matching local column-slab moves in every process row, and those
// same-sequence moves are batched into one raw-NCCL send/recv round when their
// ranks do not conflict.

NativeLocalRect ColumnSlabRect(int64_t local_rows, int64_t tile_cols,
                               int64_t local_col_block) {
  return NativeLocalRect{/*row_start=*/0,
                         /*col_start=*/local_col_block * tile_cols,
                         /*row_count=*/local_rows,
                         /*col_count=*/tile_cols};
}

NativeLocalRect RowSlabRect(int64_t tile_rows, int64_t local_cols,
                            int64_t local_row_block) {
  return NativeLocalRect{/*row_start=*/local_row_block * tile_rows,
                         /*col_start=*/0,
                         /*row_count=*/tile_rows,
                         /*col_count=*/local_cols};
}

struct Native2DSlot {
  int64_t rank;
  NativeLocalRect rect;
};

Native2DSlot ColumnPhaseSlotToRankLocal(int64_t slot, int64_t process_cols,
                                        int64_t col_blocks_per_rank,
                                        int64_t local_rows,
                                        int64_t tile_cols,
                                        absl::Span<const int64_t> rank_map) {
  const int64_t slots_per_process_row = process_cols * col_blocks_per_rank;
  const int64_t process_row = slot / slots_per_process_row;
  const int64_t row_slot = slot % slots_per_process_row;
  const int64_t process_col = row_slot / col_blocks_per_rank;
  const int64_t local_col_block = row_slot % col_blocks_per_rank;
  const int64_t process_rank = process_row * process_cols + process_col;
  return Native2DSlot{
      /*rank=*/rank_map[process_rank],
      /*rect=*/ColumnSlabRect(local_rows, tile_cols, local_col_block),
  };
}

Native2DSlot RowPhaseSlotToRankLocal(int64_t slot, int64_t process_rows,
                                     int64_t process_cols,
                                     int64_t row_blocks_per_rank,
                                     int64_t tile_rows,
                                     int64_t local_cols,
                                     absl::Span<const int64_t> rank_map) {
  const int64_t slots_per_process_col = process_rows * row_blocks_per_rank;
  const int64_t process_col = slot / slots_per_process_col;
  const int64_t col_slot = slot % slots_per_process_col;
  const int64_t process_row = col_slot / row_blocks_per_rank;
  const int64_t local_row_block = col_slot % row_blocks_per_rank;
  const int64_t process_rank = process_row * process_cols + process_col;
  return Native2DSlot{
      /*rank=*/rank_map[process_rank],
      /*rect=*/RowSlabRect(tile_rows, local_cols, local_row_block),
  };
}

absl::StatusOr<std::vector<int64_t>> BuildColumnSlabSlotMap(
    int64_t process_rows, int64_t process_cols, int64_t col_blocks_per_rank) {
  const int64_t slots_per_process_row = process_cols * col_blocks_per_rank;
  std::vector<int64_t> target_for_source(process_rows * slots_per_process_row,
                                         -1);

  for (int64_t process_row = 0; process_row < process_rows; ++process_row) {
    const int64_t row_base = process_row * slots_per_process_row;
    for (int64_t global_tile_col = 0; global_tile_col < slots_per_process_row;
         ++global_tile_col) {
      const int64_t source_process_col =
          global_tile_col / col_blocks_per_rank;
      const int64_t source_local_col_block =
          global_tile_col % col_blocks_per_rank;
      const int64_t target_process_col = global_tile_col % process_cols;
      const int64_t target_local_col_block =
          global_tile_col / process_cols;

      const int64_t source_slot =
          row_base + source_process_col * col_blocks_per_rank +
          source_local_col_block;
      const int64_t target_slot =
          row_base + target_process_col * col_blocks_per_rank +
          target_local_col_block;
      target_for_source[source_slot] = target_slot;
    }
  }

  return target_for_source;
}

absl::StatusOr<std::vector<int64_t>> BuildRowSlabSlotMap(
    int64_t process_rows, int64_t process_cols, int64_t row_blocks_per_rank) {
  const int64_t slots_per_process_col = process_rows * row_blocks_per_rank;
  std::vector<int64_t> target_for_source(process_cols * slots_per_process_col,
                                         -1);

  for (int64_t process_col = 0; process_col < process_cols; ++process_col) {
    const int64_t col_base = process_col * slots_per_process_col;
    for (int64_t global_tile_row = 0; global_tile_row < slots_per_process_col;
         ++global_tile_row) {
      const int64_t source_process_row =
          global_tile_row / row_blocks_per_rank;
      const int64_t source_local_row_block =
          global_tile_row % row_blocks_per_rank;
      const int64_t target_process_row = global_tile_row % process_rows;
      const int64_t target_local_row_block =
          global_tile_row / process_rows;

      const int64_t source_slot =
          col_base + source_process_row * row_blocks_per_rank +
          source_local_row_block;
      const int64_t target_slot =
          col_base + target_process_row * row_blocks_per_rank +
          target_local_row_block;
      target_for_source[source_slot] = target_slot;
    }
  }

  return target_for_source;
}

absl::StatusOr<std::vector<int64_t>> InvertSlotMap(
    absl::Span<const int64_t> target_for_source) {
  std::vector<int64_t> inverse(target_for_source.size(), -1);
  for (int64_t source = 0;
       source < static_cast<int64_t>(target_for_source.size()); ++source) {
    const int64_t target = target_for_source[source];
    if (target < 0 ||
        target >= static_cast<int64_t>(target_for_source.size())) {
      return absl::InternalError(absl::StrFormat(
          "native 2D redistribution cannot invert invalid target slot %d",
          target));
    }
    if (inverse[target] >= 0) {
      return absl::InternalError(absl::StrFormat(
          "native 2D redistribution cannot invert non-bijective slot map: "
          "target slot %d has sources %d and %d",
          target, inverse[target], source));
    }
    inverse[target] = source;
  }
  for (int64_t slot = 0; slot < static_cast<int64_t>(inverse.size()); ++slot) {
    if (inverse[slot] < 0) {
      return absl::InternalError(absl::StrFormat(
          "native 2D redistribution cannot invert slot map with missing "
          "target slot %d",
          slot));
    }
  }
  return inverse;
}

absl::StatusOr<std::map<int64_t, std::vector<int64_t>>> BuildNative2DCycles(
    absl::Span<const int64_t> target_for_source) {
  std::vector<uint8_t> visited(target_for_source.size(), 0);
  std::map<int64_t, std::vector<int64_t>> cycles;

  for (int64_t key = 0; key < static_cast<int64_t>(target_for_source.size());
       ++key) {
    int64_t target = target_for_source[key];
    if (target < 0 || visited[key]) {
      continue;
    }
    if (target == key) {
      visited[key] = 1;
      continue;
    }

    std::vector<int64_t> cycle = {key};
    visited[key] = 1;
    while (true) {
      if (target < 0 ||
          target >= static_cast<int64_t>(target_for_source.size())) {
        return absl::InternalError(absl::StrFormat(
            "native 2D redistribution reached invalid target slot %d",
            target));
      }
      const int64_t next_target = target_for_source[target];
      if (next_target < 0) {
        cycle.push_back(target);
        break;
      }

      const bool dst_visited = visited[target] != 0;
      if (next_target == key) {
        cycle.push_back(target);
        visited[target] = 1;
        cycle.push_back(next_target);
        break;
      }
      if (dst_visited) {
        auto prior = cycles.find(target);
        if (prior != cycles.end()) {
          cycle.insert(cycle.end(), prior->second.begin(),
                       prior->second.end());
          cycles.erase(prior);
        } else {
          cycle.push_back(target);
        }
        break;
      }

      cycle.push_back(target);
      visited[target] = 1;
      target = next_target;
    }

    if (cycle.size() > 1) {
      cycles.emplace(key, std::move(cycle));
    }
  }

  return cycles;
}

using SlotDecoder = std::function<Native2DSlot(int64_t)>;

using Native2DSlotKey =
    std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t>;

Native2DSlotKey PhysicalSlotKey(const Native2DSlot& slot) {
  return Native2DSlotKey{slot.rank, slot.rect.row_start, slot.rect.col_start,
                         slot.rect.row_count, slot.rect.col_count};
}

bool RankMapsEqual(absl::Span<const int64_t> lhs,
                   absl::Span<const int64_t> rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (size_t i = 0; i < lhs.size(); ++i) {
    if (lhs[i] != rhs[i]) {
      return false;
    }
  }
  return true;
}

absl::StatusOr<std::vector<int64_t>> BuildPhysicalSlotMap(
    absl::Span<const int64_t> logical_target_for_source,
    const SlotDecoder& source_decode_slot,
    const SlotDecoder& target_decode_slot) {
  std::map<Native2DSlotKey, int64_t> source_slot_by_physical_key;
  for (int64_t slot = 0;
       slot < static_cast<int64_t>(logical_target_for_source.size());
       ++slot) {
    const Native2DSlot source = source_decode_slot(slot);
    const Native2DSlotKey key = PhysicalSlotKey(source);
    const auto [_, inserted] = source_slot_by_physical_key.emplace(key, slot);
    if (!inserted) {
      return absl::InternalError(
          "native 2D redistribution generated duplicate source physical "
          "slots while applying rank maps");
    }
  }

  std::vector<int64_t> physical_target_for_source(
      logical_target_for_source.size(), -1);
  for (int64_t source_slot = 0;
       source_slot < static_cast<int64_t>(logical_target_for_source.size());
       ++source_slot) {
    const int64_t logical_target = logical_target_for_source[source_slot];
    if (logical_target < 0 ||
        logical_target >=
            static_cast<int64_t>(logical_target_for_source.size())) {
      return absl::InternalError(absl::StrFormat(
          "native 2D redistribution generated invalid logical target slot %d",
          logical_target));
    }

    const Native2DSlot target = target_decode_slot(logical_target);
    const Native2DSlotKey target_key = PhysicalSlotKey(target);
    const auto source_physical_slot =
        source_slot_by_physical_key.find(target_key);
    if (source_physical_slot == source_slot_by_physical_key.end()) {
      return absl::InternalError(
          "native 2D redistribution could not map target physical slot back "
          "to a source physical slot");
    }
    physical_target_for_source[source_slot] = source_physical_slot->second;
  }

  return physical_target_for_source;
}

absl::StatusOr<int64_t> AppendCycleSteps(int64_t phase,
                                         const SlotDecoder& decode_slot,
                                         const std::vector<int64_t>& slots,
                                         int64_t sequence_offset,
                                         std::vector<Native2DStep>* steps) {
  const bool is_closed = slots.size() > 1 && slots.front() == slots.back();
  if (is_closed) {
    const Native2DSlot saved = decode_slot(slots[slots.size() - 2]);
    steps->push_back(Native2DStep{phase, sequence_offset,
                                  Native2DStepKind::kSaveScratch, saved.rank,
                                  -1, saved.rect,
                                  NativeLocalRect{0, 0, 0, 0}});

    int64_t sequence = 1;
    for (int64_t index = static_cast<int64_t>(slots.size()) - 3; index >= 0;
         --index, ++sequence) {
      const Native2DSlot source = decode_slot(slots[index]);
      const Native2DSlot target = decode_slot(slots[index + 1]);
      steps->push_back(Native2DStep{
          phase, sequence_offset + sequence, Native2DStepKind::kMove,
          source.rank, target.rank, source.rect, target.rect});
    }

    const Native2DSlot target = decode_slot(slots[0]);
    steps->push_back(Native2DStep{phase, sequence_offset + sequence,
                                  Native2DStepKind::kRestoreScratch,
                                  saved.rank, target.rank,
                                  NativeLocalRect{0, 0, 0, 0}, target.rect});
    return sequence + 1;
  }

  int64_t sequence = 0;
  for (int64_t index = static_cast<int64_t>(slots.size()) - 2; index >= 0;
       --index, ++sequence) {
    const Native2DSlot source = decode_slot(slots[index]);
    const Native2DSlot target = decode_slot(slots[index + 1]);
    steps->push_back(Native2DStep{
        phase, sequence_offset + sequence, Native2DStepKind::kMove,
        source.rank, target.rank, source.rect, target.rect});
  }
  return sequence;
}

absl::Status AppendSerialCycles(
    int64_t phase, const SlotDecoder& decode_slot,
    const std::map<int64_t, std::vector<int64_t>>& cycles,
    std::vector<Native2DStep>* steps) {
  int64_t sequence_offset = 0;
  for (const auto& [_, cycle] : cycles) {
    absl::StatusOr<int64_t> sequence_count =
        AppendCycleSteps(phase, decode_slot, cycle, sequence_offset, steps);
    if (!sequence_count.ok()) {
      return sequence_count.status();
    }
    sequence_offset += *sequence_count;
  }
  return absl::OkStatus();
}

using SlotGroup = std::function<int64_t(int64_t)>;

absl::Status AppendAxisGroupSerialCycles(
    int64_t phase, const SlotDecoder& decode_slot, const SlotGroup& slot_group,
    const std::map<int64_t, std::vector<int64_t>>& cycles,
    std::vector<Native2DStep>* steps) {
  std::map<int64_t, std::vector<std::vector<int64_t>>> cycles_by_axis_group;
  for (const auto& [_, cycle] : cycles) {
    if (cycle.empty()) {
      continue;
    }
    const int64_t group = slot_group(cycle.front());
    for (int64_t slot : cycle) {
      if (slot_group(slot) != group) {
        return absl::InternalError(absl::StrFormat(
            "native 2D redistribution cycle crosses axis groups: first group "
            "%d, slot %d is in group %d",
            group, slot, slot_group(slot)));
      }
    }
    cycles_by_axis_group[group].push_back(cycle);
  }

  for (const auto& [_, group_cycles] : cycles_by_axis_group) {
    int64_t sequence_offset = 0;
    for (const std::vector<int64_t>& cycle : group_cycles) {
      absl::StatusOr<int64_t> sequence_count =
          AppendCycleSteps(phase, decode_slot, cycle, sequence_offset, steps);
      if (!sequence_count.ok()) {
        return sequence_count.status();
      }
      sequence_offset += *sequence_count;
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<Native2DStep>> BuildSlabNative2DSteps(
    int64_t process_rows, int64_t process_cols, int64_t tile_rows,
    int64_t tile_cols, int64_t local_rows, int64_t local_cols,
    absl::Span<const int64_t> block_rank_map,
    absl::Span<const int64_t> cyclic_rank_map,
    bool reverse = false) {
  if (process_rows <= 0 || process_cols <= 0 || tile_rows <= 0 ||
      tile_cols <= 0) {
    return absl::InvalidArgumentError(
        "native 2D redistribution requires positive grid and tile sizes");
  }
  if (local_rows % tile_rows != 0 || local_cols % tile_cols != 0) {
    return absl::InvalidArgumentError(
        "native 2D redistribution currently requires tile-aligned local "
        "shards");
  }

  const int64_t row_blocks_per_rank = local_rows / tile_rows;
  const int64_t col_blocks_per_rank = local_cols / tile_cols;
  std::vector<Native2DStep> steps;

  auto append_column_phase = [&](int64_t phase,
                                 absl::Span<const int64_t> source_rank_map,
                                 absl::Span<const int64_t> target_rank_map,
                                 bool invert) -> absl::Status {
    absl::StatusOr<std::vector<int64_t>> slot_map =
        BuildColumnSlabSlotMap(process_rows, process_cols,
                               col_blocks_per_rank);
    if (!slot_map.ok()) {
      return slot_map.status();
    }
    if (invert) {
      absl::StatusOr<std::vector<int64_t>> inverse = InvertSlotMap(*slot_map);
      if (!inverse.ok()) {
        return inverse.status();
      }
      slot_map = std::move(*inverse);
    }
    absl::StatusOr<std::map<int64_t, std::vector<int64_t>>> cycles =
        BuildNative2DCycles(*slot_map);
    if (!cycles.ok()) {
      return cycles.status();
    }
    SlotDecoder source_decode_slot = [&](int64_t slot) {
      return ColumnPhaseSlotToRankLocal(
          slot, process_cols, col_blocks_per_rank, local_rows, tile_cols,
          source_rank_map);
    };
    SlotDecoder target_decode_slot = [&](int64_t slot) {
      return ColumnPhaseSlotToRankLocal(
          slot, process_cols, col_blocks_per_rank, local_rows, tile_cols,
          target_rank_map);
    };
    if (!RankMapsEqual(source_rank_map, target_rank_map)) {
      absl::StatusOr<std::vector<int64_t>> physical_slot_map =
          BuildPhysicalSlotMap(*slot_map, source_decode_slot,
                               target_decode_slot);
      if (!physical_slot_map.ok()) {
        return physical_slot_map.status();
      }
      cycles = BuildNative2DCycles(*physical_slot_map);
      if (!cycles.ok()) {
        return cycles.status();
      }
      return AppendSerialCycles(phase, source_decode_slot, *cycles, &steps);
    }
    SlotDecoder decode_slot = source_decode_slot;
    const int64_t slots_per_process_row =
        process_cols * col_blocks_per_rank;
    SlotGroup axis_group = [&](int64_t slot) {
      return slot / slots_per_process_row;
    };
    return AppendAxisGroupSerialCycles(phase, decode_slot, axis_group,
                                       *cycles, &steps);
  };

  auto append_row_phase = [&](int64_t phase,
                              absl::Span<const int64_t> source_rank_map,
                              absl::Span<const int64_t> target_rank_map,
                              bool invert) -> absl::Status {
    absl::StatusOr<std::vector<int64_t>> slot_map =
        BuildRowSlabSlotMap(process_rows, process_cols, row_blocks_per_rank);
    if (!slot_map.ok()) {
      return slot_map.status();
    }
    if (invert) {
      absl::StatusOr<std::vector<int64_t>> inverse = InvertSlotMap(*slot_map);
      if (!inverse.ok()) {
        return inverse.status();
      }
      slot_map = std::move(*inverse);
    }
    absl::StatusOr<std::map<int64_t, std::vector<int64_t>>> cycles =
        BuildNative2DCycles(*slot_map);
    if (!cycles.ok()) {
      return cycles.status();
    }
    SlotDecoder source_decode_slot = [&](int64_t slot) {
      return RowPhaseSlotToRankLocal(slot, process_rows, process_cols,
                                     row_blocks_per_rank, tile_rows,
                                     local_cols, source_rank_map);
    };
    SlotDecoder target_decode_slot = [&](int64_t slot) {
      return RowPhaseSlotToRankLocal(slot, process_rows, process_cols,
                                     row_blocks_per_rank, tile_rows,
                                     local_cols, target_rank_map);
    };
    if (!RankMapsEqual(source_rank_map, target_rank_map)) {
      absl::StatusOr<std::vector<int64_t>> physical_slot_map =
          BuildPhysicalSlotMap(*slot_map, source_decode_slot,
                               target_decode_slot);
      if (!physical_slot_map.ok()) {
        return physical_slot_map.status();
      }
      cycles = BuildNative2DCycles(*physical_slot_map);
      if (!cycles.ok()) {
        return cycles.status();
      }
      return AppendSerialCycles(phase, source_decode_slot, *cycles, &steps);
    }
    SlotDecoder decode_slot = source_decode_slot;
    const int64_t slots_per_process_col =
        process_rows * row_blocks_per_rank;
    SlotGroup axis_group = [&](int64_t slot) {
      return slot / slots_per_process_col;
    };
    return AppendAxisGroupSerialCycles(phase, decode_slot, axis_group,
                                       *cycles, &steps);
  };

  if (reverse) {
    // Forward redistribution applies column-owner movement before row-owner
    // movement. The inverse must undo those phases in the opposite order.
    JAXMG_RETURN_IF_ERROR(append_row_phase(/*phase=*/0, cyclic_rank_map,
                                           cyclic_rank_map, /*invert=*/true));
    JAXMG_RETURN_IF_ERROR(append_column_phase(
        /*phase=*/1, cyclic_rank_map, block_rank_map, /*invert=*/true));
  } else {
    JAXMG_RETURN_IF_ERROR(append_column_phase(
        /*phase=*/0, block_rank_map, cyclic_rank_map, /*invert=*/false));
    JAXMG_RETURN_IF_ERROR(append_row_phase(/*phase=*/1, cyclic_rank_map,
                                           cyclic_rank_map, /*invert=*/false));
  }

  return steps;
}

struct AxisEdgeMove {
  int64_t wave;
  int64_t source_start;
  int64_t target_start;
  int64_t extent;
};

int64_t AxisPadding(int64_t logical_per_block, int64_t tile_size) {
  const int64_t remainder = logical_per_block % tile_size;
  return remainder == 0 ? 0 : tile_size - remainder;
}

absl::StatusOr<std::vector<AxisEdgeMove>> BuildAxisEdgePaddingMoves(
    int64_t block_count, int64_t logical_per_block,
    int64_t physical_per_block) {
  if (block_count <= 0 || logical_per_block <= 0 ||
      physical_per_block <= 0) {
    return absl::InvalidArgumentError(
        "edge-padding compaction requires positive axis extents");
  }
  if (logical_per_block > physical_per_block) {
    return absl::InvalidArgumentError(
        "edge-padding compaction logical extent exceeds physical extent");
  }

  const int64_t total = block_count * physical_per_block;
  const int64_t logical_total = block_count * logical_per_block;
  std::vector<uint8_t> is_real(total, 0);
  for (int64_t block = 0; block < block_count; ++block) {
    const int64_t start = block * physical_per_block;
    for (int64_t offset = 0; offset < logical_per_block; ++offset) {
      is_real[start + offset] = 1;
    }
  }

  std::vector<AxisEdgeMove> moves;
  int64_t wave = 0;
  while (true) {
    int64_t target_start = -1;
    for (int64_t index = 0; index < logical_total; ++index) {
      if (!is_real[index]) {
        target_start = index;
        break;
      }
    }
    if (target_start < 0) {
      break;
    }

    int64_t target_stop = target_start;
    while (target_stop < total && !is_real[target_stop]) {
      ++target_stop;
    }

    int64_t source_start = -1;
    for (int64_t index = target_stop; index < total; ++index) {
      if (is_real[index]) {
        source_start = index;
        break;
      }
    }
    if (source_start < 0) {
      break;
    }

    int64_t source_stop = source_start;
    while (source_stop < total && is_real[source_stop]) {
      ++source_stop;
    }

    const int64_t target_block_stop =
        (target_start / physical_per_block + 1) * physical_per_block;
    const int64_t source_block_stop =
        (source_start / physical_per_block + 1) * physical_per_block;
    const int64_t extent =
        std::min({target_stop - target_start, source_stop - source_start,
                  target_block_stop - target_start,
                  source_block_stop - source_start});
    if (extent <= 0) {
      return absl::InternalError(
          "edge-padding compaction generated an empty move");
    }

    moves.push_back(AxisEdgeMove{wave, source_start, target_start, extent});
    for (int64_t offset = 0; offset < extent; ++offset) {
      if (is_real[target_start + offset] ||
          !is_real[source_start + offset]) {
        return absl::InternalError(
            "edge-padding compaction occupancy invariant failed");
      }
      is_real[target_start + offset] = 1;
      is_real[source_start + offset] = 0;
    }
    ++wave;
  }

  return moves;
}

absl::StatusOr<std::vector<Native2DStep>> BuildEdgePaddingNative2DSteps(
    int64_t process_rows, int64_t process_cols, int64_t tile_rows,
    int64_t tile_cols, int64_t logical_rows, int64_t logical_cols,
    int64_t local_rows, int64_t local_cols,
    absl::Span<const int64_t> block_rank_map) {
  if (process_rows <= 0 || process_cols <= 0 || tile_rows <= 0 ||
      tile_cols <= 0 || logical_rows <= 0 || logical_cols <= 0) {
    return absl::InvalidArgumentError(
        "padded native 2D redistribution requires positive dimensions");
  }
  if (logical_rows % process_rows != 0 ||
      logical_cols % process_cols != 0) {
    return absl::InvalidArgumentError(
        "logical matrix shape must divide evenly over the process grid");
  }

  const int64_t local_logical_rows = logical_rows / process_rows;
  const int64_t local_logical_cols = logical_cols / process_cols;
  const int64_t expected_local_rows =
      local_logical_rows + AxisPadding(local_logical_rows, tile_rows);
  const int64_t expected_local_cols =
      local_logical_cols + AxisPadding(local_logical_cols, tile_cols);
  if (local_rows != expected_local_rows || local_cols != expected_local_cols) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "padded native 2D redistribution expected local padded shape "
        "(%d, %d), got (%d, %d)",
        expected_local_rows, expected_local_cols, local_rows, local_cols));
  }

  std::vector<Native2DStep> steps;
  absl::StatusOr<std::vector<AxisEdgeMove>> horizontal_moves =
      BuildAxisEdgePaddingMoves(process_cols, local_logical_cols, local_cols);
  if (!horizontal_moves.ok()) {
    return horizontal_moves.status();
  }
  for (const AxisEdgeMove& move : *horizontal_moves) {
    const int64_t source_process_col = move.source_start / local_cols;
    const int64_t target_process_col = move.target_start / local_cols;
    const int64_t source_local_col = move.source_start % local_cols;
    const int64_t target_local_col = move.target_start % local_cols;
    for (int64_t process_row = 0; process_row < process_rows; ++process_row) {
      const int64_t source_process_rank =
          process_row * process_cols + source_process_col;
      const int64_t target_process_rank =
          process_row * process_cols + target_process_col;
      steps.push_back(Native2DStep{
          /*phase=*/0,
          /*sequence=*/move.wave,
          Native2DStepKind::kMove,
          /*source_rank=*/block_rank_map[source_process_rank],
          /*target_rank=*/block_rank_map[target_process_rank],
          NativeLocalRect{/*row_start=*/0,
                          /*col_start=*/source_local_col,
                          /*row_count=*/local_rows,
                          /*col_count=*/move.extent},
          NativeLocalRect{/*row_start=*/0,
                          /*col_start=*/target_local_col,
                          /*row_count=*/local_rows,
                          /*col_count=*/move.extent}});
    }
  }

  absl::StatusOr<std::vector<AxisEdgeMove>> vertical_moves =
      BuildAxisEdgePaddingMoves(process_rows, local_logical_rows, local_rows);
  if (!vertical_moves.ok()) {
    return vertical_moves.status();
  }
  for (const AxisEdgeMove& move : *vertical_moves) {
    const int64_t source_process_row = move.source_start / local_rows;
    const int64_t target_process_row = move.target_start / local_rows;
    const int64_t source_local_row = move.source_start % local_rows;
    const int64_t target_local_row = move.target_start % local_rows;
    for (int64_t process_col = 0; process_col < process_cols; ++process_col) {
      const int64_t source_process_rank =
          source_process_row * process_cols + process_col;
      const int64_t target_process_rank =
          target_process_row * process_cols + process_col;
      steps.push_back(Native2DStep{
          /*phase=*/1,
          /*sequence=*/move.wave,
          Native2DStepKind::kMove,
          /*source_rank=*/block_rank_map[source_process_rank],
          /*target_rank=*/block_rank_map[target_process_rank],
          NativeLocalRect{/*row_start=*/source_local_row,
                          /*col_start=*/0,
                          /*row_count=*/move.extent,
                          /*col_count=*/local_cols},
          NativeLocalRect{/*row_start=*/target_local_row,
                          /*col_start=*/0,
                          /*row_count=*/move.extent,
                          /*col_count=*/local_cols}});
    }
  }

  return steps;
}

std::vector<Native2DStep> ReverseEdgePaddingSteps(
    const std::vector<Native2DStep>& forward_steps) {
  // Edge-padding compaction is an open-chain movement: real slabs are pulled
  // into earlier padding holes and the old source bytes are left undefined.
  // The inverse used for solver output does not need to restore padding bytes;
  // it only needs to move the logical matrix entries back into the per-shard
  // block-sharded positions. Swapping source/target rectangles and reversing
  // the wave order gives that inverse while preserving the same bounded
  // scratch policy as the forward pass.
  int64_t max_horizontal_wave = 0;
  int64_t max_vertical_wave = 0;
  for (const Native2DStep& step : forward_steps) {
    if (step.phase == 0) {
      max_horizontal_wave = std::max(max_horizontal_wave, step.sequence);
    } else {
      max_vertical_wave = std::max(max_vertical_wave, step.sequence);
    }
  }

  std::vector<Native2DStep> reverse_steps;
  reverse_steps.reserve(forward_steps.size());
  for (const Native2DStep& step : forward_steps) {
    const bool was_horizontal = step.phase == 0;
    const int64_t reverse_phase = was_horizontal ? 1 : 0;
    const int64_t reverse_sequence =
        (was_horizontal ? max_horizontal_wave : max_vertical_wave) -
        step.sequence;
    reverse_steps.push_back(Native2DStep{
        reverse_phase,
        reverse_sequence,
        Native2DStepKind::kMove,
        step.target_rank,
        step.source_rank,
        step.target,
        step.source});
  }
  return reverse_steps;
}

std::vector<Native2DStepBatch> BatchNative2DSteps(
    const std::vector<Native2DStep>& steps) {
  // Group by phase, dependency sequence, and operation kind. This deliberately
  // batches only the same step of the same cycle shape across independent
  // process rows/columns. It does not combine different sequence numbers, since
  // those represent dependent moves inside a cycle.
  std::map<std::tuple<int64_t, int64_t, int64_t>, std::vector<Native2DStep>>
      steps_by_round;
  for (const Native2DStep& step : steps) {
    steps_by_round[{step.phase, step.sequence,
                    static_cast<int64_t>(step.kind)}]
        .push_back(step);
  }

  std::vector<Native2DStepBatch> batches;

  for (const auto& [key, round_steps] : steps_by_round) {
    const int64_t phase = std::get<0>(key);
    const Native2DStepKind kind =
        static_cast<Native2DStepKind>(std::get<2>(key));
    std::vector<Native2DStepBatch> round_batches;
    for (const Native2DStep& step : round_steps) {
      bool conflicts = false;
      for (Native2DStepBatch& batch : round_batches) {
        conflicts = false;
        for (const Native2DStep& existing : batch.steps) {
          if (step.kind == Native2DStepKind::kSaveScratch) {
            conflicts = conflicts || existing.source_rank == step.source_rank;
          } else {
            conflicts = conflicts || existing.source_rank == step.source_rank ||
                        existing.target_rank == step.target_rank;
          }
        }
        if (!conflicts) {
          batch.steps.push_back(step);
          break;
        }
      }
      if (conflicts || round_batches.empty()) {
        round_batches.push_back(Native2DStepBatch{phase, kind, {step}});
      }
    }
    batches.insert(batches.end(), round_batches.begin(), round_batches.end());
  }

  return batches;
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

absl::Status XlaRect2DNativePlanPrepare(
    const CollectiveParams* collective_params,
    CollectiveCliqueRequests* clique_requests) {
  if (collective_params == nullptr || clique_requests == nullptr) {
    return absl::InvalidArgumentError(
        "xla_rect_2d_native_plan requires XLA collective prepare contexts");
  }

  // cuSOLVERMp redistribution must be able to move tiles between every rank in
  // the distributed mesh. A node-scoped clique would only contain local GPUs
  // inside the current Python process, which is correct for the legacy
  // cuSolverMg path but insufficient for one-process-per-node multi-node jobs.
  absl::StatusOr<GpuCliqueKey> clique_key =
      AllAssignedDevicesP2PCliqueKey(*collective_params);
  if (!clique_key.ok()) {
    return clique_key.status();
  }

  absl::StatusOr<std::vector<GlobalDeviceId>> device_group =
      AllAssignedGlobalDeviceGroup(*collective_params);
  if (!device_group.ok()) {
    return device_group.status();
  }
  return clique_requests->RequestClique(*clique_key, {*device_group});
}

absl::Status Rect2DNativePlanDispatchImpl(
    se::Stream* stream, se::Stream* comm_stream, cudaStream_t cuda_stream,
    int64_t process_rows, int64_t process_cols, int64_t tile_rows,
    int64_t tile_cols, absl::Span<const int64_t> source_rank_map,
    absl::Span<const int64_t> target_rank_map, ffi::AnyBuffer matrix,
    ffi::AnyBuffer scratch, ffi::Result<ffi::AnyBuffer> matrix_out,
    ffi::Result<ffi::AnyBuffer> scratch_out,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques) {
  if (stream == nullptr || comm_stream == nullptr || cuda_stream == nullptr) {
    return absl::InvalidArgumentError(
        "xla_rect_2d_native_plan requires XLA and CUDA stream contexts");
  }
  if (collective_params == nullptr || collective_cliques == nullptr) {
    return absl::InvalidArgumentError(
        "xla_rect_2d_native_plan requires XLA collective contexts");
  }
  if (matrix.dimensions().size() != 2 ||
      matrix_out->dimensions().size() != 2 ||
      matrix.dimensions()[0] != matrix_out->dimensions()[0] ||
      matrix.dimensions()[1] != matrix_out->dimensions()[1]) {
    return absl::InvalidArgumentError(
        "xla_rect_2d_native_plan expects matching rank-2 matrix input/output");
  }
  if (scratch.dimensions().size() != 1 ||
      scratch_out->dimensions().size() != 1 ||
      scratch.dimensions()[0] != scratch_out->dimensions()[0]) {
    return absl::InvalidArgumentError(
        "xla_rect_2d_native_plan expects matching rank-1 scratch input/output");
  }
  if (matrix.element_type() != matrix_out->element_type() ||
      matrix.element_type() != scratch.element_type() ||
      scratch.element_type() != scratch_out->element_type()) {
    return absl::InvalidArgumentError(
        "xla_rect_2d_native_plan requires matrix and scratch dtypes to match");
  }

  absl::StatusOr<GpuCliqueKey> clique_key =
      AllAssignedDevicesP2PCliqueKey(*collective_params);
  if (!clique_key.ok()) {
    return clique_key.status();
  }
  const int64_t num_ranks = static_cast<int64_t>(clique_key->num_devices());
  if (process_rows * process_cols != num_ranks) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "xla_rect_2d_native_plan grid %d x %d does not match clique size %d",
        process_rows, process_cols, num_ranks));
  }
  JAXMG_RETURN_IF_ERROR(ValidateRankMap(
      "xla_rect_2d_native_plan source_rank_map", source_rank_map, num_ranks));
  JAXMG_RETURN_IF_ERROR(ValidateRankMap(
      "xla_rect_2d_native_plan target_rank_map", target_rank_map, num_ranks));

  std::optional<RankId> rank =
      clique_key->rank(collective_params->global_device_id);
  if (!rank.has_value()) {
    return absl::InvalidArgumentError(
        "xla_rect_2d_native_plan could not resolve this device rank");
  }
  const int64_t rank_value = static_cast<int64_t>(rank->value());

  absl::StatusOr<GpuCommunicator*> comm = collective_cliques->GetComm(
      *clique_key, collective_params->global_device_id);
  if (!comm.ok()) {
    return comm.status();
  }

  const int64_t local_rows = matrix.dimensions()[0];
  const int64_t local_cols = matrix.dimensions()[1];
  absl::StatusOr<std::vector<Native2DStep>> steps =
      BuildSlabNative2DSteps(process_rows, process_cols, tile_rows, tile_cols,
                             local_rows, local_cols, source_rank_map,
                             target_rank_map);
  if (!steps.ok()) {
    return steps.status();
  }
  std::vector<Native2DStepBatch> batches =
      BatchNative2DSteps(*steps);

  const int64_t slab_elements = MaxStepElementCount(*steps);
  if (scratch.dimensions()[0] < 3 * slab_elements) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "xla_rect_2d_native_plan scratch length %d is smaller than "
        "3 * max(native 2D step elements) = %d",
        scratch.dimensions()[0], 3 * slab_elements));
  }
  const size_t element_bytes =
      matrix.size_bytes() / static_cast<size_t>(matrix.element_count());

  se::DeviceAddressBase matrix_out_base = matrix_out->device_memory();
  se::DeviceAddressBase scratch_out_base = scratch_out->device_memory();

  JAXMG_RETURN_IF_ERROR(CopyMatrixIfNeeded(cuda_stream, matrix, matrix_out));
  JAXMG_RETURN_IF_ERROR(CopyScratchIfNeeded(cuda_stream, scratch, scratch_out));

  return ExecuteNative2DStepBatches(
      stream, comm_stream, cuda_stream, batches, local_rows, local_cols,
      rank_value, num_ranks, element_bytes, slab_elements, matrix,
      matrix_out_base, scratch_out_base, *comm);
}

absl::Status XlaRect2DNativePlanDispatch(
    se::Stream* stream, se::Stream* comm_stream, cudaStream_t cuda_stream,
    int64_t process_rows, int64_t process_cols, int64_t tile_rows,
    int64_t tile_cols, absl::Span<const int64_t> source_rank_map,
    absl::Span<const int64_t> target_rank_map, ffi::AnyBuffer matrix,
    ffi::AnyBuffer scratch, ffi::Result<ffi::AnyBuffer> matrix_out,
    ffi::Result<ffi::AnyBuffer> scratch_out,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques) {
  return Rect2DNativePlanDispatchImpl(
      stream, comm_stream, cuda_stream, process_rows, process_cols, tile_rows,
      tile_cols, source_rank_map, target_rank_map, matrix, scratch, matrix_out,
      scratch_out, collective_params, collective_cliques);
}

absl::Status XlaRectPadded2DNativePlanPrepare(
    const CollectiveParams* collective_params,
    CollectiveCliqueRequests* clique_requests) {
  return XlaRect2DNativePlanPrepare(collective_params, clique_requests);
}

absl::Status RectPadded2DNativePlanDispatchImpl(
    se::Stream* stream, se::Stream* comm_stream, cudaStream_t cuda_stream,
    int64_t process_rows, int64_t process_cols, int64_t tile_rows,
    int64_t tile_cols, int64_t logical_rows, int64_t logical_cols,
    int64_t reverse, absl::Span<const int64_t> block_rank_map,
    absl::Span<const int64_t> cyclic_rank_map, ffi::AnyBuffer matrix,
    ffi::AnyBuffer scratch,
    ffi::Result<ffi::AnyBuffer> matrix_out,
    ffi::Result<ffi::AnyBuffer> scratch_out,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques) {
  if (stream == nullptr || comm_stream == nullptr || cuda_stream == nullptr) {
    return absl::InvalidArgumentError(
        "xla_rect_padded_2d_native_plan requires XLA and CUDA stream contexts");
  }
  if (collective_params == nullptr || collective_cliques == nullptr) {
    return absl::InvalidArgumentError(
        "xla_rect_padded_2d_native_plan requires XLA collective contexts");
  }
  if (matrix.dimensions().size() != 2 ||
      matrix_out->dimensions().size() != 2 ||
      matrix.dimensions()[0] != matrix_out->dimensions()[0] ||
      matrix.dimensions()[1] != matrix_out->dimensions()[1]) {
    return absl::InvalidArgumentError(
        "xla_rect_padded_2d_native_plan expects matching rank-2 matrix "
        "input/output");
  }
  if (scratch.dimensions().size() != 1 ||
      scratch_out->dimensions().size() != 1 ||
      scratch.dimensions()[0] != scratch_out->dimensions()[0]) {
    return absl::InvalidArgumentError(
        "xla_rect_padded_2d_native_plan expects matching rank-1 scratch "
        "input/output");
  }
  if (matrix.element_type() != matrix_out->element_type() ||
      matrix.element_type() != scratch.element_type() ||
      scratch.element_type() != scratch_out->element_type()) {
    return absl::InvalidArgumentError(
        "xla_rect_padded_2d_native_plan requires matrix and scratch dtypes to "
        "match");
  }

  absl::StatusOr<GpuCliqueKey> clique_key =
      AllAssignedDevicesP2PCliqueKey(*collective_params);
  if (!clique_key.ok()) {
    return clique_key.status();
  }
  const int64_t num_ranks = static_cast<int64_t>(clique_key->num_devices());
  if (process_rows * process_cols != num_ranks) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "xla_rect_padded_2d_native_plan grid %d x %d does not match clique "
        "size %d",
        process_rows, process_cols, num_ranks));
  }
  JAXMG_RETURN_IF_ERROR(ValidateRankMap(
      "xla_rect_padded_2d_native_plan block_rank_map", block_rank_map,
      num_ranks));
  JAXMG_RETURN_IF_ERROR(ValidateRankMap(
      "xla_rect_padded_2d_native_plan cyclic_rank_map", cyclic_rank_map,
      num_ranks));

  std::optional<RankId> rank =
      clique_key->rank(collective_params->global_device_id);
  if (!rank.has_value()) {
    return absl::InvalidArgumentError(
        "xla_rect_padded_2d_native_plan could not resolve this device rank");
  }
  const int64_t rank_value = static_cast<int64_t>(rank->value());

  absl::StatusOr<GpuCommunicator*> comm = collective_cliques->GetComm(
      *clique_key, collective_params->global_device_id);
  if (!comm.ok()) {
    return comm.status();
  }

  const int64_t local_rows = matrix.dimensions()[0];
  const int64_t local_cols = matrix.dimensions()[1];
  absl::StatusOr<std::vector<Native2DStep>> edge_steps =
      BuildEdgePaddingNative2DSteps(
          process_rows, process_cols, tile_rows, tile_cols, logical_rows,
          logical_cols, local_rows, local_cols, block_rank_map);
  if (!edge_steps.ok()) {
    return edge_steps.status();
  }
  absl::StatusOr<std::vector<Native2DStep>> slab_steps =
      BuildSlabNative2DSteps(process_rows, process_cols, tile_rows, tile_cols,
                             local_rows, local_cols, block_rank_map,
                             cyclic_rank_map, reverse != 0);
  if (!slab_steps.ok()) {
    return slab_steps.status();
  }

  std::vector<Native2DStep> reverse_edge_steps;
  if (reverse != 0) {
    reverse_edge_steps = ReverseEdgePaddingSteps(*edge_steps);
  }

  std::vector<Native2DStepBatch> edge_batches =
      BatchNative2DSteps(reverse != 0 ? reverse_edge_steps : *edge_steps);
  std::vector<Native2DStepBatch> slab_batches =
      BatchNative2DSteps(*slab_steps);
  const int64_t max_step_elements =
      std::max(MaxStepElementCount(*edge_steps),
               MaxStepElementCount(*slab_steps));
  if (max_step_elements > 0 && scratch.dimensions()[0] < 3 * max_step_elements) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "xla_rect_padded_2d_native_plan scratch length %d is smaller than "
        "3 * max(native step elements) = %d",
        scratch.dimensions()[0], 3 * max_step_elements));
  }

  const size_t element_bytes =
      matrix.size_bytes() / static_cast<size_t>(matrix.element_count());
  se::DeviceAddressBase matrix_out_base = matrix_out->device_memory();
  se::DeviceAddressBase scratch_out_base = scratch_out->device_memory();

  JAXMG_RETURN_IF_ERROR(CopyMatrixIfNeeded(cuda_stream, matrix, matrix_out));
  JAXMG_RETURN_IF_ERROR(CopyScratchIfNeeded(cuda_stream, scratch, scratch_out));
  if (max_step_elements == 0) {
    return absl::OkStatus();
  }

  if (reverse != 0) {
    JAXMG_RETURN_IF_ERROR(ExecuteNative2DStepBatches(
        stream, comm_stream, cuda_stream, slab_batches, local_rows, local_cols,
        rank_value, num_ranks, element_bytes, max_step_elements, matrix,
        matrix_out_base, scratch_out_base, *comm));
    return ExecuteNative2DStepBatches(
        stream, comm_stream, cuda_stream, edge_batches, local_rows, local_cols,
        rank_value, num_ranks, element_bytes, max_step_elements, matrix,
        matrix_out_base, scratch_out_base, *comm);
  }

  JAXMG_RETURN_IF_ERROR(ExecuteNative2DStepBatches(
      stream, comm_stream, cuda_stream, edge_batches, local_rows, local_cols,
      rank_value, num_ranks, element_bytes, max_step_elements, matrix,
      matrix_out_base, scratch_out_base, *comm));
  return ExecuteNative2DStepBatches(
      stream, comm_stream, cuda_stream, slab_batches, local_rows, local_cols,
      rank_value, num_ranks, element_bytes, max_step_elements, matrix,
      matrix_out_base, scratch_out_base, *comm);
}

absl::Status XlaRectPadded2DNativePlanDispatch(
    se::Stream* stream, se::Stream* comm_stream, cudaStream_t cuda_stream,
    int64_t process_rows, int64_t process_cols, int64_t tile_rows,
    int64_t tile_cols, int64_t logical_rows, int64_t logical_cols,
    int64_t reverse, absl::Span<const int64_t> block_rank_map,
    absl::Span<const int64_t> cyclic_rank_map, ffi::AnyBuffer matrix,
    ffi::AnyBuffer scratch,
    ffi::Result<ffi::AnyBuffer> matrix_out,
    ffi::Result<ffi::AnyBuffer> scratch_out,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques) {
  return RectPadded2DNativePlanDispatchImpl(
      stream, comm_stream, cuda_stream, process_rows, process_cols, tile_rows,
      tile_cols, logical_rows, logical_cols, reverse, block_rank_map,
      cyclic_rank_map, matrix, scratch, matrix_out, scratch_out,
      collective_params, collective_cliques);
}

}  // namespace xla::gpu
