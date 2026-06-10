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

#include <algorithm>
#include <functional>

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
// slab. Independent cycles across process rows/columns are then batched so a
// single CollectivePermute moves one slab per participating rank.

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
                                        int64_t tile_cols) {
  const int64_t slots_per_process_row = process_cols * col_blocks_per_rank;
  const int64_t process_row = slot / slots_per_process_row;
  const int64_t row_slot = slot % slots_per_process_row;
  const int64_t process_col = row_slot / col_blocks_per_rank;
  const int64_t local_col_block = row_slot % col_blocks_per_rank;
  return Native2DSlot{
      /*rank=*/process_row * process_cols + process_col,
      /*rect=*/ColumnSlabRect(local_rows, tile_cols, local_col_block),
  };
}

Native2DSlot RowPhaseSlotToRankLocal(int64_t slot, int64_t process_rows,
                                     int64_t process_cols,
                                     int64_t row_blocks_per_rank,
                                     int64_t tile_rows,
                                     int64_t local_cols) {
  const int64_t slots_per_process_col = process_rows * row_blocks_per_rank;
  const int64_t process_col = slot / slots_per_process_col;
  const int64_t col_slot = slot % slots_per_process_col;
  const int64_t process_row = col_slot / row_blocks_per_rank;
  const int64_t local_row_block = col_slot % row_blocks_per_rank;
  return Native2DSlot{
      /*rank=*/process_row * process_cols + process_col,
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

absl::Status AppendCycleSteps(int64_t phase, const SlotDecoder& decode_slot,
                              const std::vector<int64_t>& slots,
                              std::vector<Native2DStep>* steps) {
  const bool is_closed = slots.size() > 1 && slots.front() == slots.back();
  if (is_closed) {
    const Native2DSlot saved = decode_slot(slots[slots.size() - 2]);
    steps->push_back(Native2DStep{
        phase, Native2DStepKind::kSaveScratch, saved.rank, -1, saved.rect,
        NativeLocalRect{0, 0, 0, 0}});

    for (int64_t index = static_cast<int64_t>(slots.size()) - 3; index >= 0;
         --index) {
      const Native2DSlot source = decode_slot(slots[index]);
      const Native2DSlot target = decode_slot(slots[index + 1]);
      steps->push_back(Native2DStep{
          phase, Native2DStepKind::kMove, source.rank, target.rank,
          source.rect, target.rect});
    }

    const Native2DSlot target = decode_slot(slots[0]);
    steps->push_back(Native2DStep{
        phase, Native2DStepKind::kRestoreScratch, saved.rank, target.rank,
        NativeLocalRect{0, 0, 0, 0}, target.rect});
    return absl::OkStatus();
  }

  for (int64_t index = static_cast<int64_t>(slots.size()) - 2; index >= 0;
       --index) {
    const Native2DSlot source = decode_slot(slots[index]);
    const Native2DSlot target = decode_slot(slots[index + 1]);
    steps->push_back(Native2DStep{phase, Native2DStepKind::kMove, source.rank,
                                  target.rank, source.rect, target.rect});
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<Native2DStep>> BuildSlabNative2DSteps(
    int64_t process_rows, int64_t process_cols, int64_t tile_rows,
    int64_t tile_cols, int64_t local_rows, int64_t local_cols) {
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

  absl::StatusOr<std::vector<int64_t>> column_slot_map =
      BuildColumnSlabSlotMap(process_rows, process_cols, col_blocks_per_rank);
  if (!column_slot_map.ok()) {
    return column_slot_map.status();
  }
  absl::StatusOr<std::map<int64_t, std::vector<int64_t>>> column_cycles =
      BuildNative2DCycles(*column_slot_map);
  if (!column_cycles.ok()) {
    return column_cycles.status();
  }
  SlotDecoder decode_column_slot = [&](int64_t slot) {
    return ColumnPhaseSlotToRankLocal(slot, process_cols, col_blocks_per_rank,
                                      local_rows, tile_cols);
  };
  for (const auto& cycle : *column_cycles) {
    JAXMG_RETURN_IF_ERROR(
        AppendCycleSteps(/*phase=*/0, decode_column_slot, cycle.second,
                         &steps));
  }

  absl::StatusOr<std::vector<int64_t>> row_slot_map =
      BuildRowSlabSlotMap(process_rows, process_cols, row_blocks_per_rank);
  if (!row_slot_map.ok()) {
    return row_slot_map.status();
  }
  absl::StatusOr<std::map<int64_t, std::vector<int64_t>>> row_cycles =
      BuildNative2DCycles(*row_slot_map);
  if (!row_cycles.ok()) {
    return row_cycles.status();
  }
  SlotDecoder decode_row_slot = [&](int64_t slot) {
    return RowPhaseSlotToRankLocal(slot, process_rows, process_cols,
                                   row_blocks_per_rank, tile_rows, local_cols);
  };
  for (const auto& cycle : *row_cycles) {
    JAXMG_RETURN_IF_ERROR(
        AppendCycleSteps(/*phase=*/1, decode_row_slot, cycle.second, &steps));
  }

  return steps;
}

std::vector<Native2DStepBatch> BatchNative2DSteps(
    const std::vector<Native2DStep>& steps) {
  std::vector<Native2DStepBatch> batches;

  for (const Native2DStep& step : steps) {
    bool placed = false;
    for (Native2DStepBatch& batch : batches) {
      if (batch.phase != step.phase || batch.kind != step.kind) {
        continue;
      }

      bool conflicts = false;
      for (const Native2DStep& existing : batch.steps) {
        if (step.kind == Native2DStepKind::kSaveScratch) {
          conflicts = conflicts || existing.source_rank == step.source_rank;
        } else {
          conflicts = conflicts || existing.source_rank == step.source_rank ||
                      existing.target_rank == step.target_rank;
        }
      }
      if (conflicts) {
        continue;
      }
      batch.steps.push_back(step);
      placed = true;
      break;
    }

    if (!placed) {
      batches.push_back(Native2DStepBatch{step.phase, step.kind, {step}});
    }
  }

  return batches;
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

absl::Status XlaRect2DNativePlanPrepare(
    const CollectiveParams* collective_params,
    CollectiveCliqueRequests* clique_requests) {
  if (collective_params == nullptr || clique_requests == nullptr) {
    return absl::InvalidArgumentError(
        "xla_rect_2d_native_plan requires XLA collective prepare contexts");
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

absl::Status XlaRect2DNativePlanDispatch(
    se::Stream* stream, se::Stream* comm_stream, cudaStream_t cuda_stream,
    int64_t layout, int64_t process_rows, int64_t process_cols,
    int64_t tile_rows, int64_t tile_cols, ffi::AnyBuffer matrix,
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

  absl::StatusOr<RectLayout> decoded_layout = DecodeRectLayout(layout);
  JAXMG_RETURN_IF_ERROR(decoded_layout.status());

  absl::StatusOr<GpuCliqueKey> clique_key =
      NodeScopedP2PCliqueKey(*collective_params);
  if (!clique_key.ok()) {
    return clique_key.status();
  }
  const int64_t num_ranks = static_cast<int64_t>(clique_key->num_devices());
  if (process_rows * process_cols != num_ranks) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "xla_rect_2d_native_plan grid %d x %d does not match clique size %d",
        process_rows, process_cols, num_ranks));
  }

  std::optional<RankId> rank =
      clique_key->rank(collective_params->global_device_id);
  if (!rank.has_value()) {
    return absl::InvalidArgumentError(
        "xla_rect_2d_native_plan could not resolve this device rank");
  }
  const int64_t rank_value = static_cast<int64_t>(rank->value());

  absl::StatusOr<Communicator*> comm = collective_cliques->GetComm(
      *clique_key, collective_params->global_device_id);
  if (!comm.ok()) {
    return comm.status();
  }

  const int64_t local_rows = matrix.dimensions()[0];
  const int64_t local_cols = matrix.dimensions()[1];
  absl::StatusOr<std::vector<Native2DStep>> steps =
      BuildSlabNative2DSteps(process_rows, process_cols, tile_rows, tile_cols,
                             local_rows, local_cols);
  if (!steps.ok()) {
    return steps.status();
  }
  std::vector<Native2DStepBatch> batches =
      BatchNative2DSteps(*steps);

  const int64_t column_slab_elements = local_rows * tile_cols;
  const int64_t row_slab_elements = tile_rows * local_cols;
  const int64_t slab_elements =
      std::max(column_slab_elements, row_slab_elements);
  if (scratch.dimensions()[0] < 3 * slab_elements) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "xla_rect_2d_native_plan scratch length %d is smaller than "
        "3 * max(local_rows * tile_cols, tile_rows * local_cols) = %d",
        scratch.dimensions()[0], 3 * slab_elements));
  }
  const size_t element_bytes =
      matrix.size_bytes() / static_cast<size_t>(matrix.element_count());
  const uint64_t slab_bytes =
      static_cast<uint64_t>(slab_elements) * element_bytes;

  se::DeviceAddressBase matrix_out_base = matrix_out->device_memory();
  se::DeviceAddressBase scratch_out_base = scratch_out->device_memory();
  se::DeviceAddressBase saved_slot =
      scratch_out_base.GetByteSlice(0, slab_bytes);
  se::DeviceAddressBase send_slot =
      scratch_out_base.GetByteSlice(slab_bytes, slab_bytes);
  se::DeviceAddressBase recv_slot =
      scratch_out_base.GetByteSlice(2 * slab_bytes, slab_bytes);

  JAXMG_RETURN_IF_ERROR(CopyMatrixIfNeeded(cuda_stream, matrix, matrix_out));
  JAXMG_RETURN_IF_ERROR(CopyScratchIfNeeded(cuda_stream, scratch, scratch_out));

  // Execute the slab-cycle schedule in conflict-free batches. Sources are read
  // from the mutable output buffer: phase 1 consumes the column-cyclic layout
  // written by phase 0, and later cycle steps consume earlier cycle moves.
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
          cuda_stream, *decoded_layout, local_rows, local_cols,
          send_step->source.row_start, send_step->source.col_start,
          send_step->source.row_count, send_step->source.col_count,
          element_bytes, matrix_out_base, saved_slot));
      continue;
    }

    if (batch.kind == Native2DStepKind::kMove && send_step != nullptr) {
      JAXMG_RETURN_IF_ERROR(PackRect(
          cuda_stream, *decoded_layout, local_rows, local_cols,
          send_step->source.row_start, send_step->source.col_start,
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
                   : recv_step->target.row_count * recv_step->target.col_count);
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
          cuda_stream, *decoded_layout, local_rows, local_cols,
          recv_step->target.row_start, recv_step->target.col_start,
          recv_step->target.row_count, recv_step->target.col_count,
          element_bytes, local_source, matrix_out_base));
      continue;
    }

    absl::Status status = comm_stream->WaitFor(stream);
    if (!status.ok()) {
      return status;
    }
    Future<> future = (*comm)->CollectivePermute(
        collective_send_slot, recv_slot, matrix.element_type(),
        static_cast<size_t>(collective_elements), source_rank, target_ranks,
        GpuCollectives::On(*comm_stream));
    status = future.Await();
    if (!status.ok()) {
      return status;
    }
    status = stream->WaitFor(comm_stream);
    if (!status.ok()) {
      return status;
    }

    if (recv_step != nullptr) {
      JAXMG_RETURN_IF_ERROR(UnpackRect(
          cuda_stream, *decoded_layout, local_rows, local_cols,
          recv_step->target.row_start, recv_step->target.col_start,
          recv_step->target.row_count, recv_step->target.col_count,
          element_bytes, recv_slot, matrix_out_base));
    }
  }

  return absl::OkStatus();
}

}  // namespace xla::gpu
