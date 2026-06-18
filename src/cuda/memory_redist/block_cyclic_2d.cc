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
// Native 2D block-cyclic redistribution for cuSOLVERMp.
//
// This file owns the tile/slab ownership permutation that converts a
// tile-aligned, edge-padded JAX block-sharded matrix into the 2D block-cyclic
// layout expected by cuSOLVERMp. It composes the edge-padding phase from
// edge_padding_2d.cc with the low-level rectangle transport in rectangle_pack.cc.
//
// File workflow:
//   1. Validate the distributed process grid and row-major/column-major rank map.
//   2. Build the column-owner phase: within each process row, move tile-column
//      slabs to the process column required by tile_col % process_cols.
//   3. Build the row-owner phase: within each process column, move tile-row slabs
//      to the process row required by tile_row % process_rows.
//   4. Decompose each phase into closed permutation cycles, using the same
//      one-slab scratch policy as JAXMg's original 1D reshuffler.
//   5. Batch only independent same-sequence moves across different process rows
//      or columns, preserving a conservative fixed scratch bound.
//   6. Expose the public padded forward/reverse redistribution FFI target used by
//      the Python cuSOLVERMp wrapper.

#include <algorithm>
#include <functional>
#include <map>
#include <tuple>
#include <vector>

#include "memory_redist.h"

namespace xla::gpu {

namespace {

absl::Status ValidateStandardGridRankMap(const char* caller,
                                         absl::Span<const int64_t> rank_map,
                                         int64_t process_rows,
                                         int64_t process_cols) {
  const int64_t num_ranks = process_rows * process_cols;
  if (rank_map.size() != static_cast<size_t>(num_ranks)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s expected rank_map length %d, got %d", caller, num_ranks,
        rank_map.size()));
  }
  bool row_major = true;
  bool column_major = true;
  for (int64_t process_rank = 0; process_rank < num_ranks; ++process_rank) {
    const int64_t communicator_rank = rank_map[process_rank];
    const int64_t process_row = process_rank / process_cols;
    const int64_t process_col = process_rank % process_cols;
    const int64_t column_major_rank = process_col * process_rows + process_row;
    row_major = row_major && communicator_rank == process_rank;
    column_major = column_major && communicator_rank == column_major_rank;
    if (communicator_rank < 0 || communicator_rank >= num_ranks) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "%s rank_map[%d]=%d is outside [0, %d)", caller, process_rank,
          communicator_rank, num_ranks));
    }
  }
  if (!row_major && !column_major) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s requires a row-major or column-major cuSOLVERMp rank_map",
        caller));
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
  // Decode a global "column-slab slot" into the rank-local rectangle that
  // currently owns it.  Slots are grouped first by process row, then by
  // process column, then by local tile-column index.
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
  // Decode a global "row-slab slot" into the rank-local rectangle that
  // currently owns it.  Slots are grouped first by process column, then by
  // process row, then by local tile-row index.
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
  // Build source_slot -> target_slot for the column-owner phase.  Initially
  // each process column owns contiguous global tile columns; cuSOLVERMp wants
  // tile column k on process column k % process_cols.
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
  // Build source_slot -> target_slot for the row-owner phase.  After column
  // ownership is correct, this performs the analogous tile-row modulo
  // ownership within each process column.
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
  // Convert the slot permutation into closed cycles. Fixed points are skipped.
  // Non-trivial cycles are stored as slot sequences ending at the starting
  // slot, which lets AppendCycleSteps generate save/move/restore operations
  // with one bounded slab in scratch.
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

absl::StatusOr<int64_t> AppendCycleSteps(int64_t phase,
                                         const SlotDecoder& decode_slot,
                                         const std::vector<int64_t>& slots,
                                         int64_t sequence_offset,
                                         std::vector<Native2DStep>* steps) {
  // Emit the low-memory movement program for one cycle. A closed cycle uses
  // scratch to protect the last source slot, rotates the remaining slots
  // backward, then restores the saved payload into the first target.
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

using SlotGroup = std::function<int64_t(int64_t)>;

absl::Status AppendAxisGroupSerialCycles(
    int64_t phase, const SlotDecoder& decode_slot, const SlotGroup& slot_group,
    const std::map<int64_t, std::vector<int64_t>>& cycles,
    std::vector<Native2DStep>* steps) {
  // Cycles inside one process row/column group can touch the same ranks and
  // scratch slots, so they are appended serially by increasing sequence number.
  // Cycles in different groups can share sequence numbers and may later be
  // batched if their ranks are conflict-free.
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


}  // namespace

absl::StatusOr<std::vector<Native2DStep>> BuildSlabNative2DSteps(
    int64_t process_rows, int64_t process_cols, int64_t tile_rows,
    int64_t tile_cols, int64_t local_rows, int64_t local_cols,
    absl::Span<const int64_t> rank_map, bool reverse) {
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
                                 bool invert) -> absl::Status {
    // Column-owner phase: move full-height tile-column slabs horizontally
    // inside each process row.  `invert` switches the map to the reverse
    // redistribution used after a solve.
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
    SlotDecoder decode_slot = [&](int64_t slot) {
      return ColumnPhaseSlotToRankLocal(
          slot, process_cols, col_blocks_per_rank, local_rows, tile_cols,
          rank_map);
    };
    const int64_t slots_per_process_row =
        process_cols * col_blocks_per_rank;
    SlotGroup axis_group = [&](int64_t slot) {
      return slot / slots_per_process_row;
    };
    return AppendAxisGroupSerialCycles(phase, decode_slot, axis_group,
                                       *cycles, &steps);
  };

  auto append_row_phase = [&](int64_t phase,
                              bool invert) -> absl::Status {
    // Row-owner phase: move full-width tile-row slabs vertically inside each
    // process column.  This is the separable second half of the 2D
    // block-cyclic permutation.
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
    SlotDecoder decode_slot = [&](int64_t slot) {
      return RowPhaseSlotToRankLocal(slot, process_rows, process_cols,
                                     row_blocks_per_rank, tile_rows,
                                     local_cols, rank_map);
    };
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
    JAXMG_RETURN_IF_ERROR(append_row_phase(/*phase=*/0, /*invert=*/true));
    JAXMG_RETURN_IF_ERROR(append_column_phase(/*phase=*/1, /*invert=*/true));
  } else {
    JAXMG_RETURN_IF_ERROR(append_column_phase(/*phase=*/0, /*invert=*/false));
    JAXMG_RETURN_IF_ERROR(append_row_phase(/*phase=*/1, /*invert=*/false));
  }

  return steps;
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

absl::StatusOr<int64_t> RequiredCyclicSlotElements(
    int64_t tile_rows, int64_t tile_cols, int64_t local_rows,
    int64_t local_cols) {
  if (tile_rows <= 0 || tile_cols <= 0 || local_rows <= 0 ||
      local_cols <= 0) {
    return absl::InvalidArgumentError(
        "cyclic redistribution scratch requires positive dimensions");
  }
  return std::max(tile_cols * local_rows, tile_rows * local_cols);
}

absl::StatusOr<int64_t> RequiredPadded2DNativePlanScratchElements(
    int64_t process_rows, int64_t process_cols, int64_t tile_rows,
    int64_t tile_cols, int64_t logical_rows, int64_t logical_cols,
    int64_t local_rows, int64_t local_cols,
    absl::Span<const int64_t> rank_map) {
  JAXMG_RETURN_IF_ERROR(ValidateStandardGridRankMap(
      "required_padded_2d_native_plan_scratch", rank_map, process_rows,
      process_cols));

  // Validate the slab schedule and then use the closed-form cyclic bound.  The
  // schedule validation catches incompatible local shapes; the scratch formula
  // remains deterministic and independent of how many cycles are generated.
  absl::StatusOr<std::vector<Native2DStep>> slab_steps =
      BuildSlabNative2DSteps(process_rows, process_cols, tile_rows, tile_cols,
                             local_rows, local_cols, rank_map);
  if (!slab_steps.ok()) {
    return slab_steps.status();
  }
  absl::StatusOr<int64_t> cyclic_slot_elements =
      RequiredCyclicSlotElements(tile_rows, tile_cols, local_rows, local_cols);
  if (!cyclic_slot_elements.ok()) {
    return cyclic_slot_elements.status();
  }
  const int64_t scratch_elements = 3 * *cyclic_slot_elements;

  // Validate edge-padding with the same fixed scratch budget.  The planner
  // chunks open-chain moves rather than increasing the allocation.
  absl::StatusOr<std::vector<Native2DStep>> edge_steps =
      BuildEdgePaddingNative2DSteps(
          process_rows, process_cols, tile_rows, tile_cols, logical_rows,
          logical_cols, local_rows, local_cols, scratch_elements, rank_map);
  if (!edge_steps.ok()) {
    return edge_steps.status();
  }

  return scratch_elements;
}

namespace {

absl::Status ExecutePadded2DNativePlanRawImpl(
    const char* caller, se::Stream* stream, se::Stream* comm_stream,
    cudaStream_t cuda_stream, int64_t process_rows, int64_t process_cols,
    int64_t tile_rows, int64_t tile_cols, int64_t logical_rows,
    int64_t logical_cols, int64_t reverse,
    absl::Span<const int64_t> rank_map, ffi::AnyBuffer matrix,
    se::DeviceAddressBase matrix_out_base,
    se::DeviceAddressBase scratch_base, int64_t scratch_elements,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques) {
  if (stream == nullptr || comm_stream == nullptr || cuda_stream == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s requires XLA and CUDA stream contexts", caller));
  }
  if (collective_params == nullptr || collective_cliques == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s requires XLA collective contexts", caller));
  }
  if (matrix.dimensions().size() != 2) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s expects a rank-2 matrix buffer", caller));
  }

  // Production fused solvers enter through this raw executor after local
  // layout conversion. Resolve the all-assigned communicator once and use it
  // for both edge-padding and 2D block-cyclic movement.
  absl::StatusOr<GpuCliqueKey> clique_key =
      AllAssignedDevicesP2PCliqueKey(*collective_params);
  if (!clique_key.ok()) {
    return clique_key.status();
  }
  const int64_t num_ranks = static_cast<int64_t>(clique_key->num_devices());
  if (process_rows * process_cols != num_ranks) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s grid %d x %d does not match clique size %d", caller,
        process_rows, process_cols, num_ranks));
  }
  JAXMG_RETURN_IF_ERROR(ValidateStandardGridRankMap(
      caller, rank_map, process_rows, process_cols));

  std::optional<RankId> rank =
      clique_key->rank(collective_params->global_device_id);
  if (!rank.has_value()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s could not resolve this device rank", caller));
  }
  const int64_t rank_value = static_cast<int64_t>(rank->value());

  absl::StatusOr<GpuCommunicator*> comm = collective_cliques->GetComm(
      *clique_key, collective_params->global_device_id);
  if (!comm.ok()) {
    return comm.status();
  }

  const int64_t local_rows = matrix.dimensions()[0];
  const int64_t local_cols = matrix.dimensions()[1];
  absl::StatusOr<int64_t> cyclic_slot_elements =
      RequiredCyclicSlotElements(tile_rows, tile_cols, local_rows, local_cols);
  if (!cyclic_slot_elements.ok()) {
    return cyclic_slot_elements.status();
  }
  if (scratch_elements < 3 * *cyclic_slot_elements) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s scratch length %d is smaller than the fixed cyclic "
        "redistribution requirement 3 * %d = %d",
        caller, scratch_elements, *cyclic_slot_elements,
        3 * *cyclic_slot_elements));
  }

  // Build both movement programs before touching buffers.  Reverse execution
  // inverts the order: undo the cyclic block redistribution first, then undo
  // edge-padding compaction.
  absl::StatusOr<std::vector<Native2DStep>> edge_steps =
      BuildEdgePaddingNative2DSteps(
          process_rows, process_cols, tile_rows, tile_cols, logical_rows,
          logical_cols, local_rows, local_cols, scratch_elements, rank_map);
  if (!edge_steps.ok()) {
    return edge_steps.status();
  }
  absl::StatusOr<std::vector<Native2DStep>> slab_steps =
      BuildSlabNative2DSteps(process_rows, process_cols, tile_rows, tile_cols,
                             local_rows, local_cols, rank_map, reverse != 0);
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

  const size_t element_bytes =
      matrix.size_bytes() / static_cast<size_t>(matrix.element_count());
  if (matrix.device_memory().opaque() != matrix_out_base.opaque()) {
    JAXMG_RETURN_IF_CUDA_ERROR(cudaMemcpyAsync(
        matrix_out_base.opaque(), matrix.untyped_data(), matrix.size_bytes(),
        cudaMemcpyDeviceToDevice, cuda_stream));
  }
  if (edge_batches.empty() && slab_batches.empty()) {
    return absl::OkStatus();
  }

  if (reverse != 0) {
    // Output path: cuSOLVERMp layout -> edge-padded JAX-local layout ->
    // per-shard padded JAX layout.
    JAXMG_RETURN_IF_ERROR(ExecuteNative2DStepBatches(
        stream, comm_stream, cuda_stream, slab_batches, local_rows, local_cols,
        rank_value, num_ranks, element_bytes, *cyclic_slot_elements, matrix,
        matrix_out_base, scratch_base, *comm));
    return ExecuteEdgePaddingBatches(
        stream, comm_stream, cuda_stream, edge_batches, local_rows, local_cols,
        rank_value, num_ranks, element_bytes, scratch_elements, matrix,
        matrix_out_base, scratch_base, *comm);
  }

  // Input path: per-shard padded JAX layout -> edge-padded global layout ->
  // cuSOLVERMp 2D block-cyclic layout.
  JAXMG_RETURN_IF_ERROR(ExecuteEdgePaddingBatches(
      stream, comm_stream, cuda_stream, edge_batches, local_rows, local_cols,
      rank_value, num_ranks, element_bytes, scratch_elements, matrix,
      matrix_out_base, scratch_base, *comm));
  return ExecuteNative2DStepBatches(
      stream, comm_stream, cuda_stream, slab_batches, local_rows, local_cols,
      rank_value, num_ranks, element_bytes, *cyclic_slot_elements, matrix,
      matrix_out_base, scratch_base, *comm);
}

}  // namespace

absl::Status ExecutePadded2DNativePlanRaw(
    const char* caller, se::Stream* stream, se::Stream* comm_stream,
    cudaStream_t cuda_stream, int64_t process_rows, int64_t process_cols,
    int64_t tile_rows, int64_t tile_cols, int64_t logical_rows,
    int64_t logical_cols, int64_t reverse,
    absl::Span<const int64_t> rank_map, ffi::AnyBuffer matrix,
    se::DeviceAddressBase matrix_out_base,
    se::DeviceAddressBase scratch_base, int64_t scratch_elements,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques) {
  return ExecutePadded2DNativePlanRawImpl(
      caller, stream, comm_stream, cuda_stream, process_rows, process_cols,
      tile_rows, tile_cols, logical_rows, logical_cols, reverse, rank_map,
      matrix, matrix_out_base, scratch_base, scratch_elements,
      collective_params, collective_cliques);
}

}  // namespace xla::gpu
