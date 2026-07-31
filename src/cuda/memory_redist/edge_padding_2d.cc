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
// Edge-padding compaction for cuSOLVERMp 2D redistribution.
//
// JAX pads each local shard independently so native code has enough tile-aligned
// storage. cuSOLVERMp, however, expects padding to behave like global edge
// padding: real matrix entries occupy the global top-left region and unused
// storage lives on the right and bottom edges. This file builds the open-chain
// movement schedule that performs that compaction before the block-cyclic phase,
// and builds the inverse schedule used after the solve.
//
// File workflow:
//   1. Along the process-column axis, map each real local column interval to its
//      final compacted global coordinate.  This pushes all column padding to the
//      global right edge.
//   2. Along the process-row axis, perform the same direct mapping for real row
//      intervals, pushing all row padding to the global bottom edge.
//   3. Split each direct move at process-boundaries and at the caller-provided
//      padding scratch budget.  Padding is an open-chain shift, so it can use
//      the whole scratch allocation as one temporary buffer instead of the
//      three-slot cycle layout used by the block-cyclic permutation.
//   4. Build the reverse schedule by swapping source/target rectangles and
//      reversing wave order; padding bytes themselves are never semantically
//      restored.

#include <algorithm>
#include <vector>

#include "memory_redist.h"

namespace xla::gpu {
namespace {

// One chunk of a 1D edge-padding shift. The 2D planner lifts this movement
// across the orthogonal process-grid axis to create rectangle moves.
struct AxisEdgeMove {
  // One 1D open-chain move along a global process-grid axis. `wave` is a
  // dependency sequence number: moves in the same wave represent the same
  // logical shift applied independently across the orthogonal process axis.
  int64_t wave;
  int64_t source_start;
  int64_t target_start;
  int64_t extent;
};

// Returns how many physical storage entries must be appended to one logical
// local block so it is divisible by the requested tile size.
int64_t AxisPadding(int64_t logical_per_block, int64_t tile_size) {
  const int64_t remainder = logical_per_block % tile_size;
  return remainder == 0 ? 0 : tile_size - remainder;
}

// Builds the open-chain direct moves that compact one global axis from
// per-shard padded storage into global edge-padded storage.
absl::StatusOr<std::vector<AxisEdgeMove>> BuildAxisEdgePaddingMoves(
    int64_t block_count, int64_t logical_per_block,
    int64_t physical_per_block, int64_t max_extent) {
  if (block_count <= 0 || logical_per_block <= 0 ||
      physical_per_block <= 0) {
    return absl::InvalidArgumentError(
        "edge-padding compaction requires positive axis extents");
  }
  if (max_extent <= 0) {
    return absl::InvalidArgumentError(
        "edge-padding compaction requires a positive maximum move extent");
  }
  if (logical_per_block > physical_per_block) {
    return absl::InvalidArgumentError(
        "edge-padding compaction logical extent exceeds physical extent");
  }

  if (logical_per_block == physical_per_block) {
    return std::vector<AxisEdgeMove>();
  }

  // Each block's real interval [block * physical, block * physical + logical)
  // must end up at [block * logical, (block + 1) * logical) in the compacted
  // global address line.  Processing source blocks from low to high is safe for
  // left/up compaction: earlier target positions are either already finalized or
  // padding holes produced by previous moves.  Local same-rank overlap is safe
  // because the rectangle executor packs the full source chunk into scratch
  // before unpacking it to the destination.
  std::vector<AxisEdgeMove> moves;
  int64_t wave = 0;
  for (int64_t source_block = 0; source_block < block_count; ++source_block) {
    int64_t consumed = 0;
    while (consumed < logical_per_block) {
      const int64_t source_start =
          source_block * physical_per_block + consumed;
      const int64_t target_start =
          source_block * logical_per_block + consumed;

      const int64_t remaining = logical_per_block - consumed;
      const int64_t source_block_stop =
          (source_start / physical_per_block + 1) * physical_per_block;
      const int64_t target_block_stop =
          (target_start / physical_per_block + 1) * physical_per_block;
      const int64_t extent =
          std::min({remaining, source_block_stop - source_start,
                    target_block_stop - target_start, max_extent});
      if (extent <= 0) {
        return absl::InternalError(
            "edge-padding compaction generated an empty direct move");
      }

      if (source_start != target_start) {
        moves.push_back(AxisEdgeMove{wave, source_start, target_start,
                                     extent});
        ++wave;
      }
      consumed += extent;
    }
  }

  return moves;
}


}  // namespace

// Builds the 2D edge-padding compaction schedule: horizontal column compaction
// followed by vertical row compaction, each split to fit the scratch budget.
absl::StatusOr<std::vector<Native2DStep>> BuildEdgePaddingNative2DSteps(
    int64_t process_rows, int64_t process_cols, int64_t tile_rows,
    int64_t tile_cols, int64_t logical_rows, int64_t logical_cols,
    int64_t local_rows, int64_t local_cols,
    int64_t padding_slot_elements, absl::Span<const int64_t> rank_map) {
  if (process_rows <= 0 || process_cols <= 0 || tile_rows <= 0 ||
      tile_cols <= 0 || logical_rows <= 0 || logical_cols <= 0) {
    return absl::InvalidArgumentError(
        "padded native 2D redistribution requires positive dimensions");
  }
  if (padding_slot_elements <= 0) {
    return absl::InvalidArgumentError(
        "padded native 2D redistribution requires positive padding scratch");
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
  if (padding_slot_elements < local_rows ||
      padding_slot_elements < local_cols) {
    return absl::InvalidArgumentError(
        "edge-padding compaction scratch must be large enough for at least one "
        "full-height column or one full-width row");
  }
  const int64_t max_horizontal_extent =
      std::max<int64_t>(1, padding_slot_elements / local_rows);
  const int64_t max_vertical_extent =
      std::max<int64_t>(1, padding_slot_elements / local_cols);

  // Horizontal compaction pushes column padding to the global right edge.  Each
  // 1D axis move is lifted into one rectangle move for every process row, so
  // the same shift can run independently across rows when ranks do not
  // conflict.
  absl::StatusOr<std::vector<AxisEdgeMove>> horizontal_moves =
      BuildAxisEdgePaddingMoves(process_cols, local_logical_cols, local_cols,
                                max_horizontal_extent);
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
          /*source_rank=*/rank_map[source_process_rank],
          /*target_rank=*/rank_map[target_process_rank],
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

  // Vertical compaction is the row analogue: after columns are edge-compacted,
  // push row padding to the global bottom edge.  Each 1D move is lifted across
  // every process column.
  absl::StatusOr<std::vector<AxisEdgeMove>> vertical_moves =
      BuildAxisEdgePaddingMoves(process_rows, local_logical_rows, local_rows,
                                max_vertical_extent);
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
          /*source_rank=*/rank_map[source_process_rank],
          /*target_rank=*/rank_map[target_process_rank],
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

// Builds the inverse of BuildEdgePaddingNative2DSteps for solver outputs,
// moving only real matrix entries back to the original per-shard padded layout.
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


}  // namespace xla::gpu
