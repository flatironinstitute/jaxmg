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
//   1. Along the process-column axis, pull later real column slabs into earlier
//      padding holes so horizontal padding is consolidated at the global right
//      edge.
//   2. Along the process-row axis, pull later real row slabs upward so vertical
//      padding is consolidated at the global bottom edge.
//   3. Express each pull as Native2DStep moves that can be executed by the shared
//      rectangle transport layer.
//   4. Build the reverse schedule by swapping source/target rectangles and
//      reversing wave order; padding bytes themselves are never semantically
//      restored.

#include <algorithm>
#include <vector>

#include "include/xla_comm_backend.h"

namespace xla::gpu {
namespace {

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


}  // namespace

absl::StatusOr<std::vector<Native2DStep>> BuildEdgePaddingNative2DSteps(
    int64_t process_rows, int64_t process_cols, int64_t tile_rows,
    int64_t tile_cols, int64_t logical_rows, int64_t logical_cols,
    int64_t local_rows, int64_t local_cols,
    absl::Span<const int64_t> rank_map) {
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
