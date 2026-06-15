"""Test-only Python sharding wrappers for native 2D redistribution handlers.

The production package enters cuSOLVERMp through one solver FFI call, so these
helpers intentionally live in the test tree instead of ``src/jaxmg``. They
remain useful for exercising the lower-level redistribution diagnostics:
shape/scratch validation happens in Python, while the native backend performs
pack, NCCL transfer, unpack, and cycle scheduling work.

The lower-level rectangle-transfer functions are retained for diagnostics.  The
production cuSOLVERMp path uses the padded native handlers, which compact
per-shard padding to the global bottom/right edges and then perform the
tile-aligned column-owner plus row-owner redistribution in one native call.
"""

from __future__ import annotations

from collections import defaultdict

from jax import Array
from jax.sharding import Mesh, PartitionSpec as P

from jaxmg._block_cyclic_2d_plan import (
    EdgePaddingCompactionBatch,
    EdgePaddingCompactionMove,
    ExecutableFragmentTransfer,
    ExecutableFragmentTransferBatch,
    ProcessGrid,
    ProcessRankMap,
    TileShape,
    batch_edge_padding_compaction_moves,
    build_edge_padding_compaction_plan,
)
from jaxmg._xla_comm_probe import (
    xla_rect_2d_native_plan_shardmap,
    xla_rect_padded_2d_native_plan_shardmap,
    xla_rect_transfer_probe_shardmap,
)


def required_rect_transfer_scratch_size(
    batches: tuple[ExecutableFragmentTransferBatch, ...],
) -> int:
    """Return the per-rank scratch length needed by rectangle transfers.

    The current native rectangle-transfer primitive uses two dense slots per
    rank: one packed send slot and one receive slot. Batches may contain
    different rectangle shapes, but each FFI call executes one shape group, so
    the maximum fragment area determines the required scratch length.
    """
    max_elements = 0
    for batch in batches:
        for transfer in batch.transfers:
            max_elements = max(
                max_elements,
                transfer.row_count * transfer.col_count,
            )
    return 2 * max_elements


def required_edge_padding_compaction_scratch_size(
    batches: tuple[EdgePaddingCompactionBatch, ...],
) -> int:
    """Return the per-rank scratch length for edge-padding compaction.

    The executor reuses ``xla_rect_transfer_probe``. Each FFI call moves one
    shape group from one dependency wave, and the native probe uses one packed
    send slot plus one receive slot per rank.
    """
    max_elements = 0
    for batch in batches:
        for move in batch.moves:
            max_elements = max(max_elements, move.element_count)
    return 2 * max_elements


def required_native_2d_plan_scratch_size(
    *,
    local_rows: int,
    local_cols: int,
    tile_rows: int,
    tile_cols: int,
) -> int:
    """Return the per-rank scratch length for the native slab scheduler.

    The fused native path performs closed-cycle permutations over larger slabs:
    ``local_rows x tile_cols`` for the column-owner phase and
    ``tile_rows x local_cols`` for the row-owner phase. One scratch slot keeps
    the saved cycle payload live while another send slot and receive slot are
    used for the current raw-NCCL send/recv round.
    """
    local_rows = int(local_rows)
    local_cols = int(local_cols)
    tile_rows = int(tile_rows)
    tile_cols = int(tile_cols)
    if min(local_rows, local_cols, tile_rows, tile_cols) <= 0:
        raise ValueError("local and tile dimensions must be positive.")
    return 3 * max(local_rows * tile_cols, tile_rows * local_cols)


def required_padded_block_cyclic_2d_scratch_size(
    *,
    logical_rows: int,
    logical_cols: int,
    grid: ProcessGrid,
    tile_rows: int,
    tile_cols: int,
) -> int:
    """Return scratch needed by the fused padded native 2D scheduler.

    The native FFI handler reuses one scratch allocation for both the
    edge-padding compaction stage and the tile-aligned slab redistribution
    stage. The movement engine reserves three equally sized slots: saved cycle
    payload, send payload, and receive payload.
    """
    tile_shape = TileShape(rows=int(tile_rows), cols=int(tile_cols))
    compaction_plan = build_edge_padding_compaction_plan(
        logical_rows=logical_rows,
        logical_cols=logical_cols,
        tile_shape=tile_shape,
        grid=grid,
    )
    compaction_batches = batch_edge_padding_compaction_moves(
        compaction_plan.moves,
    )
    max_compaction_elements = 0
    for batch in compaction_batches:
        for move in batch.moves:
            max_compaction_elements = max(
                max_compaction_elements,
                move.element_count,
            )
    max_slab_elements = max(
        compaction_plan.padding.local_physical_rows * tile_shape.cols,
        tile_shape.rows * compaction_plan.padding.local_physical_cols,
    )
    return 3 * max(max_compaction_elements, max_slab_elements)


RectangleMove = ExecutableFragmentTransfer | EdgePaddingCompactionMove


def _shape_groups(
    transfers: tuple[RectangleMove, ...],
) -> tuple[tuple[tuple[int, int], tuple[RectangleMove, ...]], ...]:
    grouped: dict[tuple[int, int], list[RectangleMove]] = defaultdict(list)
    for transfer in transfers:
        grouped[(transfer.row_count, transfer.col_count)].append(transfer)
    return tuple((shape, tuple(grouped[shape])) for shape in sorted(grouped))


def _attrs_for_shape_group(
    transfers: tuple[RectangleMove, ...],
    *,
    num_ranks: int,
) -> tuple[list[int], list[int], list[int], list[int], list[int]]:
    targets = [-1] * num_ranks
    src_row_starts = [0] * num_ranks
    src_col_starts = [0] * num_ranks
    dst_row_starts = [0] * num_ranks
    dst_col_starts = [0] * num_ranks

    used_sources: set[int] = set()
    used_targets: set[int] = set()
    for transfer in transfers:
        if transfer.source_rank in used_sources:
            raise ValueError("shape group contains a duplicate source rank.")
        if transfer.target_rank in used_targets:
            raise ValueError("shape group contains a duplicate target rank.")
        used_sources.add(transfer.source_rank)
        used_targets.add(transfer.target_rank)

        targets[transfer.source_rank] = transfer.target_rank
        src_row_starts[transfer.source_rank] = transfer.source_rect.row_start
        src_col_starts[transfer.source_rank] = transfer.source_rect.col_start
        dst_row_starts[transfer.source_rank] = transfer.target_rect.row_start
        dst_col_starts[transfer.source_rank] = transfer.target_rect.col_start

    return (
        targets,
        src_row_starts,
        src_col_starts,
        dst_row_starts,
        dst_col_starts,
    )


def _execute_rectangle_batches_shardmap(
    matrix: Array,
    scratch: Array,
    mesh: Mesh,
    matrix_specs: P,
    scratch_specs: P,
    batches,
    *,
    grid: ProcessGrid,
) -> tuple[Array, Array]:
    for batch in batches:
        transfers = getattr(batch, "transfers", None)
        if transfers is None:
            transfers = batch.moves
        for (row_count, col_count), shape_transfers in _shape_groups(transfers):
            (
                targets,
                src_row_starts,
                src_col_starts,
                dst_row_starts,
                dst_col_starts,
            ) = _attrs_for_shape_group(
                shape_transfers,
                num_ranks=grid.num_processes,
            )
            matrix, scratch = xla_rect_transfer_probe_shardmap(
                matrix,
                scratch,
                mesh,
                matrix_specs,
                scratch_specs,
                targets=targets,
                src_row_starts=src_row_starts,
                src_col_starts=src_col_starts,
                dst_row_starts=dst_row_starts,
                dst_col_starts=dst_col_starts,
                row_count=row_count,
                col_count=col_count,
            )
    return matrix, scratch


def execute_fragment_transfer_batches_shardmap(
    matrix: Array,
    scratch: Array,
    mesh: Mesh,
    matrix_specs: P,
    scratch_specs: P,
    batches: tuple[ExecutableFragmentTransferBatch, ...],
    *,
    grid: ProcessGrid,
) -> tuple[Array, Array]:
    """Execute planned 2D rectangle-transfer batches through the NCCL FFI path.

    This is still an investigation path, but it is no longer a single
    hard-coded transfer. It takes the CPU-planned executable schedule, turns
    each conflict-free batch into the static rank arrays consumed by the native
    rectangle-transfer primitive, and applies those batches sequentially.
    """
    if grid.num_processes <= 0:
        raise ValueError("grid must contain at least one process.")
    required_scratch = required_rect_transfer_scratch_size(batches)
    if required_scratch and scratch.shape[0] // grid.num_processes < required_scratch:
        raise ValueError(
            "scratch is too small for the largest executable rectangle transfer."
        )

    return _execute_rectangle_batches_shardmap(
        matrix,
        scratch,
        mesh,
        matrix_specs,
        scratch_specs,
        batches,
        grid=grid,
    )


def execute_edge_padding_compaction_batches_shardmap(
    matrix: Array,
    scratch: Array,
    mesh: Mesh,
    matrix_specs: P,
    scratch_specs: P,
    batches: tuple[EdgePaddingCompactionBatch, ...],
    *,
    grid: ProcessGrid,
) -> tuple[Array, Array]:
    """Execute edge-padding compaction waves through rectangle transfers.

    This is the first GPU checkpoint for top-left padding normalization. It
    executes the CPU-planned horizontal and vertical waves in order, using the
    existing rectangle-transfer FFI for both inter-rank and same-rank moves.
    Source holes are logical holes; this diagnostic executor does not clear the
    old source bytes after moving them.
    """
    if grid.num_processes <= 0:
        raise ValueError("grid must contain at least one process.")
    required_scratch = required_edge_padding_compaction_scratch_size(batches)
    if required_scratch and scratch.shape[0] // grid.num_processes < required_scratch:
        raise ValueError(
            "scratch is too small for the largest edge-padding compaction move."
        )
    return _execute_rectangle_batches_shardmap(
        matrix,
        scratch,
        mesh,
        matrix_specs,
        scratch_specs,
        batches,
        grid=grid,
    )


def execute_tile_aligned_native_2d_plan_shardmap(
    matrix: Array,
    scratch: Array,
    mesh: Mesh,
    matrix_specs: P,
    scratch_specs: P,
    *,
    grid: ProcessGrid,
    rank_map: ProcessRankMap | None = None,
    tile_rows: int,
    tile_cols: int,
) -> tuple[Array, Array]:
    """Execute the first fused native 2D redistribution checkpoint.

    Unlike :func:`execute_fragment_transfer_batches_shardmap`, this path does
    not pass a Python-built transfer schedule into repeated FFI calls. The
    native handler builds the tile-level two-phase schedule itself, groups
    conflict-free transfers, and executes the full redistribution in one FFI
    call. It is currently restricted to tile-aligned local shards so that every
    moved fragment has shape ``tile_rows x tile_cols``.
    """
    if grid.num_processes <= 0:
        raise ValueError("grid must contain at least one process.")
    if rank_map is None:
        rank_map = ProcessRankMap.row_major(grid)
    if rank_map.grid != grid:
        raise ValueError("rank_map must match the execution grid.")
    rank_map.require_cusolvermp_grid_mapping(
        "execute_tile_aligned_native_2d_plan_shardmap"
    )
    tile_rows = int(tile_rows)
    tile_cols = int(tile_cols)
    if tile_rows <= 0 or tile_cols <= 0:
        raise ValueError("tile_rows and tile_cols must be positive.")
    scratch_per_rank = scratch.shape[0] // grid.num_processes
    if matrix.shape[0] % grid.process_rows or matrix.shape[1] % grid.process_cols:
        raise ValueError("matrix shape must divide evenly over the process grid.")
    local_rows = matrix.shape[0] // grid.process_rows
    local_cols = matrix.shape[1] // grid.process_cols
    required_scratch = required_native_2d_plan_scratch_size(
        local_rows=local_rows,
        local_cols=local_cols,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
    )
    if scratch_per_rank < required_scratch:
        raise ValueError(
            "scratch is too small for the native slab-aligned 2D plan."
        )

    return xla_rect_2d_native_plan_shardmap(
        matrix,
        scratch,
        mesh,
        matrix_specs,
        scratch_specs,
        process_rows=grid.process_rows,
        process_cols=grid.process_cols,
        rank_map=rank_map.ranks,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
    )


def execute_padded_block_cyclic_2d_shardmap(
    matrix: Array,
    scratch: Array,
    mesh: Mesh,
    matrix_specs: P,
    scratch_specs: P,
    *,
    logical_rows: int,
    logical_cols: int,
    grid: ProcessGrid,
    rank_map: ProcessRankMap | None = None,
    tile_rows: int,
    tile_cols: int,
) -> tuple[Array, Array]:
    """Run the fused native padded 2D redistribution checkpoint end to end.

    ``matrix`` is expected to be the physical, per-rank padded buffer: each
    local shard has enough row and column capacity for an integer number of
    cuSOLVERMp tiles. The native handler compacts local padding holes to the
    global bottom/right edges and then applies the tile-aligned block-cyclic
    slab scheduler in the same FFI call.
    """
    if grid.num_processes <= 0:
        raise ValueError("grid must contain at least one process.")
    if rank_map is None:
        rank_map = ProcessRankMap.row_major(grid)
    if rank_map.grid != grid:
        raise ValueError("rank_map must match the execution grid.")
    rank_map.require_cusolvermp_grid_mapping(
        "execute_padded_block_cyclic_2d_shardmap"
    )
    tile_shape = TileShape(rows=int(tile_rows), cols=int(tile_cols))
    compaction_plan = build_edge_padding_compaction_plan(
        logical_rows=logical_rows,
        logical_cols=logical_cols,
        tile_shape=tile_shape,
        grid=grid,
    )
    padding = compaction_plan.padding
    expected_shape = (padding.physical_rows, padding.physical_cols)
    if tuple(matrix.shape) != expected_shape:
        raise ValueError(
            "matrix shape must be the physical padded shape "
            f"{expected_shape}, got {tuple(matrix.shape)}."
        )

    required_scratch = required_padded_block_cyclic_2d_scratch_size(
        logical_rows=logical_rows,
        logical_cols=logical_cols,
        grid=grid,
        tile_rows=tile_shape.rows,
        tile_cols=tile_shape.cols,
    )
    if required_scratch and scratch.shape[0] // grid.num_processes < required_scratch:
        raise ValueError(
            "scratch is too small for padded 2D block-cyclic redistribution."
        )

    return xla_rect_padded_2d_native_plan_shardmap(
        matrix,
        scratch,
        mesh,
        matrix_specs,
        scratch_specs,
        process_rows=grid.process_rows,
        process_cols=grid.process_cols,
        rank_map=rank_map.ranks,
        tile_rows=tile_shape.rows,
        tile_cols=tile_shape.cols,
        logical_rows=logical_rows,
        logical_cols=logical_cols,
        reverse=False,
    )


def execute_reverse_padded_block_cyclic_2d_shardmap(
    matrix: Array,
    scratch: Array,
    mesh: Mesh,
    matrix_specs: P,
    scratch_specs: P,
    *,
    logical_rows: int,
    logical_cols: int,
    grid: ProcessGrid,
    rank_map: ProcessRankMap | None = None,
    tile_rows: int,
    tile_cols: int,
) -> tuple[Array, Array]:
    """Undo the fused native padded 2D redistribution.

    ``matrix`` is expected to be in the local cuSOLVERMp 2D block-cyclic
    layout produced by :func:`execute_padded_block_cyclic_2d_shardmap`. The
    native handler first applies the inverse tile-slab scheduler and then
    reverses edge-padding compaction, returning a per-rank padded block-sharded
    layout. Padding bytes are not meaningful and should be sliced away by the
    caller.
    """
    if grid.num_processes <= 0:
        raise ValueError("grid must contain at least one process.")
    if rank_map is None:
        rank_map = ProcessRankMap.row_major(grid)
    if rank_map.grid != grid:
        raise ValueError("rank_map must match the execution grid.")
    rank_map.require_cusolvermp_grid_mapping(
        "execute_reverse_padded_block_cyclic_2d_shardmap"
    )
    tile_shape = TileShape(rows=int(tile_rows), cols=int(tile_cols))
    compaction_plan = build_edge_padding_compaction_plan(
        logical_rows=logical_rows,
        logical_cols=logical_cols,
        tile_shape=tile_shape,
        grid=grid,
    )
    padding = compaction_plan.padding
    expected_shape = (padding.physical_rows, padding.physical_cols)
    if tuple(matrix.shape) != expected_shape:
        raise ValueError(
            "matrix shape must be the physical padded shape "
            f"{expected_shape}, got {tuple(matrix.shape)}."
        )

    required_scratch = required_padded_block_cyclic_2d_scratch_size(
        logical_rows=logical_rows,
        logical_cols=logical_cols,
        grid=grid,
        tile_rows=tile_shape.rows,
        tile_cols=tile_shape.cols,
    )
    if required_scratch and scratch.shape[0] // grid.num_processes < required_scratch:
        raise ValueError(
            "scratch is too small for reverse padded 2D block-cyclic "
            "redistribution."
        )

    return xla_rect_padded_2d_native_plan_shardmap(
        matrix,
        scratch,
        mesh,
        matrix_specs,
        scratch_specs,
        process_rows=grid.process_rows,
        process_cols=grid.process_cols,
        rank_map=rank_map.ranks,
        tile_rows=tile_shape.rows,
        tile_cols=tile_shape.cols,
        logical_rows=logical_rows,
        logical_cols=logical_cols,
        reverse=True,
    )
