from __future__ import annotations

from collections import defaultdict

from jax import Array
from jax.sharding import Mesh, PartitionSpec as P

from ._block_cyclic_2d_plan import (
    ExecutableFragmentTransfer,
    ExecutableFragmentTransferBatch,
    ProcessGrid,
)
from ._xla_comm_probe import (
    xla_rect_2d_native_plan_shardmap,
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
    used for the current CollectivePermute round.
    """
    local_rows = int(local_rows)
    local_cols = int(local_cols)
    tile_rows = int(tile_rows)
    tile_cols = int(tile_cols)
    if min(local_rows, local_cols, tile_rows, tile_cols) <= 0:
        raise ValueError("local and tile dimensions must be positive.")
    return 3 * max(local_rows * tile_cols, tile_rows * local_cols)


def _shape_groups(
    transfers: tuple[ExecutableFragmentTransfer, ...],
) -> tuple[tuple[tuple[int, int], tuple[ExecutableFragmentTransfer, ...]], ...]:
    grouped: dict[tuple[int, int], list[ExecutableFragmentTransfer]] = defaultdict(list)
    for transfer in transfers:
        grouped[(transfer.row_count, transfer.col_count)].append(transfer)
    return tuple((shape, tuple(grouped[shape])) for shape in sorted(grouped))


def _attrs_for_shape_group(
    transfers: tuple[ExecutableFragmentTransfer, ...],
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


def execute_fragment_transfer_batches_shardmap(
    matrix: Array,
    scratch: Array,
    mesh: Mesh,
    matrix_specs: P,
    scratch_specs: P,
    batches: tuple[ExecutableFragmentTransferBatch, ...],
    *,
    grid: ProcessGrid,
    layout="row_major",
) -> tuple[Array, Array]:
    """Execute planned 2D rectangle-transfer batches through XLA FFI.

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

    for batch in batches:
        for (row_count, col_count), transfers in _shape_groups(batch.transfers):
            (
                targets,
                src_row_starts,
                src_col_starts,
                dst_row_starts,
                dst_col_starts,
            ) = _attrs_for_shape_group(
                transfers,
                num_ranks=grid.num_processes,
            )
            matrix, scratch = xla_rect_transfer_probe_shardmap(
                matrix,
                scratch,
                mesh,
                matrix_specs,
                scratch_specs,
                layout=layout,
                targets=targets,
                src_row_starts=src_row_starts,
                src_col_starts=src_col_starts,
                dst_row_starts=dst_row_starts,
                dst_col_starts=dst_col_starts,
                row_count=row_count,
                col_count=col_count,
            )

    return matrix, scratch


def execute_tile_aligned_native_2d_plan_shardmap(
    matrix: Array,
    scratch: Array,
    mesh: Mesh,
    matrix_specs: P,
    scratch_specs: P,
    *,
    grid: ProcessGrid,
    tile_rows: int,
    tile_cols: int,
    layout="row_major",
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
        layout=layout,
        process_rows=grid.process_rows,
        process_cols=grid.process_cols,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
    )
