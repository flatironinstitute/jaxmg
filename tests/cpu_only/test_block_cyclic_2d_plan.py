import numpy as np
import pytest

from jaxmg._block_cyclic_2d_plan import (
    ProcessGrid,
    TileShape,
    batch_edge_padding_compaction_moves,
    batch_executable_fragment_transfers,
    batch_fragment_transfers,
    block_cyclic_local_rect,
    block_cyclic_tile_owner,
    block_sharded_local_rect,
    build_edge_padding_compaction_plan,
    build_executable_fragment_transfer_schedule,
    build_two_phase_fragment_transfer_schedule,
    build_two_phase_2d_plan,
    calculate_2d_padding,
    classify_proposed_phase_regions,
    classify_rectangular_region,
    transfer_phase_groups,
)
from jaxmg._block_cyclic_2d_execute import required_rect_transfer_scratch_size


def _rect_slice(rect):
    return (
        slice(rect.row_start, rect.row_stop),
        slice(rect.col_start, rect.col_stop),
    )


def _initial_padded_buffers(host, plan):
    buffers = {}
    padding = plan.padding
    for process_row in range(plan.grid.process_rows):
        for process_col in range(plan.grid.process_cols):
            rank = plan.grid.rank(process_row, process_col)
            local = np.full(
                (padding.local_physical_rows, padding.local_physical_cols),
                -1,
                dtype=host.dtype,
            )
            local[: padding.local_logical_rows, : padding.local_logical_cols] = host[
                process_row
                * padding.local_logical_rows : (process_row + 1)
                * padding.local_logical_rows,
                process_col
                * padding.local_logical_cols : (process_col + 1)
                * padding.local_logical_cols,
            ]
            buffers[rank] = local
    return buffers


def _apply_edge_padding_batches(buffers, batches):
    for batch in batches:
        payloads = [
            (move, buffers[move.source_rank][_rect_slice(move.source_rect)].copy())
            for move in batch.moves
        ]
        for move, _ in payloads:
            buffers[move.source_rank][_rect_slice(move.source_rect)] = -1
        for move, payload in payloads:
            buffers[move.target_rank][_rect_slice(move.target_rect)] = payload


def _assemble_physical_buffers(buffers, plan):
    padding = plan.padding
    out = np.empty((padding.physical_rows, padding.physical_cols), dtype=int)
    for process_row in range(plan.grid.process_rows):
        for process_col in range(plan.grid.process_cols):
            rank = plan.grid.rank(process_row, process_col)
            out[
                process_row
                * padding.local_physical_rows : (process_row + 1)
                * padding.local_physical_rows,
                process_col
                * padding.local_physical_cols : (process_col + 1)
                * padding.local_physical_cols,
            ] = buffers[rank]
    return out


def _cusolvermp_numroc(n, block_size, process_coord, source_process, process_count):
    count = 0
    for index in range(n):
        tile = index // block_size
        if (source_process + tile) % process_count == process_coord:
            count += 1
    return count


def _cusolvermp_row_major_rank(process_row, process_col, grid):
    return process_row * grid.process_cols + process_col


def _cusolvermp_block_cyclic_reference_buffers(
    host,
    *,
    grid,
    tile_shape,
    lld,
    local_col_capacity,
    rsrc=0,
    csrc=0,
):
    """Reference local buffers for a row-major cuSOLVERMp device grid.

    This mirrors the relevant parts of the NVIDIA sample setup for the common
    ``RSRC_A = CSRC_A = 0`` case: ownership is 2D block-cyclic, ranks are mapped
    with ``CUSOLVERMP_GRID_MAPPING_ROW_MAJOR``, and each local matrix is indexed
    as a column-major array with leading dimension ``lld``. The padded capacity
    can be larger than the minimum ``NUMROC`` result; entries outside the
    descriptor-visible logical matrix stay as ``-1``.
    """
    buffers = {
        rank: np.full((lld, local_col_capacity), -1, dtype=host.dtype)
        for rank in range(grid.num_processes)
    }
    local_rows_by_process = [
        _cusolvermp_numroc(
            host.shape[0],
            tile_shape.rows,
            process_row,
            rsrc,
            grid.process_rows,
        )
        for process_row in range(grid.process_rows)
    ]
    local_cols_by_process = [
        _cusolvermp_numroc(
            host.shape[1],
            tile_shape.cols,
            process_col,
            csrc,
            grid.process_cols,
        )
        for process_col in range(grid.process_cols)
    ]
    assert lld >= max(local_rows_by_process)
    assert local_col_capacity >= max(local_cols_by_process)

    for global_row in range(host.shape[0]):
        tile_row = global_row // tile_shape.rows
        row_in_tile = global_row % tile_shape.rows
        process_row = (rsrc + tile_row) % grid.process_rows
        local_row = (tile_row // grid.process_rows) * tile_shape.rows + row_in_tile

        for global_col in range(host.shape[1]):
            tile_col = global_col // tile_shape.cols
            col_in_tile = global_col % tile_shape.cols
            process_col = (csrc + tile_col) % grid.process_cols
            local_col = (
                (tile_col // grid.process_cols) * tile_shape.cols + col_in_tile
            )
            rank = _cusolvermp_row_major_rank(process_row, process_col, grid)
            buffers[rank][local_row, local_col] = host[global_row, global_col]

    return buffers


def test_2d_padding_matches_independent_1d_axis_rule():
    padding = calculate_2d_padding(
        logical_rows=20,
        logical_cols=20,
        grid=ProcessGrid(process_rows=2, process_cols=4),
        tile_shape=TileShape(rows=3, cols=2),
    )

    assert padding.local_logical_rows == 10
    assert padding.local_logical_cols == 5
    assert padding.row_padding_per_process == 2
    assert padding.col_padding_per_process == 1
    assert padding.local_physical_rows == 12
    assert padding.local_physical_cols == 6
    assert padding.physical_rows == 24
    assert padding.physical_cols == 24
    assert padding.needs_padding
    assert not padding.source_blocks_are_tile_aligned


def test_tile_aligned_source_blocks_can_move_whole_tiles():
    plan = build_two_phase_2d_plan(
        logical_rows=16,
        logical_cols=32,
        tile_shape=TileShape(rows=4, cols=4),
        grid=ProcessGrid(process_rows=2, process_cols=4),
    )

    assert plan.can_move_whole_tiles_from_initial_layout
    assert plan.split_tiles == ()
    assert not plan.padding.needs_padding


def test_non_tile_aligned_source_blocks_report_split_tiles():
    plan = build_two_phase_2d_plan(
        logical_rows=10,
        logical_cols=10,
        tile_shape=TileShape(rows=4, cols=4),
        grid=ProcessGrid(process_rows=2, process_cols=2),
    )

    assert plan.padding.row_padding_per_process == 3
    assert plan.padding.col_padding_per_process == 3
    assert not plan.can_move_whole_tiles_from_initial_layout
    assert (1, 0) in plan.split_tiles
    assert (0, 1) in plan.split_tiles
    assert (1, 1) in plan.split_tiles


def test_edge_padding_compaction_is_empty_when_source_is_tile_aligned():
    plan = build_edge_padding_compaction_plan(
        logical_rows=16,
        logical_cols=16,
        tile_shape=TileShape(rows=4, cols=4),
        grid=ProcessGrid(process_rows=2, process_cols=2),
    )

    assert not plan.padding.needs_padding
    assert plan.moves == ()
    assert batch_edge_padding_compaction_moves(plan.moves) == ()


def test_edge_padding_compaction_uses_maximal_column_intervals():
    plan = build_edge_padding_compaction_plan(
        logical_rows=2,
        logical_cols=9,
        tile_shape=TileShape(rows=2, cols=5),
        grid=ProcessGrid(process_rows=1, process_cols=3),
    )

    assert [move.source_rank for move in plan.horizontal_moves] == [1, 1, 2]
    assert [move.target_rank for move in plan.horizontal_moves] == [0, 1, 1]
    assert [move.source_rect.col_start for move in plan.horizontal_moves] == [0, 2, 0]
    assert [move.target_rect.col_start for move in plan.horizontal_moves] == [3, 0, 1]
    assert [move.source_rect.col_count for move in plan.horizontal_moves] == [2, 1, 3]
    assert plan.vertical_moves == ()


def test_edge_padding_compaction_batches_independent_process_rows_by_wave():
    plan = build_edge_padding_compaction_plan(
        logical_rows=4,
        logical_cols=9,
        tile_shape=TileShape(rows=2, cols=5),
        grid=ProcessGrid(process_rows=2, process_cols=3),
    )
    batches = batch_edge_padding_compaction_moves(plan.moves)

    assert [batch.phase for batch in batches] == [
        "horizontal",
        "horizontal",
        "horizontal",
    ]
    assert [batch.wave for batch in batches] == [0, 1, 2]
    assert [len(batch.moves) for batch in batches] == [2, 2, 2]
    for batch in batches:
        assert len(batch.source_ranks) == len(set(batch.source_ranks))
        assert len(batch.target_ranks) == len(set(batch.target_ranks))


def test_edge_padding_compaction_moves_real_data_to_top_left():
    host = np.arange(100, dtype=int).reshape(10, 10)
    plan = build_edge_padding_compaction_plan(
        logical_rows=host.shape[0],
        logical_cols=host.shape[1],
        tile_shape=TileShape(rows=4, cols=4),
        grid=ProcessGrid(process_rows=2, process_cols=2),
    )
    buffers = _initial_padded_buffers(host, plan)
    batches = batch_edge_padding_compaction_moves(plan.moves)

    assert plan.padding.row_padding_per_process == 3
    assert plan.padding.col_padding_per_process == 3
    assert [batch.phase for batch in batches] == [
        "horizontal",
        "horizontal",
        "vertical",
        "vertical",
    ]
    assert [batch.wave for batch in batches] == [0, 1, 0, 1]

    _apply_edge_padding_batches(buffers, batches)
    physical = _assemble_physical_buffers(buffers, plan)

    np.testing.assert_array_equal(
        physical[: plan.padding.logical_rows, : plan.padding.logical_cols],
        host,
    )
    assert np.all(physical[: plan.padding.logical_rows, plan.padding.logical_cols :] == -1)
    assert np.all(physical[plan.padding.logical_rows :, :] == -1)


@pytest.mark.parametrize(
    "grid",
    [
        ProcessGrid(1, 4),
        ProcessGrid(2, 4),
        ProcessGrid(4, 1),
        ProcessGrid(4, 4),
    ],
)
def test_two_phase_owner_mapping_reaches_block_cyclic_owner(grid):
    plan = build_two_phase_2d_plan(
        logical_rows=32,
        logical_cols=32,
        tile_shape=TileShape(rows=4, cols=4),
        grid=grid,
    )

    state = {
        (fragment.tile_row, fragment.tile_col): fragment.source_owner
        for fragment in plan.tile_fragments
    }
    assert len(state) == len(plan.tile_extents)

    for move in plan.moves:
        key = (move.tile_row, move.tile_col)
        assert state[key] == move.source_owner
        state[key] = move.target_owner

    for tile_row, tile_col in state:
        assert state[(tile_row, tile_col)] == block_cyclic_tile_owner(
            tile_row, tile_col, grid
        )


def test_degenerate_process_grids_skip_unneeded_phase():
    row_wise_plan = build_two_phase_2d_plan(
        logical_rows=32,
        logical_cols=32,
        tile_shape=TileShape(rows=4, cols=4),
        grid=ProcessGrid(process_rows=4, process_cols=1),
    )
    col_wise_plan = build_two_phase_2d_plan(
        logical_rows=32,
        logical_cols=32,
        tile_shape=TileShape(rows=4, cols=4),
        grid=ProcessGrid(process_rows=1, process_cols=4),
    )

    assert row_wise_plan.column_owner_moves == ()
    assert col_wise_plan.row_owner_moves == ()


@pytest.mark.parametrize(
    "grid",
    [
        ProcessGrid(2, 2),
        ProcessGrid(2, 4),
        ProcessGrid(4, 2),
    ],
)
def test_fragment_transfer_schedule_reaches_block_cyclic_owner(grid):
    plan = build_two_phase_2d_plan(
        logical_rows=24,
        logical_cols=24,
        tile_shape=TileShape(rows=4, cols=4),
        grid=grid,
    )
    transfers = build_two_phase_fragment_transfer_schedule(plan)

    state = {
        fragment.key: fragment.source_owner
        for fragment in plan.tile_fragments
    }

    for transfer in transfers:
        assert state[transfer.fragment_key] == transfer.source_owner
        assert transfer.source_rank == transfer.source_owner.rank(grid)
        assert transfer.target_rank == transfer.target_owner.rank(grid)
        state[transfer.fragment_key] = transfer.target_owner

    for fragment in plan.tile_fragments:
        assert state[fragment.key] == block_cyclic_tile_owner(
            fragment.tile_row, fragment.tile_col, grid
        )


def test_fragment_transfer_schedule_preserves_split_tile_fragments():
    plan = build_two_phase_2d_plan(
        logical_rows=10,
        logical_cols=10,
        tile_shape=TileShape(rows=4, cols=4),
        grid=ProcessGrid(process_rows=2, process_cols=2),
    )
    transfers = build_two_phase_fragment_transfer_schedule(plan)

    assert not plan.can_move_whole_tiles_from_initial_layout
    assert len(plan.tile_fragments) > len(plan.tile_extents)
    assert len({fragment.key for fragment in plan.tile_fragments}) == len(
        plan.tile_fragments
    )
    assert {transfer.fragment_key for transfer in transfers}.issubset(
        {fragment.key for fragment in plan.tile_fragments}
    )


def test_transfer_schedule_exposes_phase_parallel_groups():
    plan = build_two_phase_2d_plan(
        logical_rows=16,
        logical_cols=32,
        tile_shape=TileShape(rows=4, cols=4),
        grid=ProcessGrid(process_rows=2, process_cols=4),
    )
    groups = transfer_phase_groups(build_two_phase_fragment_transfer_schedule(plan))

    assert groups["column_owner"] == (0, 1)
    assert groups["row_owner"] == (0, 1, 2, 3)


def test_transfer_schedule_records_rank_and_group_invariants():
    grid = ProcessGrid(process_rows=2, process_cols=4)
    plan = build_two_phase_2d_plan(
        logical_rows=16,
        logical_cols=32,
        tile_shape=TileShape(rows=4, cols=4),
        grid=grid,
    )
    transfers = build_two_phase_fragment_transfer_schedule(plan)

    assert transfers
    for transfer in transfers:
        assert transfer.source_rank != transfer.target_rank
        if transfer.phase == "column_owner":
            assert transfer.source_owner.process_row == transfer.target_owner.process_row
            assert transfer.phase_group == transfer.source_owner.process_row
        else:
            assert transfer.source_owner.process_col == transfer.target_owner.process_col
            assert transfer.phase_group == transfer.source_owner.process_col


def test_fragment_transfer_batches_preserve_phase_order_and_transfers():
    plan = build_two_phase_2d_plan(
        logical_rows=16,
        logical_cols=32,
        tile_shape=TileShape(rows=4, cols=4),
        grid=ProcessGrid(process_rows=2, process_cols=4),
    )
    transfers = build_two_phase_fragment_transfer_schedule(plan)
    batches = batch_fragment_transfers(transfers)

    assert batches
    phases = [batch.phase for batch in batches]
    assert phases == sorted(phases, key=("column_owner", "row_owner").index)

    batched_transfers = [
        transfer for batch in batches for transfer in batch.transfers
    ]
    assert sorted(t.fragment_key + (t.phase,) for t in batched_transfers) == sorted(
        t.fragment_key + (t.phase,) for t in transfers
    )


def test_fragment_transfer_batches_have_no_rank_conflicts():
    plan = build_two_phase_2d_plan(
        logical_rows=16,
        logical_cols=32,
        tile_shape=TileShape(rows=4, cols=4),
        grid=ProcessGrid(process_rows=2, process_cols=4),
    )
    batches = batch_fragment_transfers(
        build_two_phase_fragment_transfer_schedule(plan)
    )

    for batch in batches:
        assert len(batch.source_ranks) == len(set(batch.source_ranks))
        assert len(batch.target_ranks) == len(set(batch.target_ranks))
        assert batch.phase_groups


def test_fragment_transfer_batch_size_limit_is_respected():
    plan = build_two_phase_2d_plan(
        logical_rows=16,
        logical_cols=32,
        tile_shape=TileShape(rows=4, cols=4),
        grid=ProcessGrid(process_rows=2, process_cols=4),
    )
    batches = batch_fragment_transfers(
        build_two_phase_fragment_transfer_schedule(plan),
        max_transfers_per_batch=2,
    )

    assert batches
    assert all(len(batch.transfers) <= 2 for batch in batches)


def test_fragment_transfer_batch_size_rejects_non_positive_limit():
    with pytest.raises(ValueError, match="positive"):
        batch_fragment_transfers((), max_transfers_per_batch=0)


def test_current_row_major_layout_makes_column_phase_strided():
    plan = build_two_phase_2d_plan(
        logical_rows=16,
        logical_cols=32,
        tile_shape=TileShape(rows=4, cols=4),
        grid=ProcessGrid(process_rows=2, process_cols=4),
    )

    classes = classify_proposed_phase_regions(plan)

    assert classes["column_tile"].requires_pack
    assert "strided" in classes["column_tile"].reason
    assert classes["row_slab"].contiguous
    assert "full-width row slab" in classes["row_slab"].reason
    assert classes["single_tile"].requires_pack


def test_raw_fragment_schedule_marks_tile_transfers_as_pack_required():
    plan = build_two_phase_2d_plan(
        logical_rows=16,
        logical_cols=32,
        tile_shape=TileShape(rows=4, cols=4),
        grid=ProcessGrid(process_rows=2, process_cols=4),
    )
    transfers = build_two_phase_fragment_transfer_schedule(plan)

    assert transfers
    assert any(transfer.phase == "column_owner" for transfer in transfers)
    assert any(transfer.phase == "row_owner" for transfer in transfers)
    assert all(transfer.access.requires_pack for transfer in transfers)


def test_rectangular_region_classification_rejects_impossible_regions():
    with pytest.raises(ValueError, match="region cannot exceed"):
        classify_rectangular_region(
            row_count=5,
            col_count=2,
            local_rows=4,
            local_cols=4,
        )


def test_column_major_layout_flips_contiguous_axis():
    column = classify_rectangular_region(
        row_count=4,
        col_count=1,
        local_rows=8,
        local_cols=8,
        layout="column_major",
    )
    row = classify_rectangular_region(
        row_count=1,
        col_count=4,
        local_rows=8,
        local_cols=8,
        layout="column_major",
    )

    assert column.contiguous
    assert row.requires_pack


def test_executable_schedule_records_local_source_and_target_rectangles():
    grid = ProcessGrid(process_rows=1, process_cols=2)
    plan = build_two_phase_2d_plan(
        logical_rows=4,
        logical_cols=8,
        tile_shape=TileShape(rows=2, cols=2),
        grid=grid,
    )
    transfers = build_executable_fragment_transfer_schedule(plan)

    moved_tile_1 = next(
        transfer
        for transfer in transfers
        if transfer.tile_row == 0 and transfer.tile_col == 1
    )
    moved_tile_2 = next(
        transfer
        for transfer in transfers
        if transfer.tile_row == 0 and transfer.tile_col == 2
    )

    assert moved_tile_1.source_rank == 0
    assert moved_tile_1.target_rank == 1
    assert moved_tile_1.source_rect.row_start == 0
    assert moved_tile_1.source_rect.col_start == 2
    assert moved_tile_1.target_rect.row_start == 0
    assert moved_tile_1.target_rect.col_start == 0

    assert moved_tile_2.source_rank == 1
    assert moved_tile_2.target_rank == 0
    assert moved_tile_2.source_rect.row_start == 0
    assert moved_tile_2.source_rect.col_start == 0
    assert moved_tile_2.target_rect.row_start == 0
    assert moved_tile_2.target_rect.col_start == 2


def test_executable_schedule_includes_same_rank_local_relocations():
    grid = ProcessGrid(process_rows=1, process_cols=2)
    plan = build_two_phase_2d_plan(
        logical_rows=4,
        logical_cols=12,
        tile_shape=TileShape(rows=2, cols=2),
        grid=grid,
    )
    transfers = build_executable_fragment_transfer_schedule(plan)

    assert any(
        transfer.source_rank == transfer.target_rank
        and transfer.source_rect != transfer.target_rect
        for transfer in transfers
    )


def test_executable_transfer_batches_have_no_rank_conflicts():
    plan = build_two_phase_2d_plan(
        logical_rows=8,
        logical_cols=8,
        tile_shape=TileShape(rows=2, cols=2),
        grid=ProcessGrid(process_rows=2, process_cols=2),
    )
    batches = batch_executable_fragment_transfers(
        build_executable_fragment_transfer_schedule(plan)
    )

    assert batches
    for batch in batches:
        assert len(batch.source_ranks) == len(set(batch.source_ranks))
        assert len(batch.target_ranks) == len(set(batch.target_ranks))


def test_required_rect_transfer_scratch_size_uses_largest_fragment():
    plan = build_two_phase_2d_plan(
        logical_rows=4,
        logical_cols=8,
        tile_shape=TileShape(rows=2, cols=2),
        grid=ProcessGrid(process_rows=1, process_cols=2),
    )
    batches = batch_executable_fragment_transfers(
        build_executable_fragment_transfer_schedule(plan)
    )

    assert required_rect_transfer_scratch_size(batches) == 8


def test_executable_schedule_reaches_expected_block_cyclic_local_buffers():
    grid = ProcessGrid(process_rows=1, process_cols=2)
    plan = build_two_phase_2d_plan(
        logical_rows=4,
        logical_cols=8,
        tile_shape=TileShape(rows=2, cols=2),
        grid=grid,
    )
    transfers = build_executable_fragment_transfer_schedule(plan)
    batches = batch_executable_fragment_transfers(transfers)

    host = [
        [[row * 8 + col for col in range(4)] for row in range(4)],
        [[row * 8 + col for col in range(4, 8)] for row in range(4)],
    ]
    buffers = [[row[:] for row in rank] for rank in host]
    for batch in batches:
        payloads = []
        for transfer in batch.transfers:
            payload = [
                row[
                    transfer.source_rect.col_start : transfer.source_rect.col_stop
                ]
                for row in buffers[transfer.source_rank][
                    transfer.source_rect.row_start : transfer.source_rect.row_stop
                ]
            ]
            payloads.append((transfer, [row[:] for row in payload]))
        for transfer, payload in payloads:
            for row_offset, row in enumerate(payload):
                target_row = transfer.target_rect.row_start + row_offset
                target_col = transfer.target_rect.col_start
                buffers[transfer.target_rank][target_row][
                    target_col : target_col + transfer.col_count
                ] = row

    expected = [
        [[-1 for _ in range(4)] for _ in range(4)],
        [[-1 for _ in range(4)] for _ in range(4)],
    ]
    for fragment in plan.tile_fragments:
        source = block_sharded_local_rect(fragment, fragment.source_owner, plan)
        target = block_cyclic_local_rect(fragment, plan)
        target_rank = block_cyclic_tile_owner(
            fragment.tile_row, fragment.tile_col, grid
        ).rank(grid)
        source_rank = fragment.source_owner.rank(grid)
        for row_offset in range(fragment.row_count):
            for col_offset in range(fragment.col_count):
                expected[target_rank][target.row_start + row_offset][
                    target.col_start + col_offset
                ] = host[source_rank][source.row_start + row_offset][
                    source.col_start + col_offset
                ]

    assert buffers == expected


def test_padded_plan_matches_cusolvermp_row_major_local_buffer_reference():
    grid = ProcessGrid(process_rows=2, process_cols=2)
    tile_shape = TileShape(rows=4, cols=4)
    host = np.arange(100, dtype=int).reshape(10, 10)
    compaction_plan = build_edge_padding_compaction_plan(
        logical_rows=host.shape[0],
        logical_cols=host.shape[1],
        tile_shape=tile_shape,
        grid=grid,
    )
    buffers = _initial_padded_buffers(host, compaction_plan)
    _apply_edge_padding_batches(
        buffers,
        batch_edge_padding_compaction_moves(compaction_plan.moves),
    )

    physical_plan = build_two_phase_2d_plan(
        logical_rows=compaction_plan.padding.physical_rows,
        logical_cols=compaction_plan.padding.physical_cols,
        tile_shape=tile_shape,
        grid=grid,
    )
    physical_batches = batch_executable_fragment_transfers(
        build_executable_fragment_transfer_schedule(physical_plan)
    )
    for batch in physical_batches:
        payloads = [
            (
                transfer,
                buffers[transfer.source_rank][_rect_slice(transfer.source_rect)].copy(),
            )
            for transfer in batch.transfers
        ]
        for transfer, payload in payloads:
            buffers[transfer.target_rank][_rect_slice(transfer.target_rect)] = payload

    reference = _cusolvermp_block_cyclic_reference_buffers(
        host,
        grid=grid,
        tile_shape=tile_shape,
        lld=compaction_plan.padding.local_physical_rows,
        local_col_capacity=compaction_plan.padding.local_physical_cols,
    )

    for rank, expected in reference.items():
        np.testing.assert_array_equal(buffers[rank], expected)


def test_process_grid_rank_matches_cusolvermp_row_major_mapping():
    grid = ProcessGrid(process_rows=2, process_cols=3)

    assert [
        grid.rank(process_row, process_col)
        for process_row in range(grid.process_rows)
        for process_col in range(grid.process_cols)
    ] == [0, 1, 2, 3, 4, 5]
