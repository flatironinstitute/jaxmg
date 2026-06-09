import pytest

from jaxmg._block_cyclic_2d_plan import (
    ProcessGrid,
    TileShape,
    batch_executable_fragment_transfers,
    batch_fragment_transfers,
    block_cyclic_local_rect,
    block_cyclic_tile_owner,
    block_sharded_local_rect,
    build_executable_fragment_transfer_schedule,
    build_two_phase_fragment_transfer_schedule,
    build_two_phase_2d_plan,
    calculate_2d_padding,
    classify_proposed_phase_regions,
    classify_rectangular_region,
    transfer_phase_groups,
)
from jaxmg._block_cyclic_2d_execute import required_rect_transfer_scratch_size


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
