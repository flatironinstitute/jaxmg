import pytest

from jaxmg._block_cyclic_2d_plan import (
    ProcessGrid,
    TileShape,
    block_cyclic_tile_owner,
    build_two_phase_fragment_transfer_schedule,
    build_two_phase_2d_plan,
    calculate_2d_padding,
    classify_proposed_phase_regions,
    classify_rectangular_region,
    transfer_phase_groups,
)


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
