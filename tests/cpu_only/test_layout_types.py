import pytest

from jaxmg._cusolvermp_layout import rhs_distribution_columns
from jaxmg._layout_types import (
    ProcessGrid,
    ProcessRankMap,
    TileShape,
    calculate_2d_padding,
    validate_nonempty_block_cyclic_ownership,
)


def test_process_grid_rejects_empty_axes():
    """Both cuSOLVERMp process-grid axes must be positive."""
    with pytest.raises(ValueError, match="process_rows"):
        ProcessGrid(process_rows=0, process_cols=1)
    with pytest.raises(ValueError, match="process_cols"):
        ProcessGrid(process_rows=1, process_cols=0)


def test_process_rank_map_accepts_row_major_mapping():
    """Row-major mesh order maps directly to cuSOLVERMp row-major grids."""
    grid = ProcessGrid(process_rows=2, process_cols=3)
    rank_map = ProcessRankMap.row_major(grid)

    assert rank_map.ranks == (0, 1, 2, 3, 4, 5)
    assert rank_map.grid_mapping == "row_major"
    assert rank_map.cusolvermp_grid_mapping == 1


def test_process_rank_map_accepts_column_major_mapping():
    """Column-major mesh order maps directly to cuSOLVERMp column-major grids."""
    grid = ProcessGrid(process_rows=2, process_cols=3)
    rank_map = ProcessRankMap.column_major(grid)

    assert rank_map.ranks == (0, 2, 4, 1, 3, 5)
    assert rank_map.grid_mapping == "column_major"
    assert rank_map.cusolvermp_grid_mapping == 0


def test_process_rank_map_rejects_exotic_mapping():
    """Arbitrary mesh permutations are rejected before native code runs."""
    rank_map = ProcessRankMap(
        grid=ProcessGrid(process_rows=2, process_cols=2),
        ranks=(0, 2, 3, 1),
    )

    with pytest.raises(ValueError, match="row-major or column-major"):
        rank_map.require_cusolvermp_grid_mapping("test")


def test_calculate_2d_padding_matches_local_tile_capacity():
    """Padding is computed per local block, then scaled back globally."""
    padding = calculate_2d_padding(
        logical_rows=18,
        logical_cols=14,
        grid=ProcessGrid(process_rows=3, process_cols=2),
        tile_shape=TileShape(rows=4, cols=4),
    )

    assert padding.local_logical_rows == 6
    assert padding.local_logical_cols == 7
    assert padding.row_padding_per_process == 2
    assert padding.col_padding_per_process == 1
    assert padding.local_physical_rows == 8
    assert padding.local_physical_cols == 8
    assert padding.physical_rows == 24
    assert padding.physical_cols == 16


def test_rhs_distribution_columns_pads_skinny_rhs_for_process_columns():
    """POTRS pads RHS routing columns when NRHS is thinner than the grid."""
    assert rhs_distribution_columns(1, process_cols=4, pad=True) == 4
    assert rhs_distribution_columns(5, process_cols=4, pad=True) == 8
    assert rhs_distribution_columns(8, process_cols=4, pad=True) == 8


def test_rhs_distribution_columns_rejects_required_padding_when_disabled():
    """Skinny RHS support requires pad=True for multi-column grids."""
    with pytest.raises(ValueError, match="RHS column count"):
        rhs_distribution_columns(1, process_cols=4, pad=False)


def test_validate_nonempty_block_cyclic_ownership_accepts_full_tile_grid():
    """A grid is valid when each process row and column owns a tile."""
    validate_nonempty_block_cyclic_ownership(
        logical_rows=768,
        logical_cols=384,
        grid=ProcessGrid(process_rows=4, process_cols=2),
        tile_shape=TileShape(rows=96, cols=96),
        caller="test",
    )


def test_validate_nonempty_block_cyclic_ownership_rejects_empty_process_column():
    """Oversized tiles can leave process columns with no logical tiles."""
    with pytest.raises(ValueError, match="logical_tile_grid=\\(4, 1\\)"):
        validate_nonempty_block_cyclic_ownership(
            logical_rows=384,
            logical_cols=64,
            grid=ProcessGrid(process_rows=4, process_cols=2),
            tile_shape=TileShape(rows=96, cols=96),
            caller="test",
        )


def test_validate_nonempty_block_cyclic_ownership_rejects_empty_process_row():
    """Oversized tiles can leave process rows with no logical tiles."""
    with pytest.raises(ValueError, match="logical_tile_grid=\\(1, 4\\)"):
        validate_nonempty_block_cyclic_ownership(
            logical_rows=64,
            logical_cols=384,
            grid=ProcessGrid(process_rows=2, process_cols=4),
            tile_shape=TileShape(rows=96, cols=96),
            caller="test",
        )
