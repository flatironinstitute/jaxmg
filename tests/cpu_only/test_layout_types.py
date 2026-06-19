import pytest

from jaxmg._layout_types import (
    ProcessGrid,
    TileShape,
    validate_nonempty_block_cyclic_ownership,
)


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
