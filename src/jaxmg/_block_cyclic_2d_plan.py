from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


MemoryLayout = Literal["row_major", "column_major"]
RedistributionPhase = Literal["column_owner", "row_owner"]


@dataclass(frozen=True)
class ProcessGrid:
    """2D process grid used by a cuSOLVERMp descriptor."""

    process_rows: int
    process_cols: int

    def __post_init__(self) -> None:
        if self.process_rows <= 0:
            raise ValueError("process_rows must be positive.")
        if self.process_cols <= 0:
            raise ValueError("process_cols must be positive.")

    @property
    def num_processes(self) -> int:
        return self.process_rows * self.process_cols

    def rank(self, process_row: int, process_col: int) -> int:
        if not 0 <= process_row < self.process_rows:
            raise ValueError("process_row out of range.")
        if not 0 <= process_col < self.process_cols:
            raise ValueError("process_col out of range.")
        return process_row * self.process_cols + process_col


@dataclass(frozen=True)
class TileShape:
    """cuSOLVERMp matrix tile shape."""

    rows: int
    cols: int

    def __post_init__(self) -> None:
        if self.rows <= 0:
            raise ValueError("tile row size must be positive.")
        if self.cols <= 0:
            raise ValueError("tile column size must be positive.")


@dataclass(frozen=True)
class Owner2D:
    """Process-grid coordinate for a tile or tile fragment."""

    process_row: int
    process_col: int

    def rank(self, grid: ProcessGrid) -> int:
        return grid.rank(self.process_row, self.process_col)


@dataclass(frozen=True)
class MatrixPadding2D:
    """Per-process padding needed to make both local axes tile-aligned."""

    logical_rows: int
    logical_cols: int
    local_logical_rows: int
    local_logical_cols: int
    row_padding_per_process: int
    col_padding_per_process: int
    local_physical_rows: int
    local_physical_cols: int
    physical_rows: int
    physical_cols: int

    @property
    def needs_padding(self) -> bool:
        return self.row_padding_per_process != 0 or self.col_padding_per_process != 0

    @property
    def source_blocks_are_tile_aligned(self) -> bool:
        return self.row_padding_per_process == 0 and self.col_padding_per_process == 0


@dataclass(frozen=True)
class TileExtent:
    """Logical matrix extent covered by one MB_A x NB_A tile."""

    tile_row: int
    tile_col: int
    row_start: int
    row_stop: int
    col_start: int
    col_stop: int

    @property
    def row_count(self) -> int:
        return self.row_stop - self.row_start

    @property
    def col_count(self) -> int:
        return self.col_stop - self.col_start


@dataclass(frozen=True)
class TileFragment:
    """Part of a logical tile that is initially owned by one block-sharded rank."""

    tile_row: int
    tile_col: int
    source_owner: Owner2D
    row_start: int
    row_stop: int
    col_start: int
    col_stop: int

    @property
    def row_count(self) -> int:
        return self.row_stop - self.row_start

    @property
    def col_count(self) -> int:
        return self.col_stop - self.col_start


@dataclass(frozen=True)
class TwoPhaseTileMove:
    """One owner-coordinate move in the proposed separable 2D redistribution."""

    phase: RedistributionPhase
    tile_row: int
    tile_col: int
    source_owner: Owner2D
    target_owner: Owner2D


@dataclass(frozen=True)
class MemoryAccessClassification:
    """Whether a rectangular region is contiguous under a local memory layout."""

    layout: MemoryLayout
    row_count: int
    col_count: int
    local_rows: int
    local_cols: int
    contiguous: bool
    reason: str

    @property
    def requires_pack(self) -> bool:
        return not self.contiguous


@dataclass(frozen=True)
class BlockCyclic2DPlan:
    """CPU-only investigation plan for a 2D block-sharded to block-cyclic map."""

    grid: ProcessGrid
    tile_shape: TileShape
    padding: MatrixPadding2D
    tile_extents: tuple[TileExtent, ...]
    tile_fragments: tuple[TileFragment, ...]
    moves: tuple[TwoPhaseTileMove, ...]

    @property
    def split_tiles(self) -> tuple[tuple[int, int], ...]:
        counts: dict[tuple[int, int], int] = {}
        for fragment in self.tile_fragments:
            key = (fragment.tile_row, fragment.tile_col)
            counts[key] = counts.get(key, 0) + 1
        return tuple(key for key, count in counts.items() if count > 1)

    @property
    def can_move_whole_tiles_from_initial_layout(self) -> bool:
        return not self.split_tiles

    @property
    def column_owner_moves(self) -> tuple[TwoPhaseTileMove, ...]:
        return tuple(move for move in self.moves if move.phase == "column_owner")

    @property
    def row_owner_moves(self) -> tuple[TwoPhaseTileMove, ...]:
        return tuple(move for move in self.moves if move.phase == "row_owner")


def ceil_div(lhs: int, rhs: int) -> int:
    if rhs <= 0:
        raise ValueError("rhs must be positive.")
    return -(-lhs // rhs)


def calculate_axis_padding(local_size: int, tile_size: int) -> int:
    """Match the 1D JAXMg padding rule for one local shard axis."""
    if local_size < 0:
        raise ValueError("local_size must be non-negative.")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive.")
    return (-local_size) % tile_size


def calculate_2d_padding(
    logical_rows: int,
    logical_cols: int,
    grid: ProcessGrid,
    tile_shape: TileShape,
) -> MatrixPadding2D:
    """Return per-process padding for a block-sharded 2D JAX input layout."""
    if logical_rows <= 0 or logical_cols <= 0:
        raise ValueError("logical dimensions must be positive.")
    if logical_rows % grid.process_rows != 0:
        raise ValueError("logical_rows must be divisible by process_rows.")
    if logical_cols % grid.process_cols != 0:
        raise ValueError("logical_cols must be divisible by process_cols.")

    local_logical_rows = logical_rows // grid.process_rows
    local_logical_cols = logical_cols // grid.process_cols
    row_padding = calculate_axis_padding(local_logical_rows, tile_shape.rows)
    col_padding = calculate_axis_padding(local_logical_cols, tile_shape.cols)
    local_physical_rows = local_logical_rows + row_padding
    local_physical_cols = local_logical_cols + col_padding

    return MatrixPadding2D(
        logical_rows=logical_rows,
        logical_cols=logical_cols,
        local_logical_rows=local_logical_rows,
        local_logical_cols=local_logical_cols,
        row_padding_per_process=row_padding,
        col_padding_per_process=col_padding,
        local_physical_rows=local_physical_rows,
        local_physical_cols=local_physical_cols,
        physical_rows=local_physical_rows * grid.process_rows,
        physical_cols=local_physical_cols * grid.process_cols,
    )


def block_sharded_owner(
    row: int,
    col: int,
    grid: ProcessGrid,
    padding: MatrixPadding2D,
) -> Owner2D:
    """Owner of a logical element in the initial 2D block-sharded layout."""
    if not 0 <= row < padding.logical_rows:
        raise ValueError("row out of logical range.")
    if not 0 <= col < padding.logical_cols:
        raise ValueError("col out of logical range.")
    process_row = min(row // padding.local_logical_rows, grid.process_rows - 1)
    process_col = min(col // padding.local_logical_cols, grid.process_cols - 1)
    return Owner2D(process_row, process_col)


def block_cyclic_tile_owner(
    tile_row: int,
    tile_col: int,
    grid: ProcessGrid,
) -> Owner2D:
    """cuSOLVERMp-style 2D block-cyclic tile owner."""
    if tile_row < 0 or tile_col < 0:
        raise ValueError("tile indices must be non-negative.")
    return Owner2D(tile_row % grid.process_rows, tile_col % grid.process_cols)


def build_logical_tile_extents(
    logical_rows: int,
    logical_cols: int,
    tile_shape: TileShape,
) -> tuple[TileExtent, ...]:
    """Enumerate the logical tile grid, including partial matrix-edge tiles."""
    if logical_rows <= 0 or logical_cols <= 0:
        raise ValueError("logical dimensions must be positive.")
    extents: list[TileExtent] = []
    for tile_row in range(ceil_div(logical_rows, tile_shape.rows)):
        row_start = tile_row * tile_shape.rows
        row_stop = min(row_start + tile_shape.rows, logical_rows)
        for tile_col in range(ceil_div(logical_cols, tile_shape.cols)):
            col_start = tile_col * tile_shape.cols
            col_stop = min(col_start + tile_shape.cols, logical_cols)
            extents.append(
                TileExtent(
                    tile_row=tile_row,
                    tile_col=tile_col,
                    row_start=row_start,
                    row_stop=row_stop,
                    col_start=col_start,
                    col_stop=col_stop,
                )
            )
    return tuple(extents)


def split_tile_over_initial_owners(
    extent: TileExtent,
    grid: ProcessGrid,
    padding: MatrixPadding2D,
) -> tuple[TileFragment, ...]:
    """Split a logical tile where it crosses initial block-sharding boundaries."""
    row_cuts = {extent.row_start, extent.row_stop}
    col_cuts = {extent.col_start, extent.col_stop}

    for process_row in range(1, grid.process_rows):
        boundary = process_row * padding.local_logical_rows
        if extent.row_start < boundary < extent.row_stop:
            row_cuts.add(boundary)
    for process_col in range(1, grid.process_cols):
        boundary = process_col * padding.local_logical_cols
        if extent.col_start < boundary < extent.col_stop:
            col_cuts.add(boundary)

    rows = sorted(row_cuts)
    cols = sorted(col_cuts)
    fragments: list[TileFragment] = []
    for row_start, row_stop in zip(rows, rows[1:]):
        for col_start, col_stop in zip(cols, cols[1:]):
            owner = block_sharded_owner(row_start, col_start, grid, padding)
            fragments.append(
                TileFragment(
                    tile_row=extent.tile_row,
                    tile_col=extent.tile_col,
                    source_owner=owner,
                    row_start=row_start,
                    row_stop=row_stop,
                    col_start=col_start,
                    col_stop=col_stop,
                )
            )
    return tuple(fragments)


def build_two_phase_2d_plan(
    logical_rows: int,
    logical_cols: int,
    tile_shape: TileShape,
    grid: ProcessGrid,
) -> BlockCyclic2DPlan:
    """Plan the proposed column-owner then row-owner 2D redistribution.

    This is deliberately CPU-only and owner-coordinate based. It proves whether
    the 2D ownership map factors into two 1D-style permutations before any CUDA
    or NCCL implementation details are added.
    """
    padding = calculate_2d_padding(logical_rows, logical_cols, grid, tile_shape)
    extents = build_logical_tile_extents(logical_rows, logical_cols, tile_shape)
    fragments: list[TileFragment] = []
    moves: list[TwoPhaseTileMove] = []

    for extent in extents:
        tile_fragments = split_tile_over_initial_owners(extent, grid, padding)
        fragments.extend(tile_fragments)

        target_owner = block_cyclic_tile_owner(
            extent.tile_row, extent.tile_col, grid
        )
        for fragment in tile_fragments:
            after_column_owner = Owner2D(
                fragment.source_owner.process_row, target_owner.process_col
            )
            if fragment.source_owner.process_col != target_owner.process_col:
                moves.append(
                    TwoPhaseTileMove(
                        phase="column_owner",
                        tile_row=extent.tile_row,
                        tile_col=extent.tile_col,
                        source_owner=fragment.source_owner,
                        target_owner=after_column_owner,
                    )
                )
            if after_column_owner.process_row != target_owner.process_row:
                moves.append(
                    TwoPhaseTileMove(
                        phase="row_owner",
                        tile_row=extent.tile_row,
                        tile_col=extent.tile_col,
                        source_owner=after_column_owner,
                        target_owner=target_owner,
                    )
                )

    return BlockCyclic2DPlan(
        grid=grid,
        tile_shape=tile_shape,
        padding=padding,
        tile_extents=extents,
        tile_fragments=tuple(fragments),
        moves=tuple(moves),
    )


def classify_rectangular_region(
    row_count: int,
    col_count: int,
    local_rows: int,
    local_cols: int,
    *,
    layout: MemoryLayout = "row_major",
) -> MemoryAccessClassification:
    """Classify whether a local rectangular region can be moved contiguously."""
    if row_count <= 0 or col_count <= 0:
        raise ValueError("region dimensions must be positive.")
    if local_rows <= 0 or local_cols <= 0:
        raise ValueError("local matrix dimensions must be positive.")
    if row_count > local_rows or col_count > local_cols:
        raise ValueError("region cannot exceed local matrix dimensions.")

    if layout == "row_major":
        if row_count == 1:
            return MemoryAccessClassification(
                layout, row_count, col_count, local_rows, local_cols, True,
                "one row is contiguous in the current native offset model",
            )
        if col_count == local_cols:
            return MemoryAccessClassification(
                layout, row_count, col_count, local_rows, local_cols, True,
                "full-width row slab is contiguous in the current native offset model",
            )
        return MemoryAccessClassification(
            layout, row_count, col_count, local_rows, local_cols, False,
            "multi-row sub-column region is strided and needs pack/unpack",
        )

    if layout == "column_major":
        if col_count == 1:
            return MemoryAccessClassification(
                layout, row_count, col_count, local_rows, local_cols, True,
                "one column is contiguous in column-major layout",
            )
        if row_count == local_rows:
            return MemoryAccessClassification(
                layout, row_count, col_count, local_rows, local_cols, True,
                "full-height column slab is contiguous in column-major layout",
            )
        return MemoryAccessClassification(
            layout, row_count, col_count, local_rows, local_cols, False,
            "multi-column sub-row region is strided and needs pack/unpack",
        )

    raise ValueError(f"unknown memory layout {layout!r}.")


def classify_proposed_phase_regions(
    plan: BlockCyclic2DPlan,
    *,
    layout: MemoryLayout = "row_major",
) -> dict[str, MemoryAccessClassification]:
    """Classify the natural grouped regions for the two separable phases.

    For the current row-major native model, moving a column tile through a
    process row is a strided sub-column region, while moving a complete row slab
    inside one process column is contiguous if grouped over all local columns.
    """
    padding = plan.padding
    tile_shape = plan.tile_shape
    return {
        "column_tile": classify_rectangular_region(
            padding.local_physical_rows,
            min(tile_shape.cols, padding.local_physical_cols),
            padding.local_physical_rows,
            padding.local_physical_cols,
            layout=layout,
        ),
        "row_slab": classify_rectangular_region(
            min(tile_shape.rows, padding.local_physical_rows),
            padding.local_physical_cols,
            padding.local_physical_rows,
            padding.local_physical_cols,
            layout=layout,
        ),
        "single_tile": classify_rectangular_region(
            min(tile_shape.rows, padding.local_physical_rows),
            min(tile_shape.cols, padding.local_physical_cols),
            padding.local_physical_rows,
            padding.local_physical_cols,
            layout=layout,
        ),
    }
