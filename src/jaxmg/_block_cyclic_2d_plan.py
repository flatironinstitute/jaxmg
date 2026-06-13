"""CPU planning utilities for JAX block-sharded to cuSOLVERMp layout changes.

cuSOLVERMp expects 2D block-cyclic ownership over a process grid and
column-major local matrices.  The objects in this module describe that mapping
without touching device memory.  They are used for three purposes:

1. validating the ownership mathematics against small CPU references;
2. computing local padding and scratch requirements for the public wrapper; and
3. documenting the dependency waves used by the native C++ redistribution.

The production cuSOLVERMp path no longer sends a Python-built move list to C++.
The native handler rebuilds the equivalent edge-padding and slab schedules from
scalar metadata, while this module remains the readable reference model and the
source of Python-side validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


RedistributionPhase = Literal["column_owner", "row_owner"]
EdgePaddingPhase = Literal["horizontal", "vertical"]


@dataclass(frozen=True)
class ProcessGrid:
    """2D process grid used by a cuSOLVERMp descriptor.

    The current planner maps ``(process_row, process_col)`` to a flat rank in
    row-major order. A cuSOLVERMp caller must therefore create the device grid
    with ``CUSOLVERMP_GRID_MAPPING_ROW_MAJOR`` unless this planner is extended to
    carry an explicit grid-mapping choice.
    """

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
class ProcessRankMap:
    """Row-major mapping from cuSOLVERMp process-grid coordinates to ranks.

    cuSOLVERMp itself uses the dense row-major rank convention encoded by its
    grid descriptor:

        rank = process_row * process_cols + process_col

    JAXMg requires the user-facing JAX mesh to follow that same device order.
    The object remains explicit at Python/native boundaries so validation can
    fail early with a readable error if a permuted mesh is supplied.
    """

    grid: ProcessGrid
    ranks: tuple[int, ...]

    def __post_init__(self) -> None:
        ranks = tuple(int(rank) for rank in self.ranks)
        object.__setattr__(self, "ranks", ranks)
        if len(ranks) != self.grid.num_processes:
            raise ValueError(
                "rank map length must match the process-grid size "
                f"{self.grid.num_processes}, got {len(ranks)}."
            )
        expected = set(range(self.grid.num_processes))
        actual = set(ranks)
        if actual != expected:
            raise ValueError(
                "rank map must be a permutation of communicator ranks "
                f"0..{self.grid.num_processes - 1}, got {ranks}."
            )

    @classmethod
    def row_major(cls, grid: ProcessGrid) -> "ProcessRankMap":
        return cls(grid=grid, ranks=tuple(range(grid.num_processes)))

    def rank(self, process_row: int, process_col: int) -> int:
        return self.ranks[self.grid.rank(process_row, process_col)]

    @property
    def is_row_major_identity(self) -> bool:
        return self.ranks == tuple(range(self.grid.num_processes))

    def require_row_major_identity(self, caller: str) -> None:
        if not self.is_row_major_identity:
            raise ValueError(
                f"{caller} requires the JAX mesh device order to match "
                "cuSOLVERMp row-major communicator rank order. Got process-grid "
                f"rank map {self.ranks}. Construct the mesh with devices in "
                "row-major order, for example "
                "Mesh(np.asarray(jax.devices()).reshape(process_rows, "
                "process_cols), axis_names), or use jax.make_mesh only when "
                "the resulting mesh has row-major device order."
            )


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

    @property
    def key(self) -> tuple[int, int, int, int, int, int]:
        return (
            self.tile_row,
            self.tile_col,
            self.row_start,
            self.row_stop,
            self.col_start,
            self.col_stop,
        )


@dataclass(frozen=True)
class TwoPhaseTileMove:
    """One owner-coordinate move in the proposed separable 2D redistribution."""

    phase: RedistributionPhase
    tile_row: int
    tile_col: int
    source_owner: Owner2D
    target_owner: Owner2D


@dataclass(frozen=True)
class TwoPhaseFragmentTransfer:
    """Concrete tile-fragment transfer used by the reference scheduler."""

    phase: RedistributionPhase
    tile_row: int
    tile_col: int
    row_start: int
    row_stop: int
    col_start: int
    col_stop: int
    source_owner: Owner2D
    target_owner: Owner2D
    source_rank: int
    target_rank: int
    phase_group: int
    access: MemoryAccessClassification

    @property
    def fragment_key(self) -> tuple[int, int, int, int, int, int]:
        return (
            self.tile_row,
            self.tile_col,
            self.row_start,
            self.row_stop,
            self.col_start,
            self.col_stop,
        )

    @property
    def row_count(self) -> int:
        return self.row_stop - self.row_start

    @property
    def col_count(self) -> int:
        return self.col_stop - self.col_start


@dataclass(frozen=True)
class FragmentTransferBatch:
    """One conflict-free round of fragment transfers.

    The native NCCL implementation submits equivalent conflict-free transfer
    groups while using a bounded per-rank scratch policy:
    no rank appears as a source or target more than once in the batch.
    """

    phase: RedistributionPhase
    round_index: int
    transfers: tuple[TwoPhaseFragmentTransfer, ...]

    @property
    def source_ranks(self) -> tuple[int, ...]:
        return tuple(transfer.source_rank for transfer in self.transfers)

    @property
    def target_ranks(self) -> tuple[int, ...]:
        return tuple(transfer.target_rank for transfer in self.transfers)

    @property
    def phase_groups(self) -> tuple[int, ...]:
        return tuple(
            sorted({transfer.phase_group for transfer in self.transfers})
        )


@dataclass(frozen=True)
class MemoryAccessClassification:
    """Whether a rectangular region is contiguous under a local memory layout."""

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
class LocalRect:
    """Rectangle coordinates inside one rank-local matrix buffer."""

    row_start: int
    col_start: int
    row_count: int
    col_count: int

    @property
    def row_stop(self) -> int:
        return self.row_start + self.row_count

    @property
    def col_stop(self) -> int:
        return self.col_start + self.col_count


@dataclass(frozen=True)
class ExecutableFragmentTransfer:
    """Fragment movement with concrete rank-local source and target rectangles."""

    phase: RedistributionPhase
    tile_row: int
    tile_col: int
    source_rank: int
    target_rank: int
    source_rect: LocalRect
    target_rect: LocalRect
    phase_group: int
    access: MemoryAccessClassification

    @property
    def row_count(self) -> int:
        return self.source_rect.row_count

    @property
    def col_count(self) -> int:
        return self.source_rect.col_count


@dataclass(frozen=True)
class ExecutableFragmentTransferBatch:
    """Conflict-free execution round for concrete local rectangle transfers."""

    phase: RedistributionPhase
    round_index: int
    transfers: tuple[ExecutableFragmentTransfer, ...]

    @property
    def source_ranks(self) -> tuple[int, ...]:
        return tuple(transfer.source_rank for transfer in self.transfers)

    @property
    def target_ranks(self) -> tuple[int, ...]:
        return tuple(transfer.target_rank for transfer in self.transfers)


@dataclass(frozen=True)
class EdgePaddingCompactionMove:
    """One maximal rectangle move in the edge-padding normalization pass."""

    phase: EdgePaddingPhase
    wave: int
    phase_group: int
    source_rank: int
    target_rank: int
    source_rect: LocalRect
    target_rect: LocalRect

    @property
    def element_count(self) -> int:
        return self.source_rect.row_count * self.source_rect.col_count

    @property
    def row_count(self) -> int:
        return self.source_rect.row_count

    @property
    def col_count(self) -> int:
        return self.source_rect.col_count


@dataclass(frozen=True)
class EdgePaddingCompactionBatch:
    """Dependency wave of matching edge-padding moves.

    A batch represents the same compaction step applied across independent
    process rows for horizontal compaction, or independent process columns for
    vertical compaction. Different waves are not merged because wave ``k + 1``
    consumes the holes created by wave ``k``.
    """

    phase: EdgePaddingPhase
    wave: int
    moves: tuple[EdgePaddingCompactionMove, ...]

    @property
    def source_ranks(self) -> tuple[int, ...]:
        return tuple(move.source_rank for move in self.moves)

    @property
    def target_ranks(self) -> tuple[int, ...]:
        return tuple(move.target_rank for move in self.moves)


@dataclass(frozen=True)
class EdgePaddingCompactionPlan:
    """CPU plan for converting per-shard padding into global edge padding."""

    grid: ProcessGrid
    tile_shape: TileShape
    padding: MatrixPadding2D
    moves: tuple[EdgePaddingCompactionMove, ...]

    @property
    def horizontal_moves(self) -> tuple[EdgePaddingCompactionMove, ...]:
        return tuple(move for move in self.moves if move.phase == "horizontal")

    @property
    def vertical_moves(self) -> tuple[EdgePaddingCompactionMove, ...]:
        return tuple(move for move in self.moves if move.phase == "vertical")


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


def _axis_edge_padding_moves(
    *,
    block_count: int,
    logical_per_block: int,
    physical_per_block: int,
) -> tuple[tuple[int, int, int, int], ...]:
    """Return maximal open-chain moves for one padded axis.

    Each returned tuple is ``(wave, source_start, target_start, extent)`` in
    global physical axis coordinates. The occupancy model treats initial
    per-block padding as holes and stable-compacts real intervals to the left.
    Move extents are clipped at block boundaries so every move maps to one
    source rank and one target rank in the 2D process grid.
    """
    if block_count <= 0 or logical_per_block <= 0 or physical_per_block <= 0:
        raise ValueError("block count and axis extents must be positive.")
    if logical_per_block > physical_per_block:
        raise ValueError("logical_per_block cannot exceed physical_per_block.")

    total = block_count * physical_per_block
    logical_total = block_count * logical_per_block
    is_real = [False] * total
    for block in range(block_count):
        start = block * physical_per_block
        for offset in range(logical_per_block):
            is_real[start + offset] = True

    moves: list[tuple[int, int, int, int]] = []
    wave = 0
    while True:
        target_start = None
        for index in range(logical_total):
            if not is_real[index]:
                target_start = index
                break
        if target_start is None:
            break

        target_stop = target_start
        while target_stop < total and not is_real[target_stop]:
            target_stop += 1

        source_start = None
        for index in range(target_stop, total):
            if is_real[index]:
                source_start = index
                break
        if source_start is None:
            break

        source_stop = source_start
        while source_stop < total and is_real[source_stop]:
            source_stop += 1

        target_block_stop = (
            target_start // physical_per_block + 1
        ) * physical_per_block
        source_block_stop = (
            source_start // physical_per_block + 1
        ) * physical_per_block
        extent = min(
            target_stop - target_start,
            source_stop - source_start,
            target_block_stop - target_start,
            source_block_stop - source_start,
        )
        if extent <= 0:
            raise RuntimeError("edge-padding compaction produced an empty move.")

        moves.append((wave, source_start, target_start, extent))
        for offset in range(extent):
            if is_real[target_start + offset] or not is_real[source_start + offset]:
                raise RuntimeError("invalid edge-padding compaction occupancy.")
            is_real[target_start + offset] = True
            is_real[source_start + offset] = False
        wave += 1

    return tuple(moves)


def build_edge_padding_compaction_plan(
    logical_rows: int,
    logical_cols: int,
    tile_shape: TileShape,
    grid: ProcessGrid,
) -> EdgePaddingCompactionPlan:
    """Plan the top-left normalization needed after local shard padding.

    The returned moves first compact column padding to the global right edge
    inside every process row, then compact row padding to the global bottom edge
    inside every process column. Moves are maximal axis-aligned rectangles whose
    destinations are known padding holes at the time they are issued.
    """
    padding = calculate_2d_padding(logical_rows, logical_cols, grid, tile_shape)
    moves: list[EdgePaddingCompactionMove] = []

    horizontal_axis_moves = _axis_edge_padding_moves(
        block_count=grid.process_cols,
        logical_per_block=padding.local_logical_cols,
        physical_per_block=padding.local_physical_cols,
    )
    for wave, source_col, target_col, extent in horizontal_axis_moves:
        source_process_col = source_col // padding.local_physical_cols
        target_process_col = target_col // padding.local_physical_cols
        source_local_col = source_col % padding.local_physical_cols
        target_local_col = target_col % padding.local_physical_cols
        for process_row in range(grid.process_rows):
            moves.append(
                EdgePaddingCompactionMove(
                    phase="horizontal",
                    wave=wave,
                    phase_group=process_row,
                    source_rank=grid.rank(process_row, source_process_col),
                    target_rank=grid.rank(process_row, target_process_col),
                    source_rect=LocalRect(
                        row_start=0,
                        col_start=source_local_col,
                        row_count=padding.local_physical_rows,
                        col_count=extent,
                    ),
                    target_rect=LocalRect(
                        row_start=0,
                        col_start=target_local_col,
                        row_count=padding.local_physical_rows,
                        col_count=extent,
                    ),
                )
            )

    vertical_axis_moves = _axis_edge_padding_moves(
        block_count=grid.process_rows,
        logical_per_block=padding.local_logical_rows,
        physical_per_block=padding.local_physical_rows,
    )
    for wave, source_row, target_row, extent in vertical_axis_moves:
        source_process_row = source_row // padding.local_physical_rows
        target_process_row = target_row // padding.local_physical_rows
        source_local_row = source_row % padding.local_physical_rows
        target_local_row = target_row % padding.local_physical_rows
        for process_col in range(grid.process_cols):
            moves.append(
                EdgePaddingCompactionMove(
                    phase="vertical",
                    wave=wave,
                    phase_group=process_col,
                    source_rank=grid.rank(source_process_row, process_col),
                    target_rank=grid.rank(target_process_row, process_col),
                    source_rect=LocalRect(
                        row_start=source_local_row,
                        col_start=0,
                        row_count=extent,
                        col_count=padding.local_physical_cols,
                    ),
                    target_rect=LocalRect(
                        row_start=target_local_row,
                        col_start=0,
                        row_count=extent,
                        col_count=padding.local_physical_cols,
                    ),
                )
            )

    return EdgePaddingCompactionPlan(
        grid=grid,
        tile_shape=tile_shape,
        padding=padding,
        moves=tuple(moves),
    )


def batch_edge_padding_compaction_moves(
    moves: tuple[EdgePaddingCompactionMove, ...],
) -> tuple[EdgePaddingCompactionBatch, ...]:
    """Group edge-padding moves by phase and dependency wave.

    This exposes the intended parallelism: all process rows/columns perform the
    same wave together when their source and target ranks are disjoint, while
    later waves remain ordered behind earlier hole-propagation steps.
    """
    batches: list[EdgePaddingCompactionBatch] = []
    for phase in ("horizontal", "vertical"):
        phase_moves = [move for move in moves if move.phase == phase]
        waves = sorted({move.wave for move in phase_moves})
        for wave in waves:
            wave_moves = tuple(move for move in phase_moves if move.wave == wave)
            sources = [move.source_rank for move in wave_moves]
            targets = [move.target_rank for move in wave_moves]
            if len(sources) != len(set(sources)) or len(targets) != len(set(targets)):
                raise ValueError(
                    "edge-padding wave contains repeated source or target ranks."
                )
            batches.append(
                EdgePaddingCompactionBatch(
                    phase=phase,
                    wave=wave,
                    moves=wave_moves,
                )
            )
    return tuple(batches)


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


def build_two_phase_fragment_transfer_schedule(
    plan: BlockCyclic2DPlan,
) -> tuple[TwoPhaseFragmentTransfer, ...]:
    """Build a concrete fragment transfer schedule for the two-phase plan.

    The schedule is still CPU-only, but it has the shape needed by the eventual
    CUDA/NCCL implementation: phase, source rank, target rank, phase-parallel
    group, logical rectangle, and whether that rectangle is contiguous in the
    local memory model.
    """
    transfers: list[TwoPhaseFragmentTransfer] = []
    local_rows = plan.padding.local_physical_rows
    local_cols = plan.padding.local_physical_cols

    for fragment in plan.tile_fragments:
        target_owner = block_cyclic_tile_owner(
            fragment.tile_row, fragment.tile_col, plan.grid
        )
        after_column_owner = Owner2D(
            fragment.source_owner.process_row, target_owner.process_col
        )
        access = classify_rectangular_region(
            fragment.row_count,
            fragment.col_count,
            local_rows,
            local_cols,
        )

        if fragment.source_owner.process_col != target_owner.process_col:
            transfers.append(
                TwoPhaseFragmentTransfer(
                    phase="column_owner",
                    tile_row=fragment.tile_row,
                    tile_col=fragment.tile_col,
                    row_start=fragment.row_start,
                    row_stop=fragment.row_stop,
                    col_start=fragment.col_start,
                    col_stop=fragment.col_stop,
                    source_owner=fragment.source_owner,
                    target_owner=after_column_owner,
                    source_rank=fragment.source_owner.rank(plan.grid),
                    target_rank=after_column_owner.rank(plan.grid),
                    phase_group=fragment.source_owner.process_row,
                    access=access,
                )
            )

        if after_column_owner.process_row != target_owner.process_row:
            transfers.append(
                TwoPhaseFragmentTransfer(
                    phase="row_owner",
                    tile_row=fragment.tile_row,
                    tile_col=fragment.tile_col,
                    row_start=fragment.row_start,
                    row_stop=fragment.row_stop,
                    col_start=fragment.col_start,
                    col_stop=fragment.col_stop,
                    source_owner=after_column_owner,
                    target_owner=target_owner,
                    source_rank=after_column_owner.rank(plan.grid),
                    target_rank=target_owner.rank(plan.grid),
                    phase_group=after_column_owner.process_col,
                    access=access,
                )
            )

    return tuple(transfers)


def transfer_phase_groups(
    transfers: tuple[TwoPhaseFragmentTransfer, ...],
) -> dict[RedistributionPhase, tuple[int, ...]]:
    """Return the independent phase groups present in a transfer schedule."""
    grouped: dict[RedistributionPhase, set[int]] = {
        "column_owner": set(),
        "row_owner": set(),
    }
    for transfer in transfers:
        grouped[transfer.phase].add(transfer.phase_group)
    return {
        phase: tuple(sorted(groups))
        for phase, groups in grouped.items()
    }


def batch_fragment_transfers(
    transfers: tuple[TwoPhaseFragmentTransfer, ...],
    *,
    max_transfers_per_batch: int | None = None,
) -> tuple[FragmentTransferBatch, ...]:
    """Pack fragment transfers into rank-conflict-free execution rounds.

    This scheduler is intentionally conservative. It keeps the required global
    ordering of the proposed algorithm by executing every ``column_owner`` batch
    before every ``row_owner`` batch. Within one phase it greedily places
    transfers into the earliest batch whose source and target ranks are both
    free. This gives the CUDA/NCCL layer a simple invariant: one packed send
    buffer and one receive target per rank are enough for each batch.
    """
    if max_transfers_per_batch is not None and max_transfers_per_batch <= 0:
        raise ValueError("max_transfers_per_batch must be positive when set.")

    batches: list[FragmentTransferBatch] = []
    for phase in ("column_owner", "row_owner"):
        phase_transfers = [
            transfer for transfer in transfers if transfer.phase == phase
        ]
        phase_batches: list[list[TwoPhaseFragmentTransfer]] = []
        used_sources: list[set[int]] = []
        used_targets: list[set[int]] = []

        for transfer in phase_transfers:
            placed = False
            for batch_index, batch in enumerate(phase_batches):
                if (
                    transfer.source_rank in used_sources[batch_index]
                    or transfer.target_rank in used_targets[batch_index]
                    or (
                        max_transfers_per_batch is not None
                        and len(batch) >= max_transfers_per_batch
                    )
                ):
                    continue
                batch.append(transfer)
                used_sources[batch_index].add(transfer.source_rank)
                used_targets[batch_index].add(transfer.target_rank)
                placed = True
                break

            if not placed:
                phase_batches.append([transfer])
                used_sources.append({transfer.source_rank})
                used_targets.append({transfer.target_rank})

        for round_index, batch in enumerate(phase_batches):
            batches.append(
                FragmentTransferBatch(
                    phase=phase,
                    round_index=round_index,
                    transfers=tuple(batch),
                )
            )

    return tuple(batches)


def _check_local_rect(
    rect: LocalRect,
    *,
    local_rows: int,
    local_cols: int,
) -> LocalRect:
    if (
        rect.row_start < 0
        or rect.col_start < 0
        or rect.row_stop > local_rows
        or rect.col_stop > local_cols
    ):
        raise ValueError("local rectangle does not fit in rank-local buffer.")
    return rect


def block_sharded_local_rect(
    fragment: TileFragment,
    owner: Owner2D,
    plan: BlockCyclic2DPlan,
) -> LocalRect:
    """Initial JAX block-sharded local coordinates for ``fragment``."""
    return _check_local_rect(
        LocalRect(
            row_start=fragment.row_start
            - owner.process_row * plan.padding.local_logical_rows,
            col_start=fragment.col_start
            - owner.process_col * plan.padding.local_logical_cols,
            row_count=fragment.row_count,
            col_count=fragment.col_count,
        ),
        local_rows=plan.padding.local_physical_rows,
        local_cols=plan.padding.local_physical_cols,
    )


def block_cyclic_local_rect(
    fragment: TileFragment,
    plan: BlockCyclic2DPlan,
) -> LocalRect:
    """Final cuSOLVERMp block-cyclic local coordinates for ``fragment``."""
    row_in_tile = fragment.row_start - fragment.tile_row * plan.tile_shape.rows
    col_in_tile = fragment.col_start - fragment.tile_col * plan.tile_shape.cols
    return _check_local_rect(
        LocalRect(
            row_start=(fragment.tile_row // plan.grid.process_rows)
            * plan.tile_shape.rows
            + row_in_tile,
            col_start=(fragment.tile_col // plan.grid.process_cols)
            * plan.tile_shape.cols
            + col_in_tile,
            row_count=fragment.row_count,
            col_count=fragment.col_count,
        ),
        local_rows=plan.padding.local_physical_rows,
        local_cols=plan.padding.local_physical_cols,
    )


def column_owner_intermediate_local_rect(
    fragment: TileFragment,
    owner: Owner2D,
    plan: BlockCyclic2DPlan,
) -> LocalRect:
    """Intermediate coordinates after column ownership has been fixed.

    At this point the process column matches the final block-cyclic owner, but
    the process row is still the original block-sharded row owner. The local
    column is therefore in final block-cyclic coordinates while the local row is
    still in the original block-sharded coordinates.
    """
    col_in_tile = fragment.col_start - fragment.tile_col * plan.tile_shape.cols
    return _check_local_rect(
        LocalRect(
            row_start=fragment.row_start
            - owner.process_row * plan.padding.local_logical_rows,
            col_start=(fragment.tile_col // plan.grid.process_cols)
            * plan.tile_shape.cols
            + col_in_tile,
            row_count=fragment.row_count,
            col_count=fragment.col_count,
        ),
        local_rows=plan.padding.local_physical_rows,
        local_cols=plan.padding.local_physical_cols,
    )


def build_executable_fragment_transfer_schedule(
    plan: BlockCyclic2DPlan,
) -> tuple[ExecutableFragmentTransfer, ...]:
    """Build concrete local-rectangle moves for the two-phase redistribution.

    Unlike :func:`build_two_phase_fragment_transfer_schedule`, this schedule is
    not only an owner-coordinate proof. It also records same-rank local
    relocations whenever a fragment already lives on the correct rank but needs
    to move to its intermediate or final cuSOLVERMp local coordinates.
    """
    transfers: list[ExecutableFragmentTransfer] = []
    local_rows = plan.padding.local_physical_rows
    local_cols = plan.padding.local_physical_cols

    for fragment in plan.tile_fragments:
        target_owner = block_cyclic_tile_owner(
            fragment.tile_row, fragment.tile_col, plan.grid
        )
        current_owner = fragment.source_owner
        current_rect = block_sharded_local_rect(fragment, current_owner, plan)
        after_column_owner = Owner2D(
            current_owner.process_row, target_owner.process_col
        )
        after_column_rect = column_owner_intermediate_local_rect(
            fragment, after_column_owner, plan
        )
        access = classify_rectangular_region(
            fragment.row_count,
            fragment.col_count,
            local_rows,
            local_cols,
        )

        if (
            current_owner != after_column_owner
            or current_rect != after_column_rect
        ):
            transfers.append(
                ExecutableFragmentTransfer(
                    phase="column_owner",
                    tile_row=fragment.tile_row,
                    tile_col=fragment.tile_col,
                    source_rank=current_owner.rank(plan.grid),
                    target_rank=after_column_owner.rank(plan.grid),
                    source_rect=current_rect,
                    target_rect=after_column_rect,
                    phase_group=current_owner.process_row,
                    access=access,
                )
            )
            current_owner = after_column_owner
            current_rect = after_column_rect

        final_rect = block_cyclic_local_rect(fragment, plan)
        if current_owner != target_owner or current_rect != final_rect:
            transfers.append(
                ExecutableFragmentTransfer(
                    phase="row_owner",
                    tile_row=fragment.tile_row,
                    tile_col=fragment.tile_col,
                    source_rank=current_owner.rank(plan.grid),
                    target_rank=target_owner.rank(plan.grid),
                    source_rect=current_rect,
                    target_rect=final_rect,
                    phase_group=current_owner.process_col,
                    access=access,
                )
            )

    return tuple(transfers)


def batch_executable_fragment_transfers(
    transfers: tuple[ExecutableFragmentTransfer, ...],
    *,
    max_transfers_per_batch: int | None = None,
) -> tuple[ExecutableFragmentTransferBatch, ...]:
    """Pack executable rectangle transfers into conflict-free rounds."""
    if max_transfers_per_batch is not None and max_transfers_per_batch <= 0:
        raise ValueError("max_transfers_per_batch must be positive when set.")

    batches: list[ExecutableFragmentTransferBatch] = []
    for phase in ("column_owner", "row_owner"):
        phase_transfers = [
            transfer for transfer in transfers if transfer.phase == phase
        ]
        phase_batches: list[list[ExecutableFragmentTransfer]] = []
        used_sources: list[set[int]] = []
        used_targets: list[set[int]] = []

        for transfer in phase_transfers:
            placed = False
            for batch_index, batch in enumerate(phase_batches):
                if (
                    transfer.source_rank in used_sources[batch_index]
                    or transfer.target_rank in used_targets[batch_index]
                    or (
                        max_transfers_per_batch is not None
                        and len(batch) >= max_transfers_per_batch
                    )
                ):
                    continue
                batch.append(transfer)
                used_sources[batch_index].add(transfer.source_rank)
                used_targets[batch_index].add(transfer.target_rank)
                placed = True
                break

            if not placed:
                phase_batches.append([transfer])
                used_sources.append({transfer.source_rank})
                used_targets.append({transfer.target_rank})

        for round_index, batch in enumerate(phase_batches):
            batches.append(
                ExecutableFragmentTransferBatch(
                    phase=phase,
                    round_index=round_index,
                    transfers=tuple(batch),
                )
            )

    return tuple(batches)


def classify_rectangular_region(
    row_count: int,
    col_count: int,
    local_rows: int,
    local_cols: int,
) -> MemoryAccessClassification:
    """Classify whether a column-major local region can move contiguously."""
    if row_count <= 0 or col_count <= 0:
        raise ValueError("region dimensions must be positive.")
    if local_rows <= 0 or local_cols <= 0:
        raise ValueError("local matrix dimensions must be positive.")
    if row_count > local_rows or col_count > local_cols:
        raise ValueError("region cannot exceed local matrix dimensions.")

    if col_count == 1:
        return MemoryAccessClassification(
            row_count, col_count, local_rows, local_cols, True,
            "one column is contiguous in column-major layout",
        )
    if row_count == local_rows:
        return MemoryAccessClassification(
            row_count, col_count, local_rows, local_cols, True,
            "full-height column slab is contiguous in column-major layout",
        )
    return MemoryAccessClassification(
        row_count, col_count, local_rows, local_cols, False,
        "multi-column sub-row region is strided and needs pack/unpack",
    )


def classify_proposed_phase_regions(
    plan: BlockCyclic2DPlan,
) -> dict[str, MemoryAccessClassification]:
    """Classify the natural grouped regions for the two separable phases.

    For cuSOLVERMp's column-major local model, moving a complete column slab
    through a process row is contiguous if grouped over all local rows, while
    smaller multi-column row fragments still need pack/unpack scratch.
    """
    padding = plan.padding
    tile_shape = plan.tile_shape
    return {
        "column_tile": classify_rectangular_region(
            padding.local_physical_rows,
            min(tile_shape.cols, padding.local_physical_cols),
            padding.local_physical_rows,
            padding.local_physical_cols,
        ),
        "row_slab": classify_rectangular_region(
            min(tile_shape.rows, padding.local_physical_rows),
            padding.local_physical_cols,
            padding.local_physical_rows,
            padding.local_physical_cols,
        ),
        "single_tile": classify_rectangular_region(
            min(tile_shape.rows, padding.local_physical_rows),
            min(tile_shape.cols, padding.local_physical_cols),
            padding.local_physical_rows,
            padding.local_physical_cols,
        ),
    }
