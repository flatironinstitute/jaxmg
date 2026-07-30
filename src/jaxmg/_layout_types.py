"""Runtime layout types for the cuSOLVERMp Python wrappers.

This module describes process-grid dimensions, supported cuSOLVERMp rank
mappings, tile geometry, and the local capacity required for tile-aligned
padding. The solver wrappers use these values to establish the static layout
contract passed to the native backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ProcessGridMapping = Literal["row_major", "column_major"]


@dataclass(frozen=True)
class ProcessGrid:
    """2D process grid used by a cuSOLVERMp descriptor."""

    process_rows: int
    process_cols: int

    def __post_init__(self) -> None:
        """Validate that both process-grid dimensions are non-empty."""
        if self.process_rows <= 0:
            raise ValueError("process_rows must be positive.")
        if self.process_cols <= 0:
            raise ValueError("process_cols must be positive.")

    @property
    def num_processes(self) -> int:
        """Return the total number of participating process-grid slots."""
        return self.process_rows * self.process_cols

    def rank(self, process_row: int, process_col: int) -> int:
        """Return the row-major process-grid slot for one grid coordinate."""
        if not 0 <= process_row < self.process_rows:
            raise ValueError("process_row out of range.")
        if not 0 <= process_col < self.process_cols:
            raise ValueError("process_col out of range.")
        return process_row * self.process_cols + process_col


@dataclass(frozen=True)
class ProcessRankMap:
    """Supported process-grid slot to communicator-rank mapping.

    ``ranks`` is indexed by row-major process-grid slot:

        process_rank = process_row * process_cols + process_col

    and stores the communicator rank at that grid coordinate.  JAXMg supports
    only the two mappings cuSOLVERMp can describe in a grid descriptor:
    row-major and column-major.
    """

    grid: ProcessGrid
    ranks: tuple[int, ...]

    def __post_init__(self) -> None:
        """Validate and normalize a process-grid rank permutation."""
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
        """Construct cuSOLVERMp's row-major rank mapping for a grid."""
        return cls(grid=grid, ranks=tuple(range(grid.num_processes)))

    @classmethod
    def column_major(cls, grid: ProcessGrid) -> "ProcessRankMap":
        """Construct cuSOLVERMp's column-major rank mapping for a grid."""
        return cls(
            grid=grid,
            ranks=tuple(
                process_col * grid.process_rows + process_row
                for process_row in range(grid.process_rows)
                for process_col in range(grid.process_cols)
            ),
        )

    def rank(self, process_row: int, process_col: int) -> int:
        """Return the communicator rank at one process-grid coordinate."""
        return self.ranks[self.grid.rank(process_row, process_col)]

    @property
    def is_row_major_identity(self) -> bool:
        """Return True when the rank map matches row-major grid order."""
        return self.ranks == tuple(range(self.grid.num_processes))

    @property
    def is_column_major_identity(self) -> bool:
        """Return True when the rank map matches column-major grid order."""
        return self.ranks == ProcessRankMap.column_major(self.grid).ranks

    @property
    def grid_mapping(self) -> ProcessGridMapping | None:
        """Return the cuSOLVERMp mapping name, or None for unsupported maps."""
        if self.is_row_major_identity:
            return "row_major"
        if self.is_column_major_identity:
            return "column_major"
        return None

    @property
    def cusolvermp_grid_mapping(self) -> int:
        """cuSOLVERMp enum value for this rank map.

        NVIDIA's headers define ``CUSOLVERMP_GRID_MAPPING_COL_MAJOR = 0`` and
        ``CUSOLVERMP_GRID_MAPPING_ROW_MAJOR = 1``.  The integer is passed as an
        FFI attribute so native code can select the same cuSOLVERMp grid
        mapping as the JAX mesh.
        """
        mapping = self.grid_mapping
        if mapping == "column_major":
            return 0
        if mapping == "row_major":
            return 1
        raise ValueError("rank map is not a cuSOLVERMp-supported grid mapping.")

    def require_cusolvermp_grid_mapping(self, caller: str) -> None:
        """Raise if the rank map cannot be represented by cuSOLVERMp."""
        if self.grid_mapping is None:
            raise ValueError(
                f"{caller} requires the JAX mesh device order to match either "
                "cuSOLVERMp row-major or column-major communicator rank order. "
                f"Got process-grid rank map {self.ranks}. Construct the mesh "
                "with a regular row-major or column-major device layout; "
                "arbitrary mesh permutations are not supported."
            )


@dataclass(frozen=True)
class TileShape:
    """cuSOLVERMp matrix tile shape."""

    rows: int
    cols: int

    def __post_init__(self) -> None:
        """Validate positive row and column tile dimensions."""
        if self.rows <= 0:
            raise ValueError("tile row size must be positive.")
        if self.cols <= 0:
            raise ValueError("tile column size must be positive.")


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
        """Return True when at least one local axis needs padding."""
        return self.row_padding_per_process != 0 or self.col_padding_per_process != 0

    @property
    def source_blocks_are_tile_aligned(self) -> bool:
        """Return True when all source local blocks are already tile-aligned."""
        return self.row_padding_per_process == 0 and self.col_padding_per_process == 0


def calculate_axis_padding(local_size: int, tile_size: int) -> int:
    """Match JAXMg's local-shard padding rule for one axis."""
    if local_size < 0:
        raise ValueError("local_size must be non-negative.")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive.")
    return (-local_size) % tile_size


def validate_nonempty_block_cyclic_ownership(
    logical_rows: int,
    logical_cols: int,
    grid: ProcessGrid,
    tile_shape: TileShape,
    *,
    caller: str,
) -> None:
    """Require every cuSOLVERMp process-grid row and column to own data tiles.

    cuSOLVERMp can be fragile when a process-grid row or column owns no logical
    matrix tiles.  This happens for very small matrices, oversized tile sizes,
    or overly large process grids.  The check is closed-form: in a cyclic
    distribution every process row owns a tile iff the number of logical tile
    rows is at least the number of process rows; columns follow the same rule.
    """
    if logical_rows <= 0 or logical_cols <= 0:
        raise ValueError("logical dimensions must be positive.")

    num_tile_rows = (int(logical_rows) + int(tile_shape.rows) - 1) // int(
        tile_shape.rows
    )
    num_tile_cols = (int(logical_cols) + int(tile_shape.cols) - 1) // int(
        tile_shape.cols
    )

    if num_tile_rows < grid.process_rows or num_tile_cols < grid.process_cols:
        raise ValueError(
            f"{caller} requires every cuSOLVERMp process-grid row and column "
            "to own at least one logical block-cyclic tile. "
            f"Got logical_shape=({logical_rows}, {logical_cols}), "
            f"tile_shape=({tile_shape.rows}, {tile_shape.cols}), "
            f"process_grid=({grid.process_rows}, {grid.process_cols}), "
            f"logical_tile_grid=({num_tile_rows}, {num_tile_cols}). "
            "Use a smaller process grid, a larger matrix, or a smaller tile size."
        )


def calculate_2d_padding(
    logical_rows: int,
    logical_cols: int,
    grid: ProcessGrid,
    tile_shape: TileShape,
) -> MatrixPadding2D:
    """Calculate the per-process padding required for a 2D block-sharded array.

    To participate in cuSOLVERMp operations, each local shard in a 2D block
    distribution must be tile-aligned. This function determines how many
    padding rows and columns must be added to each local process's shard to
    reach the next multiple of the tile size ``T_A``.

    Args:
        logical_rows: Total number of logical rows in the global matrix.
        logical_cols: Total number of logical columns in the global matrix.
        grid: The 2D process grid dimensions (rows x cols).
        tile_shape: The 2D tile shape used by cuSOLVERMp.

    Returns:
        A ``MatrixPadding2D`` object containing the calculated local and
        global physical (padded) dimensions.

    Raises:
        ValueError: If logical dimensions are not divisible by the process
            grid dimensions.
    """
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
