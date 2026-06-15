"""Small runtime layout types for the cuSOLVERMp Python wrappers.

The full block-cyclic reference planner lives in the test tree.  Production
Python only needs a compact description of the process grid, the supported
cuSOLVERMp rank mappings, the square tile size, and the local shard padding
rule.  Keeping those definitions here prevents the installed package from
carrying the large CPU reference planner used for redistribution tests.
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

    @classmethod
    def column_major(cls, grid: ProcessGrid) -> "ProcessRankMap":
        return cls(
            grid=grid,
            ranks=tuple(
                process_col * grid.process_rows + process_row
                for process_row in range(grid.process_rows)
                for process_col in range(grid.process_cols)
            ),
        )

    def rank(self, process_row: int, process_col: int) -> int:
        return self.ranks[self.grid.rank(process_row, process_col)]

    @property
    def is_row_major_identity(self) -> bool:
        return self.ranks == tuple(range(self.grid.num_processes))

    @property
    def is_column_major_identity(self) -> bool:
        return self.ranks == ProcessRankMap.column_major(self.grid).ranks

    @property
    def grid_mapping(self) -> ProcessGridMapping | None:
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
        FFI attribute because the native backend dynamically loads cuSOLVERMp.
        """
        mapping = self.grid_mapping
        if mapping == "column_major":
            return 0
        if mapping == "row_major":
            return 1
        raise ValueError("rank map is not a cuSOLVERMp-supported grid mapping.")

    def require_cusolvermp_grid_mapping(self, caller: str) -> None:
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
        return self.row_padding_per_process != 0 or self.col_padding_per_process != 0

    @property
    def source_blocks_are_tile_aligned(self) -> bool:
        return self.row_padding_per_process == 0 and self.col_padding_per_process == 0


def calculate_axis_padding(local_size: int, tile_size: int) -> int:
    """Match JAXMg's local-shard padding rule for one axis."""
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
    """Calculate the per-process padding required for a 2D block-sharded array.

    To participate in cuSOLVERMp operations, each local shard in a 2D block
    distribution must be tile-aligned. This function determines how many
    padding rows and columns must be added to each local process's shard to
    reach the next multiple of the tile size ``T_A``.

    Args:
        logical_rows: Total number of logical rows in the global matrix.
        logical_cols: Total number of logical columns in the global matrix.
        grid: The 2D process grid dimensions (rows x cols).
        tile_shape: The square tile shape used by cuSOLVERMp.

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
