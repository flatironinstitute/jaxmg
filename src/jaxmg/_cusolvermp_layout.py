"""Shared Python layout helpers for the cuSOLVERMp backend.

This module defines four parts of the Python-to-native layout contract shared
by the JAXMg solver and decomposition wrappers:

1. infer and validate the JAX ``Mesh`` and matrix ``PartitionSpec``;
2. place solve inputs in the regular native work layout and restore outputs to
   their user-facing sharding;
3. map the partitions of the JAX mesh onto the cuSOLVERMp process grid; and
4. describe status-buffer sharding and local tile-capacity padding.

All GPU-to-GPU redistribution and all cuSOLVERMp calls are implemented in
C++/CUDA.  The helpers here only describe shapes and sharding to JAX.
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import jax
from jax import Array
from jax.sharding import AbstractMesh, AxisType, Mesh, NamedSharding, PartitionSpec as P

from ._layout_types import (
    MatrixPadding2D,
    ProcessGrid,
    TileShape,
    calculate_2d_padding,
    validate_nonempty_block_cyclic_ownership,
)


class PreparedMatrixLayout(NamedTuple):
    """Distributed layout metadata shared by every cuSOLVERMp routine."""

    mesh: Mesh | AbstractMesh
    matrix_specs: P
    native_status_specs: P
    grid: ProcessGrid
    partition_slots: tuple[int, ...]
    tile_shape: TileShape
    padding: MatrixPadding2D


# -----------------------------------------------------------------------------
# Matrix mesh contract
# -----------------------------------------------------------------------------


def normalize_matrix_specs(
    matrix_specs: P | tuple[P, ...] | list[P] | None,
    *,
    in_specs: P | tuple[P, ...] | list[P] | None = None,
) -> P | None:
    """Normalize the accepted matrix-sharding specification forms.

    ``in_specs`` is retained as an alias for ``matrix_specs``. Either argument
    accepts a ``PartitionSpec`` directly or inside a one-element tuple/list,
    but they cannot be supplied together.

    Args:
        matrix_specs: Matrix sharding specification used by the cuSOLVERMp
            wrappers, or ``None`` when it should be inferred from the input.
        in_specs: Alias for ``matrix_specs`` retained for API compatibility.

    Returns:
        The normalized ``PartitionSpec``, or ``None`` when neither argument was
        provided.

    Raises:
        ValueError: If both arguments are supplied or a sequence does not
            contain exactly one specification.
        TypeError: If the normalized value is not a ``PartitionSpec``.
    """
    if matrix_specs is not None and in_specs is not None:
        raise ValueError("Specify only one of matrix_specs=... or in_specs=...")
    specs = matrix_specs if matrix_specs is not None else in_specs
    if isinstance(specs, (tuple, list)):
        if len(specs) != 1:
            raise ValueError(
                "matrix_specs/in_specs must be a PartitionSpec or a "
                "one-element tuple/list containing one."
            )
        specs = specs[0]
    if specs is not None and not isinstance(specs, P):
        raise TypeError(
            "matrix_specs/in_specs must be a PartitionSpec or a one-element "
            "tuple/list containing one."
        )
    return specs


def mesh_axis_size(mesh: Mesh | AbstractMesh, axis_name: str) -> int:
    """Return the number of devices along a named JAX mesh axis.

    Args:
        mesh: JAX device mesh containing the requested axis.
        axis_name: Name of the axis whose extent is required.

    Returns:
        The positive integer extent recorded in ``mesh.shape``.

    Raises:
        ValueError: If ``axis_name`` is absent from the mesh.
    """
    try:
        return int(mesh.shape[axis_name])
    except KeyError as exc:
        raise ValueError(f"mesh does not contain axis {axis_name!r}.") from exc


def use_abstract_mesh_decorator(mesh: Mesh | AbstractMesh):
    """The decorated function is called within the context of ``use_abstract_mesh``.

    Needed because ``jax.shard_map`` rejects a mesh that is not the context mesh,
    so a caller keeping its own mesh in context could not call JAXMg at all.
    """

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with jax.sharding.use_abstract_mesh(mesh.abstract_mesh):
                return fn(*args, **kwargs)

        return wrapper

    return decorator


def validate_2d_matrix_specs(
    mesh: Mesh | AbstractMesh,
    matrix_specs: P,
) -> tuple[str | None, str | None, ProcessGrid]:
    """Validate and describe a two-dimensional matrix sharding.

    The first ``PartitionSpec`` entry defines the process-row axis and the
    second defines the process-column axis. Either one may be ``None``, which
    describes a degenerate ``P_r x 1`` or ``1 x P_c`` process grid: the
    corresponding matrix dimension is then not distributed, and every process
    owns it whole. That is the natural sharding for callers whose mesh has a
    single axis, such as ``Mesh(jax.devices(), ('x',))`` with ``P('x', None)``.

    Args:
        mesh: JAX mesh used to shard the matrix.
        matrix_specs: Rank-2 matrix ``PartitionSpec``.

    Returns:
        ``(row_axis, col_axis, grid)``, where ``grid`` contains the extents of
        the selected mesh axes, and an axis left unsharded is reported as
        ``None`` with an extent of one.

    Raises:
        TypeError: If ``matrix_specs`` is not a ``PartitionSpec``.
        ValueError: If the specification is not rank two, both matrix
            dimensions are unsharded, or a referenced axis is absent.
    """
    if not isinstance(matrix_specs, P):
        raise TypeError("matrix_specs must be a PartitionSpec.")
    if len(matrix_specs._partitions) != 2:
        raise ValueError("matrix_specs must describe a rank-2 sharding.")
    row_axis, col_axis = matrix_specs._partitions
    if row_axis is None and col_axis is None:
        raise ValueError(
            "cuSOLVERMp requires at least one matrix axis to be sharded by a "
            "named mesh axis, but matrix_specs leaves both unsharded."
        )
    for axis in (row_axis, col_axis):
        if not (isinstance(axis, str) or axis is None):
            raise ValueError(
                "cuSOLVERMp requires each matrix axis to be sharded by a single "
                "named mesh axis or left unsharded, for example P('pr', 'pc') "
                "or P('pr', None)."
            )
    grid = ProcessGrid(
        process_rows=1 if row_axis is None else mesh_axis_size(mesh, row_axis),
        process_cols=1 if col_axis is None else mesh_axis_size(mesh, col_axis),
    )
    return row_axis, col_axis, grid


# -----------------------------------------------------------------------------
# Right-hand-side layout
# -----------------------------------------------------------------------------


def rhs_distribution_columns(nrhs: int, *, process_cols: int, pad: bool) -> int:
    """Choose the JAX-visible RHS width used before local tile padding.

    cuSOLVERMp represents a skinny solve input ``B`` using its logical
    ``NRHS`` columns, whose block-cyclic ownership need not match JAX's even
    block sharding. The JAX-facing input must first have a global column
    dimension that is divisible by ``process_cols``.

    To bridge the two models, JAXMg pads the JAX-visible RHS width to the next
    multiple of the process-column count before applying local tile padding.
    Native code still receives the original ``NRHS`` and passes that logical
    value to cuSOLVERMp; the extra columns provide only routing capacity and
    are removed after the solved RHS is restored to its user-facing sharding.

    Args:
        nrhs: Logical number of solve-input columns.
        process_cols: Number of process columns in the cuSOLVERMp grid.
        pad: Whether routing columns may be added when ``nrhs`` is not evenly
            divisible by ``process_cols``.

    Returns:
        The smallest JAX-visible width greater than or equal to ``nrhs`` that
        is divisible by ``process_cols``.

    Raises:
        ValueError: If either size is non-positive, or padding is required while
            ``pad`` is ``False``.
    """
    nrhs = int(nrhs)
    process_cols = int(process_cols)
    if nrhs <= 0:
        raise ValueError("nrhs must be positive.")
    if process_cols <= 0:
        raise ValueError("process_cols must be positive.")

    remainder = nrhs % process_cols
    if remainder == 0:
        return nrhs
    if not pad:
        raise ValueError(
            "pad=False requires the RHS column count to be divisible by the "
            "process-grid column count. Set pad=True to add routing columns "
            "for skinny RHS matrices."
        )
    return nrhs + (process_cols - remainder)


def _place_for_matrix_axis_mode(
    value: Array,
    *,
    mesh: Mesh | AbstractMesh,
    matrix_specs: P,
    target_specs: P,
) -> Array:
    """Place an array according to the matrix mesh-axis mode.

    Explicit JAX mesh axes require ``jax.reshard`` because their sharding is
    fixed by the caller. Auto axes instead require
    ``jax.lax.with_sharding_constraint`` inside the compiled computation.

    Args:
        value: Array to place in the target sharding.
        mesh: JAX mesh that owns the matrix axes.
        matrix_specs: Matrix ``PartitionSpec`` used to identify the relevant
            mesh-axis modes.
        target_specs: ``PartitionSpec`` required for ``value``.

    Returns:
        ``value`` constrained or resharded to ``NamedSharding(mesh,
        target_specs)``.

    Raises:
        ValueError: If a matrix axis is absent from ``mesh`` or the two matrix
            axes do not share a supported Explicit or Auto mode.
    """
    axis_types = dict(zip(mesh.axis_names, mesh.axis_types))
    # Only axes actually mapped onto the mesh have a mode to agree on.
    matrix_axes = tuple(x for x in matrix_specs._partitions if x is not None)
    try:
        matrix_axis_types = tuple(axis_types[axis] for axis in matrix_axes)
    except KeyError as exc:
        raise ValueError(
            "matrix specs reference an axis that is absent from the JAX mesh."
        ) from exc

    target_sharding = NamedSharding(mesh, target_specs)
    if all(axis_type is AxisType.Explicit for axis_type in matrix_axis_types):
        return jax.reshard(value, target_sharding)
    if all(axis_type is AxisType.Auto for axis_type in matrix_axis_types):
        return jax.lax.with_sharding_constraint(value, target_sharding)
    raise ValueError(
        "cuSOLVERMp requires matrix mesh axes to be either all Explicit or "
        "all Auto; mixed Explicit/Auto/Manual mesh axes are unsupported."
    )


def place_rhs_for_native_work(
    rhs: Array,
    *,
    mesh: Mesh | AbstractMesh,
    matrix_specs: P,
) -> Array:
    """Place an RHS in the matrix's native work sharding.

    The public solvers accept a replicated RHS-column axis, for example
    ``P('pr', None)`` for an ``N x 1`` solve input. Before the shard-local FFI,
    the RHS is placed in the matrix's work sharding. JAX has two different APIs
    for that placement depending on how the mesh was created:
    ``jax.make_mesh`` creates explicit axes and requires ``jax.reshard``,
    while the conventional ``Mesh(...)`` constructor creates auto axes and
    requires ``with_sharding_constraint`` inside a compiled computation.

    Mixed explicit/auto process-grid axes are rejected because neither
    placement primitive can represent the mixed target consistently.

    Args:
        rhs: Solve input in its user-facing sharding.
        mesh: JAX mesh used by the matrix and native work buffers.
        matrix_specs: Matrix sharding required by the native FFI.

    Returns:
        The solve input placed in ``NamedSharding(mesh, matrix_specs)``.

    Raises:
        ValueError: If the matrix axes are absent or use an unsupported mixture
            of mesh-axis modes.
    """
    return _place_for_matrix_axis_mode(
        rhs,
        mesh=mesh,
        matrix_specs=matrix_specs,
        target_specs=matrix_specs,
    )


def infer_rhs_specs(rhs: Array, *, matrix_specs: P) -> P:
    """Return the RHS sharding to restore after native redistribution.

    Capture this specification before entering an internal ``jax.jit`` because
    tracing on Auto mesh axes may no longer expose the input array's original
    ``NamedSharding``.
    """
    sharding = getattr(rhs, "sharding", None)
    if isinstance(sharding, NamedSharding):
        return sharding.spec
    # Under jit, the type of the RHS only carries its sharding along Explicit
    # mesh axes; with Auto axes it reads as replicated, which it need not be.
    sharding = getattr(jax.typeof(rhs), "sharding", None)
    if isinstance(sharding, NamedSharding) and all(
        axis_type == AxisType.Explicit for axis_type in sharding.mesh.axis_types
    ):
        return sharding.spec
    row_axis, _ = matrix_specs._partitions
    return P(row_axis, None)


def restore_rhs_from_native_work(
    rhs: Array,
    *,
    rhs_specs: P,
    mesh: Mesh | AbstractMesh,
    matrix_specs: P,
) -> Array:
    """Restore a solved RHS to its user-facing sharding before shape slicing.

    Native work buffers use the matrix's sharding and may contain
    extra columns so every process column owns data. On explicit mesh axes,
    slicing those columns away while they remain partitioned can produce a
    dimension that is not divisible by the process-column count. Restore the
    input RHS ``PartitionSpec`` first so the logical-width slice is valid and
    the solver returns the same JAX-facing layout it received.

    Args:
        rhs: Solved native work buffer before removal of routing columns.
        rhs_specs: Original solve-input sharding to restore.
        mesh: JAX mesh used by the matrix and solve input.
        matrix_specs: Matrix sharding used by the native work buffer.

    Returns:
        The solved input restored to the original named sharding.

    Raises:
        ValueError: If the matrix axes are absent or use an unsupported mixture
            of mesh-axis modes.
    """
    return _place_for_matrix_axis_mode(
        rhs,
        mesh=mesh,
        matrix_specs=matrix_specs,
        target_specs=rhs_specs,
    )


# -----------------------------------------------------------------------------
# Process-grid slots
# -----------------------------------------------------------------------------


def partition_slots_from_mesh(
    mesh: Mesh | AbstractMesh,
    *,
    row_axis: str | None,
    col_axis: str | None,
    grid: ProcessGrid,
    caller: str,
) -> tuple[int, ...]:
    """Map every SPMD partition of ``mesh`` to its cuSOLVERMp grid slot.

    The native backend resolves each partition's physical device and
    communicator rank at run time from XLA's device assignment, so this helper
    needs only the abstract mesh axis names and sizes.
    """
    axis_names = tuple(mesh.axis_names)
    axis_sizes = tuple(int(size) for size in mesh.axis_sizes)
    matrix_axes = tuple(axis for axis in (row_axis, col_axis) if axis is not None)
    if any(axis not in axis_names for axis in matrix_axes):
        raise ValueError("matrix sharding axes must be present in the mesh.")
    if row_axis is not None and row_axis == col_axis:
        raise ValueError("matrix row and column axes must be distinct.")
    num_partitions = int(np.prod(axis_sizes, dtype=np.int64))
    if num_partitions != grid.num_processes:
        raise ValueError(
            f"{caller} currently expects the JAX mesh to contain exactly the "
            "axes used by the cuSOLVERMp process grid. Got "
            f"{num_partitions} mesh devices for a {grid.process_rows} x "
            f"{grid.process_cols} process grid."
        )

    coords = np.unravel_index(np.arange(num_partitions), axis_sizes)

    def coord(axis: str | None) -> np.ndarray:
        if axis is None:
            return np.zeros(num_partitions, dtype=np.int64)
        return np.asarray(coords[axis_names.index(axis)], dtype=np.int64)

    slots = coord(row_axis) * grid.process_cols + coord(col_axis)
    return tuple(int(slot) for slot in slots)


def infer_mesh_and_matrix_specs(
    a: Array,
    *,
    mesh: Mesh | AbstractMesh | None,
    matrix_specs: P | tuple[P, ...] | list[P] | None,
    in_specs: P | tuple[P, ...] | list[P] | None = None,
) -> tuple[Mesh | AbstractMesh, P]:
    """Resolve the JAX mesh and matrix sharding used by a solver call.

    Explicit ``mesh`` and ``matrix_specs`` values take precedence. Any missing
    value is inferred from the input matrix when it carries ``NamedSharding``.
    The mesh may be abstract because native code resolves devices at run time.

    Args:
        a: Input matrix whose named sharding may provide the mesh contract.
        mesh: Explicit concrete or abstract JAX mesh, or ``None`` to infer it
            from ``a``.
        matrix_specs: Explicit matrix ``PartitionSpec``, or ``None`` to infer
            it from ``a``.
        in_specs: Alias for ``matrix_specs``.

    Returns:
        The resolved ``(mesh, matrix_specs)`` pair, with a rank-1
        specification padded to rank two.

    Raises:
        ValueError: If inference is required but ``a`` has no
            ``NamedSharding``, or both sharding-specification aliases are set.
        TypeError: If the supplied matrix specification has an invalid type.
    """
    matrix_specs = normalize_matrix_specs(matrix_specs, in_specs=in_specs)
    if mesh is None or matrix_specs is None:
        sharding = getattr(a, "sharding", None)
        if not isinstance(sharding, NamedSharding):
            raise ValueError(
                "cuSOLVERMp routine could not infer mesh/matrix_specs from A. "
                "Shard A with jax.sharding.NamedSharding or pass mesh=... and "
                "matrix_specs=..."
            )
        if mesh is None:
            mesh = sharding.mesh
        if matrix_specs is None:
            matrix_specs = sharding.spec

    # ``P('x')`` and ``P('x', None)`` describe the same layout of a 2D array;
    # pad the short form so the rest of the layout code stays rank-2 throughout.
    partitions = matrix_specs._partitions
    return mesh, P(*partitions, *(None,) * (2 - len(partitions)))


# -----------------------------------------------------------------------------
# JAX-visible status and matrix layouts
# -----------------------------------------------------------------------------


def status_specs(
    row_axis: str | None, col_axis: str | None, grid: ProcessGrid
) -> P:
    """Choose the sharding for a per-rank native status vector.

    The status vector has one logical axis but must retain one distinct shard per
    process rank. A degenerate process grid uses its non-trivial mesh axis; a
    two-dimensional grid maps the vector axis over the product of both axes.

    Args:
        row_axis: Name of the process-row mesh axis.
        col_axis: Name of the process-column mesh axis.
        grid: Shape of the cuSOLVERMp process grid.

    Returns:
        A rank-1 ``PartitionSpec`` covering every process rank exactly once.
    """
    if grid.process_rows == 1:
        return P(col_axis)
    if grid.process_cols == 1:
        return P(row_axis)
    return P((row_axis, col_axis))


def prepare_matrix_padding(
    logical_rows: int,
    logical_cols: int,
    grid: ProcessGrid,
    tile_shape: TileShape,
    *,
    pad: bool,
    caller: str,
) -> MatrixPadding2D:
    """Return the tile-aligned local capacity for a distributed array.

    Args:
        logical_rows: Global logical row count.
        logical_cols: Global logical column count.
        grid: cuSOLVERMp process-grid shape.
        tile_shape: Native block-cyclic tile dimensions.
        pad: Whether local shards may be padded to tile-aligned capacity.
        caller: Routine-specific array label used in validation errors.

    Returns:
        The logical and physical local dimensions required on every rank.

    Raises:
        ValueError: If the global shape cannot be represented by the JAX
            sharding, or padding is required while ``pad=False``.
    """
    try:
        padding = calculate_2d_padding(
            logical_rows=logical_rows,
            logical_cols=logical_cols,
            grid=grid,
            tile_shape=tile_shape,
        )
    except ValueError as exc:
        raise ValueError(
            f"{caller} shape ({logical_rows}, {logical_cols}) must be divisible "
            f"by process grid ({grid.process_rows}, {grid.process_cols}) "
            "before local tile padding."
        ) from exc
    if not pad and padding.needs_padding:
        raise ValueError(
            f"{caller} requires tile-aligned local shards when pad=False. "
            "Use a tile size that divides both local dimensions or set pad=True."
        )
    return padding


def prepare_rectangular_matrix_layout(
    logical_rows: int,
    logical_cols: int,
    grid: ProcessGrid,
    tile_shape: TileShape,
    *,
    pad: bool,
    caller: str,
) -> MatrixPadding2D:
    """Validate block-cyclic ownership and return uniform local capacity."""
    validate_nonempty_block_cyclic_ownership(
        logical_rows=logical_rows,
        logical_cols=logical_cols,
        grid=grid,
        tile_shape=tile_shape,
        caller=caller,
    )
    return prepare_matrix_padding(
        logical_rows,
        logical_cols,
        grid,
        tile_shape,
        pad=pad,
        caller=caller,
    )


def prepare_input_matrix_layout(
    a: Array,
    tile_size: int,
    *,
    mesh: Mesh | AbstractMesh | None,
    matrix_specs: P | tuple[P, ...] | list[P] | None,
    in_specs: P | tuple[P, ...] | list[P] | None,
    pad: bool,
    caller: str,
) -> PreparedMatrixLayout:
    """Derive the common mesh, process-grid, partition, and padding metadata.

    Routine-specific wrappers validate their mathematical contracts before
    calling this helper. This function handles the distributed layout contract
    shared by every native cuSOLVERMp operation.
    """
    mesh, matrix_specs = infer_mesh_and_matrix_specs(
        a,
        mesh=mesh,
        matrix_specs=matrix_specs,
        in_specs=in_specs,
    )
    row_axis, col_axis, grid = validate_2d_matrix_specs(mesh, matrix_specs)
    partition_slots = partition_slots_from_mesh(
        mesh,
        row_axis=row_axis,
        col_axis=col_axis,
        grid=grid,
        caller=caller,
    )
    tile_shape = TileShape(rows=int(tile_size), cols=int(tile_size))
    padding = prepare_rectangular_matrix_layout(
        int(a.shape[0]),
        int(a.shape[1]),
        grid,
        tile_shape,
        pad=pad,
        caller=f"{caller}(A)",
    )
    return PreparedMatrixLayout(
        mesh,
        matrix_specs,
        status_specs(row_axis, col_axis, grid),
        grid,
        partition_slots,
        tile_shape,
        padding,
    )


# -----------------------------------------------------------------------------
# Local padding transforms
# -----------------------------------------------------------------------------


def _pad_local_2d(block: Array, *, row_padding: int, col_padding: int) -> Array:
    """Pad one local shard on its bottom and right edges.

    Args:
        block: Rank-2 local matrix shard.
        row_padding: Number of zero rows appended to the shard.
        col_padding: Number of zero columns appended to the shard.

    Returns:
        The original block when no padding is needed, otherwise a zero-padded
        block with shape increased by the requested amounts.
    """
    if row_padding == 0 and col_padding == 0:
        return block
    return jnp.pad(block, ((0, row_padding), (0, col_padding)))


def _unpad_local_2d(block: Array, *, local_rows: int, local_cols: int) -> Array:
    """Slice one local shard back to its logical unpadded shape.

    Args:
        block: Rank-2 local matrix shard containing optional edge padding.
        local_rows: Number of logical rows retained from the top edge.
        local_cols: Number of logical columns retained from the left edge.

    Returns:
        The leading ``(local_rows, local_cols)`` logical region.
    """
    return block[:local_rows, :local_cols]


def make_local_pad_fn(
    mesh: Mesh | AbstractMesh, matrix_specs: P, padding: MatrixPadding2D
):
    """Build the shard-local bottom/right padding transform for a matrix."""
    if not padding.needs_padding:
        return lambda block: block
    return jax.shard_map(
        functools.partial(
            _pad_local_2d,
            row_padding=padding.row_padding_per_process,
            col_padding=padding.col_padding_per_process,
        ),
        mesh=mesh,
        in_specs=matrix_specs,
        out_specs=matrix_specs,
        check_vma=True,
    )


def make_local_unpad_fn(
    mesh: Mesh | AbstractMesh, matrix_specs: P, padding: MatrixPadding2D
):
    """Build the shard-local slice that restores a matrix's logical shape."""
    return jax.shard_map(
        functools.partial(
            _unpad_local_2d,
            local_rows=padding.local_logical_rows,
            local_cols=padding.local_logical_cols,
        ),
        mesh=mesh,
        in_specs=matrix_specs,
        out_specs=matrix_specs,
        check_vma=True,
    )
