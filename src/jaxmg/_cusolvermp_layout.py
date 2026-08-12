"""Shared Python layout helpers for the cuSOLVERMp backend.

This module defines four parts of the Python-to-native layout contract shared
by the POTRS, LU-solve, and SYEVD wrappers:

1. infer and validate the JAX ``Mesh`` and matrix ``PartitionSpec``;
2. place solve inputs in the regular native work layout and restore outputs to
   their user-facing sharding;
3. map JAX devices onto a cuSOLVERMp-supported process grid; and
4. describe status-buffer sharding and local tile-capacity padding.

All GPU-to-GPU redistribution and all cuSOLVERMp calls are implemented in
C++/CUDA.  The helpers here only describe shapes and sharding to JAX.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import jax
from jax import Array
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

from ._layout_types import (
    ProcessGrid,
    ProcessRankMap,
)


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


def mesh_axis_size(mesh: Mesh, axis_name: str) -> int:
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


def validate_2d_matrix_specs(
    mesh: Mesh,
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

    cuSOLVERMp accepts a skinny solve-input matrix ``B`` with ``NRHS``
    columns, even when ``NRHS`` is smaller than the process-grid column count.
    In that ScaLAPACK-style layout, some process columns simply own zero real
    RHS columns. The JAX-facing block-sharded input cannot express that zero
    ownership with the simple ``PartitionSpec(row_axis, col_axis)`` contract
    used by this backend, because the global column dimension must first be
    splittable over ``process_cols``.

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
    mesh: Mesh,
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
    mesh: Mesh,
    matrix_specs: P,
) -> Array:
    """Place an RHS in the native backend's regular 2D work sharding.

    The public solvers accept a replicated RHS-column axis, for example
    ``P('pr', None)`` for an ``N x 1`` solve input. The shard-local native
    FFI instead receives a regular ``P('pr', 'pc')`` work buffer. JAX has two
    different APIs for that placement depending on how the mesh was created:
    ``jax.make_mesh`` creates explicit axes and requires ``jax.reshard``,
    while the conventional ``Mesh(...)`` constructor creates auto axes and
    requires ``with_sharding_constraint`` inside a compiled computation.

    Mixed explicit/auto process-grid axes are rejected because neither
    placement primitive can represent the mixed target consistently.

    Args:
        rhs: Solve input in its user-facing sharding.
        mesh: JAX mesh used by the matrix and native work buffers.
        matrix_specs: Regular 2D matrix sharding required by the native FFI.

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


def restore_rhs_from_native_work(
    rhs: Array,
    *,
    reference_rhs: Array,
    mesh: Mesh,
    matrix_specs: P,
) -> Array:
    """Restore a solved RHS to its user-facing sharding before shape slicing.

    Native work buffers use the matrix's regular 2D sharding and may contain
    extra columns so every process column owns data. On explicit mesh axes,
    slicing those columns away while they remain partitioned can produce a
    dimension that is not divisible by the process-column count. Restore the
    input RHS ``PartitionSpec`` first so the logical-width slice is valid and
    the solver returns the same JAX-facing layout it received.

    Args:
        rhs: Solved native work buffer before removal of routing columns.
        reference_rhs: Original solve input whose sharding should be restored.
        mesh: JAX mesh used by the matrix and solve input.
        matrix_specs: Regular 2D sharding used by the native work buffer.

    Returns:
        The solved input restored to the original named sharding. If the
        reference has no ``NamedSharding``, the row axis remains sharded and
        the column axis is replicated.

    Raises:
        ValueError: If the matrix axes are absent or use an unsupported mixture
            of mesh-axis modes.
    """
    reference_sharding = getattr(jax.typeof(reference_rhs), "sharding", None)
    if isinstance(reference_sharding, NamedSharding):
        target_specs = reference_sharding.spec
    else:
        row_axis, _ = matrix_specs._partitions
        target_specs = P(row_axis, None)

    return _place_for_matrix_axis_mode(
        rhs,
        mesh=mesh,
        matrix_specs=matrix_specs,
        target_specs=target_specs,
    )


# -----------------------------------------------------------------------------
# Process-grid rank mapping
# -----------------------------------------------------------------------------


def _device_process_index(device) -> int:
    """Return the host-process index that owns a JAX device.

    JAX versions expose ``process_index`` either as an integer property or a
    zero-argument method, so both forms are accepted.

    Raises:
        AttributeError: If the device exposes no process index.
    """
    value = getattr(device, "process_index", None)
    if callable(value):
        return int(value())
    if value is None:
        raise AttributeError(f"device {device!r} has no process_index")
    return int(value)


def _device_id(device) -> int:
    """Return JAX's stable integer device id for communicator ordering.

    Raises:
        AttributeError: If the device exposes no integer ``id``.
    """
    value = getattr(device, "id", None)
    if value is None:
        raise AttributeError(f"device {device!r} has no id")
    return int(value)


def _device_local_hardware_id(device) -> int:
    """Return the local GPU hardware id used to disambiguate device order.

    JAX does not expose ``local_hardware_id`` on every backend. In that case,
    the globally stable device id provides the fallback ordering key.
    """
    value = getattr(device, "local_hardware_id", None)
    if value is None:
        return _device_id(device)
    return int(value)


def _device_rank_key(device) -> tuple[int, int, int]:
    """Return the device ordering key used to model XLA communicator ranks.

    Devices are ordered first by host process, then by JAX device id, and
    finally by local hardware id. This matches the regular rank layouts accepted
    by the native cuSOLVERMp process grid.
    """
    return (
        _device_process_index(device),
        _device_id(device),
        _device_local_hardware_id(device),
    )


def process_rank_map_from_mesh(
    mesh: Mesh,
    *,
    row_axis: str | None,
    col_axis: str | None,
    grid: ProcessGrid,
    caller: str,
) -> ProcessRankMap:
    """Map JAX mesh coordinates to cuSOLVERMp communicator ranks.

    Users construct meshes with normal JAX APIs.  JAXMg inspects the resulting
    device array and accepts it only when the process-grid coordinates match a
    cuSOLVERMp-supported row-major or column-major rank mapping.  Arbitrary
    mesh permutations are rejected before native code runs, because cuSOLVERMp
    cannot represent those permutations in its grid descriptor.

    Args:
        mesh: JAX mesh containing the process-grid devices.
        row_axis: Mesh axis mapped to cuSOLVERMp process rows, or ``None`` for a
            degenerate ``1 x P_c`` grid.
        col_axis: Mesh axis mapped to cuSOLVERMp process columns, or ``None``
            for a degenerate ``P_r x 1`` grid.
        grid: Expected two-dimensional process-grid shape.
        caller: Routine name included in validation errors.

    Returns:
        A process-grid-slot to communicator-rank mapping in row-major slot
        order.

    Raises:
        ValueError: If the matrix axes are invalid, the mesh shape differs from
            ``grid``, devices are duplicated, or device order is not one of
            cuSOLVERMp's supported grid mappings.
    """
    axis_names = tuple(mesh.axis_names)
    matrix_axes = tuple(axis for axis in (row_axis, col_axis) if axis is not None)
    if any(axis not in axis_names for axis in matrix_axes):
        raise ValueError("matrix sharding axes must be present in the mesh.")
    if row_axis is not None and row_axis == col_axis:
        raise ValueError("matrix row and column axes must be distinct.")

    devices = np.asarray(mesh.devices, dtype=object)
    if devices.ndim != len(axis_names):
        raise ValueError(
            "mesh device array rank does not match the number of mesh axes."
        )
    if devices.size != grid.num_processes:
        raise ValueError(
            f"{caller} currently expects the JAX mesh to contain exactly the "
            "axes used by the cuSOLVERMp process grid. Got "
            f"{devices.size} mesh devices for a {grid.process_rows} x "
            f"{grid.process_cols} process grid."
        )

    # A degenerate grid leaves one matrix dimension unsharded: the mesh has no
    # axis for it, so neither array below carries that extent-one dimension.
    positions = tuple(axis_names.index(axis) for axis in matrix_axes)
    devices_by_matrix_axis = np.moveaxis(devices, positions, range(len(positions)))
    grid_shape = (grid.process_rows, grid.process_cols)
    expected_shape = tuple(
        size
        for axis, size in zip((row_axis, col_axis), grid_shape)
        if axis is not None
    )
    if devices_by_matrix_axis.shape != expected_shape:
        raise ValueError(
            "mesh device grid does not match the matrix process-grid shape "
            f"{expected_shape}, got {devices_by_matrix_axis.shape}."
        )

    process_devices = list(devices_by_matrix_axis.reshape(-1))
    communicator_devices = sorted(process_devices, key=_device_rank_key)
    rank_by_device = {
        _device_rank_key(device): rank
        for rank, device in enumerate(communicator_devices)
    }
    if len(rank_by_device) != len(process_devices):
        raise ValueError("mesh contains duplicate process devices.")

    rank_map = ProcessRankMap(
        grid=grid,
        ranks=tuple(
            rank_by_device[_device_rank_key(device)] for device in process_devices
        ),
    )
    rank_map.require_cusolvermp_grid_mapping(caller)
    return rank_map


def standard_grid_rank_map_attr(
    rank_map,
    *,
    process_rows: int,
    process_cols: int,
    caller: str,
) -> np.ndarray:
    """Validate and encode a process-slot to communicator-rank mapping.

    Args:
        rank_map: ``ProcessRankMap``, one-dimensional rank sequence, or ``None``
            for row-major identity order.
        process_rows: Number of process-grid rows.
        process_cols: Number of process-grid columns.
        caller: Routine name included in validation errors.

    Returns:
        A contiguous ``int64`` array indexed by row-major process-grid slot.

    Raises:
        ValueError: If the map has the wrong shape or is not a permutation of
            all communicator ranks.
        NotImplementedError: If the permutation is neither cuSOLVERMp row-major
            nor column-major order.
    """
    process_rows = int(process_rows)
    process_cols = int(process_cols)
    num_ranks = process_rows * process_cols
    if rank_map is None:
        rank_array = np.arange(num_ranks, dtype=np.int64)
    elif hasattr(rank_map, "ranks"):
        rank_array = np.asarray(rank_map.ranks, dtype=np.int64)
    else:
        rank_array = np.asarray(rank_map, dtype=np.int64)
    if rank_array.shape != (num_ranks,):
        raise ValueError(
            f"{caller} rank_map must have shape ({num_ranks},), got "
            f"{rank_array.shape}."
        )
    if sorted(rank_array.tolist()) != list(range(num_ranks)):
        raise ValueError(
            f"{caller} rank_map must be a permutation of [0, {num_ranks})."
        )

    row_major = np.arange(num_ranks, dtype=np.int64)
    column_major = np.asarray(
        [
            process_col * process_rows + process_row
            for process_row in range(process_rows)
            for process_col in range(process_cols)
        ],
        dtype=np.int64,
    )
    if not np.array_equal(rank_array, row_major) and not np.array_equal(
        rank_array, column_major
    ):
        raise NotImplementedError(
            f"{caller} supports only row-major or column-major cuSOLVERMp "
            f"rank maps. Got {rank_array.tolist()}."
        )
    return np.ascontiguousarray(rank_array)


def cusolvermp_grid_mapping_attr(
    rank_map,
    grid_mapping,
    *,
    process_rows: int,
    process_cols: int,
    caller: str,
) -> int:
    """Resolve cuSOLVERMp's grid-mapping enum for a validated rank map.

    NVIDIA defines column-major mapping as ``0`` and row-major mapping as ``1``.
    When ``grid_mapping`` is omitted, the value is inferred from ``rank_map``.

    Args:
        rank_map: Process-slot to communicator-rank mapping.
        grid_mapping: Explicit cuSOLVERMp enum value, or ``None`` to infer it.
        process_rows: Number of process-grid rows.
        process_cols: Number of process-grid columns.
        caller: Routine name included in validation errors.

    Returns:
        ``0`` for column-major rank order or ``1`` for row-major rank order.

    Raises:
        ValueError: If the enum is unsupported or disagrees with ``rank_map``.
        NotImplementedError: If ``rank_map`` is not a supported dense order.
    """
    rank_array = standard_grid_rank_map_attr(
        rank_map,
        process_rows=process_rows,
        process_cols=process_cols,
        caller=caller,
    )
    if grid_mapping is None and hasattr(rank_map, "cusolvermp_grid_mapping"):
        grid_mapping = rank_map.cusolvermp_grid_mapping
    if grid_mapping is None:
        row_major = np.arange(int(process_rows) * int(process_cols), dtype=np.int64)
        grid_mapping = 1 if np.array_equal(rank_array, row_major) else 0
    grid_mapping = int(grid_mapping)
    if grid_mapping not in (0, 1):
        raise ValueError(
            f"{caller} grid_mapping must be 0 (column-major) or 1 (row-major), "
            f"got {grid_mapping}."
        )

    expected = (
        np.arange(int(process_rows) * int(process_cols), dtype=np.int64)
        if grid_mapping == 1
        else np.asarray(
            [
                process_col * int(process_rows) + process_row
                for process_row in range(int(process_rows))
                for process_col in range(int(process_cols))
            ],
            dtype=np.int64,
        )
    )
    if not np.array_equal(rank_array, expected):
        raise ValueError(
            f"{caller} rank_map does not match grid_mapping={grid_mapping}."
        )
    return grid_mapping


def infer_mesh_and_matrix_specs(
    a: Array,
    *,
    mesh: Mesh | None,
    matrix_specs: P | tuple[P, ...] | list[P] | None,
    in_specs: P | tuple[P, ...] | list[P] | None = None,
) -> tuple[Mesh, P]:
    """Resolve the JAX mesh and matrix sharding used by a solver call.

    Explicit ``mesh`` and ``matrix_specs`` values take precedence. Any missing
    value is inferred from the input matrix when it carries ``NamedSharding``.

    Args:
        a: Input matrix whose named sharding may provide the mesh contract.
        mesh: Explicit JAX mesh, or ``None`` to infer it from ``a``.
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
# JAX-visible status and local padding layouts
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
