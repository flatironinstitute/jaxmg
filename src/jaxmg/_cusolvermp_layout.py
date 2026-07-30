"""Shared Python layout helpers for the cuSOLVERMp backend.

The public solvers in :mod:`jaxmg._potrs`, :mod:`jaxmg._lu_solve`, and
:mod:`jaxmg._syevd` should read like user-facing numerical routines. This
module holds the common JAX mesh, right-hand-side placement, process-grid, and
local-padding mechanics required before entering the fused native FFI calls.

The module covers four parts of the Python-to-native layout contract:

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
    """Normalize legacy ``in_specs`` and current ``matrix_specs`` inputs.

    Older JAXMg entry points accepted ``in_specs`` as either a single
    ``PartitionSpec`` or a one-element tuple/list.  The cuSOLVERMp backend uses
    the clearer name ``matrix_specs`` because the same 2D sharding contract is
    applied to all matrix-like arguments.  Keeping this tiny compatibility
    bridge avoids spreading argument-normalization code across the public
    wrappers.
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
    """Return the size of one named JAX mesh axis."""
    try:
        return int(mesh.shape[axis_name])
    except KeyError as exc:
        raise ValueError(f"mesh does not contain axis {axis_name!r}.") from exc


def validate_2d_matrix_specs(
    mesh: Mesh,
    matrix_specs: P,
) -> tuple[str, str, ProcessGrid]:
    """Validate that a matrix is sharded over two named mesh axes."""
    if not isinstance(matrix_specs, P):
        raise TypeError("matrix_specs must be a PartitionSpec.")
    if len(matrix_specs._partitions) != 2:
        raise ValueError("matrix_specs must describe a rank-2 sharding.")
    row_axis, col_axis = matrix_specs._partitions
    if not isinstance(row_axis, str) or not isinstance(col_axis, str):
        raise ValueError(
            "cuSOLVERMp currently requires both matrix axes to be sharded by "
            "named mesh axes, for example P('pr', 'pc')."
        )
    grid = ProcessGrid(
        process_rows=mesh_axis_size(mesh, row_axis),
        process_cols=mesh_axis_size(mesh, col_axis),
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
    """Place an array using the primitive required by the matrix mesh axes."""
    axis_types = dict(zip(mesh.axis_names, mesh.axis_types))
    matrix_axes = tuple(matrix_specs._partitions)
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

    Mixed explicit/auto process-grid axes are rejected. They are not a normal
    JAX mesh construction for this backend and neither placement primitive can
    express the mixed target consistently.
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
    """Return the host-process index that owns a JAX device."""
    value = getattr(device, "process_index", None)
    if callable(value):
        return int(value())
    if value is None:
        raise AttributeError(f"device {device!r} has no process_index")
    return int(value)


def _device_id(device) -> int:
    """Return JAX's stable integer device id for communicator ordering."""
    value = getattr(device, "id", None)
    if value is None:
        raise AttributeError(f"device {device!r} has no id")
    return int(value)


def _device_local_hardware_id(device) -> int:
    """Return the local GPU hardware id when JAX exposes one."""
    value = getattr(device, "local_hardware_id", None)
    if value is None:
        return _device_id(device)
    return int(value)


def _device_rank_key(device) -> tuple[int, int, int]:
    """Best available Python-side model of XLA communicator rank order."""
    return (
        _device_process_index(device),
        _device_id(device),
        _device_local_hardware_id(device),
    )


def process_rank_map_from_mesh(
    mesh: Mesh,
    *,
    row_axis: str,
    col_axis: str,
    grid: ProcessGrid,
    caller: str,
) -> ProcessRankMap:
    """Map JAX mesh coordinates to cuSOLVERMp communicator ranks.

    Users construct meshes with normal JAX APIs.  JAXMg inspects the resulting
    device array and accepts it only when the process-grid coordinates match a
    cuSOLVERMp-supported row-major or column-major rank mapping.  Arbitrary
    mesh permutations are rejected before native code runs, because cuSOLVERMp
    cannot represent those permutations in its grid descriptor.
    """
    axis_names = tuple(mesh.axis_names)
    if row_axis not in axis_names or col_axis not in axis_names:
        raise ValueError("matrix sharding axes must be present in the mesh.")
    if row_axis == col_axis:
        raise ValueError("matrix row and column axes must be distinct.")

    devices = np.asarray(mesh.devices, dtype=object)
    if devices.ndim != len(axis_names):
        raise ValueError(
            "mesh device array rank does not match the number of mesh axes."
        )
    if devices.size != grid.num_processes:
        raise ValueError(
            f"{caller} currently expects the JAX mesh to contain exactly the "
            "two axes used by the cuSOLVERMp process grid. Got "
            f"{devices.size} mesh devices for a {grid.process_rows} x "
            f"{grid.process_cols} process grid."
        )

    row_position = axis_names.index(row_axis)
    col_position = axis_names.index(col_axis)
    devices_by_matrix_axis = np.moveaxis(
        devices,
        (row_position, col_position),
        (0, 1),
    )
    expected_shape = (grid.process_rows, grid.process_cols)
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
    """Return a dense cuSOLVERMp-compatible process-slot -> rank map."""
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
    """Return cuSOLVERMp's grid-mapping enum for the validated rank map."""
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
    """Infer the cuSOLVERMp mesh contract from an input JAX array."""
    matrix_specs = normalize_matrix_specs(matrix_specs, in_specs=in_specs)
    if mesh is not None and matrix_specs is not None:
        return mesh, matrix_specs

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
    return mesh, matrix_specs


# -----------------------------------------------------------------------------
# JAX-visible status and local padding layouts
# -----------------------------------------------------------------------------


def status_specs(row_axis: str, col_axis: str, grid: ProcessGrid) -> P:
    """Choose a sharding for one small per-rank native status vector."""
    if grid.process_rows == 1:
        return P(col_axis)
    if grid.process_cols == 1:
        return P(row_axis)
    return P((row_axis, col_axis))


def _pad_local_2d(block: Array, *, row_padding: int, col_padding: int) -> Array:
    """Pad a single local shard on its bottom and right edges."""
    if row_padding == 0 and col_padding == 0:
        return block
    return jnp.pad(block, ((0, row_padding), (0, col_padding)))


def _unpad_local_2d(block: Array, *, local_rows: int, local_cols: int) -> Array:
    """Slice a local shard back to its unpadded logical shape."""
    return block[:local_rows, :local_cols]
