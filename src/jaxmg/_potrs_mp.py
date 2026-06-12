"""Public cuSOLVERMp Cholesky solve wrapper.

The native cuSOLVERMp backend consumes a 2D block-cyclic, column-major local
layout, while users naturally construct ordinary JAX block-sharded arrays. This
module owns the high-level Python orchestration around the native handlers:

1. validate that ``A`` and ``B`` are compatible with a 2D JAX process grid;
2. locally pad each JAX shard so both local axes are tile-aligned;
3. allocate bounded per-rank scratch for native redistribution;
4. call the native forward 2D redistribution handler;
5. call the production ``cusolvermp_potrs`` FFI target; and
6. reverse-redistribute and unpad the solved right-hand side.

The expensive data movement and cuSOLVERMp calls are native C++/CUDA work. The
Python layer keeps only shape, sharding, and scratch-size policy visible.
"""

from __future__ import annotations

from functools import partial
from typing import Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from ._block_cyclic_2d_execute import (
    execute_padded_block_cyclic_2d_shardmap,
    execute_reverse_padded_block_cyclic_2d_shardmap,
    required_padded_block_cyclic_2d_scratch_size,
)
from ._block_cyclic_2d_plan import (
    ProcessGrid,
    ProcessRankMap,
    TileShape,
    calculate_2d_padding,
)
from ._setup import ensure_init_jaxmg_backend
from ._xla_comm_probe import cusolvermp_potrs_shardmap


def _mesh_axis_size(mesh: Mesh, axis_name: str) -> int:
    try:
        return int(mesh.shape[axis_name])
    except KeyError as exc:
        raise ValueError(f"mesh does not contain axis {axis_name!r}.") from exc


def _validate_2d_matrix_specs(mesh: Mesh, matrix_specs: P) -> tuple[str, str, ProcessGrid]:
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
        process_rows=_mesh_axis_size(mesh, row_axis),
        process_cols=_mesh_axis_size(mesh, col_axis),
    )
    return row_axis, col_axis, grid


def _device_process_index(device) -> int:
    value = getattr(device, "process_index", None)
    if callable(value):
        return int(value())
    if value is None:
        raise AttributeError(f"device {device!r} has no process_index")
    return int(value)


def _device_id(device) -> int:
    value = getattr(device, "id", None)
    if value is None:
        raise AttributeError(f"device {device!r} has no id")
    return int(value)


def _device_local_hardware_id(device) -> int:
    value = getattr(device, "local_hardware_id", None)
    if value is None:
        return _device_id(device)
    return int(value)


def _device_rank_key(device) -> tuple[int, int, int]:
    """Best available Python-side model of XLA's communicator rank order."""
    return (
        _device_process_index(device),
        _device_id(device),
        _device_local_hardware_id(device),
    )


def _process_rank_map_from_mesh(
    mesh: Mesh,
    *,
    row_axis: str,
    col_axis: str,
    grid: ProcessGrid,
) -> ProcessRankMap:
    """Infer the process-grid rank map implied by a JAX mesh.

    Native redistribution and cuSOLVERMp operate on communicator ranks.  The
    public API operates on a JAX ``Mesh``.  This helper is the bridge: it reads
    the mesh device array in the matrix row/column axis order and maps each
    process-grid slot to the rank it will have in the all-assigned communicator
    order used by JAXMg's native backend.

    The redistribution layer uses this map to translate between the user's JAX
    mesh order and the dense row-major rank order used by cuSOLVERMp.  Physical
    GPU ids do not have to be contiguous; they only need to form a unique set
    of devices participating in this JAX mesh.
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
            "potrs_mp currently expects the JAX mesh to contain exactly the "
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
    return ProcessRankMap(
        grid=grid,
        ranks=tuple(
            rank_by_device[_device_rank_key(device)] for device in process_devices
        ),
    )


def _infer_mesh_and_matrix_specs(
    a: Array,
    *,
    mesh: Mesh | None,
    matrix_specs: P | None,
) -> tuple[Mesh, P]:
    """Infer the cuSOLVERMp mesh contract from the input JAX sharding.

    Users normally create a JAX ``Mesh`` and ``NamedSharding`` before calling
    JAXMg.  The solver should therefore consume the sharding already attached
    to ``A`` instead of requiring the same mesh/spec to be repeated at the call
    site.  Explicit ``mesh`` and ``matrix_specs`` remain available for
    diagnostics and for tests that exercise validation before arrays are
    physically sharded.
    """
    if mesh is not None and matrix_specs is not None:
        return mesh, matrix_specs

    sharding = getattr(a, "sharding", None)
    if not isinstance(sharding, NamedSharding):
        raise ValueError(
            "potrs_mp could not infer mesh/matrix_specs from A. Shard A with "
            "jax.sharding.NamedSharding or pass mesh=... and matrix_specs=..."
        )

    if mesh is None:
        mesh = sharding.mesh
    if matrix_specs is None:
        matrix_specs = sharding.spec
    return mesh, matrix_specs


def _status_specs(row_axis: str, col_axis: str, grid: ProcessGrid) -> P:
    if grid.process_rows == 1:
        return P(col_axis)
    if grid.process_cols == 1:
        return P(row_axis)
    return P((row_axis, col_axis))


def _pad_local_2d(block: Array, *, row_padding: int, col_padding: int) -> Array:
    if row_padding == 0 and col_padding == 0:
        return block
    return jnp.pad(block, ((0, row_padding), (0, col_padding)))


def _unpad_local_2d(
    block: Array, *, local_rows: int, local_cols: int
) -> Array:
    return block[:local_rows, :local_cols]


def _pad_block_sharded_2d(
    matrix: Array,
    *,
    mesh: Mesh,
    matrix_specs: P,
    grid: ProcessGrid,
    tile_shape: TileShape,
    pad: bool,
) -> tuple[Array, tuple[int, int]]:
    padding = calculate_2d_padding(
        logical_rows=matrix.shape[0],
        logical_cols=matrix.shape[1],
        grid=grid,
        tile_shape=tile_shape,
    )
    if not pad and padding.needs_padding:
        raise ValueError(
            "potrs_mp requires tile-aligned local shards when pad=False. "
            "Use a tile size that divides each local shard or set pad=True."
        )

    if not padding.needs_padding:
        return matrix, (padding.local_logical_rows, padding.local_logical_cols)

    pad_fn = jax.shard_map(
        partial(
            _pad_local_2d,
            row_padding=padding.row_padding_per_process,
            col_padding=padding.col_padding_per_process,
        ),
        mesh=mesh,
        in_specs=matrix_specs,
        out_specs=matrix_specs,
        check_vma=True,
    )
    padded = pad_fn(matrix)
    return padded, (padding.local_logical_rows, padding.local_logical_cols)


def _unpad_block_sharded_2d(
    matrix: Array,
    *,
    mesh: Mesh,
    matrix_specs: P,
    local_rows: int,
    local_cols: int,
) -> Array:
    unpad_fn = jax.shard_map(
        partial(_unpad_local_2d, local_rows=local_rows, local_cols=local_cols),
        mesh=mesh,
        in_specs=matrix_specs,
        out_specs=matrix_specs,
        check_vma=True,
    )
    return unpad_fn(matrix)


def _redistribute_to_cusolvermp(
    matrix: Array,
    scratch: Array,
    *,
    mesh: Mesh,
    matrix_specs: P,
    scratch_specs: P,
    grid: ProcessGrid,
    block_rank_map: ProcessRankMap,
    cyclic_rank_map: ProcessRankMap,
    tile_shape: TileShape,
    logical_rows: int,
    logical_cols: int,
) -> tuple[Array, Array]:
    return execute_padded_block_cyclic_2d_shardmap(
        matrix,
        scratch,
        mesh,
        matrix_specs,
        scratch_specs,
        logical_rows=logical_rows,
        logical_cols=logical_cols,
        grid=grid,
        block_rank_map=block_rank_map,
        cyclic_rank_map=cyclic_rank_map,
        tile_rows=tile_shape.rows,
        tile_cols=tile_shape.cols,
    )


def _redistribute_from_cusolvermp(
    matrix: Array,
    scratch: Array,
    *,
    mesh: Mesh,
    matrix_specs: P,
    scratch_specs: P,
    grid: ProcessGrid,
    block_rank_map: ProcessRankMap,
    cyclic_rank_map: ProcessRankMap,
    tile_shape: TileShape,
    logical_rows: int,
    logical_cols: int,
) -> tuple[Array, Array]:
    return execute_reverse_padded_block_cyclic_2d_shardmap(
        matrix,
        scratch,
        mesh,
        matrix_specs,
        scratch_specs,
        logical_rows=logical_rows,
        logical_cols=logical_cols,
        grid=grid,
        block_rank_map=block_rank_map,
        cyclic_rank_map=cyclic_rank_map,
        tile_rows=tile_shape.rows,
        tile_cols=tile_shape.cols,
    )


def potrs_mp(
    a: Array,
    b: Array,
    T_A: int,
    mesh: Mesh | None = None,
    matrix_specs: P | None = None,
    return_status: bool = False,
    pad: bool = True,
) -> Union[Array, Tuple[Array, Array]]:
    """Solve ``A x = B`` with cuSOLVERMp on a 2D JAX process grid.

    This is the first production cuSOLVERMp path in JAXMg. It accepts a normal
    block-sharded JAX layout, pads each local shard to the tile size, performs
    the GPU-to-GPU 2D block-cyclic redistribution, runs cuSOLVERMp ``potrf`` and
    ``potrs`` using the XLA-owned NCCL communicator, then reverse-redistributes
    the solved RHS back to the original block-sharded layout.

    By default, the process grid is inferred from the ``NamedSharding`` attached
    to ``a``.  The row-major order of that mesh is used as the cuSOLVERMp rank
    order: rank = process_row * process_cols + process_col.  That must match
    the row-major cuSOLVERMp grid descriptor used natively.

    Args:
        a: Square Hermitian/symmetric positive-definite matrix, normally
            sharded with ``NamedSharding(mesh, P(row_axis, col_axis))``.
        b: Rank-1 or rank-2 right-hand side, sharded with the same
            mesh/spec as ``a``. The current implementation requires both
            logical dimensions to divide evenly over the corresponding
            process-grid dimensions before local tile padding is applied.
        T_A: Square cuSOLVERMp tile size. JAXMg currently uses
            ``MB_A == NB_A == T_A`` for the Cholesky solve path.
        mesh: Optional JAX mesh override. If omitted, inferred from
            ``a.sharding.mesh``.
        matrix_specs: Optional PartitionSpec override for both ``a`` and ``b``.
            If omitted, inferred from ``a.sharding.spec``.
        return_status: If true, return the per-rank native status vector
            alongside the solved RHS.
        pad: If true, add local row/column capacity so every local shard is
            tile-aligned. If false, incompatible local shard shapes raise.
    """
    if a.ndim != 2:
        raise ValueError("potrs_mp expects a rank-2 matrix A.")
    if b.ndim == 1:
        b = jnp.expand_dims(b, axis=1)
    if b.ndim != 2:
        raise ValueError("potrs_mp expects a rank-1 or rank-2 RHS B.")
    if a.dtype != b.dtype:
        raise TypeError("potrs_mp requires matching A/B dtypes.")
    if a.dtype not in (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128):
        raise TypeError(
            "potrs_mp supports float32, float64, complex64, and complex128."
        )
    if a.shape[0] != a.shape[1]:
        raise ValueError("potrs_mp expects A to be square.")
    if a.shape[0] != b.shape[0]:
        raise ValueError("A and B must have matching leading dimensions.")
    if int(T_A) <= 0:
        raise ValueError("T_A must be positive.")

    mesh, matrix_specs = _infer_mesh_and_matrix_specs(
        a,
        mesh=mesh,
        matrix_specs=matrix_specs,
    )
    row_axis, col_axis, grid = _validate_2d_matrix_specs(mesh, matrix_specs)
    jax_rank_map = _process_rank_map_from_mesh(
        mesh,
        row_axis=row_axis,
        col_axis=col_axis,
        grid=grid,
    )
    solver_rank_map = ProcessRankMap.row_major(grid)
    scratch_specs = _status_specs(row_axis, col_axis, grid)
    tile_shape = TileShape(rows=int(T_A), cols=int(T_A))

    ensure_init_jaxmg_backend()

    a_padded, _ = _pad_block_sharded_2d(
        a,
        mesh=mesh,
        matrix_specs=matrix_specs,
        grid=grid,
        tile_shape=tile_shape,
        pad=pad,
    )
    b_padded, (b_local_rows, b_local_cols) = _pad_block_sharded_2d(
        b,
        mesh=mesh,
        matrix_specs=matrix_specs,
        grid=grid,
        tile_shape=tile_shape,
        pad=pad,
    )

    scratch_per_rank = max(
        required_padded_block_cyclic_2d_scratch_size(
            logical_rows=a.shape[0],
            logical_cols=a.shape[1],
            grid=grid,
            tile_rows=tile_shape.rows,
            tile_cols=tile_shape.cols,
        ),
        required_padded_block_cyclic_2d_scratch_size(
            logical_rows=b.shape[0],
            logical_cols=b.shape[1],
            grid=grid,
            tile_rows=tile_shape.rows,
            tile_cols=tile_shape.cols,
        ),
    )
    scratch = jax.device_put(
        jnp.zeros((grid.num_processes * scratch_per_rank,), dtype=a.dtype),
        NamedSharding(mesh, scratch_specs),
    )

    a_cyclic, scratch = _redistribute_to_cusolvermp(
        a_padded,
        scratch,
        mesh=mesh,
        matrix_specs=matrix_specs,
        scratch_specs=scratch_specs,
        grid=grid,
        block_rank_map=jax_rank_map,
        cyclic_rank_map=solver_rank_map,
        tile_shape=tile_shape,
        logical_rows=a.shape[0],
        logical_cols=a.shape[1],
    )
    b_cyclic, scratch = _redistribute_to_cusolvermp(
        b_padded,
        scratch,
        mesh=mesh,
        matrix_specs=matrix_specs,
        scratch_specs=scratch_specs,
        grid=grid,
        block_rank_map=jax_rank_map,
        cyclic_rank_map=solver_rank_map,
        tile_shape=tile_shape,
        logical_rows=b.shape[0],
        logical_cols=b.shape[1],
    )

    _, b_solved_cyclic, status = cusolvermp_potrs_shardmap(
        a_cyclic,
        b_cyclic,
        mesh,
        matrix_specs,
        scratch_specs,
        process_rows=grid.process_rows,
        process_cols=grid.process_cols,
        n=a.shape[0],
        nrhs=b.shape[1],
        tile_size=tile_shape.rows,
        rank_map=solver_rank_map,
    )

    b_solved_padded, scratch = _redistribute_from_cusolvermp(
        b_solved_cyclic,
        scratch,
        mesh=mesh,
        matrix_specs=matrix_specs,
        scratch_specs=scratch_specs,
        grid=grid,
        block_rank_map=jax_rank_map,
        cyclic_rank_map=solver_rank_map,
        tile_shape=tile_shape,
        logical_rows=b.shape[0],
        logical_cols=b.shape[1],
    )
    out = _unpad_block_sharded_2d(
        b_solved_padded,
        mesh=mesh,
        matrix_specs=matrix_specs,
        local_rows=b_local_rows,
        local_cols=b_local_cols,
    )
    if return_status:
        return out, status
    return out
