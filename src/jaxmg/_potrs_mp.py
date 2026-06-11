from __future__ import annotations

from functools import partial
from typing import Tuple, Union

import jax
import jax.numpy as jnp
from jax import Array
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from ._block_cyclic_2d_execute import (
    execute_padded_block_cyclic_2d_shardmap,
    execute_reverse_padded_block_cyclic_2d_shardmap,
    required_padded_block_cyclic_2d_scratch_size,
)
from ._block_cyclic_2d_plan import (
    ProcessGrid,
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
        tile_rows=tile_shape.rows,
        tile_cols=tile_shape.cols,
    )


def potrs_mp(
    a: Array,
    b: Array,
    T_A: int,
    mesh: Mesh,
    matrix_specs: P,
    return_status: bool = False,
    pad: bool = True,
) -> Union[Array, Tuple[Array, Array]]:
    """Solve ``A x = B`` with cuSOLVERMp on a 2D JAX process grid.

    This is the first production cuSOLVERMp path in JAXMg. It accepts a normal
    block-sharded JAX layout, pads each local shard to the tile size, performs
    the GPU-to-GPU 2D block-cyclic redistribution, runs cuSOLVERMp ``potrf`` and
    ``potrs`` using the XLA-owned NCCL communicator, then reverse-redistributes
    the solved RHS back to the original block-sharded layout.

    The process grid is inferred from ``matrix_specs`` and ``mesh`` using
    row-major rank mapping: rank = process_row * process_cols + process_col.
    That must match the row-major cuSOLVERMp grid descriptor used natively.

    Args:
        a: Square Hermitian/symmetric positive-definite matrix, sharded with
            ``matrix_specs`` over a 2D mesh.
        b: Rank-1 or rank-2 right-hand side, sharded with the same
            ``matrix_specs``. The current implementation requires both logical
            dimensions to divide evenly over the corresponding process-grid
            dimensions before local tile padding is applied.
        T_A: Square cuSOLVERMp tile size. JAXMg currently uses
            ``MB_A == NB_A == T_A`` for the Cholesky solve path.
        mesh: JAX mesh whose row-major device order defines the cuSOLVERMp
            process-grid rank order.
        matrix_specs: PartitionSpec for both ``a`` and ``b``; normally
            ``P('pr', 'pc')``.
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

    row_axis, col_axis, grid = _validate_2d_matrix_specs(mesh, matrix_specs)
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
    )

    b_solved_padded, scratch = _redistribute_from_cusolvermp(
        b_solved_cyclic,
        scratch,
        mesh=mesh,
        matrix_specs=matrix_specs,
        scratch_specs=scratch_specs,
        grid=grid,
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
