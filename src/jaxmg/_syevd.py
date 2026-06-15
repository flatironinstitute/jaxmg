"""Public cuSOLVERMp symmetric/Hermitian eigensolver wrapper.

``syevd`` uses the same fused layout pipeline as :func:`jaxmg.potrs`:

1. validate the input matrix and its 2D JAX sharding;
2. pad local shards so both matrix axes are tile-aligned;
3. enter one native C++/CUDA FFI call;
4. let native code redistribute into cuSOLVERMp's 2D block-cyclic layout and
   call vector-producing ``cusolverMpSyevd``; and
5. reverse-redistribute eigenvectors and remove the local padding.

Only the eigenvector-producing mode is exposed.  Current validated cuSOLVERMp
runtimes reject the no-vector mode, and computing eigenvectors only to discard
them would hide a very different cost model from users.
"""

from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp
from jax import Array
from jax.sharding import Mesh, PartitionSpec as P

from ._block_cyclic_2d_plan import TileShape
from ._cusolvermp_ffi import cusolvermp_syevd_shardmap
from ._cusolvermp_layout import (
    infer_mesh_and_matrix_specs,
    pad_block_sharded_2d,
    process_rank_map_from_mesh,
    status_specs,
    unpad_block_sharded_2d,
    validate_2d_matrix_specs,
)
from ._setup import ensure_init_jaxmg_backend


def syevd(
    a: Array,
    T_A: int,
    mesh: Mesh | None = None,
    matrix_specs: P | tuple[P, ...] | list[P] | None = None,
    *,
    in_specs: P | tuple[P, ...] | list[P] | None = None,
    return_status: bool = False,
    pad: bool = True,
) -> Tuple[Array, Array] | Tuple[Array, Array, Array]:
    """Compute eigenvalues and eigenvectors on a 2D cuSOLVERMp process grid.

    Args:
        a: Square symmetric/Hermitian matrix, normally sharded with
            ``NamedSharding(mesh, P(row_axis, col_axis))``.
        T_A: Square cuSOLVERMp tile size.  JAXMg currently uses
            ``MB_A == NB_A == T_A``.
        mesh: Optional JAX mesh override.  If omitted, inferred from
            ``a.sharding.mesh``.
        matrix_specs: Optional 2D ``PartitionSpec`` override.  If omitted,
            inferred from ``a.sharding.spec``.
        in_specs: Backwards-compatible alias for ``matrix_specs``.
        return_status: If true, append the native per-rank status array.
        pad: If true, locally pad shards so every local row/column capacity is
            tile-aligned.  If false, incompatible shapes raise.

    Returns:
        ``(w, v)`` where ``w`` contains replicated eigenvalues and ``v`` is
        returned in the same JAX-facing block-sharded layout as ``a``.  If
        ``return_status=True``, returns ``(w, v, status)``.
    """
    if a.ndim != 2:
        raise ValueError("syevd expects a rank-2 matrix A.")
    if a.dtype not in (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128):
        raise TypeError("syevd supports float32, float64, complex64, and complex128.")
    if a.shape[0] != a.shape[1]:
        raise ValueError("syevd expects A to be square.")
    if int(T_A) <= 0:
        raise ValueError("T_A must be positive.")

    mesh, matrix_specs = infer_mesh_and_matrix_specs(
        a,
        mesh=mesh,
        matrix_specs=matrix_specs,
        in_specs=in_specs,
    )
    row_axis, col_axis, grid = validate_2d_matrix_specs(mesh, matrix_specs)
    rank_map = process_rank_map_from_mesh(
        mesh,
        row_axis=row_axis,
        col_axis=col_axis,
        grid=grid,
        caller="syevd",
    )
    native_status_specs = status_specs(row_axis, col_axis, grid)
    tile_shape = TileShape(rows=int(T_A), cols=int(T_A))

    ensure_init_jaxmg_backend()

    a_padded, (a_local_rows, a_local_cols) = pad_block_sharded_2d(
        a,
        mesh=mesh,
        matrix_specs=matrix_specs,
        grid=grid,
        tile_shape=tile_shape,
        pad=pad,
    )

    eigenvalues, vectors_padded, native_status = cusolvermp_syevd_shardmap(
        a_padded,
        mesh,
        matrix_specs,
        native_status_specs,
        process_rows=grid.process_rows,
        process_cols=grid.process_cols,
        n=a.shape[0],
        tile_size=tile_shape.rows,
        rank_map=rank_map,
        grid_mapping=rank_map.cusolvermp_grid_mapping,
    )

    vectors = unpad_block_sharded_2d(
        vectors_padded,
        mesh=mesh,
        matrix_specs=matrix_specs,
        local_rows=a_local_rows,
        local_cols=a_local_cols,
    )
    if return_status:
        return eigenvalues, vectors, native_status
    return eigenvalues, vectors
