"""Public cuSOLVERMp Cholesky solve wrapper.

``potrs`` is the high-level JAX entry point for the fused cuSOLVERMp backend.
Users provide ordinary JAX arrays sharded over a 2D ``Mesh``.  The Python layer
does the parts JAX must see directly:

1. validate the logical matrix/RHS shapes and dtypes;
2. infer or validate the JAX mesh and ``PartitionSpec``;
3. check that the mesh order is compatible with a cuSOLVERMp process grid;
4. pad each local shard so both axes are tile-aligned; and
5. remove local padding from the solved RHS after native code returns.

The expensive work is a single fused C++/CUDA FFI call.  Native code allocates
its own scratch, redistributes the padded JAX layout into cuSOLVERMp's 2D
block-cyclic layout, calls ``cusolverMpPotrf``/``cusolverMpPotrs`` using the
XLA-owned NCCL communicator, and redistributes the result back to the original
JAX-facing layout.
"""

from __future__ import annotations

from typing import Union

import jax.numpy as jnp
from jax import Array
from jax.sharding import Mesh, PartitionSpec as P

from ._block_cyclic_2d_plan import TileShape
from ._cusolvermp_ffi import cusolvermp_potrs_shardmap
from ._cusolvermp_layout import (
    infer_mesh_and_matrix_specs,
    pad_block_sharded_2d,
    process_rank_map_from_mesh,
    status_specs,
    unpad_block_sharded_2d,
    validate_2d_matrix_specs,
)
from ._setup import ensure_init_jaxmg_backend


def potrs(
    a: Array,
    b: Array,
    T_A: int,
    mesh: Mesh | None = None,
    matrix_specs: P | tuple[P, ...] | list[P] | None = None,
    *,
    in_specs: P | tuple[P, ...] | list[P] | None = None,
    return_status: bool = False,
    pad: bool = True,
) -> Union[Array, tuple[Array, Array]]:
    """Solve ``A x = B`` with cuSOLVERMp on a 2D JAX process grid.

    Args:
        a: Square Hermitian/symmetric positive-definite matrix, normally
            sharded with ``NamedSharding(mesh, P(row_axis, col_axis))``.
        b: Rank-1 or rank-2 right-hand side.  The current cuSOLVERMp path uses
            the same 2D mesh/spec contract as ``a`` for the RHS.
        T_A: Square cuSOLVERMp tile size.  JAXMg currently uses
            ``MB_A == NB_A == T_A``.
        mesh: Optional JAX mesh override.  If omitted, inferred from
            ``a.sharding.mesh``.
        matrix_specs: Optional 2D ``PartitionSpec`` override.  If omitted,
            inferred from ``a.sharding.spec``.
        in_specs: Backwards-compatible alias for ``matrix_specs``.
        return_status: If true, return the native per-rank status array along
            with the solved RHS.
        pad: If true, locally pad shards so every local row/column capacity is
            tile-aligned.  If false, incompatible shapes raise.

    Returns:
        The solved RHS in the same JAX-facing block-sharded layout as ``b``.
        If ``return_status=True``, returns ``(x, status)``.
    """
    if a.ndim != 2:
        raise ValueError("potrs expects a rank-2 matrix A.")
    if b.ndim == 1:
        b = jnp.expand_dims(b, axis=1)
    if b.ndim != 2:
        raise ValueError("potrs expects a rank-1 or rank-2 RHS B.")
    if a.dtype != b.dtype:
        raise TypeError("potrs requires matching A/B dtypes.")
    if a.dtype not in (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128):
        raise TypeError("potrs supports float32, float64, complex64, and complex128.")
    if a.shape[0] != a.shape[1]:
        raise ValueError("potrs expects A to be square.")
    if a.shape[0] != b.shape[0]:
        raise ValueError("A and B must have matching leading dimensions.")
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
        caller="potrs",
    )
    native_status_specs = status_specs(row_axis, col_axis, grid)
    tile_shape = TileShape(rows=int(T_A), cols=int(T_A))

    ensure_init_jaxmg_backend()

    # JAX owns visible padding because it changes array shapes.  Once padded,
    # native code can move complete tile rectangles without allocating a second
    # full distributed matrix in Python.
    a_padded, _ = pad_block_sharded_2d(
        a,
        mesh=mesh,
        matrix_specs=matrix_specs,
        grid=grid,
        tile_shape=tile_shape,
        pad=pad,
    )
    b_padded, (b_local_rows, b_local_cols) = pad_block_sharded_2d(
        b,
        mesh=mesh,
        matrix_specs=matrix_specs,
        grid=grid,
        tile_shape=tile_shape,
        pad=pad,
    )

    b_solved_padded, native_status = cusolvermp_potrs_shardmap(
        a_padded,
        b_padded,
        mesh,
        matrix_specs,
        native_status_specs,
        process_rows=grid.process_rows,
        process_cols=grid.process_cols,
        n=a.shape[0],
        nrhs=b.shape[1],
        tile_size=tile_shape.rows,
        rank_map=rank_map,
        grid_mapping=rank_map.cusolvermp_grid_mapping,
    )

    out = unpad_block_sharded_2d(
        b_solved_padded,
        mesh=mesh,
        matrix_specs=matrix_specs,
        local_rows=b_local_rows,
        local_cols=b_local_cols,
    )
    if return_status:
        return out, native_status
    return out
