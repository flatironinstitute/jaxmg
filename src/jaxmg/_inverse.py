"""Distributed inversion through a fused factorization/inversion call.

POTRI and GETRI accept the original square matrix, not precomputed factors.
Their public and context interfaces share layout preparation and one native
target; only the factorization and inversion algorithm differs.
"""

from __future__ import annotations

from functools import lru_cache, partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.sharding import AbstractMesh, Mesh, PartitionSpec as P

from ._cusolvermp_layout import (
    mark_varying,
    pad_local_2d,
    unpad_local_2d,
    place_for_native_work,
    prepare_input_matrix_layout,
    prepare_local_matrix_layout,
    restore_matrix_result,
)
from ._cusolvermp_status import _CUSOLVERMP_INVERSE_STATUS_SIZE
from ._setup import ensure_init_jaxmg_backend


def potri(
    a: Array,
    T_A: int,
    mesh: Mesh | AbstractMesh | None = None,
    matrix_specs: P | tuple[P] | list[P] | None = None,
    *,
    in_specs: P | tuple[P] | list[P] | None = None,
    return_status: bool = False,
    pad: bool = True,
    donate: bool = True,
) -> Array | tuple[Array, Array]:
    """Compute the distributed inverse of a symmetric or Hermitian positive-definite matrix.

    Returns the full inverse of a square matrix ``A``. Unlike NVIDIA's
    factor-based POTRI interface, this routine accepts the original matrix
    and performs Cholesky factorization and inversion in one native call.

    The lower triangle defines the Hermitian matrix; the returned inverse
    contains both triangles, with conjugate symmetry for complex inputs.

    Args:
        a: Original square matrix, sharded over a one- or two-axis mesh or
            replicated. Supports float32, float64, complex64 and complex128.
        T_A: Square tile dimension used by cuSOLVERMp.
        mesh: Device mesh. If omitted, inferred from A or the active context.
        matrix_specs: Matrix partition specification. If omitted, inferred
            from A, defaulting to the mesh axes in order.
        in_specs: Backwards-compatible alias for ``matrix_specs``.
        return_status: Return ``(inverse, status)`` rather than just the inverse.
        pad: Allow local tile-capacity padding; otherwise require alignment.
        donate: Reuse A's storage by default. Donated inputs cannot be used
            again. Pass False to preserve A with an additional matrix buffer.

    Returns:
        The full inverse with A's shape and result sharding, including
        replication when A was replicated. Optional status contains per-rank
        native diagnostics. Numerical failure returns NaNs and nonzero status.

    Raises:
        TypeError: If the dtype or partition specification is unsupported.
        ValueError: If shapes, tile sizes, or mesh layouts are incompatible.

    Notes:
        Prefer :func:`jaxmg.potrs` when only ``A x = B`` is required.
        POTRF and POTRI reuse one matrix allocation and release factorization
        workspace before allocating inversion workspace.
    """
    layout = _prepare_inverse_call(
        a, T_A, mesh, matrix_specs, in_specs=in_specs, pad=pad, caller="potri",
    )
    inverse, status = _inverse_compiled(
        layout, int(T_A), cholesky=True, pad=pad, donate=donate,
    )(a)
    return (inverse, status) if return_status else inverse


def potri_jit_ctx(
    a: Array,
    T_A: int,
    mesh: Mesh | AbstractMesh | None = None,
    matrix_specs: P | tuple[P] | list[P] | None = None,
    *,
    in_specs: P | tuple[P] | list[P] | None = None,
    pad: bool = True,
    _prepared=None,
) -> tuple[Array, Array]:
    """Invert A inside a caller-owned ``jax.jit`` using Cholesky.

    Accepts the original symmetric or Hermitian positive-definite matrix, as in
    :func:`jaxmg.potri`. The caller controls compilation and donation;
    ``jax.jit(..., donate_argnums=(0,))`` can reuse A's storage.

    Args:
        a: Original square global matrix.
        T_A: Square tile dimension.
        mesh: Mesh inferred from A or the active context when omitted.
        matrix_specs: Matrix partition specification, inferred when omitted.
        in_specs: Backwards-compatible alias for ``matrix_specs``.
        pad: Allow tile-capacity padding.

    Returns:
        ``(inverse, status)``. The full inverse follows A's result sharding,
        including replication. Status retains diagnostics from every rank.

    Raises:
        TypeError: If the dtype or partition specification is unsupported.
        ValueError: If shapes, tile sizes, or mesh layouts are incompatible.
    """
    return _inverse_jit_ctx(
        a, T_A, mesh, matrix_specs, in_specs=in_specs, pad=pad,
        cholesky=True, _prepared=_prepared,
    )


def potri_shardmap_ctx(
    a: Array,
    T_A: int,
    matrix_specs: P | tuple[P] | list[P] | None = None,
    *,
    in_specs: P | tuple[P] | list[P] | None = None,
    pad: bool = True,
) -> tuple[Array, Array]:
    """Invert local matrix blocks inside a caller's ``jax.shard_map``.

    Uses the same fused Cholesky factorization and inversion as
    :func:`jaxmg.potri`, with compilation and sharding owned by the caller.

    Args:
        a: Local block of the original symmetric or Hermitian positive-definite matrix.
        T_A: Square tile dimension.
        matrix_specs: A's specification in the enclosing shard map; defaults
            to the context mesh axes in order. Must describe the actual blocks,
            since the global matrix shape is reconstructed from this metadata.
        in_specs: Backwards-compatible alias for ``matrix_specs``.
        pad: Allow tile-capacity padding.

    Returns:
        ``(inverse, status)``: the unpadded local inverse block and this rank's
        diagnostic vector. The inverse aliases the input work buffer, so no
        separate ``a_work`` result is required.

    Raises:
        TypeError: If the dtype or partition specification is unsupported.
        ValueError: If called outside shard_map or the global matrix is invalid.
    """
    return _inverse_shardmap_ctx(
        a, T_A, matrix_specs, in_specs=in_specs, pad=pad, cholesky=True,
    )


def getri(
    a: Array,
    T_A: int,
    mesh: Mesh | AbstractMesh | None = None,
    matrix_specs: P | tuple[P] | list[P] | None = None,
    *,
    in_specs: P | tuple[P] | list[P] | None = None,
    return_status: bool = False,
    pad: bool = True,
    donate: bool = True,
) -> Array | tuple[Array, Array]:
    """Compute the distributed inverse of a general nonsingular matrix.

    Returns the full inverse of a square matrix ``A``. Unlike NVIDIA's
    factor-based GETRI interface, this routine accepts the original matrix
    and performs pivoted LU factorization and inversion in one native call.

    Args:
        a: Original square matrix, sharded over a one- or two-axis mesh or
            replicated. Supports float32, float64, complex64 and complex128.
        T_A: Square tile dimension used by cuSOLVERMp.
        mesh: Device mesh. If omitted, inferred from A or the active context.
        matrix_specs: Matrix partition specification. If omitted, inferred
            from A, defaulting to the mesh axes in order.
        in_specs: Backwards-compatible alias for ``matrix_specs``.
        return_status: Return ``(inverse, status)`` rather than just the inverse.
        pad: Allow local tile-capacity padding; otherwise require alignment.
        donate: Reuse A's storage by default. Donated inputs cannot be used
            again. Pass False to preserve A with an additional matrix buffer.

    Returns:
        The full inverse with A's shape and result sharding, including
        replication when A was replicated. Optional status contains per-rank
        native diagnostics. Numerical failure returns NaNs and nonzero status.

    Raises:
        TypeError: If the dtype or partition specification is unsupported.
        ValueError: If shapes, tile sizes, or mesh layouts are incompatible.

    Notes:
        Prefer :func:`jaxmg.lu_solve` when only ``A x = B`` is required.
        GETRF and GETRI reuse one matrix allocation and release factorization
        workspace before allocating inversion workspace.
    """
    layout = _prepare_inverse_call(
        a, T_A, mesh, matrix_specs, in_specs=in_specs, pad=pad, caller="getri",
    )
    inverse, status = _inverse_compiled(
        layout, int(T_A), cholesky=False, pad=pad, donate=donate,
    )(a)
    return (inverse, status) if return_status else inverse


def getri_jit_ctx(
    a: Array,
    T_A: int,
    mesh: Mesh | AbstractMesh | None = None,
    matrix_specs: P | tuple[P] | list[P] | None = None,
    *,
    in_specs: P | tuple[P] | list[P] | None = None,
    pad: bool = True,
    _prepared=None,
) -> tuple[Array, Array]:
    """Invert A inside a caller-owned ``jax.jit`` using pivoted LU.

    Accepts the original general nonsingular matrix, as in
    :func:`jaxmg.getri`. The caller controls compilation and donation;
    ``jax.jit(..., donate_argnums=(0,))`` can reuse A's storage.

    Args:
        a: Original square global matrix.
        T_A: Square tile dimension.
        mesh: Mesh inferred from A or the active context when omitted.
        matrix_specs: Matrix partition specification, inferred when omitted.
        in_specs: Backwards-compatible alias for ``matrix_specs``.
        pad: Allow tile-capacity padding.

    Returns:
        ``(inverse, status)``. The full inverse follows A's result sharding,
        including replication. Status retains diagnostics from every rank.

    Raises:
        TypeError: If the dtype or partition specification is unsupported.
        ValueError: If shapes, tile sizes, or mesh layouts are incompatible.
    """
    return _inverse_jit_ctx(
        a, T_A, mesh, matrix_specs, in_specs=in_specs, pad=pad,
        cholesky=False, _prepared=_prepared,
    )


def getri_shardmap_ctx(
    a: Array,
    T_A: int,
    matrix_specs: P | tuple[P] | list[P] | None = None,
    *,
    in_specs: P | tuple[P] | list[P] | None = None,
    pad: bool = True,
) -> tuple[Array, Array]:
    """Invert local matrix blocks inside a caller's ``jax.shard_map``.

    Uses the same fused pivoted LU factorization and inversion as
    :func:`jaxmg.getri`, with compilation and sharding owned by the caller.

    Args:
        a: Local block of the original general nonsingular matrix.
        T_A: Square tile dimension.
        matrix_specs: A's specification in the enclosing shard map; defaults
            to the context mesh axes in order. Must describe the actual blocks,
            since the global matrix shape is reconstructed from this metadata.
        in_specs: Backwards-compatible alias for ``matrix_specs``.
        pad: Allow tile-capacity padding.

    Returns:
        ``(inverse, status)``: the unpadded local inverse block and this rank's
        diagnostic vector. The inverse aliases the input work buffer, so no
        separate ``a_work`` result is required.

    Raises:
        TypeError: If the dtype or partition specification is unsupported.
        ValueError: If called outside shard_map or the global matrix is invalid.
    """
    return _inverse_shardmap_ctx(
        a, T_A, matrix_specs, in_specs=in_specs, pad=pad, cholesky=False,
    )


def _prepare_inverse_call(
    a, tile_size, mesh, matrix_specs, *, in_specs, pad, caller, local=False,
):
    """Validate inversion and derive its distributed matrix layout.

    1. Check matrix rank, supported dtype, square shape and positive tile size.
    2. Resolve the mesh and tile-aligned capacity; local calls reconstruct
       the global shape from their enclosing mesh and partition specification.
    """
    if a.ndim != 2:
        raise ValueError(f"{caller} expects a rank-2 matrix A.")
    if a.dtype not in (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128):
        raise TypeError(f"{caller} supports float32, float64, complex64, and complex128.")
    if not local and a.shape[0] != a.shape[1]:
        raise ValueError(f"{caller} expects A to be square.")
    if int(tile_size) <= 0:
        raise ValueError("T_A must be positive.")
    if local:
        layout = prepare_local_matrix_layout(
            a, tile_size, matrix_specs=matrix_specs, in_specs=in_specs,
            pad=pad, caller=caller,
        )
        if layout.padding.logical_rows != layout.padding.logical_cols:
            raise ValueError(f"{caller} expects A to be square.")
        return layout
    return prepare_input_matrix_layout(
        a, tile_size, mesh=mesh, matrix_specs=matrix_specs,
        in_specs=in_specs, pad=pad, caller=caller,
    )


def _inverse_jit_ctx(
    a, T_A, mesh=None, matrix_specs=None, *, in_specs=None, pad=True,
    cholesky, _prepared=None,
):
    """Place global inputs, invoke the local context, and restore result sharding."""
    layout = _prepared if _prepared is not None else _prepare_inverse_call(
        a, T_A, mesh, matrix_specs, in_specs=in_specs, pad=pad,
        caller="potri_jit_ctx" if cholesky else "getri_jit_ctx",
    )
    local_inverse = potri_shardmap_ctx if cholesky else getri_shardmap_ctx
    with jax.sharding.use_abstract_mesh(layout.mesh.abstract_mesh):
        a = place_for_native_work(a, mesh=layout.mesh, matrix_specs=layout.matrix_specs)
        inverse, status = jax.shard_map(
            partial(local_inverse, T_A=T_A, matrix_specs=layout.matrix_specs, pad=pad),
            mesh=layout.mesh, in_specs=layout.matrix_specs,
            out_specs=(layout.matrix_specs, layout.native_status_specs),
        )(a)
        inverse = restore_matrix_result(
            inverse, result_specs=layout.matrix_result_specs,
            mesh=layout.mesh, matrix_specs=layout.matrix_specs,
        )
    return inverse, status


def _inverse_shardmap_ctx(
    a, T_A, matrix_specs, *, in_specs, pad, cholesky,
):
    """Pad local capacity, run the fused inverse, and remove capacity padding."""
    layout = _prepare_inverse_call(
        a, T_A, None, matrix_specs, in_specs=in_specs, pad=pad,
        caller="potri_shardmap_ctx" if cholesky else "getri_shardmap_ctx",
        local=True,
    )
    ensure_init_jaxmg_backend()
    padding = layout.padding
    pr, pc = layout.grid.process_rows, layout.grid.process_cols

    # Stage 1: provide tile-aligned capacity without changing the logical order.
    a = pad_local_2d(a, row_padding=padding.row_padding_per_process,
                    col_padding=padding.col_padding_per_process)
    # Stage 2: factor, invert, and restore the complete inverse in native code.
    inverse, status = jax.ffi.ffi_call(
        "cusolvermp_inverse",
        (jax.ShapeDtypeStruct(a.shape, a.dtype),
         jax.ShapeDtypeStruct((_CUSOLVERMP_INVERSE_STATUS_SIZE,), jnp.int32)),
        input_layouts=((0, 1),), output_layouts=((0, 1), (0,)),
        input_output_aliases={0: 0},
    )(
        a, process_rows=pr, process_cols=pc,
        partition_slots=np.asarray(layout.partition_slots, dtype=np.int64),
        n=int(padding.logical_rows), tile_size=int(T_A), cholesky=cholesky,
    )
    # Stage 3: return the caller's local shape and declare rank-varying outputs.
    inverse = unpad_local_2d(inverse, local_rows=padding.local_logical_rows,
                            local_cols=padding.local_logical_cols)
    axes = tuple(axis for axis, size in zip(layout.matrix_specs, (pr, pc))
                 if axis is not None and size > 1)
    return mark_varying(inverse, axes), mark_varying(status, axes)


@lru_cache(maxsize=None)
def _inverse_compiled(layout, tile_size, *, cholesky, pad, donate):
    """Cache the public compilation boundary around the global context."""
    context = potri_jit_ctx if cholesky else getri_jit_ctx
    return jax.jit(
        partial(context, T_A=tile_size, pad=pad, _prepared=layout),
        donate_argnums=(0,) if donate else (),
    )
