"""Public cuSOLVERMp Cholesky-factorization wrapper.

The Python layer validates a square JAX array and describes the local padding
required by the native backend. The fused FFI call redistributes the matrix,
computes the lower Cholesky factor with ``cusolverMpPotrf``, zeros its strict
upper triangle, and restores the factor to the original JAX sharding.
"""

from __future__ import annotations

from functools import lru_cache, partial
from typing import List, Tuple, Union

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
    use_abstract_mesh_decorator,
)
from ._cusolvermp_status import _CUSOLVERMP_POTRF_STATUS_SIZE
from ._layout_types import MatrixPadding2D, ProcessGrid
from ._setup import ensure_init_jaxmg_backend


def potrf(
    a: Array,
    T_A: int,
    mesh: Mesh | AbstractMesh | None = None,
    matrix_specs: P | Tuple[P] | List[P] | None = None,
    *,
    in_specs: P | Tuple[P] | List[P] | None = None,
    return_status: bool = False,
    pad: bool = True,
    donate: bool = True,
) -> Union[Array, Tuple[Array, Array]]:
    """Compute the distributed Cholesky factor of a positive-definite matrix.

    For a symmetric or Hermitian positive-definite ``n x n`` matrix ``A``,
    returns the lower-triangular matrix ``L`` such that ``A = L @ L^H``. The
    strict upper triangle of ``L`` is zero, as in
    :func:`jax.numpy.linalg.cholesky`.

    Note:
        If a local shard dimension is not divisible by ``T_A``, ``pad=True``
        allocates additional tile-aligned capacity before the native call.
        Choosing a tile size that divides the local dimensions avoids this
        allocation. Performance depends on the matrix size, process grid, and
        tile size.

    Args:
        a (Array): 2D, symmetric positive-definite input matrix sharded over a
            one- or two-axis device mesh, for example with
            ``P(<row_axis>)`` or ``P(<row_axis>, <col_axis>)``.
        T_A (int): Square tile width used by cuSOLVERMp. Each local shard
            dimension must be a multiple of ``T_A`` after padding.
        mesh (Mesh or AbstractMesh, optional): JAX mesh used for ``jax.shard_map``.
            If omitted, read off the sharding of ``a`` (its type inside
            ``jax.jit``), or taken from the context mesh.
        matrix_specs (PartitionSpec or tuple/list[PartitionSpec], optional):
            PartitionSpec describing the matrix sharding. If omitted, read
            off the sharding of ``a``, defaulting to the mesh axes in order.
        in_specs: Backwards-compatible alias for ``matrix_specs``.
        return_status (bool, optional): If True return ``(L, status)`` where
            ``status`` is the native per-rank diagnostic vector. If False
            return ``L`` only. Default is False.
        pad (bool, optional): If True (default) apply per-device padding so
            each local shard length is compatible with ``T_A``; if False the
            caller must ensure shapes already match the kernel's requirements.
        donate (bool, optional): If True (default) the input buffer may be
            donated and overwritten by ``L``, which means it is deleted and
            cannot be used again. Pass False to preserve it, at the cost of an
            additional A-sized allocation.

    Returns:
        Array or (Array, Array): The Cholesky factor ``L``, restored to
            replicated sharding when ``a`` is replicated and otherwise sharded
            like ``a``. If ``return_status=True`` also return the native solver
            status.

    Raises:
        TypeError: If dtypes or ``PartitionSpec`` inputs are unsupported.
        ValueError: If shapes, tile sizes, or mesh layouts are incompatible.

    Notes:
        - Unless ``donate=False``, the ``a`` buffer is donated for zero-copy
          interaction with the native library, so it is deleted and cannot be
          used after the call.
        - Native code converts row-major JAX local storage to cuSOLVERMp's
          column-major local layout, redistributes to 2D block-cyclic layout,
          calls ``cusolverMpPotrf``, and redistributes the factor back.
        - If the native solver fails, for example because ``a`` is not
          positive definite, ``L`` is filled with NaN and ``status`` is
          non-zero.
    """
    layout = _prepare_potrf_call(
        a,
        T_A,
        mesh,
        matrix_specs,
        in_specs=in_specs,
        pad=pad,
        caller="potrf",
    )
    ensure_init_jaxmg_backend()
    impl = _potrf_compiled(
        layout.mesh,
        layout.matrix_specs,
        layout.native_status_specs,
        layout.grid,
        layout.partition_slots,
        layout.padding,
        matrix_result_specs=layout.matrix_result_specs,
        n=int(a.shape[0]),
        tile_size=int(T_A),
        dtype=a.dtype,
        donate=donate,
    )
    factor, native_status = impl(a)
    return (factor, native_status) if return_status else factor


def potrf_jit_ctx(
    a: Array,
    T_A: int,
    mesh: Mesh | AbstractMesh | None = None,
    matrix_specs: P | Tuple[P] | List[P] | None = None,
    *,
    in_specs: P | Tuple[P] | List[P] | None = None,
    pad: bool = True,
) -> Tuple[Array, Array]:
    """Compute the Cholesky factor inside a caller-owned ``jax.jit``.

    This interface performs the same validation, redistribution, and native
    computation as :func:`jaxmg.potrf`, but leaves the outer compilation and
    donation boundary to the caller. The native factor output aliases the input
    work buffer, so an outer ``jax.jit(..., donate_argnums=(0,))`` can donate
    ``a`` into ``L``.

    Args:
        a (Array): 2D, symmetric positive-definite input matrix sharded over a
            one- or two-axis device mesh, for example with
            ``P(<row_axis>)`` or ``P(<row_axis>, <col_axis>)``.
        T_A (int): Square tile width used by cuSOLVERMp.
        mesh (Mesh or AbstractMesh, optional): JAX mesh used for ``jax.shard_map``.
            If omitted, read off the sharding of ``a`` (its type inside
            ``jax.jit``), or taken from the context mesh.
        matrix_specs (PartitionSpec or tuple/list[PartitionSpec], optional):
            PartitionSpec describing the matrix sharding. If omitted, read
            off the sharding of ``a``, defaulting to the mesh axes in order.
        in_specs: Backwards-compatible alias for ``matrix_specs``.
        pad (bool, optional): If True (default) apply per-device padding so
            each local shard length is compatible with ``T_A``; if False the
            caller must ensure shapes already match the kernel's requirements.

    Returns:
        ``(L, status)``. The status vector is always returned so an enclosing
        compiled function can propagate native diagnostics. ``L`` is restored
        when ``a`` is replicated; otherwise it uses ``matrix_specs``.

    Raises:
        TypeError: If dtypes or ``PartitionSpec`` inputs are unsupported.
        ValueError: If shapes, tile sizes, or mesh layouts are incompatible.

    Notes:
        - If the native solver fails, ``L`` is filled with NaN and the status
          is non-zero.
    """
    layout = _prepare_potrf_call(
        a,
        T_A,
        mesh,
        matrix_specs,
        in_specs=in_specs,
        pad=pad,
        caller="potrf_jit_ctx",
    )
    ensure_init_jaxmg_backend()
    return _potrf_pipeline(
        layout.mesh,
        layout.matrix_specs,
        layout.native_status_specs,
        layout.grid,
        layout.partition_slots,
        layout.padding,
        matrix_result_specs=layout.matrix_result_specs,
        n=int(a.shape[0]),
        tile_size=int(T_A),
        dtype=a.dtype,
    )(a)


def potrf_shardmap_ctx(
    a: Array,
    T_A: int,
    matrix_specs: P | Tuple[P] | List[P] | None = None,
    *,
    in_specs: P | Tuple[P] | List[P] | None = None,
    pad: bool = True,
) -> Tuple[Array, Array]:
    """Compute local Cholesky-factor blocks inside a caller's ``jax.shard_map``.

    The enclosing mesh contains the matrix axes. The same native factorization
    is used by :func:`potrf` and :func:`potrf_jit_ctx`; only the caller's
    compilation and sharding context differs.

    Args:
        a: Local block of a symmetric or Hermitian positive-definite matrix.
        T_A: Square tile width used by cuSOLVERMp.
        matrix_specs: A's partition specification in the enclosing shard map;
            defaults to the context mesh axes in order.
        in_specs: Backwards-compatible alias for ``matrix_specs``.
        pad: Allow tile-capacity padding; otherwise require aligned blocks.

    Returns:
        ``(L, status)``: the unpadded local factor block and this rank's
        diagnostic vector. The factor reuses the input work buffer, so no
        separate ``a_work`` output is needed for donation.

    Raises:
        TypeError: If the dtype or partition specification is unsupported.
        ValueError: If called outside shard_map or the global matrix is invalid.
    """
    layout = _prepare_potrf_call(
        a, T_A, None, matrix_specs, in_specs=in_specs, pad=pad,
        caller="potrf_shardmap_ctx", local=True,
    )
    ensure_init_jaxmg_backend()
    return _potrf_local_pipeline(
        layout.matrix_specs, layout.grid, layout.partition_slots, layout.padding,
        n=layout.padding.logical_rows, tile_size=int(T_A),
    )(a)


def _prepare_potrf_call(
    a: Array,
    tile_size: int,
    mesh: Mesh | AbstractMesh | None,
    matrix_specs: P | Tuple[P] | List[P] | None,
    *,
    in_specs: P | Tuple[P] | List[P] | None,
    pad: bool,
    caller: str,
    local: bool = False,
):
    """Validate a Cholesky factorization and derive its distributed layout.

    Preparation proceeds as follows:

    1. Validate the input rank, dtype, square shape, and tile size.
    2. Resolve the mesh, process grid, rank map, and tile-aligned layout of A;
       local blocks use the enclosing mesh to reconstruct the global shape.
    """
    if a.ndim != 2:
        raise ValueError(f"{caller} expects a rank-2 matrix A.")
    _check_supported_potrf_dtype(a.dtype)
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
        a,
        tile_size,
        mesh=mesh,
        matrix_specs=matrix_specs,
        in_specs=in_specs,
        pad=pad,
        caller=caller,
    )


def _check_supported_potrf_dtype(dtype) -> None:
    """Validate that ``dtype`` maps to a cuSOLVERMp POTRF entry point."""
    if dtype not in (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128):
        raise TypeError("potrf supports float32, float64, complex64, and complex128.")


_ROW_MAJOR_JAX_LAYOUT = (0, 1)


def _potrf_local_pipeline(
    matrix_specs: P,
    grid: ProcessGrid,
    partition_slots: tuple[int, ...],
    a_padding: MatrixPadding2D,
    *,
    n: int,
    tile_size: int,
):
    """Build the local padding, factorization, and factor-restoration stages."""
    process_rows = grid.process_rows
    process_cols = grid.process_cols
    slots_attr = np.asarray(partition_slots, dtype=np.int64)

    def potrf_ffi(_a: Array):
        """Declare local FFI buffers and static Cholesky metadata."""
        # Stage 1: provide tile-aligned capacity for the local input.
        _a = pad_local_2d(_a, row_padding=a_padding.row_padding_per_process,
                          col_padding=a_padding.col_padding_per_process)
        if _a.ndim != 2:
            raise ValueError("cusolvermp_potrf expects a rank-2 matrix buffer.")
        _check_supported_potrf_dtype(_a.dtype)
        # Stage 2: factorize and restore the local memory layout in native code.
        ffi_fn = partial(
            jax.ffi.ffi_call(
                "cusolvermp_potrf",
                (
                    jax.ShapeDtypeStruct(_a.shape, _a.dtype),
                    jax.ShapeDtypeStruct((_CUSOLVERMP_POTRF_STATUS_SIZE,), jnp.int32),
                ),
                input_layouts=(_ROW_MAJOR_JAX_LAYOUT,),
                output_layouts=(_ROW_MAJOR_JAX_LAYOUT, (0,)),
                input_output_aliases={0: 0},
            ),
            process_rows=process_rows,
            process_cols=process_cols,
            partition_slots=slots_attr,
            n=int(n),
            tile_size=int(tile_size),
        )
        factor, status = ffi_fn(_a)
        # Stage 3: remove tile padding and retain rank-varying outputs.
        factor = unpad_local_2d(factor, local_rows=a_padding.local_logical_rows,
                               local_cols=a_padding.local_logical_cols)
        axes = tuple(axis for axis, size in zip(matrix_specs, (process_rows, process_cols))
                     if axis is not None and size > 1)
        return mark_varying(factor, axes), mark_varying(status, axes)

    return potrf_ffi


@lru_cache(maxsize=None)
def _potrf_pipeline(
    mesh: Mesh | AbstractMesh,
    matrix_specs: P,
    native_status_specs: P,
    grid: ProcessGrid,
    partition_slots: tuple[int, ...],
    a_padding: MatrixPadding2D,
    *,
    matrix_result_specs: P,
    n: int,
    tile_size: int,
    dtype,
):
    """Map the shared local factorization and restore global output sharding."""

    potrf_shardmap = jax.shard_map(
        _potrf_local_pipeline(matrix_specs, grid, partition_slots, a_padding,
                              n=n, tile_size=tile_size),
        mesh=mesh,
        in_specs=matrix_specs,
        out_specs=(matrix_specs, native_status_specs),
        check_vma=True,
    )

    @use_abstract_mesh_decorator(mesh)
    def impl(_a: Array):
        """Apply local padding, fused Cholesky factorization, and unpadding."""
        _a = place_for_native_work(_a, mesh=mesh, matrix_specs=matrix_specs)
        factor, native_status = potrf_shardmap(_a)
        return restore_matrix_result(factor, result_specs=matrix_result_specs,
                                     mesh=mesh, matrix_specs=matrix_specs), native_status

    return impl


@lru_cache(maxsize=None)
def _potrf_compiled(
    mesh: Mesh | AbstractMesh,
    matrix_specs: P,
    native_status_specs: P,
    grid: ProcessGrid,
    partition_slots: tuple[int, ...],
    a_padding: MatrixPadding2D,
    *,
    matrix_result_specs: P,
    n: int,
    tile_size: int,
    dtype,
    donate: bool,
):
    """Build and cache the internally jitted public POTRF pipeline."""
    pipeline = _potrf_pipeline(
        mesh,
        matrix_specs,
        native_status_specs,
        grid,
        partition_slots,
        a_padding,
        matrix_result_specs=matrix_result_specs,
        n=n,
        tile_size=tile_size,
        dtype=dtype,
    )

    @partial(jax.jit, donate_argnums=(0,) if donate else ())
    def impl(_a: Array):
        return pipeline(_a)

    return impl
