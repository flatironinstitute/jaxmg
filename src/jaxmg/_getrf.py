"""Public cuSOLVERMp LU-factorization wrapper.

The Python layer validates a square JAX array and describes the local padding
required by the native backend. The fused FFI call redistributes the matrix,
computes the pivoted LU factorization with ``cusolverMpGetrf``, gathers the
distributed pivots into one replicated vector, and restores the packed factor
to the original JAX sharding.
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
    make_local_pad_fn,
    make_local_unpad_fn,
    prepare_input_matrix_layout,
    restore_matrix_result,
    use_abstract_mesh_decorator,
)
from ._cusolvermp_status import _CUSOLVERMP_GETRF_STATUS_SIZE
from ._layout_types import MatrixPadding2D, ProcessGrid
from ._setup import ensure_init_jaxmg_backend


def getrf(
    a: Array,
    T_A: int,
    mesh: Mesh | AbstractMesh | None = None,
    matrix_specs: P | Tuple[P] | List[P] | None = None,
    *,
    in_specs: P | Tuple[P] | List[P] | None = None,
    return_status: bool = False,
    pad: bool = True,
    donate: bool = True,
) -> Union[Tuple[Array, Array], Tuple[Array, Array, Array]]:
    """Compute the distributed pivoted LU factorization of a square matrix.

    For a nonsingular ``n x n`` matrix ``A``, computes ``P @ A = L @ U`` with
    partial pivoting. The outputs follow :func:`jax.scipy.linalg.lu_factor`:
    ``lu`` holds the strictly lower part of the unit-lower ``L`` and the upper
    triangle of ``U``, and ``piv`` holds 0-based pivot indices, meaning row
    ``i`` was interchanged with row ``piv[i]``. Use
    :func:`jax.lax.linalg.lu_pivots_to_permutation` to obtain the permutation.

    Note:
        If a local shard dimension is not divisible by ``T_A``, ``pad=True``
        allocates additional tile-aligned capacity before the native call.
        Choosing a tile size that divides the local dimensions avoids this
        allocation. Performance depends on the matrix size, process grid, and
        tile size.

    Args:
        a (Array): 2D, nonsingular input matrix sharded over a one- or two-axis
            device mesh, for example with ``P(<row_axis>)`` or
            ``P(<row_axis>, <col_axis>)``.
        T_A (int): Square tile width used by cuSOLVERMp. Each local shard
            dimension must be a multiple of ``T_A`` after padding.
        mesh (Mesh or AbstractMesh, optional): JAX mesh used for ``jax.shard_map``.
            If omitted, read off the sharding of ``a`` (its type inside
            ``jax.jit``), or taken from the context mesh.
        matrix_specs (PartitionSpec or tuple/list[PartitionSpec], optional):
            PartitionSpec describing the matrix sharding. If omitted, read
            off the sharding of ``a``, defaulting to the mesh axes in order.
        in_specs: Backwards-compatible alias for ``matrix_specs``.
        return_status (bool, optional): If True return ``(lu, piv, status)``
            where ``status`` is the native per-rank diagnostic vector. If False
            return ``(lu, piv)`` only. Default is False.
        pad (bool, optional): If True (default) apply per-device padding so
            each local shard length is compatible with ``T_A``; if False the
            caller must ensure shapes already match the kernel's requirements.
        donate (bool, optional): If True (default) the input buffer may be
            donated and overwritten by ``lu``, which means it is deleted and
            cannot be used again. Pass False to preserve it, at the cost of an
            additional A-sized allocation.

    Returns:
        ``(lu, piv)`` by default, or ``(lu, piv, status)`` when
        ``return_status=True``. ``lu`` is restored to replicated sharding when
        ``a`` is replicated and otherwise sharded like ``a``. ``piv`` is a
        replicated ``int32`` vector of length ``n``.

    Raises:
        TypeError: If dtypes or ``PartitionSpec`` inputs are unsupported.
        ValueError: If shapes, tile sizes, or mesh layouts are incompatible.

    Notes:
        - Unless ``donate=False``, the ``a`` buffer is donated for zero-copy
          interaction with the native library, so it is deleted and cannot be
          used after the call.
        - Native code converts row-major JAX local storage to cuSOLVERMp's
          column-major local layout, redistributes to 2D block-cyclic layout,
          calls ``cusolverMpGetrf``, and redistributes the factor back.
        - If the native solver fails, including when ``U`` has an exactly zero
          diagonal entry, ``lu`` is filled with NaN, ``piv`` is filled with
          ``-1``, and ``status`` is non-zero.
    """
    layout = _prepare_getrf_call(
        a,
        T_A,
        mesh,
        matrix_specs,
        in_specs=in_specs,
        pad=pad,
        caller="getrf",
    )
    ensure_init_jaxmg_backend()
    impl = _getrf_compiled(
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
    lu, piv, native_status = impl(a)
    return (lu, piv, native_status) if return_status else (lu, piv)


def getrf_shardmap_ctx(
    a: Array,
    T_A: int,
    mesh: Mesh | AbstractMesh | None = None,
    matrix_specs: P | Tuple[P] | List[P] | None = None,
    *,
    in_specs: P | Tuple[P] | List[P] | None = None,
    pad: bool = True,
) -> Tuple[Array, Array, Array]:
    """Compute the pivoted LU factorization inside a caller-owned ``jax.jit``.

    This interface performs the same validation, redistribution, and native
    computation as :func:`jaxmg.getrf`, but leaves the outer compilation and
    donation boundary to the caller. The native ``lu`` output aliases the input
    work buffer, so an outer ``jax.jit(..., donate_argnums=(0,))`` can donate
    ``a`` into ``lu``.

    Args:
        a (Array): 2D, nonsingular input matrix sharded over a one- or two-axis
            device mesh, for example with ``P(<row_axis>)`` or
            ``P(<row_axis>, <col_axis>)``.
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
        ``(lu, piv, status)``. The status vector is always returned so an
        enclosing compiled function can propagate native diagnostics. ``lu``
        is restored when ``a`` is replicated; otherwise it uses
        ``matrix_specs``. ``piv`` is a replicated ``int32`` vector.

    Raises:
        TypeError: If dtypes or ``PartitionSpec`` inputs are unsupported.
        ValueError: If shapes, tile sizes, or mesh layouts are incompatible.

    Notes:
        - If the native solver fails, ``lu`` is filled with NaN, ``piv`` is
          filled with ``-1``, and the status is non-zero.
    """
    layout = _prepare_getrf_call(
        a,
        T_A,
        mesh,
        matrix_specs,
        in_specs=in_specs,
        pad=pad,
        caller="getrf_shardmap_ctx",
    )
    ensure_init_jaxmg_backend()
    return _getrf_pipeline(
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


def _prepare_getrf_call(
    a: Array,
    tile_size: int,
    mesh: Mesh | AbstractMesh | None,
    matrix_specs: P | Tuple[P] | List[P] | None,
    *,
    in_specs: P | Tuple[P] | List[P] | None,
    pad: bool,
    caller: str,
):
    """Validate an LU factorization and derive its distributed layout.

    Preparation proceeds as follows:

    1. Validate the input rank, dtype, square shape, and tile size.
    2. Resolve the mesh, process grid, rank map, and tile-aligned layout of A.
    """
    if a.ndim != 2:
        raise ValueError(f"{caller} expects a rank-2 matrix A.")
    _check_supported_getrf_dtype(a.dtype)
    if a.shape[0] != a.shape[1]:
        raise ValueError(f"{caller} expects A to be square.")
    if int(tile_size) <= 0:
        raise ValueError("T_A must be positive.")

    return prepare_input_matrix_layout(
        a,
        tile_size,
        mesh=mesh,
        matrix_specs=matrix_specs,
        in_specs=in_specs,
        pad=pad,
        caller=caller,
    )


def _check_supported_getrf_dtype(dtype) -> None:
    """Validate that ``dtype`` maps to a cuSOLVERMp GETRF entry point."""
    if dtype not in (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128):
        raise TypeError("getrf supports float32, float64, complex64, and complex128.")


_ROW_MAJOR_JAX_LAYOUT = (0, 1)


@lru_cache(maxsize=None)
def _getrf_pipeline(
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
    """Build and cache the unjitted JAX-visible GETRF execution pipeline."""
    process_rows = grid.process_rows
    process_cols = grid.process_cols
    slots_attr = np.asarray(partition_slots, dtype=np.int64)
    pad_a = make_local_pad_fn(mesh, matrix_specs, a_padding)
    unpad_lu = make_local_unpad_fn(mesh, matrix_specs, a_padding)
    restore_result = partial(
        restore_matrix_result,
        result_specs=matrix_result_specs,
        mesh=mesh,
        matrix_specs=matrix_specs,
    )

    def getrf_ffi(_a: Array):
        """Declare local FFI buffers and static LU metadata."""
        if _a.ndim != 2:
            raise ValueError("cusolvermp_getrf expects a rank-2 matrix buffer.")
        _check_supported_getrf_dtype(_a.dtype)
        ffi_fn = partial(
            jax.ffi.ffi_call(
                "cusolvermp_getrf",
                (
                    jax.ShapeDtypeStruct(_a.shape, _a.dtype),
                    jax.ShapeDtypeStruct((int(n),), jnp.int32),
                    jax.ShapeDtypeStruct((_CUSOLVERMP_GETRF_STATUS_SIZE,), jnp.int32),
                ),
                input_layouts=(_ROW_MAJOR_JAX_LAYOUT,),
                output_layouts=(_ROW_MAJOR_JAX_LAYOUT, (0,), (0,)),
                input_output_aliases={0: 0},
            ),
            process_rows=process_rows,
            process_cols=process_cols,
            partition_slots=slots_attr,
            n=int(n),
            tile_size=int(tile_size),
        )
        return ffi_fn(_a)

    # The native call replicates the pivots across every rank before it
    # returns, so the pivot vector leaves the shard map unsharded.
    getrf_shardmap = jax.shard_map(
        getrf_ffi,
        mesh=mesh,
        in_specs=matrix_specs,
        out_specs=(matrix_specs, P(), native_status_specs),
        check_vma=False,
    )

    @use_abstract_mesh_decorator(mesh)
    def impl(_a: Array):
        """Apply local padding, fused LU factorization, and unpadding."""
        lu_padded, piv, native_status = getrf_shardmap(pad_a(_a))
        return restore_result(unpad_lu(lu_padded)), piv, native_status

    return impl


@lru_cache(maxsize=None)
def _getrf_compiled(
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
    """Build and cache the internally jitted public GETRF pipeline."""
    pipeline = _getrf_pipeline(
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
