"""Public cuSOLVERMp reduced-QR wrapper.

The Python layer validates a tall or square JAX array and describes the local
padding required by the native backend. The fused FFI call redistributes the
matrix, computes ``A = Q @ R``, and restores both factors to their original JAX
sharding.
"""

from __future__ import annotations

from functools import lru_cache, partial
from typing import List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.sharding import AbstractMesh, Mesh, PartitionSpec as P

from ._cusolvermp_layout import (
    make_local_pad_fn,
    make_local_unpad_fn,
    prepare_input_matrix_layout,
    prepare_rectangular_matrix_layout,
    use_abstract_mesh_decorator,
)
from ._cusolvermp_status import _CUSOLVERMP_QR_STATUS_SIZE
from ._layout_types import MatrixPadding2D, ProcessGrid
from ._setup import ensure_init_jaxmg_backend


def qr(
    a: Array,
    T_A: int,
    mesh: Mesh | AbstractMesh | None = None,
    matrix_specs: P | Tuple[P] | List[P] | None = None,
    *,
    in_specs: P | Tuple[P] | List[P] | None = None,
    return_status: bool = False,
    pad: bool = True,
    donate: bool = True,
) -> tuple[Array, ...]:
    """Compute a distributed reduced QR decomposition with cuSOLVERMp.

    For a tall or square ``m x n`` matrix with ``m >= n``, returns an ``m x n``
    matrix ``Q`` with orthonormal columns and an ``n x n`` upper-triangular
    matrix ``R`` such that ``A = Q @ R``.

    Args:
        a (Array): A rank-2 real or complex tall or square matrix sharded over
            a one- or two-axis device mesh.
        T_A (int): Square cuSOLVERMp tile width.
        mesh (Mesh, optional): JAX mesh used by ``jax.shard_map``. If omitted,
            inferred from ``a.sharding.mesh``.
        matrix_specs (PartitionSpec or tuple/list[PartitionSpec], optional):
            Rank-2 matrix sharding. If omitted, inferred from
            ``a.sharding.spec``.
        in_specs: Backwards-compatible alias for ``matrix_specs``.
        return_status (bool, optional): If True, return the native per-rank
            diagnostic status vector after ``Q`` and ``R``.
        pad (bool, optional): If True (default), add tile-aligned local capacity
            where required. If False, all participating local matrix shapes
            must already be divisible by ``T_A``.
        donate (bool, optional): If True (default), the input may be donated
            and overwritten by ``Q``. Pass False to preserve it, at the cost
            of an additional A-sized allocation.

    Returns:
        ``(Q, R)`` by default, or ``(Q, R, status)`` when
        ``return_status=True``.

    Raises:
        TypeError: If the input dtype or sharding specification is unsupported.
        ValueError: If the matrix is wide or its shape, tile size, process
            grid, or output layout is incompatible with cuSOLVERMp.
    """
    layout, r_padding = _prepare_qr_call(
        a,
        T_A,
        mesh,
        matrix_specs,
        in_specs=in_specs,
        pad=pad,
        caller="qr",
    )
    m, n = map(int, a.shape)
    ensure_init_jaxmg_backend()
    impl = _qr_compiled(
        layout.mesh,
        layout.matrix_specs,
        layout.native_status_specs,
        layout.grid,
        layout.partition_slots,
        layout.padding,
        r_padding,
        m=m,
        n=n,
        tile_size=int(T_A),
        dtype=a.dtype,
        donate=donate,
    )
    q, r, native_status = impl(a)
    return (q, r, native_status) if return_status else (q, r)


def qr_shardmap_ctx(
    a: Array,
    T_A: int,
    mesh: Mesh | AbstractMesh | None = None,
    matrix_specs: P | Tuple[P] | List[P] | None = None,
    *,
    in_specs: P | Tuple[P] | List[P] | None = None,
    pad: bool = True,
) -> tuple[Array, Array, Array]:
    """Compute reduced QR inside a caller-owned ``jax.jit``.

    This interface performs the same validation, redistribution, and native
    computation as :func:`jaxmg.qr`, but leaves the outer compilation and
    donation boundary to the caller. The first output is ``Q``, allowing an
    outer ``jax.jit(..., donate_argnums=(0,))`` to alias the input matrix.

    Args:
        a (Array): A rank-2 real or complex tall or square matrix sharded over
            a one- or two-axis device mesh.
        T_A (int): Square cuSOLVERMp tile width.
        mesh (Mesh, optional): JAX mesh used by ``jax.shard_map``. If omitted,
            inferred from ``a.sharding.mesh``.
        matrix_specs (PartitionSpec or tuple/list[PartitionSpec], optional):
            Rank-2 matrix sharding. If omitted, inferred from
            ``a.sharding.spec``.
        in_specs: Backwards-compatible alias for ``matrix_specs``.
        pad (bool, optional): If True (default), add tile-aligned local capacity
            where required. If False, all participating local matrix shapes
            must already be divisible by ``T_A``.

    Returns:
        ``(Q, R, status)``. The status vector is always returned so an
        enclosing compiled function can propagate native diagnostics.

    Raises:
        TypeError: If the input dtype or sharding specification is unsupported.
        ValueError: If the matrix is wide or its shape, tile size, process
            grid, or output layout is incompatible with cuSOLVERMp.
    """
    layout, r_padding = _prepare_qr_call(
        a,
        T_A,
        mesh,
        matrix_specs,
        in_specs=in_specs,
        pad=pad,
        caller="qr_shardmap_ctx",
    )
    m, n = map(int, a.shape)
    ensure_init_jaxmg_backend()
    return _qr_pipeline(
        layout.mesh,
        layout.matrix_specs,
        layout.native_status_specs,
        layout.grid,
        layout.partition_slots,
        layout.padding,
        r_padding,
        m=m,
        n=n,
        tile_size=int(T_A),
        dtype=a.dtype,
    )(a)


def _prepare_qr_call(
    a: Array,
    tile_size: int,
    mesh: Mesh | AbstractMesh | None,
    matrix_specs: P | Tuple[P] | List[P] | None,
    *,
    in_specs: P | Tuple[P] | List[P] | None,
    pad: bool,
    caller: str,
):
    """Validate a QR call and derive its distributed layouts.

    Preparation proceeds as follows:

    1. Validate the input rank, dtype, and tile size.
    2. Require a tall or square input for the reduced QR decomposition.
    3. Resolve the mesh, process grid, rank map, and tile-aligned layout of A.
    4. Validate and prepare the square R output layout.
    """
    if a.ndim != 2:
        raise ValueError(f"{caller} expects a rank-2 matrix A.")
    _check_supported_qr_dtype(a.dtype)
    if int(tile_size) <= 0:
        raise ValueError("T_A must be positive.")
    m, n = map(int, a.shape)
    if m < n:
        raise ValueError(f"{caller} requires a tall or square matrix with m >= n.")

    layout = prepare_input_matrix_layout(
        a,
        tile_size,
        mesh=mesh,
        matrix_specs=matrix_specs,
        in_specs=in_specs,
        pad=pad,
        caller=caller,
    )
    r_padding = prepare_rectangular_matrix_layout(
        n,
        n,
        layout.grid,
        layout.tile_shape,
        pad=pad,
        caller=f"{caller}(R)",
    )
    return layout, r_padding


def _check_supported_qr_dtype(dtype) -> None:
    """Validate that ``dtype`` maps to cuSOLVERMp GEQRF and ORGQR."""
    if dtype not in (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128):
        raise TypeError("qr supports float32, float64, complex64, and complex128.")


_ROW_MAJOR_JAX_LAYOUT = (0, 1)


@lru_cache(maxsize=None)
def _qr_pipeline(
    mesh: Mesh | AbstractMesh,
    matrix_specs: P,
    native_status_specs: P,
    grid: ProcessGrid,
    partition_slots: tuple[int, ...],
    a_padding: MatrixPadding2D,
    r_padding: MatrixPadding2D,
    *,
    m: int,
    n: int,
    tile_size: int,
    dtype,
):
    """Build and cache the unjitted JAX-visible QR execution pipeline."""
    process_rows = grid.process_rows
    process_cols = grid.process_cols
    slots_attr = np.asarray(partition_slots, dtype=np.int64)
    pad_a = make_local_pad_fn(mesh, matrix_specs, a_padding)
    unpad_q = make_local_unpad_fn(mesh, matrix_specs, a_padding)
    unpad_r = make_local_unpad_fn(mesh, matrix_specs, r_padding)

    def qr_ffi(_a: Array):
        """Declare local FFI buffers and static reduced-QR metadata."""
        if _a.ndim != 2:
            raise ValueError("cusolvermp_qr expects a rank-2 matrix buffer.")
        _check_supported_qr_dtype(_a.dtype)
        ffi_fn = partial(
            jax.ffi.ffi_call(
                "cusolvermp_qr",
                (
                    jax.ShapeDtypeStruct(_a.shape, _a.dtype),
                    jax.ShapeDtypeStruct(
                        (r_padding.local_physical_rows, r_padding.local_physical_cols),
                        _a.dtype,
                    ),
                    jax.ShapeDtypeStruct((_CUSOLVERMP_QR_STATUS_SIZE,), jnp.int32),
                ),
                input_layouts=(_ROW_MAJOR_JAX_LAYOUT,),
                output_layouts=(_ROW_MAJOR_JAX_LAYOUT, _ROW_MAJOR_JAX_LAYOUT, (0,)),
                input_output_aliases={0: 0},
            ),
            process_rows=process_rows,
            process_cols=process_cols,
            partition_slots=slots_attr,
            m=int(m),
            n=int(n),
            tile_size=int(tile_size),
        )
        return ffi_fn(_a)

    qr_shardmap = jax.shard_map(
        qr_ffi,
        mesh=mesh,
        in_specs=matrix_specs,
        out_specs=(matrix_specs, matrix_specs, native_status_specs),
        check_vma=False,
    )

    @use_abstract_mesh_decorator(mesh)
    def impl(_a: Array):
        """Apply local padding, fused reduced QR, and output slicing."""
        q_padded, r_padded, native_status = qr_shardmap(pad_a(_a))
        return unpad_q(q_padded), unpad_r(r_padded), native_status

    return impl


@lru_cache(maxsize=None)
def _qr_compiled(
    mesh: Mesh | AbstractMesh,
    matrix_specs: P,
    native_status_specs: P,
    grid: ProcessGrid,
    partition_slots: tuple[int, ...],
    a_padding: MatrixPadding2D,
    r_padding: MatrixPadding2D,
    *,
    m: int,
    n: int,
    tile_size: int,
    dtype,
    donate: bool,
):
    """Build and cache the internally jitted public QR pipeline."""
    pipeline = _qr_pipeline(
        mesh,
        matrix_specs,
        native_status_specs,
        grid,
        partition_slots,
        a_padding,
        r_padding,
        m=m,
        n=n,
        tile_size=tile_size,
        dtype=dtype,
    )

    @partial(jax.jit, donate_argnums=(0,) if donate else ())
    def impl(_a: Array):
        return pipeline(_a)

    return impl
