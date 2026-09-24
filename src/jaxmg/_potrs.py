"""Public cuSOLVERMp Cholesky solve wrapper.

The Python layer validates JAX array metadata, applies per-shard tile padding,
and constructs the compiled FFI call.  The fused native C++/CUDA handler then
performs the row-major to column-major local layout conversion, 2D
redistribution, ``cusolverMpPotrf``/``cusolverMpPotrs`` calls, reverse
redistribution, and final layout restoration.
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
    infer_rhs_specs,
    make_local_pad_fn,
    make_local_unpad_fn,
    prepare_input_matrix_layout,
    prepare_matrix_padding,
    use_abstract_mesh_decorator,
    place_rhs_for_native_work,
    restore_rhs_from_native_work,
    rhs_distribution_columns,
)
from ._cusolvermp_status import _CUSOLVERMP_POTRS_STATUS_SIZE
from ._layout_types import MatrixPadding2D, ProcessGrid
from ._setup import ensure_init_jaxmg_backend


def potrs(
    a: Array,
    b: Array,
    T_A: int,
    mesh: Mesh | AbstractMesh | None = None,
    matrix_specs: P | Tuple[P] | List[P] | None = None,
    *,
    in_specs: P | Tuple[P] | List[P] | None = None,
    return_status: bool = False,
    return_logdet: bool = False,
    pad: bool = True,
    donate: bool = True,
) -> Union[Array, Tuple[Array, Array], Tuple[Array, Array, Array]]:
    """Solve the linear system A x = B using the multi-GPU potrs native kernel.

    This is the high-level JAXMg Cholesky-solve entry point.  It prepares a
    block-sharded JAX array for cuSOLVERMp, calls the fused native backend, and
    returns the solution in the same JAX-facing layout as ``b``.

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
        b (Array): 1D or 2D solve input. A vector is treated as an
            ``N x 1`` matrix.
        T_A (int): Square tile width used by cuSOLVERMp. Each local shard
            dimension must be a multiple of ``T_A`` after padding.
        mesh (Mesh or AbstractMesh, optional): JAX mesh used for ``jax.shard_map``.
            If omitted, read off the sharding of ``a`` (its type inside
            ``jax.jit``), or taken from the context mesh.
        matrix_specs (PartitionSpec or tuple/list[PartitionSpec], optional):
            PartitionSpec describing the matrix sharding. If omitted, read
            off the sharding of ``a``, defaulting to the mesh axes in order.
        in_specs: Backwards-compatible alias for ``matrix_specs``.
        return_status (bool, optional): If True return ``(x, status)`` where
            ``status`` is the native per-rank diagnostic vector. If False
            return ``x`` only. Default is False.
        return_logdet (bool, optional): If True also return ``log(det(A))``
            computed from the distributed Cholesky factor. Default is False.
        pad (bool, optional): If True (default) apply per-device padding so
            each local shard length is compatible with ``T_A``; if False the
            caller must ensure shapes already match the kernel's requirements.
        donate (bool, optional): If True (default) the input buffers may be
            donated to the native call for zero-copy execution, which means
            they are deleted and cannot be used again. Pass False to preserve
            them, at the cost of keeping the original and working buffers in
            memory simultaneously.

    Returns:
        One of ``x``, ``(x, status)``, ``(x, logdet)``, or
        ``(x, logdet, status)``. The solution retains the JAX-facing layout of
        ``b`` and ``logdet`` is a replicated real scalar.

    Raises:
        TypeError: If dtypes or ``PartitionSpec`` inputs are unsupported.
        ValueError: If shapes, tile sizes, or mesh layouts are incompatible.

    Notes:
        - Unless ``donate=False``, the ``a`` and ``b`` buffers are donated for
          zero-copy interaction with the native library, so they are deleted and
          cannot be used after the call.
        - Native code converts row-major JAX local storage to cuSOLVERMp's
          column-major local layout, redistributes to 2D block-cyclic layout,
          calls ``cusolverMpPotrf``/``cusolverMpPotrs``, and redistributes the
          result back.
        - If the native solver fails the returned solution may contain NaNs
          and ``status`` will be non-zero.
    """
    b, vector_rhs, layout, rhs_specs, b_padding, b_distribution_cols = (
        _prepare_potrs_call(
            a,
            b,
            T_A,
            mesh,
            matrix_specs,
            in_specs=in_specs,
            pad=pad,
            caller="potrs",
        )
    )
    nrhs = int(b.shape[1])

    ensure_init_jaxmg_backend()

    impl = _potrs_compiled(
        layout.mesh,
        layout.matrix_specs,
        layout.native_status_specs,
        layout.grid,
        layout.partition_slots,
        layout.padding,
        b_padding,
        rhs_specs,
        n=a.shape[0],
        nrhs=nrhs,
        b_distribution_cols=b_distribution_cols,
        tile_size=layout.tile_shape.rows,
        return_logdet=return_logdet,
        donate=donate,
    )
    result = impl(a, b)
    if return_logdet:
        _, out, logdet, native_status = result
    else:
        _, out, native_status = result
    if vector_rhs:
        out = out[:, 0]
    if return_logdet and return_status:
        return out, logdet[0], native_status
    if return_logdet:
        return out, logdet[0]
    if return_status:
        return out, native_status
    return out


def potrs_shardmap_ctx(
    a: Array,
    b: Array,
    T_A: int,
    mesh: Mesh | AbstractMesh | None = None,
    matrix_specs: P | Tuple[P] | List[P] | None = None,
    *,
    in_specs: P | Tuple[P] | List[P] | None = None,
    return_logdet: bool = False,
    pad: bool = True,
) -> Union[Tuple[Array, Array, Array], Tuple[Array, Array, Array, Array]]:
    """Solve A x = B while exposing the donated matrix work buffer.

    This helper is the lower-level variant of :func:`jaxmg.potrs` intended for
    contexts where the caller wants to control the outer ``jax.jit`` boundary.
    It performs the same validation, local padding, shard-map construction, and
    fused cuSOLVERMp FFI call as the public solver, but it does not wrap the
    pipeline in an internal ``jax.jit``.  Instead, it returns the native matrix
    work buffer alongside the solution so an outer JIT can donate ``a`` into an
    ``A``-sized output.

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
        b (Array): 1D or 2D solve input. A vector is treated as an
            ``N x 1`` matrix.
        T_A (int): Square tile width used by cuSOLVERMp. Each local shard
            dimension must be a multiple of ``T_A`` after padding.
        mesh (Mesh or AbstractMesh, optional): JAX mesh used for ``jax.shard_map``.
            If omitted, read off the sharding of ``a`` (its type inside
            ``jax.jit``), or taken from the context mesh.
        matrix_specs (PartitionSpec or tuple/list[PartitionSpec], optional):
            PartitionSpec describing the matrix sharding. If omitted, read
            off the sharding of ``a``, defaulting to the mesh axes in order.
        in_specs: Backwards-compatible alias for ``matrix_specs``.
        return_logdet (bool, optional): If True return the replicated Cholesky
            log determinant between the solution and status outputs. Default
            is False.
        pad (bool, optional): If True (default) apply per-device padding so
            each local shard length is compatible with ``T_A``; if False the
            caller must ensure shapes already match the kernel's requirements.

    Returns:
        tuple: ``(a_work, x, status)`` or
        ``(a_work, x, logdet, status)``. ``a_work`` is the padded matrix work
        buffer required for donation, ``x`` retains the JAX-facing layout of
        ``b``, and ``logdet`` is a replicated real scalar.

    Raises:
        TypeError: If dtypes or ``PartitionSpec`` inputs are unsupported.
        ValueError: If shapes, tile sizes, or mesh layouts are incompatible.

    Notes:
        - This function intentionally returns ``a_work``.  Public
          :func:`potrs` discards that buffer for convenience, but an outer
          ``jax.jit(..., donate_argnums=(0, 1))`` can only donate ``a`` if an
          ``A``-sized output is returned.  Callers that want donation must
          keep ``a_work`` in the jitted function's returned pytree.
        - Native code converts row-major JAX local storage to cuSOLVERMp's
          column-major local layout, redistributes to 2D block-cyclic layout,
          calls ``cusolverMpPotrf``/``cusolverMpPotrs``, and redistributes the
          result back.
    """
    b, vector_rhs, layout, rhs_specs, b_padding, b_distribution_cols = (
        _prepare_potrs_call(
            a,
            b,
            T_A,
            mesh,
            matrix_specs,
            in_specs=in_specs,
            pad=pad,
            caller="potrs_shardmap_ctx",
        )
    )
    nrhs = int(b.shape[1])

    ensure_init_jaxmg_backend()

    impl = _potrs_pipeline(
        layout.mesh,
        layout.matrix_specs,
        layout.native_status_specs,
        layout.grid,
        layout.partition_slots,
        layout.padding,
        b_padding,
        rhs_specs,
        n=a.shape[0],
        nrhs=nrhs,
        b_distribution_cols=b_distribution_cols,
        tile_size=layout.tile_shape.rows,
        return_logdet=return_logdet,
    )
    result = impl(a, b)
    if return_logdet:
        a_work, out, logdet, native_status = result
    else:
        a_work, out, native_status = result
    if vector_rhs:
        out = out[:, 0]
    if return_logdet:
        return a_work, out, logdet[0], native_status
    return a_work, out, native_status


def _prepare_potrs_call(
    a: Array,
    b: Array,
    tile_size: int,
    mesh: Mesh | AbstractMesh | None,
    matrix_specs: P | Tuple[P] | List[P] | None,
    *,
    in_specs: P | Tuple[P] | List[P] | None,
    pad: bool,
    caller: str,
):
    """Validate a Cholesky solve and derive its matrix and RHS layouts.

    Preparation proceeds as follows:

    1. Validate A and normalize a vector B to a one-column matrix.
    2. Require matching supported dtypes, compatible leading dimensions, a
       square A, and a positive tile size.
    3. Resolve the mesh, process grid, rank map, and tile-aligned layout of A.
    4. Infer the JAX-facing B sharding, add any required routing columns, and
       derive its tile-aligned native work layout.
    """
    if a.ndim != 2:
        raise ValueError(f"{caller} expects a rank-2 matrix A.")
    vector_rhs = b.ndim == 1
    if vector_rhs:
        b = jnp.expand_dims(b, axis=1)
    if b.ndim != 2:
        raise ValueError(f"{caller} expects a rank-1 or rank-2 RHS B.")
    if a.dtype != b.dtype:
        raise TypeError(f"{caller} requires matching A/B dtypes.")
    _check_supported_potrs_dtype(a.dtype)
    if a.shape[0] != a.shape[1]:
        raise ValueError(f"{caller} expects A to be square.")
    if a.shape[0] != b.shape[0]:
        raise ValueError("A and B must have matching leading dimensions.")
    if int(tile_size) <= 0:
        raise ValueError("T_A must be positive.")

    layout = prepare_input_matrix_layout(
        a,
        tile_size,
        mesh=mesh,
        matrix_specs=matrix_specs,
        in_specs=in_specs,
        pad=pad,
        caller=caller,
    )
    rhs_specs = infer_rhs_specs(b, matrix_specs=layout.matrix_specs)
    nrhs = int(b.shape[1])
    b_distribution_cols = rhs_distribution_columns(
        nrhs,
        process_cols=layout.grid.process_cols,
        pad=pad,
    )
    b_padding = prepare_matrix_padding(
        logical_rows=b.shape[0],
        logical_cols=b_distribution_cols,
        grid=layout.grid,
        tile_shape=layout.tile_shape,
        pad=pad,
        caller=f"{caller}(B)",
    )
    return b, vector_rhs, layout, rhs_specs, b_padding, b_distribution_cols


def _check_supported_potrs_dtype(dtype) -> None:
    """Validate that ``dtype`` maps to a cuSOLVERMp POTRS entry point.

    The native backend dispatches to the real, double, complex64, and
    complex128 cuSOLVERMp routines.  Rejecting unsupported dtypes at the Python
    boundary gives a clear user error before JAX traces a compiled FFI call.
    """
    if dtype not in (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128):
        raise TypeError("potrs supports float32, float64, complex64, and complex128.")


def _real_dtype_for_logdet(dtype):
    """Return the real-component dtype used for the log determinant."""
    if dtype == jnp.float32 or dtype == jnp.complex64:
        return jnp.float32
    if dtype == jnp.float64 or dtype == jnp.complex128:
        return jnp.float64
    raise TypeError("potrs supports float32, float64, complex64, and complex128.")


_ROW_MAJOR_JAX_LAYOUT = (0, 1)


@lru_cache(maxsize=None)
def _potrs_pipeline(
    mesh: Mesh | AbstractMesh,
    matrix_specs: P,
    native_status_specs: P,
    grid: ProcessGrid,
    partition_slots: tuple[int, ...],
    a_padding: MatrixPadding2D,
    b_padding: MatrixPadding2D,
    rhs_specs: P,
    *,
    n: int,
    nrhs: int,
    b_distribution_cols: int,
    tile_size: int,
    return_logdet: bool,
):
    """Build and cache the unjitted JAX-visible POTRS execution pipeline.

    This factory is cached by static configuration: mesh, process grid, rank
    partition mapping, padding shape, matrix size, RHS width, and tile size. Reusing the
    wrapper avoids rebuilding the same ``jax.shard_map`` structure on repeated
    solves with identical layout metadata.
    """
    process_rows = grid.process_rows
    process_cols = grid.process_cols
    slots_attr = np.asarray(partition_slots, dtype=np.int64)
    b_distribution_padding = int(b_distribution_cols) - int(nrhs)
    pad_a = make_local_pad_fn(mesh, matrix_specs, a_padding)
    pad_b = make_local_pad_fn(mesh, matrix_specs, b_padding)
    unpad_b = make_local_unpad_fn(mesh, matrix_specs, b_padding)

    def potrs_ffi(_a: Array, _b: Array):
        """Call fused native redistribution and ``potrf/potrs`` on one shard.

        The closure captures only static metadata that XLA needs at trace time:
        process-grid shape, rank map, logical dimensions, and tile size.  The
        actual matrix buffers remain donated JAX arrays and enter native code
        through ``jax.ffi.ffi_call``.
        """
        if _a.ndim != 2 or _b.ndim != 2:
            raise ValueError("cusolvermp_potrs expects rank-2 A and B buffers.")
        if _a.dtype != _b.dtype:
            raise TypeError("cusolvermp_potrs requires matching A/B dtypes.")
        _check_supported_potrs_dtype(_a.dtype)
        if _a.shape[0] != _b.shape[0]:
            raise ValueError("A and B must have matching local row capacity.")

        common_out_type = (
            jax.ShapeDtypeStruct(_a.shape, _a.dtype),
            jax.ShapeDtypeStruct(_b.shape, _b.dtype),
        )
        status_type = jax.ShapeDtypeStruct(
            (_CUSOLVERMP_POTRS_STATUS_SIZE,), jnp.int32
        )
        if return_logdet:
            out_type = common_out_type + (
                jax.ShapeDtypeStruct((1,), _real_dtype_for_logdet(_a.dtype)),
                status_type,
            )
            output_layouts = (
                _ROW_MAJOR_JAX_LAYOUT,
                _ROW_MAJOR_JAX_LAYOUT,
                (0,),
                (0,),
            )
            target_name = "cusolvermp_potrs_logdet"
        else:
            out_type = common_out_type + (status_type,)
            output_layouts = (
                _ROW_MAJOR_JAX_LAYOUT,
                _ROW_MAJOR_JAX_LAYOUT,
                (0,),
            )
            target_name = "cusolvermp_potrs"
        ffi_fn = partial(
            jax.ffi.ffi_call(
                target_name,
                out_type,
                input_layouts=(_ROW_MAJOR_JAX_LAYOUT, _ROW_MAJOR_JAX_LAYOUT),
                output_layouts=output_layouts,
                input_output_aliases={0: 0, 1: 1},
            ),
            process_rows=process_rows,
            process_cols=process_cols,
            partition_slots=slots_attr,
            n=int(n),
            nrhs=int(nrhs),
            b_distribution_cols=int(b_distribution_cols),
            tile_size=int(tile_size),
        )
        return ffi_fn(_a, _b)

    if return_logdet:
        native_out_specs = (
            matrix_specs,
            matrix_specs,
            P(),
            native_status_specs,
        )
    else:
        native_out_specs = (matrix_specs, matrix_specs, native_status_specs)
    potrs_shardmap = jax.shard_map(
        potrs_ffi,
        mesh=mesh,
        in_specs=(matrix_specs, matrix_specs),
        out_specs=native_out_specs,
        check_vma=False,
    )

    @use_abstract_mesh_decorator(mesh)
    def impl(_a: Array, _b: Array):
        """Run padding, fused native POTRS, and unpadding as one compiled path."""
        a_padded = pad_a(_a)
        if b_distribution_padding:
            b_distribution = jnp.pad(_b, ((0, 0), (0, b_distribution_padding)))
        else:
            b_distribution = _b
        # The public API permits RHS sharding that differs from A, such as a
        # replicated RHS-column axis. Native redistribution consumes the
        # matrix work sharding before shard-local tile-capacity padding.
        b_distribution = place_rhs_for_native_work(
            b_distribution,
            mesh=mesh,
            matrix_specs=matrix_specs,
        )
        b_padded = pad_b(b_distribution)
        native_result = potrs_shardmap(a_padded, b_padded)
        if return_logdet:
            a_work_padded, b_solved_padded, logdet, native_status = native_result
        else:
            a_work_padded, b_solved_padded, native_status = native_result
        out = unpad_b(b_solved_padded)
        out = restore_rhs_from_native_work(
            out,
            rhs_specs=rhs_specs,
            mesh=mesh,
            matrix_specs=matrix_specs,
        )
        if return_logdet:
            return a_work_padded, out[:, :nrhs], logdet, native_status
        return a_work_padded, out[:, :nrhs], native_status

    return impl


@lru_cache(maxsize=None)
def _potrs_compiled(
    mesh: Mesh | AbstractMesh,
    matrix_specs: P,
    native_status_specs: P,
    grid: ProcessGrid,
    partition_slots: tuple[int, ...],
    a_padding: MatrixPadding2D,
    b_padding: MatrixPadding2D,
    rhs_specs: P,
    *,
    n: int,
    nrhs: int,
    b_distribution_cols: int,
    tile_size: int,
    return_logdet: bool,
    donate: bool,
):
    """Build and cache the internally jitted public POTRS execution pipeline."""
    pipeline = _potrs_pipeline(
        mesh,
        matrix_specs,
        native_status_specs,
        grid,
        partition_slots,
        a_padding,
        b_padding,
        rhs_specs,
        n=n,
        nrhs=nrhs,
        b_distribution_cols=b_distribution_cols,
        tile_size=tile_size,
        return_logdet=return_logdet,
    )

    @partial(jax.jit, donate_argnums=(0, 1) if donate else ())
    def impl(_a: Array, _b: Array):
        """Run the cached POTRS pipeline behind the public convenience API."""
        return pipeline(_a, _b)

    return impl
