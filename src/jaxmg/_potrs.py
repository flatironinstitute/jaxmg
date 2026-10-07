"""Public cuSOLVERMp Cholesky solve wrapper.

The Python layer validates JAX array metadata, applies per-shard tile padding,
and constructs the compiled FFI call.  The fused native C++/CUDA handler then
performs the row-major to column-major local layout conversion, 2D
redistribution, ``cusolverMpPotrf``/``cusolverMpPotrs`` calls, reverse
redistribution, and final layout restoration.

The three entry points are layered:

- :func:`potrs_shardmap_ctx` runs on local blocks inside a caller's
  ``jax.shard_map``;
- :func:`potrs_jit_ctx` wraps it in ``jax.shard_map`` for global arrays inside a
  caller-owned ``jax.jit``;
- :func:`potrs` wraps that pipeline in an internal ``jax.jit`` with donation.
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
    mark_varying,
    pad_local_2d,
    place_for_native_work,
    prepare_input_matrix_layout,
    prepare_local_matrix_layout,
    prepare_matrix_padding,
    restore_rhs_from_native_work,
    rhs_distribution_columns,
    unpad_local_2d,
    use_abstract_mesh_decorator,
)
from ._cusolvermp_status import _CUSOLVERMP_POTRS_STATUS_SIZE
from ._layout_types import ProcessGrid
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
        - If the native solver fails, the solution and ``logdet`` are filled
          with NaN and ``status`` is non-zero.
    """
    b, vector_rhs, layout, rhs_specs, b_distribution_cols = (
        _prepare_global_potrs_call(
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

    ensure_init_jaxmg_backend()

    pipeline = _potrs_pipeline(
        layout.mesh,
        layout.matrix_specs,
        layout.native_status_specs,
        rhs_specs,
        nrhs=int(b.shape[1]),
        b_distribution_cols=b_distribution_cols,
        tile_size=layout.tile_shape.rows,
        return_logdet=return_logdet,
        pad=pad,
    )
    # The cached pipeline is the jit cache key, so a fresh wrapper still reuses
    # earlier traces and compilations.
    result = jax.jit(pipeline, donate_argnums=(0, 1) if donate else ())(a, b)
    if return_logdet:
        _, out, logdet, native_status = result
    else:
        _, out, native_status = result
    if vector_rhs:
        out = out[:, 0]
    if return_logdet and return_status:
        return out, logdet, native_status
    if return_logdet:
        return out, logdet
    if return_status:
        return out, native_status
    return out


def potrs_jit_ctx(
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
    """Solve A x = B on global arrays inside a caller-owned ``jax.jit``.

    This helper is the lower-level variant of :func:`jaxmg.potrs` intended for
    contexts where the caller wants to control the outer ``jax.jit`` boundary.

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
        - Up to 1.4.0 this function was named ``potrs_shardmap_ctx``.
    """
    b, vector_rhs, layout, rhs_specs, b_distribution_cols = (
        _prepare_global_potrs_call(
            a,
            b,
            T_A,
            mesh,
            matrix_specs,
            in_specs=in_specs,
            pad=pad,
            caller="potrs_jit_ctx",
        )
    )

    ensure_init_jaxmg_backend()

    impl = _potrs_pipeline(
        layout.mesh,
        layout.matrix_specs,
        layout.native_status_specs,
        rhs_specs,
        nrhs=int(b.shape[1]),
        b_distribution_cols=b_distribution_cols,
        tile_size=layout.tile_shape.rows,
        return_logdet=return_logdet,
        pad=pad,
    )
    result = impl(a, b)
    if vector_rhs:
        result = (result[0], result[1][:, 0], *result[2:])
    return result


def potrs_shardmap_ctx(
    a: Array,
    b: Array,
    T_A: int,
    matrix_specs: P | Tuple[P] | List[P] | None = None,
    *,
    in_specs: P | Tuple[P] | List[P] | None = None,
    return_logdet: bool = False,
    pad: bool = True,
) -> Union[Tuple[Array, Array, Array], Tuple[Array, Array, Array, Array]]:
    """Solve A x = B on local blocks inside a caller's ``jax.shard_map``.

    This is the per-shard core of :func:`jaxmg.potrs`. Call it from the body of
    a ``jax.shard_map`` whose mesh contains exactly the matrix axes, with
    ``a`` and ``b`` both sharded by ``matrix_specs``.

    Args:
        a (Array): Local block of a 2D, symmetric positive-definite matrix.
        b (Array): Local block of the solve input, sharded like ``a`` or
            replicated over the column axis. A local vector is treated as a
            one-column block.
        T_A (int): Square tile width used by cuSOLVERMp. Each local shard
            dimension must be a multiple of ``T_A`` after padding.
        matrix_specs (PartitionSpec or tuple/list[PartitionSpec], optional):
            PartitionSpec of ``a`` and ``b`` in the enclosing ``shard_map``.
            If omitted, defaults to the context mesh axes in order.
        in_specs: Backwards-compatible alias for ``matrix_specs``.
        return_logdet (bool, optional): If True return the replicated Cholesky
            log determinant between the solution and status outputs. Default
            is False.
        pad (bool, optional): If True (default) pad each local block so its
            dimensions are compatible with ``T_A``; if False the caller must
            ensure shapes already match the kernel's requirements.

    Returns:
        tuple: ``(a_work, x, status)`` or ``(a_work, x, logdet, status)``.
        ``a_work`` is the padded local matrix work buffer required for
        donation, ``x`` is the local solution block, ``logdet`` is a real
        scalar that is identical on every rank, and ``status`` is this rank's
        native diagnostic vector.

    Raises:
        TypeError: If dtypes or ``PartitionSpec`` inputs are unsupported.
        ValueError: If called outside ``jax.shard_map``, or if shapes, tile
            sizes, or mesh layouts are incompatible.
    """
    b, vector_rhs, layout, b_padding, nrhs = _prepare_local_potrs_call(
        a,
        b,
        T_A,
        matrix_specs,
        in_specs=in_specs,
        pad=pad,
        caller="potrs_shardmap_ctx",
    )
    b_varying = jax.typeof(b).manual_axis_type.varying

    ensure_init_jaxmg_backend()

    a_padding = layout.padding
    a = pad_local_2d(
        a,
        row_padding=a_padding.row_padding_per_process,
        col_padding=a_padding.col_padding_per_process,
    )
    b = pad_local_2d(
        b,
        row_padding=b_padding.row_padding_per_process,
        col_padding=b_padding.col_padding_per_process,
    )
    result = _potrs_native_call(
        a,
        b,
        grid=layout.grid,
        partition_slots=layout.partition_slots,
        n=a_padding.logical_rows,
        nrhs=nrhs,
        tile_size=layout.tile_shape.rows,
        return_logdet=return_logdet,
    )
    a_work, x, *logdet, status = result
    x = unpad_local_2d(
        x,
        local_rows=b_padding.local_logical_rows,
        local_cols=b_padding.local_logical_cols,
    )
    if vector_rhs:
        x = x[:, 0]
    # The native outputs differ per rank but are typed invariant.
    row_axis, _ = layout.matrix_specs
    axes = tuple(
        axis
        for axis in layout.matrix_specs
        if axis is not None and layout.mesh.shape[axis] > 1
    )
    x_axes = tuple(axis for axis in axes if axis == row_axis or axis in b_varying)
    a_work = mark_varying(a_work, axes)
    x = mark_varying(x, x_axes)
    status = mark_varying(status, axes)
    if return_logdet:
        return a_work, x, logdet[0][0], status
    return a_work, x, status


def _check_potrs_operands(a: Array, b: Array, tile_size: int, caller: str):
    """Validate operand ranks and dtypes, and normalize a vector B to a matrix."""
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
    if a.shape[0] != b.shape[0]:
        raise ValueError("A and B must have matching leading dimensions.")
    if int(tile_size) <= 0:
        raise ValueError("T_A must be positive.")
    return b, vector_rhs


def _prepare_global_potrs_call(
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
    """Validate a Cholesky solve on global arrays and derive its layouts.

    Preparation proceeds as follows:

    1. Validate A and normalize a vector B to a one-column matrix.
    2. Require matching supported dtypes, compatible leading dimensions, a
       square A, and a positive tile size.
    3. Resolve the mesh, process grid, rank map, and tile-aligned layout of A.
    4. Infer the JAX-facing B sharding, add any routing columns required to
       shard B like A, and check its tile-aligned layout.
    """
    b, vector_rhs = _check_potrs_operands(a, b, tile_size, caller)
    if a.shape[0] != a.shape[1]:
        raise ValueError(f"{caller} expects A to be square.")

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
    b_distribution_cols = rhs_distribution_columns(
        int(b.shape[1]),
        process_cols=layout.grid.process_cols,
        pad=pad,
    )
    # The shard-local layer repeats this check; running it here reports errors
    # against the caller's own entry point.
    prepare_matrix_padding(
        logical_rows=b.shape[0],
        logical_cols=b_distribution_cols,
        grid=layout.grid,
        tile_shape=layout.tile_shape,
        pad=pad,
        caller=f"{caller}(B)",
    )
    return b, vector_rhs, layout, rhs_specs, b_distribution_cols


def _prepare_local_potrs_call(
    a: Array,
    b: Array,
    tile_size: int,
    matrix_specs: P | Tuple[P] | List[P] | None,
    *,
    in_specs: P | Tuple[P] | List[P] | None,
    pad: bool,
    caller: str,
):
    """Validate a Cholesky solve on local blocks and derive its layouts.

    The global shapes follow from the local blocks and the process grid of the
    context mesh. Because B is sharded like A, every one of its global columns
    is passed to cuSOLVERMp as a right-hand side.
    """
    b, vector_rhs = _check_potrs_operands(a, b, tile_size, caller)
    layout = prepare_local_matrix_layout(
        a,
        tile_size,
        matrix_specs=matrix_specs,
        in_specs=in_specs,
        pad=pad,
        caller=caller,
    )
    n = layout.padding.logical_rows
    if n != layout.padding.logical_cols:
        raise ValueError(f"{caller} expects A to be square.")
    nrhs = int(b.shape[1]) * layout.grid.process_cols
    b_padding = prepare_matrix_padding(
        logical_rows=n,
        logical_cols=nrhs,
        grid=layout.grid,
        tile_shape=layout.tile_shape,
        pad=pad,
        caller=f"{caller}(B)",
    )
    return b, vector_rhs, layout, b_padding, nrhs


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


def _potrs_native_call(
    a: Array,
    b: Array,
    *,
    grid: ProcessGrid,
    partition_slots: tuple[int, ...],
    n: int,
    nrhs: int,
    tile_size: int,
    return_logdet: bool,
):
    """Call fused native redistribution and ``potrf/potrs`` on one shard.

    Only static metadata that XLA needs at trace time enters as attributes:
    process-grid shape, rank map, logical dimensions, and tile size.  The
    matrix buffers are aliased to the first two outputs, so donated inputs
    enter native code without a copy.
    """
    common_out_type = (
        jax.ShapeDtypeStruct(a.shape, a.dtype),
        jax.ShapeDtypeStruct(b.shape, b.dtype),
    )
    status_type = jax.ShapeDtypeStruct((_CUSOLVERMP_POTRS_STATUS_SIZE,), jnp.int32)
    if return_logdet:
        out_type = common_out_type + (
            jax.ShapeDtypeStruct((1,), _real_dtype_for_logdet(a.dtype)),
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
    ffi_fn = jax.ffi.ffi_call(
        target_name,
        out_type,
        input_layouts=(_ROW_MAJOR_JAX_LAYOUT, _ROW_MAJOR_JAX_LAYOUT),
        output_layouts=output_layouts,
        input_output_aliases={0: 0, 1: 1},
    )
    return ffi_fn(
        a,
        b,
        process_rows=grid.process_rows,
        process_cols=grid.process_cols,
        partition_slots=np.asarray(partition_slots, dtype=np.int64),
        n=int(n),
        nrhs=int(nrhs),
        b_distribution_cols=int(nrhs),
        tile_size=int(tile_size),
    )


@lru_cache(maxsize=None)
def _potrs_pipeline(
    mesh: Mesh | AbstractMesh,
    matrix_specs: P,
    native_status_specs: P,
    rhs_specs: P,
    *,
    nrhs: int,
    b_distribution_cols: int,
    tile_size: int,
    return_logdet: bool,
    pad: bool,
):
    """Build and cache the unjitted global POTRS pipeline.

    The pipeline places A and B in the matrix sharding, runs
    :func:`potrs_shardmap_ctx` under ``jax.shard_map``, and restores the
    solution to the JAX-facing RHS sharding.  It is cached by static
    configuration so repeated solves reuse the same ``jax.shard_map``.
    """
    b_distribution_padding = int(b_distribution_cols) - int(nrhs)
    if return_logdet:
        out_specs = (matrix_specs, matrix_specs, P(), native_status_specs)
    else:
        out_specs = (matrix_specs, matrix_specs, native_status_specs)
    solve = jax.shard_map(
        partial(
            potrs_shardmap_ctx,
            T_A=tile_size,
            matrix_specs=matrix_specs,
            return_logdet=return_logdet,
            pad=pad,
        ),
        mesh=mesh,
        in_specs=(matrix_specs, matrix_specs),
        out_specs=out_specs,
        check_vma=True,
    )

    @use_abstract_mesh_decorator(mesh)
    def impl(_a: Array, _b: Array):
        """Place the inputs, solve shard-locally, and restore the solution."""
        _a = place_for_native_work(_a, mesh=mesh, matrix_specs=matrix_specs)
        # The public API permits RHS sharding that differs from A, such as a
        # replicated RHS-column axis, and RHS widths that are not divisible by
        # the process-column count. Routing columns make B shardable like A.
        if b_distribution_padding:
            _b = jnp.pad(_b, ((0, 0), (0, b_distribution_padding)))
        _b = place_for_native_work(_b, mesh=mesh, matrix_specs=matrix_specs)
        a_work, out, *rest = solve(_a, _b)
        out = restore_rhs_from_native_work(
            out,
            rhs_specs=rhs_specs,
            mesh=mesh,
            matrix_specs=matrix_specs,
        )
        return (a_work, out[:, :nrhs], *rest)

    return impl
