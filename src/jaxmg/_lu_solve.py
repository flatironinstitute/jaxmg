"""Public cuSOLVERMp LU solve wrapper.

The Python layer validates JAX array metadata, applies per-shard tile padding,
and constructs the compiled FFI call.  The fused native C++/CUDA handler then
performs the row-major to column-major local layout conversion, 2D
redistribution, ``cusolverMpGetrf``/``cusolverMpGetrs`` calls, reverse
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
    mark_varying,
    pad_local_2d,
    unpad_local_2d,
    prepare_input_matrix_layout,
    prepare_local_matrix_layout,
    prepare_matrix_padding,
    use_abstract_mesh_decorator,
    place_for_native_work,
    restore_rhs_from_native_work,
    rhs_distribution_columns,
)
from ._cusolvermp_status import _CUSOLVERMP_LU_SOLVE_STATUS_SIZE
from ._layout_types import MatrixPadding2D, ProcessGrid
from ._setup import ensure_init_jaxmg_backend


def lu_solve(
    a: Array,
    b: Array,
    T_A: int,
    mesh: Mesh | AbstractMesh | None = None,
    matrix_specs: P | Tuple[P] | List[P] | None = None,
    *,
    in_specs: P | Tuple[P] | List[P] | None = None,
    return_status: bool = False,
    pad: bool = True,
    donate: bool = True,
) -> Union[Array, Tuple[Array, Array]]:
    """Solve the linear system A x = B using the multi-GPU LU native kernel.

    This is the high-level JAXMg general linear-solve entry point.  It prepares
    a block-sharded JAX array for cuSOLVERMp, calls the fused native backend,
    and returns the solution in the same JAX-facing layout as ``b``.

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
        pad (bool, optional): If True (default) apply per-device padding so
            each local shard length is compatible with ``T_A``; if False the
            caller must ensure shapes already match the kernel's requirements.
        donate (bool, optional): If True (default) the input buffers may be
            donated to the native call for zero-copy execution, which means
            they are deleted and cannot be used again. Pass False to preserve
            them, at the cost of keeping the original and working buffers in
            memory simultaneously.

    Returns:
        Array or (Array, Array): The solution ``x`` in the same JAX-facing
            block-sharded layout as ``b``. If ``return_status=True`` also
            return the native solver status.

    Raises:
        TypeError: If dtypes or ``PartitionSpec`` inputs are unsupported.
        ValueError: If shapes, tile sizes, or mesh layouts are incompatible.

    Notes:
        - Unless ``donate=False``, the ``a`` and ``b`` buffers are donated for
          zero-copy interaction with the native library, so they are deleted and
          cannot be used after the call.
        - Native code converts row-major JAX local storage to cuSOLVERMp's
          column-major local layout, redistributes to 2D block-cyclic layout,
          calls ``cusolverMpGetrf``/``cusolverMpGetrs``, and redistributes the
          result back.
        - If the native solver fails, the solution is filled with NaN and
          ``status`` is non-zero.
    """
    b, vector_rhs, layout, rhs_specs, b_padding, b_distribution_cols = (
        _prepare_lu_solve_call(
            a,
            b,
            T_A,
            mesh,
            matrix_specs,
            in_specs=in_specs,
            pad=pad,
            caller="lu_solve",
        )
    )
    nrhs = int(b.shape[1])

    ensure_init_jaxmg_backend()

    impl = _lu_solve_compiled(
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
        donate=donate,
    )
    _, out, native_status = impl(a, b)
    if vector_rhs:
        out = out[:, 0]
    if return_status:
        return out, native_status
    return out


def lu_solve_jit_ctx(
    a: Array,
    b: Array,
    T_A: int,
    mesh: Mesh | AbstractMesh | None = None,
    matrix_specs: P | Tuple[P] | List[P] | None = None,
    *,
    in_specs: P | Tuple[P] | List[P] | None = None,
    pad: bool = True,
) -> Tuple[Array, Array, Array]:
    """Solve A x = B while exposing the donated matrix work buffer.

    This helper is the lower-level variant of :func:`jaxmg.lu_solve` intended
    for contexts where the caller wants to control the outer ``jax.jit``
    boundary.  It performs the same validation, local padding, shard-map
    construction, and fused cuSOLVERMp FFI call as the public solver, but it
    does not wrap the pipeline in an internal ``jax.jit``.  Instead, it returns
    the native matrix work buffer alongside the solution so an outer JIT can
    donate ``a`` into an ``A``-sized output.

    Args:
        a (Array): 2D, nonsingular input matrix sharded over a one- or two-axis
            device mesh, for example with ``P(<row_axis>)`` or
            ``P(<row_axis>, <col_axis>)``.
        b (Array): 1D or 2D solve input. A vector is treated as an
            ``N x 1`` matrix.
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
        tuple: ``(a_work, x, status)`` where ``a_work`` is the padded matrix
        work buffer returned by the native FFI call, ``x`` is the solution in
        the same JAX-facing layout as ``b``, and ``status`` is the native
        per-rank diagnostic vector.
    """
    b, vector_rhs, layout, rhs_specs, b_padding, b_distribution_cols = (
        _prepare_lu_solve_call(
            a,
            b,
            T_A,
            mesh,
            matrix_specs,
            in_specs=in_specs,
            pad=pad,
            caller="lu_solve_jit_ctx",
        )
    )
    nrhs = int(b.shape[1])

    ensure_init_jaxmg_backend()

    impl = _lu_solve_pipeline(
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
    )
    a_work, out, native_status = impl(a, b)
    if vector_rhs:
        out = out[:, 0]
    return a_work, out, native_status


def lu_solve_shardmap_ctx(
    a: Array,
    b: Array,
    T_A: int,
    matrix_specs: P | Tuple[P] | List[P] | None = None,
    *,
    in_specs: P | Tuple[P] | List[P] | None = None,
    pad: bool = True,
) -> Tuple[Array, Array, Array]:
    """Solve A x = B on local blocks inside a caller's ``jax.shard_map``.

    A follows ``matrix_specs``; B shares its row sharding and may have
    distributed or replicated columns. Replicated copies are solved as
    duplicate right-hand sides using a single distributed LU factorization.

    Args:
        a: Local block of a square, nonsingular global matrix.
        b: Local RHS matrix or vector, with the same dtype and row layout as A.
        T_A: Square tile width used by cuSOLVERMp.
        matrix_specs: A's partition specification in the enclosing shard map;
            defaults to the context mesh axes in order.
        in_specs: Backwards-compatible alias for ``matrix_specs``.
        pad: Allow local tile-capacity padding; otherwise require aligned blocks.

    Returns:
        ``(a_work, x, status)``: the padded local work buffer, local solution
        and this rank's diagnostic vector. Return ``a_work`` from the outer
        jitted function when donating A.

    Raises:
        TypeError: If operand dtypes or partition specifications are unsupported.
        ValueError: If called outside shard_map or shapes/layouts are incompatible.
    """
    b, vector_rhs, layout, _, b_padding, nrhs = _prepare_lu_solve_call(
        a, b, T_A, None, matrix_specs, in_specs=in_specs, pad=pad,
        caller="lu_solve_shardmap_ctx", local=True,
    )
    ensure_init_jaxmg_backend()
    a_work, x, status = _lu_solve_local_pipeline(
        layout.matrix_specs, layout.grid, layout.partition_slots,
        layout.padding, b_padding, n=layout.padding.logical_rows,
        nrhs=nrhs, b_distribution_cols=nrhs, tile_size=int(T_A),
    )(a, b)
    return a_work, x[:, 0] if vector_rhs else x, status


def _prepare_lu_solve_call(
    a: Array,
    b: Array,
    tile_size: int,
    mesh: Mesh | AbstractMesh | None,
    matrix_specs: P | Tuple[P] | List[P] | None,
    *,
    in_specs: P | Tuple[P] | List[P] | None,
    pad: bool,
    caller: str,
    local: bool = False,
):
    """Validate an LU solve and derive its matrix and RHS layouts.

    Preparation proceeds as follows:

    1. Validate A and normalize a vector B to a one-column matrix.
    2. Require matching supported dtypes, compatible leading dimensions, a
       square A, and a positive tile size.
    3. Resolve the mesh, process grid, rank map, and tile-aligned layout of A;
       for local blocks, reconstruct the global dimensions from the context mesh.
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
    _check_supported_lu_solve_dtype(a.dtype)
    if not local and a.shape[0] != a.shape[1]:
        raise ValueError(f"{caller} expects A to be square.")
    if a.shape[0] != b.shape[0]:
        raise ValueError("A and B must have matching leading dimensions.")
    if int(tile_size) <= 0:
        raise ValueError("T_A must be positive.")

    if local:
        layout = prepare_local_matrix_layout(
            a, tile_size, matrix_specs=matrix_specs, in_specs=in_specs,
            pad=pad, caller=caller,
        )
        if layout.padding.logical_rows != layout.padding.logical_cols:
            raise ValueError(f"{caller} expects A to be square.")
    else:
        layout = prepare_input_matrix_layout(
            a, tile_size, mesh=mesh, matrix_specs=matrix_specs,
            in_specs=in_specs, pad=pad, caller=caller,
        )
    rhs_specs = layout.matrix_specs if local else infer_rhs_specs(b, matrix_specs=layout.matrix_specs)
    nrhs = int(b.shape[1])
    if local:
        nrhs *= layout.grid.process_cols
    b_distribution_cols = rhs_distribution_columns(
        nrhs,
        process_cols=layout.grid.process_cols,
        pad=pad,
    )
    b_padding = prepare_matrix_padding(
        logical_rows=layout.padding.logical_rows,
        logical_cols=b_distribution_cols,
        grid=layout.grid,
        tile_shape=layout.tile_shape,
        pad=pad,
        caller=f"{caller}(B)",
    )
    return b, vector_rhs, layout, rhs_specs, b_padding, b_distribution_cols


def _check_supported_lu_solve_dtype(dtype) -> None:
    """Validate that ``dtype`` maps to cuSOLVERMp GETRF/GETRS entry points."""
    if dtype not in (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128):
        raise TypeError(
            "lu_solve supports float32, float64, complex64, and complex128."
        )


_ROW_MAJOR_JAX_LAYOUT = (0, 1)


def _lu_solve_local_pipeline(
    matrix_specs: P,
    grid: ProcessGrid,
    partition_slots: tuple[int, ...],
    a_padding: MatrixPadding2D,
    b_padding: MatrixPadding2D,
    *,
    n: int,
    nrhs: int,
    b_distribution_cols: int,
    tile_size: int,
):
    """Build the local padding, native LU solve, and output-restoration stages."""
    process_rows = grid.process_rows
    process_cols = grid.process_cols
    slots_attr = np.asarray(partition_slots, dtype=np.int64)

    def lu_solve_ffi(_a: Array, _b: Array) -> tuple[Array, Array, Array]:
        """Execute the shared shard-local solve without adding a shard map."""
        b_varying = jax.typeof(_b).manual_axis_type.varying
        # Stage 1: provide tile-aligned capacity for both operands.
        _a = pad_local_2d(_a, row_padding=a_padding.row_padding_per_process,
                          col_padding=a_padding.col_padding_per_process)
        _b = pad_local_2d(_b, row_padding=b_padding.row_padding_per_process,
                          col_padding=b_padding.col_padding_per_process)
        if _a.ndim != 2 or _b.ndim != 2:
            raise ValueError("cusolvermp_lu_solve expects rank-2 A and B buffers.")
        if _a.dtype != _b.dtype:
            raise TypeError("cusolvermp_lu_solve requires matching A/B dtypes.")
        _check_supported_lu_solve_dtype(_a.dtype)
        if _a.shape[0] != _b.shape[0]:
            raise ValueError("A and B must have matching local row capacity.")

        # Stage 2: execute the fused redistribution, factorization and solve.
        out_type = (
            jax.ShapeDtypeStruct(_a.shape, _a.dtype),
            jax.ShapeDtypeStruct(_b.shape, _b.dtype),
            jax.ShapeDtypeStruct((_CUSOLVERMP_LU_SOLVE_STATUS_SIZE,), jnp.int32),
        )
        ffi_fn = partial(
            jax.ffi.ffi_call(
                "cusolvermp_lu_solve",
                out_type,
                input_layouts=(_ROW_MAJOR_JAX_LAYOUT, _ROW_MAJOR_JAX_LAYOUT),
                output_layouts=(
                    _ROW_MAJOR_JAX_LAYOUT,
                    _ROW_MAJOR_JAX_LAYOUT,
                    (0,),
                ),
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
        # FFI operands must agree on varying axes, even for replicated RHS columns.
        input_axes = tuple(jax.typeof(_a).manual_axis_type.varying | b_varying)
        a_work, b_out, status = ffi_fn(
            mark_varying(_a, input_axes), mark_varying(_b, input_axes)
        )
        # Stage 3: restore the local RHS shape and per-rank output types.
        b_out = unpad_local_2d(b_out, local_rows=b_padding.local_logical_rows,
                              local_cols=b_padding.local_logical_cols)
        axes = tuple(axis for axis, size in zip(matrix_specs, (grid.process_rows, grid.process_cols))
                     if axis is not None and size > 1)
        x_axes = tuple(axis for axis in axes if axis == matrix_specs[0] or axis in b_varying)
        return mark_varying(a_work, axes), mark_varying(b_out, x_axes), mark_varying(status, axes)

    return lu_solve_ffi


@lru_cache(maxsize=None)
def _lu_solve_pipeline(
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
):
    """Place global inputs, map the local solve, and restore the RHS layout."""
    b_distribution_padding = int(b_distribution_cols) - int(nrhs)

    lu_solve_shardmap = jax.shard_map(
        _lu_solve_local_pipeline(
            matrix_specs, grid, partition_slots, a_padding, b_padding,
            n=n, nrhs=nrhs, b_distribution_cols=b_distribution_cols, tile_size=tile_size,
        ),
        mesh=mesh,
        in_specs=(matrix_specs, matrix_specs),
        out_specs=(matrix_specs, matrix_specs, native_status_specs),
        check_vma=True,
    )

    @use_abstract_mesh_decorator(mesh)
    def impl(_a: Array, _b: Array) -> tuple[Array, Array, Array]:
        """Run padding, fused native LU solve, and unpadding."""
        _a = place_for_native_work(_a, mesh=mesh, matrix_specs=matrix_specs)
        if b_distribution_padding:
            b_distribution = jnp.pad(_b, ((0, 0), (0, b_distribution_padding)))
        else:
            b_distribution = _b
        # The public API permits RHS sharding that differs from A, such as a
        # replicated RHS-column axis. Native redistribution consumes the
        # matrix work sharding before shard-local tile-capacity padding.
        b_distribution = place_for_native_work(
            b_distribution,
            mesh=mesh,
            matrix_specs=matrix_specs,
        )
        a_work_padded, out, native_status = lu_solve_shardmap(
            _a, b_distribution
        )
        out = restore_rhs_from_native_work(
            out,
            rhs_specs=rhs_specs,
            mesh=mesh,
            matrix_specs=matrix_specs,
        )
        return a_work_padded, out[:, :nrhs], native_status

    return impl


@lru_cache(maxsize=None)
def _lu_solve_compiled(
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
    donate: bool,
):
    """Build and cache the internally jitted public LU-solve pipeline."""
    pipeline = _lu_solve_pipeline(
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
    )

    @partial(jax.jit, donate_argnums=(0, 1) if donate else ())
    def impl(_a: Array, _b: Array) -> tuple[Array, Array, Array]:
        """Run the cached LU-solve pipeline behind the public API."""
        return pipeline(_a, _b)

    return impl
