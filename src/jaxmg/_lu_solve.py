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
from jax import Array
from jax.sharding import Mesh, PartitionSpec as P

from ._cusolvermp_layout import (
    _pad_local_2d,
    _unpad_local_2d,
    cusolvermp_grid_mapping_attr,
    infer_mesh_and_matrix_specs,
    use_abstract_mesh_decorator,
    place_rhs_for_native_work,
    process_rank_map_from_mesh,
    restore_rhs_from_native_work,
    rhs_distribution_columns,
    standard_grid_rank_map_attr,
    status_specs,
    validate_2d_matrix_specs,
)
from ._cusolvermp_status import _CUSOLVERMP_LU_SOLVE_STATUS_SIZE
from ._layout_types import MatrixPadding2D, ProcessGrid, ProcessRankMap, TileShape
from ._layout_types import calculate_2d_padding
from ._layout_types import validate_nonempty_block_cyclic_ownership
from ._setup import ensure_init_jaxmg_backend


def lu_solve(
    a: Array,
    b: Array,
    T_A: int,
    mesh: Mesh | None = None,
    matrix_specs: P | Tuple[P] | List[P] | None = None,
    *,
    in_specs: P | Tuple[P] | List[P] | None = None,
    return_status: bool = False,
    pad: bool = True,
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
        mesh (Mesh, optional): JAX mesh used for ``jax.shard_map``. If omitted,
            inferred from ``a.sharding.mesh``.
        matrix_specs (PartitionSpec or tuple/list[PartitionSpec], optional):
            PartitionSpec describing the matrix sharding. If omitted, inferred
            from ``a.sharding.spec``.
        in_specs: Backwards-compatible alias for ``matrix_specs``.
        return_status (bool, optional): If True return ``(x, status)`` where
            ``status`` is the native per-rank diagnostic vector. If False
            return ``x`` only. Default is False.
        pad (bool, optional): If True (default) apply per-device padding so
            each local shard length is compatible with ``T_A``; if False the
            caller must ensure shapes already match the kernel's requirements.

    Returns:
        Array or (Array, Array): The solution ``x`` in the same JAX-facing
            block-sharded layout as ``b``. If ``return_status=True`` also
            return the native solver status.

    Raises:
        TypeError: If dtypes or ``PartitionSpec`` inputs are unsupported.
        ValueError: If shapes, tile sizes, or mesh layouts are incompatible.

    Notes:
        - The FFI call may donate the ``a`` and ``b`` buffers for zero-copy
          interaction with the native library.
        - Native code converts row-major JAX local storage to cuSOLVERMp's
          column-major local layout, redistributes to 2D block-cyclic layout,
          calls ``cusolverMpGetrf``/``cusolverMpGetrs``, and redistributes the
          result back.
        - If the native solver fails the returned solution may contain NaNs
          and ``status`` will be non-zero.
    """
    if a.ndim != 2:
        raise ValueError("lu_solve expects a rank-2 matrix A.")
    vector_rhs = b.ndim == 1
    if vector_rhs:
        b = jnp.expand_dims(b, axis=1)
    if b.ndim != 2:
        raise ValueError("lu_solve expects a rank-1 or rank-2 RHS B.")
    if a.dtype != b.dtype:
        raise TypeError("lu_solve requires matching A/B dtypes.")
    _check_supported_lu_solve_dtype(a.dtype)
    if a.shape[0] != a.shape[1]:
        raise ValueError("lu_solve expects A to be square.")
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
        caller="lu_solve",
    )
    native_status_specs = status_specs(row_axis, col_axis, grid)
    tile_shape = TileShape(rows=int(T_A), cols=int(T_A))
    validate_nonempty_block_cyclic_ownership(
        logical_rows=a.shape[0],
        logical_cols=a.shape[1],
        grid=grid,
        tile_shape=tile_shape,
        caller="lu_solve(A)",
    )
    nrhs = int(b.shape[1])
    b_distribution_cols = rhs_distribution_columns(
        nrhs,
        process_cols=grid.process_cols,
        pad=pad,
    )

    a_padding = calculate_2d_padding(
        logical_rows=a.shape[0],
        logical_cols=a.shape[1],
        grid=grid,
        tile_shape=tile_shape,
    )
    b_padding = calculate_2d_padding(
        logical_rows=b.shape[0],
        logical_cols=b_distribution_cols,
        grid=grid,
        tile_shape=tile_shape,
    )
    _check_padding_allowed(a_padding, pad=pad, caller="lu_solve(A)")
    _check_padding_allowed(b_padding, pad=pad, caller="lu_solve(B)")

    ensure_init_jaxmg_backend()

    impl = _lu_solve_compiled(
        mesh,
        matrix_specs,
        native_status_specs,
        grid,
        rank_map,
        rank_map.cusolvermp_grid_mapping,
        a_padding,
        b_padding,
        n=a.shape[0],
        nrhs=nrhs,
        b_distribution_cols=b_distribution_cols,
        tile_size=tile_shape.rows,
    )
    _, out, native_status = impl(a, b)
    if vector_rhs:
        out = out[:, 0]
    if return_status:
        return out, native_status
    return out


def lu_solve_shardmap_ctx(
    a: Array,
    b: Array,
    T_A: int,
    mesh: Mesh | None = None,
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
        mesh (Mesh, optional): JAX mesh used for ``jax.shard_map``. If omitted,
            inferred from ``a.sharding.mesh``.
        matrix_specs (PartitionSpec or tuple/list[PartitionSpec], optional):
            PartitionSpec describing the matrix sharding. If omitted, inferred
            from ``a.sharding.spec``.
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
    if a.ndim != 2:
        raise ValueError("lu_solve_shardmap_ctx expects a rank-2 matrix A.")
    vector_rhs = b.ndim == 1
    if vector_rhs:
        b = jnp.expand_dims(b, axis=1)
    if b.ndim != 2:
        raise ValueError("lu_solve_shardmap_ctx expects a rank-1 or rank-2 RHS B.")
    if a.dtype != b.dtype:
        raise TypeError("lu_solve_shardmap_ctx requires matching A/B dtypes.")
    _check_supported_lu_solve_dtype(a.dtype)
    if a.shape[0] != a.shape[1]:
        raise ValueError("lu_solve_shardmap_ctx expects A to be square.")
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
        caller="lu_solve_shardmap_ctx",
    )
    native_status_specs = status_specs(row_axis, col_axis, grid)
    tile_shape = TileShape(rows=int(T_A), cols=int(T_A))
    validate_nonempty_block_cyclic_ownership(
        logical_rows=a.shape[0],
        logical_cols=a.shape[1],
        grid=grid,
        tile_shape=tile_shape,
        caller="lu_solve_shardmap_ctx(A)",
    )
    nrhs = int(b.shape[1])
    b_distribution_cols = rhs_distribution_columns(
        nrhs,
        process_cols=grid.process_cols,
        pad=pad,
    )

    a_padding = calculate_2d_padding(
        logical_rows=a.shape[0],
        logical_cols=a.shape[1],
        grid=grid,
        tile_shape=tile_shape,
    )
    b_padding = calculate_2d_padding(
        logical_rows=b.shape[0],
        logical_cols=b_distribution_cols,
        grid=grid,
        tile_shape=tile_shape,
    )
    _check_padding_allowed(a_padding, pad=pad, caller="lu_solve_shardmap_ctx(A)")
    _check_padding_allowed(b_padding, pad=pad, caller="lu_solve_shardmap_ctx(B)")

    ensure_init_jaxmg_backend()

    impl = _lu_solve_pipeline(
        mesh,
        matrix_specs,
        native_status_specs,
        grid,
        rank_map,
        rank_map.cusolvermp_grid_mapping,
        a_padding,
        b_padding,
        n=a.shape[0],
        nrhs=nrhs,
        b_distribution_cols=b_distribution_cols,
        tile_size=tile_shape.rows,
    )
    a_work, out, native_status = impl(a, b)
    if vector_rhs:
        out = out[:, 0]
    return a_work, out, native_status


_ROW_MAJOR_JAX_LAYOUT = (0, 1)


def _check_supported_lu_solve_dtype(dtype) -> None:
    """Validate that ``dtype`` maps to cuSOLVERMp GETRF/GETRS entry points."""
    if dtype not in (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128):
        raise TypeError(
            "lu_solve supports float32, float64, complex64, and complex128."
        )


def _check_padding_allowed(
    padding: MatrixPadding2D,
    *,
    pad: bool,
    caller: str,
) -> None:
    """Enforce the public ``pad`` policy for a padded matrix-like argument."""
    if not pad and padding.needs_padding:
        raise ValueError(
            f"{caller} requires tile-aligned local shards when pad=False. "
            "Use a tile size that divides each local shard or set pad=True."
        )


def _make_local_pad_fn(mesh: Mesh, matrix_specs: P, padding: MatrixPadding2D):
    """Build the shard-local bottom/right padding transform."""
    if not padding.needs_padding:
        return lambda block: block
    return jax.shard_map(
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


def _make_local_unpad_fn(
    mesh: Mesh,
    matrix_specs: P,
    *,
    local_rows: int,
    local_cols: int,
):
    """Build the shard-local slice transform that removes visible padding."""
    return jax.shard_map(
        partial(_unpad_local_2d, local_rows=local_rows, local_cols=local_cols),
        mesh=mesh,
        in_specs=matrix_specs,
        out_specs=matrix_specs,
        check_vma=True,
    )


@lru_cache(maxsize=None)
def _lu_solve_pipeline(
    mesh: Mesh,
    matrix_specs: P,
    native_status_specs: P,
    grid: ProcessGrid,
    rank_map: ProcessRankMap,
    grid_mapping: int,
    a_padding: MatrixPadding2D,
    b_padding: MatrixPadding2D,
    *,
    n: int,
    nrhs: int,
    b_distribution_cols: int,
    tile_size: int,
):
    """Build and cache the unjitted JAX-visible LU-solve execution pipeline."""
    process_rows = grid.process_rows
    process_cols = grid.process_cols
    rank_array = standard_grid_rank_map_attr(
        rank_map,
        process_rows=process_rows,
        process_cols=process_cols,
        caller="cusolvermp_lu_solve",
    )
    grid_mapping = cusolvermp_grid_mapping_attr(
        rank_map,
        grid_mapping,
        process_rows=process_rows,
        process_cols=process_cols,
        caller="cusolvermp_lu_solve",
    )
    b_distribution_padding = int(b_distribution_cols) - int(nrhs)
    pad_a = _make_local_pad_fn(mesh, matrix_specs, a_padding)
    pad_b = _make_local_pad_fn(mesh, matrix_specs, b_padding)
    unpad_b = _make_local_unpad_fn(
        mesh,
        matrix_specs,
        local_rows=b_padding.local_logical_rows,
        local_cols=b_padding.local_logical_cols,
    )

    def lu_solve_ffi(_a: Array, _b: Array) -> tuple[Array, Array, Array]:
        """Call fused native redistribution and ``getrf/getrs`` on one shard."""
        if _a.ndim != 2 or _b.ndim != 2:
            raise ValueError("cusolvermp_lu_solve expects rank-2 A and B buffers.")
        if _a.dtype != _b.dtype:
            raise TypeError("cusolvermp_lu_solve requires matching A/B dtypes.")
        _check_supported_lu_solve_dtype(_a.dtype)
        if _a.shape[0] != _b.shape[0]:
            raise ValueError("A and B must have matching local row capacity.")

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
            grid_mapping=grid_mapping,
            rank_map=rank_array,
            n=int(n),
            nrhs=int(nrhs),
            b_distribution_cols=int(b_distribution_cols),
            tile_size=int(tile_size),
        )
        a_work, b_out, status = ffi_fn(_a, _b)
        return a_work, b_out, status

    lu_solve_shardmap = jax.shard_map(
        lu_solve_ffi,
        mesh=mesh,
        in_specs=(matrix_specs, matrix_specs),
        out_specs=(matrix_specs, matrix_specs, native_status_specs),
        check_vma=False,
    )

    @use_abstract_mesh_decorator(mesh)
    def impl(_a: Array, _b: Array) -> tuple[Array, Array, Array]:
        """Run padding, fused native LU solve, and unpadding."""
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
        a_work_padded, b_solved_padded, native_status = lu_solve_shardmap(
            a_padded, b_padded
        )
        out = unpad_b(b_solved_padded)
        out = restore_rhs_from_native_work(
            out,
            reference_rhs=_b,
            mesh=mesh,
            matrix_specs=matrix_specs,
        )
        return a_work_padded, out[:, :nrhs], native_status

    return impl


@lru_cache(maxsize=None)
def _lu_solve_compiled(
    mesh: Mesh,
    matrix_specs: P,
    native_status_specs: P,
    grid: ProcessGrid,
    rank_map: ProcessRankMap,
    grid_mapping: int,
    a_padding: MatrixPadding2D,
    b_padding: MatrixPadding2D,
    *,
    n: int,
    nrhs: int,
    b_distribution_cols: int,
    tile_size: int,
):
    """Build and cache the internally jitted public LU-solve pipeline."""
    pipeline = _lu_solve_pipeline(
        mesh,
        matrix_specs,
        native_status_specs,
        grid,
        rank_map,
        grid_mapping,
        a_padding,
        b_padding,
        n=n,
        nrhs=nrhs,
        b_distribution_cols=b_distribution_cols,
        tile_size=tile_size,
    )

    @partial(jax.jit, donate_argnums=(0, 1))
    def impl(_a: Array, _b: Array) -> tuple[Array, Array, Array]:
        """Run the cached LU-solve pipeline behind the public API."""
        return pipeline(_a, _b)

    return impl
