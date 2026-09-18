"""Public cuSOLVERMp polar-decomposition wrapper.

The Python layer validates tall or square JAX arrays and constructs one of two
compiled FFI calls. The native backend converts local storage to column-major
form, redistributes the matrix into cuSOLVERMp's 2D block-cyclic layout,
computes ``A = Up @ H``, and restores the requested factors to their original
JAX sharding.
"""

from __future__ import annotations

from functools import lru_cache, partial
from typing import List, Tuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.sharding import Mesh, PartitionSpec as P

from ._cusolvermp_layout import (
    cusolvermp_grid_mapping_attr,
    infer_mesh_and_matrix_specs,
    make_local_pad_fn,
    make_local_unpad_fn,
    prepare_rectangular_matrix_layout,
    process_rank_map_from_mesh,
    standard_grid_rank_map_attr,
    status_specs,
    use_abstract_mesh_decorator,
    validate_2d_matrix_specs,
)
from ._cusolvermp_status import _CUSOLVERMP_POLAR_STATUS_SIZE
from ._layout_types import MatrixPadding2D, ProcessGrid, ProcessRankMap, TileShape
from ._setup import ensure_init_jaxmg_backend


def polar(
    a: Array,
    T_A: int,
    mesh: Mesh | None = None,
    matrix_specs: P | Tuple[P] | List[P] | None = None,
    *,
    in_specs: P | Tuple[P] | List[P] | None = None,
    compute_h: bool = True,
    return_status: bool = False,
    pad: bool = True,
    donate: bool = True,
) -> Array | tuple[Array, ...]:
    """Compute a distributed polar decomposition with cuSOLVERMp.

    For a full-rank ``m x n`` matrix with ``m >= n``, the decomposition is
    ``A = Up @ H``. ``Up`` has orthonormal columns and ``H`` is an ``n x n``
    Hermitian positive-definite matrix. Rank-deficient inputs instead produce
    a positive-semidefinite ``H`` and may not yield orthonormal columns in the
    null-space directions of ``Up``.

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
        compute_h (bool, optional): Whether to compute and return ``H``.
            Default is True. This must be a Python ``bool`` fixed while
            tracing.
        return_status (bool, optional): If True, append the native per-rank
            diagnostic status vector to the numerical outputs.
        pad (bool, optional): If True (default), add tile-aligned local capacity
            where required. If False, all participating local matrix shapes
            must already be divisible by ``T_A``.
        donate (bool, optional): If True (default), the input may be donated
            and overwritten by ``Up``. Pass False to preserve it, at the cost
            of an additional A-sized allocation.

    Returns:
        ``(Up, H)`` by default, or only ``Up`` when ``compute_h=False``. If
        ``return_status=True``, the status vector is appended to that result.

    Raises:
        TypeError: If the dtype, static mode flag, or sharding specification is
            unsupported.
        ValueError: If the matrix is wide or its shape, tile size, process
            grid, or requested output layout is incompatible with cuSOLVERMp.
    """
    mesh, matrix_specs, native_status_specs, grid, rank_map, a_padding, h_padding = (
        _prepare_polar_call(
            a,
            T_A,
            mesh,
            matrix_specs,
            in_specs=in_specs,
            compute_h=compute_h,
            pad=pad,
            caller="polar",
        )
    )
    m, n = map(int, a.shape)
    ensure_init_jaxmg_backend()
    impl = _polar_compiled(
        mesh,
        matrix_specs,
        native_status_specs,
        grid,
        rank_map,
        rank_map.cusolvermp_grid_mapping,
        a_padding,
        h_padding,
        m=m,
        n=n,
        tile_size=int(T_A),
        dtype=a.dtype,
        compute_h=compute_h,
        donate=donate,
    )
    outputs = impl(a)
    if compute_h:
        up, h, native_status = outputs
        return (up, h, native_status) if return_status else (up, h)
    up, native_status = outputs
    return (up, native_status) if return_status else up


def polar_shardmap_ctx(
    a: Array,
    T_A: int,
    mesh: Mesh | None = None,
    matrix_specs: P | Tuple[P] | List[P] | None = None,
    *,
    in_specs: P | Tuple[P] | List[P] | None = None,
    compute_h: bool = True,
    pad: bool = True,
) -> tuple[Array, ...]:
    """Compute a polar decomposition inside a caller-owned ``jax.jit``.

    This interface performs the same validation, redistribution, and native
    computation as :func:`jaxmg.polar`, but leaves the outer compilation and
    donation boundary to the caller. The first output is ``Up``, so an outer
    ``jax.jit(..., donate_argnums=(0,))`` can alias the input matrix directly
    to a numerical result.

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
        compute_h (bool, optional): Whether to compute and return ``H``.
            Default is True. This must be a Python ``bool`` fixed while
            tracing.
        pad (bool, optional): If True (default), add tile-aligned local capacity
            where required. If False, all participating local matrix shapes
            must already be divisible by ``T_A``.

    Returns:
        ``(Up, H, status)`` when ``compute_h=True`` or ``(Up, status)``
        otherwise. The status vector is always returned so an enclosing
        compiled function can propagate native diagnostics.

    Raises:
        TypeError: If the dtype, static mode flag, or sharding specification is
            unsupported.
        ValueError: If the matrix is wide or its shape, tile size, process
            grid, or requested output layout is incompatible with cuSOLVERMp.
    """
    mesh, matrix_specs, native_status_specs, grid, rank_map, a_padding, h_padding = (
        _prepare_polar_call(
            a,
            T_A,
            mesh,
            matrix_specs,
            in_specs=in_specs,
            compute_h=compute_h,
            pad=pad,
            caller="polar_shardmap_ctx",
        )
    )
    m, n = map(int, a.shape)
    ensure_init_jaxmg_backend()
    return _polar_pipeline(
        mesh,
        matrix_specs,
        native_status_specs,
        grid,
        rank_map,
        rank_map.cusolvermp_grid_mapping,
        a_padding,
        h_padding,
        m=m,
        n=n,
        tile_size=int(T_A),
        dtype=a.dtype,
        compute_h=compute_h,
    )(a)


def _prepare_polar_call(
    a: Array,
    tile_size: int,
    mesh: Mesh | None,
    matrix_specs: P | Tuple[P] | List[P] | None,
    *,
    in_specs: P | Tuple[P] | List[P] | None,
    compute_h: bool,
    pad: bool,
    caller: str,
):
    """Validate one public polar call and derive its distributed layouts."""
    if a.ndim != 2:
        raise ValueError(f"{caller} expects a rank-2 matrix A.")
    _check_supported_polar_dtype(a.dtype)
    if int(tile_size) <= 0:
        raise ValueError("T_A must be positive.")
    if not isinstance(compute_h, bool):
        raise TypeError("compute_h must be a Python bool.")
    m, n = map(int, a.shape)
    if m < n:
        raise ValueError(f"{caller} requires a tall or square matrix with m >= n.")

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
        caller=caller,
    )
    tile_shape = TileShape(rows=int(tile_size), cols=int(tile_size))
    a_padding = prepare_rectangular_matrix_layout(
        m, n, grid, tile_shape, pad=pad, caller=f"{caller}(A)"
    )
    h_padding = (
        prepare_rectangular_matrix_layout(
            n, n, grid, tile_shape, pad=pad, caller=f"{caller}(H)"
        )
        if compute_h
        else None
    )
    return (
        mesh,
        matrix_specs,
        status_specs(row_axis, col_axis, grid),
        grid,
        rank_map,
        a_padding,
        h_padding,
    )


def _check_supported_polar_dtype(dtype) -> None:
    """Validate that ``dtype`` maps to a cuSOLVERMp polar entry point."""
    if dtype not in (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128):
        raise TypeError("polar supports float32, float64, complex64, and complex128.")


_ROW_MAJOR_JAX_LAYOUT = (0, 1)


@lru_cache(maxsize=None)
def _polar_pipeline(
    mesh: Mesh,
    matrix_specs: P,
    native_status_specs: P,
    grid: ProcessGrid,
    rank_map: ProcessRankMap,
    grid_mapping: int,
    a_padding: MatrixPadding2D,
    h_padding: MatrixPadding2D | None,
    *,
    m: int,
    n: int,
    tile_size: int,
    dtype,
    compute_h: bool,
):
    """Build and cache the unjitted JAX-visible polar execution pipeline."""
    process_rows = grid.process_rows
    process_cols = grid.process_cols
    rank_array = standard_grid_rank_map_attr(
        rank_map,
        process_rows=process_rows,
        process_cols=process_cols,
        caller="cusolvermp_polar",
    )
    grid_mapping = cusolvermp_grid_mapping_attr(
        rank_map,
        grid_mapping,
        process_rows=process_rows,
        process_cols=process_cols,
        caller="cusolvermp_polar",
    )
    pad_a = make_local_pad_fn(mesh, matrix_specs, a_padding)
    unpad_up = make_local_unpad_fn(mesh, matrix_specs, a_padding)
    unpad_h = (
        make_local_unpad_fn(mesh, matrix_specs, h_padding)
        if h_padding is not None
        else None
    )
    ffi_target = "cusolvermp_polar_uh" if compute_h else "cusolvermp_polar_u"
    out_specs = (
        (matrix_specs, matrix_specs, native_status_specs)
        if compute_h
        else (matrix_specs, native_status_specs)
    )

    def polar_ffi(_a: Array):
        """Declare local FFI buffers and static rectangular metadata."""
        if _a.ndim != 2:
            raise ValueError("cusolvermp_polar expects a rank-2 matrix buffer.")
        _check_supported_polar_dtype(_a.dtype)
        output_types = [jax.ShapeDtypeStruct(_a.shape, _a.dtype)]
        output_layouts = [_ROW_MAJOR_JAX_LAYOUT]
        if compute_h:
            output_types.append(
                jax.ShapeDtypeStruct(
                    (h_padding.local_physical_rows, h_padding.local_physical_cols),
                    _a.dtype,
                )
            )
            output_layouts.append(_ROW_MAJOR_JAX_LAYOUT)
        output_types.append(
            jax.ShapeDtypeStruct((_CUSOLVERMP_POLAR_STATUS_SIZE,), jnp.int32)
        )
        output_layouts.append((0,))
        ffi_fn = partial(
            jax.ffi.ffi_call(
                ffi_target,
                tuple(output_types),
                input_layouts=(_ROW_MAJOR_JAX_LAYOUT,),
                output_layouts=tuple(output_layouts),
                input_output_aliases={0: 0},
            ),
            process_rows=process_rows,
            process_cols=process_cols,
            grid_mapping=grid_mapping,
            rank_map=rank_array,
            m=int(m),
            n=int(n),
            tile_size=int(tile_size),
        )
        return ffi_fn(_a)

    polar_shardmap = jax.shard_map(
        polar_ffi,
        mesh=mesh,
        in_specs=matrix_specs,
        out_specs=out_specs,
        check_vma=False,
    )

    @use_abstract_mesh_decorator(mesh)
    def impl(_a: Array):
        """Apply local padding, fused polar decomposition, and output slicing."""
        outputs = polar_shardmap(pad_a(_a))
        if compute_h:
            up_padded, h_padded, native_status = outputs
            return unpad_up(up_padded), unpad_h(h_padded), native_status
        up_padded, native_status = outputs
        return unpad_up(up_padded), native_status

    return impl


@lru_cache(maxsize=None)
def _polar_compiled(
    mesh: Mesh,
    matrix_specs: P,
    native_status_specs: P,
    grid: ProcessGrid,
    rank_map: ProcessRankMap,
    grid_mapping: int,
    a_padding: MatrixPadding2D,
    h_padding: MatrixPadding2D | None,
    *,
    m: int,
    n: int,
    tile_size: int,
    dtype,
    compute_h: bool,
    donate: bool,
):
    """Build and cache the internally jitted public polar pipeline."""
    pipeline = _polar_pipeline(
        mesh,
        matrix_specs,
        native_status_specs,
        grid,
        rank_map,
        grid_mapping,
        a_padding,
        h_padding,
        m=m,
        n=n,
        tile_size=tile_size,
        dtype=dtype,
        compute_h=compute_h,
    )

    @partial(jax.jit, donate_argnums=(0,) if donate else ())
    def impl(_a: Array):
        return pipeline(_a)

    return impl
