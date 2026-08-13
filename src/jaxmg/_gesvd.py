"""Public cuSOLVERMp general singular-value decomposition wrapper.

The Python layer validates rectangular JAX array metadata, determines the
logical shapes of the requested singular-vector matrices, and constructs one
of four compiled FFI calls. The native backend converts local storage to
cuSOLVERMp's column-major layout, redistributes the matrix into 2D block-cyclic
form, runs ``cusolverMpGesvd``, and restores only the requested vector outputs
to their JAX-facing layouts.
"""

from __future__ import annotations

from functools import lru_cache, partial
from typing import List, Tuple

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
    process_rank_map_from_mesh,
    standard_grid_rank_map_attr,
    status_specs,
    validate_2d_matrix_specs,
)
from ._cusolvermp_status import _CUSOLVERMP_GESVD_STATUS_SIZE
from ._layout_types import MatrixPadding2D, ProcessGrid, ProcessRankMap, TileShape
from ._layout_types import calculate_2d_padding
from ._layout_types import validate_nonempty_block_cyclic_ownership
from ._setup import ensure_init_jaxmg_backend


def gesvd(
    a: Array,
    T_A: int,
    mesh: Mesh | None = None,
    matrix_specs: P | Tuple[P] | List[P] | None = None,
    *,
    in_specs: P | Tuple[P] | List[P] | None = None,
    compute_u: bool = True,
    compute_vh: bool = True,
    full_matrices: bool = False,
    return_status: bool = False,
    pad: bool = True,
) -> Array | tuple[Array, ...]:
    """Compute a distributed singular-value decomposition with cuSOLVERMp.

    This is the high-level JAXMg interface for rectangular or square matrices.
    For an ``m x n`` input, let ``k = min(m, n)``. By default the routine
    returns the reduced decomposition ``(U, s, Vh)`` with shapes ``(m, k)``,
    ``(k,)``, and ``(k, n)``. ``full_matrices=True`` instead requests U with
    shape ``(m, m)`` and Vh with shape ``(n, n)``. Either vector matrix can be
    disabled independently so JAX does not allocate or redistribute an output
    that the application does not require.

    The input and every requested matrix output use the same JAX mesh and
    ``PartitionSpec``. Their logical dimensions must therefore be
    divisible by the corresponding process-grid dimensions. Tile padding is
    applied separately to A, U, and Vh when their local dimensions are not
    divisible by ``T_A``.

    Args:
        a (Array): A rank-2 real or complex matrix sharded over a one- or
            two-axis device mesh.
        T_A (int): Square cuSOLVERMp tile width. GESVD supports rectangular
            matrices but requires equal row and column tile dimensions.
        mesh (Mesh, optional): JAX mesh used by ``jax.shard_map``. If omitted,
            inferred from ``a.sharding.mesh``.
        matrix_specs (PartitionSpec or tuple/list[PartitionSpec], optional):
            Rank-2 matrix sharding. If omitted, inferred from
            ``a.sharding.spec``.
        in_specs: Backwards-compatible alias for ``matrix_specs``.
        compute_u (bool, optional): Whether to compute and return left singular
            vectors. Default is True. This must be a Python ``bool`` fixed
            while tracing.
        compute_vh (bool, optional): Whether to compute and return conjugate-
            transposed right singular vectors. Default is True. This must be a
            Python ``bool`` fixed while tracing.
        full_matrices (bool, optional): If False (default), return reduced
            vector matrices. If True, return full U and Vh matrices. This must
            be fixed while tracing and affects only requested vector outputs.
        return_status (bool, optional): If True, append the native per-rank
            diagnostic status vector to the selected numerical outputs.
        pad (bool, optional): If True (default), add tile-aligned local capacity
            where required. If False, all participating local matrix shapes
            must already be divisible by ``T_A``.

    Returns:
        The selected numerical outputs follow NumPy/JAX SVD order:
        ``(U, s, Vh)`` when both vectors are requested, ``(U, s)`` for U only,
        ``(s, Vh)`` for Vh only, and ``s`` for values only. If
        ``return_status=True``, the status vector is appended to that result.

    Raises:
        TypeError: If the dtype, static mode flags, or sharding specification is
            unsupported.
        ValueError: If a shape, tile size, process grid, or requested output
            layout is incompatible with cuSOLVERMp.

    Notes:
        - The internally jitted implementation donates ``a`` to the opaque
          matrix work result returned by the native call.
        - cuSOLVERMp requires A, U, and Vh to occupy distinct storage. Donation
          therefore removes a second A-sized work allocation but cannot alias
          A to either singular-vector output.
        - If the native solver fails, numerical outputs may be incomplete; use
          ``return_status=True`` when per-rank diagnostics are required.
    """
    if a.ndim != 2:
        raise ValueError("gesvd expects a rank-2 matrix A.")
    _check_supported_gesvd_dtype(a.dtype)
    if int(T_A) <= 0:
        raise ValueError("T_A must be positive.")
    for name, value in (
        ("compute_u", compute_u),
        ("compute_vh", compute_vh),
        ("full_matrices", full_matrices),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be a Python bool.")

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
        caller="gesvd",
    )
    native_status_specs = status_specs(row_axis, col_axis, grid)
    tile_shape = TileShape(rows=int(T_A), cols=int(T_A))
    m, n = map(int, a.shape)
    k = min(m, n)
    u_shape = (m, m if full_matrices else k)
    vh_shape = (n if full_matrices else k, n)

    a_padding = _prepare_gesvd_matrix_layout(
        m, n, grid, tile_shape, pad=pad, caller="gesvd(A)"
    )
    u_padding = (
        _prepare_gesvd_matrix_layout(
            *u_shape, grid, tile_shape, pad=pad, caller="gesvd(U)"
        )
        if compute_u
        else None
    )
    vh_padding = (
        _prepare_gesvd_matrix_layout(
            *vh_shape, grid, tile_shape, pad=pad, caller="gesvd(Vh)"
        )
        if compute_vh
        else None
    )

    ensure_init_jaxmg_backend()
    impl = _gesvd_compiled(
        mesh,
        matrix_specs,
        native_status_specs,
        grid,
        rank_map,
        rank_map.cusolvermp_grid_mapping,
        a_padding,
        u_padding,
        vh_padding,
        m=m,
        n=n,
        tile_size=tile_shape.rows,
        dtype=a.dtype,
        compute_u=compute_u,
        compute_vh=compute_vh,
        full_matrices=full_matrices,
    )
    outputs = impl(a)
    if compute_u and compute_vh:
        singular_values, _, u, vh, native_status = outputs
        result = (u, singular_values, vh)
    elif compute_u:
        singular_values, _, u, native_status = outputs
        result = (u, singular_values)
    elif compute_vh:
        singular_values, _, vh, native_status = outputs
        result = (singular_values, vh)
    else:
        singular_values, _, native_status = outputs
        if return_status:
            return singular_values, native_status
        return singular_values
    if return_status:
        return (*result, native_status)
    return result


def gesvd_shardmap_ctx(
    a: Array,
    T_A: int,
    mesh: Mesh | None = None,
    matrix_specs: P | Tuple[P] | List[P] | None = None,
    *,
    in_specs: P | Tuple[P] | List[P] | None = None,
    compute_u: bool = True,
    compute_vh: bool = True,
    full_matrices: bool = False,
    pad: bool = True,
) -> tuple[Array, ...]:
    """Compute a distributed SVD while exposing donated matrix work storage.

    This lower-level interface performs the same validation, padding, native
    redistribution, and cuSOLVERMp execution as :func:`jaxmg.gesvd`, but leaves
    the outer ``jax.jit`` boundary to the caller. The first return value is the
    opaque ``a_work`` buffer overwritten by GESVD. Keeping that value in the
    outer function's returned pytree allows ``jax.jit(...,
    donate_argnums=(0,))`` to alias the input matrix to an A-sized output.

    Singular values and requested vector matrices follow the same shapes and
    selection rules as :func:`jaxmg.gesvd`. Unlike POTRS and LU solve, the
    factorized A buffer is not a numerical result. cuSOLVERMp also prohibits A,
    U, and Vh from overlapping, so requested vector matrices remain separate
    allocations even when A is donated.

    Args:
        a (Array): A rank-2 real or complex matrix sharded over a one- or
            two-axis device mesh.
        T_A (int): Square cuSOLVERMp tile width.
        mesh (Mesh, optional): JAX mesh used by ``jax.shard_map``. If omitted,
            inferred from ``a.sharding.mesh``.
        matrix_specs (PartitionSpec or tuple/list[PartitionSpec], optional):
            Rank-2 matrix sharding. If omitted, inferred from
            ``a.sharding.spec``.
        in_specs: Backwards-compatible alias for ``matrix_specs``.
        compute_u (bool, optional): Whether to compute left singular vectors.
        compute_vh (bool, optional): Whether to compute right singular vectors
            in conjugate-transposed form.
        full_matrices (bool, optional): Whether requested vector outputs use
            full rather than reduced shapes.
        pad (bool, optional): Whether JAXMg may add tile-aligned local capacity.

    Returns:
        ``(a_work, U, s, Vh, status)``, ``(a_work, U, s, status)``,
        ``(a_work, s, Vh, status)``, or ``(a_work, s, status)`` according to the
        selected vector outputs. ``status`` is always returned by the context
        interface so an enclosing compiled function can propagate diagnostics.

    Raises:
        TypeError: If the dtype, static mode flags, or sharding specification is
            unsupported.
        ValueError: If a shape, tile size, process grid, or requested output
            layout is incompatible with cuSOLVERMp.
    """
    if a.ndim != 2:
        raise ValueError("gesvd_shardmap_ctx expects a rank-2 matrix A.")
    _check_supported_gesvd_dtype(a.dtype)
    if int(T_A) <= 0:
        raise ValueError("T_A must be positive.")
    for name, value in (
        ("compute_u", compute_u),
        ("compute_vh", compute_vh),
        ("full_matrices", full_matrices),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be a Python bool.")

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
        caller="gesvd_shardmap_ctx",
    )
    native_status_specs = status_specs(row_axis, col_axis, grid)
    tile_shape = TileShape(rows=int(T_A), cols=int(T_A))
    m, n = map(int, a.shape)
    k = min(m, n)
    u_shape = (m, m if full_matrices else k)
    vh_shape = (n if full_matrices else k, n)

    a_padding = _prepare_gesvd_matrix_layout(
        m, n, grid, tile_shape, pad=pad, caller="gesvd_shardmap_ctx(A)"
    )
    u_padding = (
        _prepare_gesvd_matrix_layout(
            *u_shape,
            grid,
            tile_shape,
            pad=pad,
            caller="gesvd_shardmap_ctx(U)",
        )
        if compute_u
        else None
    )
    vh_padding = (
        _prepare_gesvd_matrix_layout(
            *vh_shape,
            grid,
            tile_shape,
            pad=pad,
            caller="gesvd_shardmap_ctx(Vh)",
        )
        if compute_vh
        else None
    )

    ensure_init_jaxmg_backend()
    impl = _gesvd_pipeline(
        mesh,
        matrix_specs,
        native_status_specs,
        grid,
        rank_map,
        rank_map.cusolvermp_grid_mapping,
        a_padding,
        u_padding,
        vh_padding,
        m=m,
        n=n,
        tile_size=tile_shape.rows,
        dtype=a.dtype,
        compute_u=compute_u,
        compute_vh=compute_vh,
        full_matrices=full_matrices,
    )
    outputs = impl(a)
    if compute_u and compute_vh:
        singular_values, a_work, u, vh, native_status = outputs
        return a_work, u, singular_values, vh, native_status
    if compute_u:
        singular_values, a_work, u, native_status = outputs
        return a_work, u, singular_values, native_status
    if compute_vh:
        singular_values, a_work, vh, native_status = outputs
        return a_work, singular_values, vh, native_status
    singular_values, a_work, native_status = outputs
    return a_work, singular_values, native_status


_ROW_MAJOR_JAX_LAYOUT = (0, 1)


def _check_supported_gesvd_dtype(dtype) -> None:
    """Validate that ``dtype`` maps to a cuSOLVERMp GESVD entry point.

    cuSOLVERMp provides real and complex singular-value decompositions in
    single and double precision. Unsupported dtypes are rejected before JAX
    tracing reaches the native FFI boundary.
    """
    if dtype not in (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128):
        raise TypeError("gesvd supports float32, float64, complex64, and complex128.")


def _real_dtype_for_singular_values(dtype):
    """Return the real component dtype used for singular values."""
    if dtype == jnp.float32 or dtype == jnp.complex64:
        return jnp.float32
    if dtype == jnp.float64 or dtype == jnp.complex128:
        return jnp.float64
    raise TypeError(
        "cuSOLVERMp GESVD supports float32, float64, complex64, and complex128."
    )


def _prepare_gesvd_matrix_layout(
    logical_rows: int,
    logical_cols: int,
    grid: ProcessGrid,
    tile_shape: TileShape,
    *,
    pad: bool,
    caller: str,
) -> MatrixPadding2D:
    """Validate one GESVD matrix and return its uniform local capacity.

    A and each requested vector output participate independently in the 2D
    block-cyclic workflow. This helper enforces non-empty tile ownership,
    checks that the logical shape can be represented by the JAX block sharding,
    and applies the caller's padding policy consistently to all matrices.
    """
    validate_nonempty_block_cyclic_ownership(
        logical_rows=logical_rows,
        logical_cols=logical_cols,
        grid=grid,
        tile_shape=tile_shape,
        caller=caller,
    )
    try:
        padding = calculate_2d_padding(
            logical_rows=logical_rows,
            logical_cols=logical_cols,
            grid=grid,
            tile_shape=tile_shape,
        )
    except ValueError as exc:
        raise ValueError(
            f"{caller} shape ({logical_rows}, {logical_cols}) must be divisible "
            f"by process grid ({grid.process_rows}, {grid.process_cols}) "
            "before local tile padding."
        ) from exc
    if not pad and padding.needs_padding:
        raise ValueError(
            f"{caller} requires tile-aligned local shards when pad=False. "
            "Use a tile size that divides both local dimensions or set pad=True."
        )
    return padding


def _make_local_pad_fn(mesh: Mesh, matrix_specs: P, padding: MatrixPadding2D):
    """Build the shard-local bottom/right padding transform for A."""
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
    padding: MatrixPadding2D,
):
    """Build the shard-local slice that restores a requested vector shape."""
    return jax.shard_map(
        partial(
            _unpad_local_2d,
            local_rows=padding.local_logical_rows,
            local_cols=padding.local_logical_cols,
        ),
        mesh=mesh,
        in_specs=matrix_specs,
        out_specs=matrix_specs,
        check_vma=True,
    )


@lru_cache(maxsize=None)
def _gesvd_pipeline(
    mesh: Mesh,
    matrix_specs: P,
    native_status_specs: P,
    grid: ProcessGrid,
    rank_map: ProcessRankMap,
    grid_mapping: int,
    a_padding: MatrixPadding2D,
    u_padding: MatrixPadding2D | None,
    vh_padding: MatrixPadding2D | None,
    *,
    m: int,
    n: int,
    tile_size: int,
    dtype,
    compute_u: bool,
    compute_vh: bool,
    full_matrices: bool,
):
    """Build and cache the unjitted JAX-visible GESVD execution pipeline.

    The cache key contains all static shape, mesh, rank-map, dtype, and output
    mode metadata. Repeated calls with the same distributed configuration reuse
    the same shard-map and FFI wrapper construction.
    """
    process_rows = grid.process_rows
    process_cols = grid.process_cols
    rank_array = standard_grid_rank_map_attr(
        rank_map,
        process_rows=process_rows,
        process_cols=process_cols,
        caller="cusolvermp_gesvd",
    )
    grid_mapping = cusolvermp_grid_mapping_attr(
        rank_map,
        grid_mapping,
        process_rows=process_rows,
        process_cols=process_cols,
        caller="cusolvermp_gesvd",
    )
    pad_a = _make_local_pad_fn(mesh, matrix_specs, a_padding)
    unpad_u = (
        _make_local_unpad_fn(mesh, matrix_specs, u_padding)
        if u_padding is not None
        else None
    )
    unpad_vh = (
        _make_local_unpad_fn(mesh, matrix_specs, vh_padding)
        if vh_padding is not None
        else None
    )

    if compute_u and compute_vh:
        ffi_target = "cusolvermp_gesvd_uv"
        out_specs = (
            P(None),
            matrix_specs,
            matrix_specs,
            matrix_specs,
            native_status_specs,
        )
    elif compute_u:
        ffi_target = "cusolvermp_gesvd_u"
        out_specs = (P(None), matrix_specs, matrix_specs, native_status_specs)
    elif compute_vh:
        ffi_target = "cusolvermp_gesvd_vh"
        out_specs = (P(None), matrix_specs, matrix_specs, native_status_specs)
    else:
        ffi_target = "cusolvermp_gesvd_values"
        out_specs = (P(None), matrix_specs, native_status_specs)

    def gesvd_ffi(_a: Array):
        """Declare the local FFI buffers and static rectangular metadata."""
        if _a.ndim != 2:
            raise ValueError("cusolvermp_gesvd expects a rank-2 matrix buffer.")
        _check_supported_gesvd_dtype(_a.dtype)

        singular_type = jax.ShapeDtypeStruct(
            (min(int(m), int(n)),), _real_dtype_for_singular_values(dtype)
        )
        work_type = jax.ShapeDtypeStruct(_a.shape, _a.dtype)
        status_type = jax.ShapeDtypeStruct(
            (_CUSOLVERMP_GESVD_STATUS_SIZE,), jnp.int32
        )
        output_types = [singular_type, work_type]
        output_layouts = [(0,), _ROW_MAJOR_JAX_LAYOUT]
        if compute_u:
            output_types.append(
                jax.ShapeDtypeStruct(
                    (u_padding.local_physical_rows, u_padding.local_physical_cols),
                    _a.dtype,
                )
            )
            output_layouts.append(_ROW_MAJOR_JAX_LAYOUT)
        if compute_vh:
            output_types.append(
                jax.ShapeDtypeStruct(
                    (
                        vh_padding.local_physical_rows,
                        vh_padding.local_physical_cols,
                    ),
                    _a.dtype,
                )
            )
            output_layouts.append(_ROW_MAJOR_JAX_LAYOUT)
        output_types.append(status_type)
        output_layouts.append((0,))

        ffi_fn = partial(
            jax.ffi.ffi_call(
                ffi_target,
                tuple(output_types),
                input_layouts=(_ROW_MAJOR_JAX_LAYOUT,),
                output_layouts=tuple(output_layouts),
                input_output_aliases={0: 1},
            ),
            process_rows=process_rows,
            process_cols=process_cols,
            grid_mapping=grid_mapping,
            rank_map=rank_array,
            m=int(m),
            n=int(n),
            tile_size=int(tile_size),
            full_matrices=int(full_matrices),
        )
        return ffi_fn(_a)

    gesvd_shardmap = jax.shard_map(
        gesvd_ffi,
        mesh=mesh,
        in_specs=matrix_specs,
        out_specs=out_specs,
        check_vma=False,
    )

    @use_abstract_mesh_decorator(mesh)
    def impl(_a: Array):
        """Apply local padding, fused GESVD, and requested output slicing."""
        outputs = gesvd_shardmap(pad_a(_a))
        if compute_u and compute_vh:
            singular_values, a_work, u_padded, vh_padded, native_status = outputs
            return (
                singular_values,
                a_work,
                unpad_u(u_padded),
                unpad_vh(vh_padded),
                native_status,
            )
        if compute_u:
            singular_values, a_work, u_padded, native_status = outputs
            return singular_values, a_work, unpad_u(u_padded), native_status
        if compute_vh:
            singular_values, a_work, vh_padded, native_status = outputs
            return singular_values, a_work, unpad_vh(vh_padded), native_status
        return outputs

    return impl


@lru_cache(maxsize=None)
def _gesvd_compiled(
    mesh: Mesh,
    matrix_specs: P,
    native_status_specs: P,
    grid: ProcessGrid,
    rank_map: ProcessRankMap,
    grid_mapping: int,
    a_padding: MatrixPadding2D,
    u_padding: MatrixPadding2D | None,
    vh_padding: MatrixPadding2D | None,
    *,
    m: int,
    n: int,
    tile_size: int,
    dtype,
    compute_u: bool,
    compute_vh: bool,
    full_matrices: bool,
):
    """Build and cache the internally jitted public GESVD pipeline."""
    pipeline = _gesvd_pipeline(
        mesh,
        matrix_specs,
        native_status_specs,
        grid,
        rank_map,
        grid_mapping,
        a_padding,
        u_padding,
        vh_padding,
        m=m,
        n=n,
        tile_size=tile_size,
        dtype=dtype,
        compute_u=compute_u,
        compute_vh=compute_vh,
        full_matrices=full_matrices,
    )

    @partial(jax.jit, donate_argnums=(0,))
    def impl(_a: Array):
        """Run the cached GESVD pipeline behind the public convenience API."""
        return pipeline(_a)

    return impl
