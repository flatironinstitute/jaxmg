"""Public cuSOLVERMp least-squares wrapper.

The Python layer validates rectangular JAX array metadata, applies per-shard
tile padding, and constructs the compiled FFI call. The fused native backend
then converts local memory layouts, redistributes ``A`` and ``B`` into
cuSOLVERMp's 2D block-cyclic layout, runs ``cusolverMpGels``, and restores the
solution to its JAX-facing layout.
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
    infer_rhs_specs,
    mesh_axis_size,
    place_rhs_for_native_work,
    process_rank_map_from_mesh,
    restore_rhs_from_native_work,
    rhs_distribution_columns,
    standard_grid_rank_map_attr,
    status_specs,
    use_abstract_mesh_decorator,
    validate_2d_matrix_specs,
)
from ._cusolvermp_status import _CUSOLVERMP_LEAST_SQUARES_STATUS_SIZE
from ._layout_types import MatrixPadding2D, ProcessGrid, ProcessRankMap, TileShape
from ._layout_types import calculate_2d_padding
from ._layout_types import validate_nonempty_block_cyclic_ownership
from ._setup import ensure_init_jaxmg_backend


def least_squares(
    a: Array,
    b: Array,
    T_A: int,
    mesh: Mesh | None = None,
    matrix_specs: P | Tuple[P] | List[P] | None = None,
    *,
    in_specs: P | Tuple[P] | List[P] | None = None,
    return_status: bool = False,
    pad: bool = True,
    donate: bool = True,
) -> Union[Array, Tuple[Array, Array]]:
    """Solve ``min_X ||A X - B||_2`` with distributed QR factorization.

    The current implementation supports overdetermined or square systems with
    ``M >= N``. For ``A`` of shape ``(M, N)`` and ``B`` of shape ``(M, K)``,
    the returned solution has shape ``(N, K)``. A rank-1 ``B`` is accepted and
    produces a rank-1 solution. Every process-grid column must own at least one
    block-cyclic tile of ``B``.

    Args:
        a (Array): Rank-2 input matrix sharded over a one- or two-axis device
            mesh.
        b (Array): Rank-1 or rank-2 solve input with ``M`` rows.
        T_A (int): Square cuSOLVERMp tile width.
        mesh (Mesh, optional): JAX mesh used by ``jax.shard_map``. If omitted,
            inferred from ``a.sharding.mesh``.
        matrix_specs (PartitionSpec or tuple/list[PartitionSpec], optional):
            Rank-2 matrix sharding. If omitted, inferred from
            ``a.sharding.spec``.
        in_specs: Backwards-compatible alias for ``matrix_specs``.
        return_status (bool, optional): If True return ``(x, status)``.
        pad (bool, optional): If True (default), add tile-aligned local
            capacity where required.
        donate (bool, optional): If True (default), the input buffers may be
            reused by the native call and must not be used afterwards. Pass
            False to preserve them at the cost of additional memory.

    Returns:
        Array or (Array, Array): The least-squares solution, and optionally the
        native per-rank diagnostic status.

    Raises:
        TypeError: If dtypes or sharding specifications are unsupported.
        ValueError: If shapes, tile sizes, or mesh layouts are incompatible.
    """
    vector_rhs = _validate_least_squares_inputs(a, b, T_A, "least_squares")
    if vector_rhs:
        b = jnp.expand_dims(b, axis=1)

    (
        mesh,
        matrix_specs,
        rhs_specs,
        native_status_specs,
        grid,
        rank_map,
        a_padding,
        b_padding,
        b_distribution_cols,
    ) = _prepare_least_squares_layout(
        a,
        b,
        T_A,
        mesh=mesh,
        matrix_specs=matrix_specs,
        in_specs=in_specs,
        pad=pad,
        caller="least_squares",
    )

    ensure_init_jaxmg_backend()
    impl = _least_squares_compiled(
        mesh,
        matrix_specs,
        native_status_specs,
        grid,
        rank_map,
        rank_map.cusolvermp_grid_mapping,
        a_padding,
        b_padding,
        rhs_specs,
        m=int(a.shape[0]),
        n=int(a.shape[1]),
        nrhs=int(b.shape[1]),
        b_distribution_cols=b_distribution_cols,
        tile_size=int(T_A),
        donate=donate,
    )
    _, _, out, native_status = impl(a, b)
    if vector_rhs:
        out = out[:, 0]
    if return_status:
        return out, native_status
    return out


def least_squares_shardmap_ctx(
    a: Array,
    b: Array,
    T_A: int,
    mesh: Mesh | None = None,
    matrix_specs: P | Tuple[P] | List[P] | None = None,
    *,
    in_specs: P | Tuple[P] | List[P] | None = None,
    pad: bool = True,
) -> Tuple[Array, Array, Array, Array]:
    """Solve a least-squares system inside a caller-owned ``jax.jit``.

    This lower-level interface performs the same work as
    :func:`jaxmg.least_squares` but leaves the outer JIT boundary to the caller.
    It returns ``a_work`` and ``b_work`` so an outer compiled function can
    donate both inputs into shape-compatible native work outputs.

    Args:
        a (Array): Rank-2 input matrix sharded over a one- or two-axis device
            mesh.
        b (Array): Rank-1 or rank-2 solve input with ``M`` rows.
        T_A (int): Square cuSOLVERMp tile width.
        mesh (Mesh, optional): JAX mesh used by ``jax.shard_map``. If omitted,
            inferred from ``a.sharding.mesh``.
        matrix_specs (PartitionSpec or tuple/list[PartitionSpec], optional):
            Rank-2 matrix sharding. If omitted, inferred from
            ``a.sharding.spec``.
        in_specs: Backwards-compatible alias for ``matrix_specs``.
        pad (bool, optional): If True (default), add tile-aligned local
            capacity where required.

    Returns:
        tuple: ``(a_work, b_work, x, status)`` containing the opaque matrix and
        solve-input work buffers, least-squares solution, and native per-rank
        status. The first ``N`` rows of ``b_work`` contain ``x``.

    Raises:
        TypeError: If dtypes or sharding specifications are unsupported.
        ValueError: If shapes, tile sizes, or mesh layouts are incompatible.
    """
    vector_rhs = _validate_least_squares_inputs(
        a, b, T_A, "least_squares_shardmap_ctx"
    )
    if vector_rhs:
        b = jnp.expand_dims(b, axis=1)

    (
        mesh,
        matrix_specs,
        rhs_specs,
        native_status_specs,
        grid,
        rank_map,
        a_padding,
        b_padding,
        b_distribution_cols,
    ) = _prepare_least_squares_layout(
        a,
        b,
        T_A,
        mesh=mesh,
        matrix_specs=matrix_specs,
        in_specs=in_specs,
        pad=pad,
        caller="least_squares_shardmap_ctx",
    )

    ensure_init_jaxmg_backend()
    impl = _least_squares_pipeline(
        mesh,
        matrix_specs,
        native_status_specs,
        grid,
        rank_map,
        rank_map.cusolvermp_grid_mapping,
        a_padding,
        b_padding,
        rhs_specs,
        m=int(a.shape[0]),
        n=int(a.shape[1]),
        nrhs=int(b.shape[1]),
        b_distribution_cols=b_distribution_cols,
        tile_size=int(T_A),
    )
    a_work, b_work, out, native_status = impl(a, b)
    if vector_rhs:
        b_work = b_work[:, 0]
        out = out[:, 0]
    return a_work, b_work, out, native_status


_ROW_MAJOR_JAX_LAYOUT = (0, 1)


def _validate_least_squares_inputs(
    a: Array, b: Array, tile_size: int, caller: str
) -> bool:
    """Validate the public least-squares array contract."""
    if a.ndim != 2:
        raise ValueError(f"{caller} expects a rank-2 matrix A.")
    if b.ndim not in (1, 2):
        raise ValueError(f"{caller} expects a rank-1 or rank-2 solve input B.")
    if a.dtype != b.dtype:
        raise TypeError(f"{caller} requires matching A/B dtypes.")
    if a.dtype not in (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128):
        raise TypeError(
            "least_squares supports float32, float64, complex64, and complex128."
        )
    if a.shape[0] != b.shape[0]:
        raise ValueError("A and B must have matching leading dimensions.")
    if a.shape[0] < a.shape[1]:
        raise ValueError("least_squares currently requires M >= N.")
    if int(tile_size) <= 0:
        raise ValueError("T_A must be positive.")
    return b.ndim == 1


def _prepare_least_squares_layout(
    a: Array,
    b: Array,
    tile_size: int,
    *,
    mesh: Mesh | None,
    matrix_specs: P | Tuple[P] | List[P] | None,
    in_specs: P | Tuple[P] | List[P] | None,
    pad: bool,
    caller: str,
):
    """Derive the common matrix, RHS, padding, and process-grid metadata."""
    mesh, matrix_specs = infer_mesh_and_matrix_specs(
        a, mesh=mesh, matrix_specs=matrix_specs, in_specs=in_specs
    )
    rhs_specs = infer_rhs_specs(b, matrix_specs=matrix_specs)
    row_axis, col_axis, grid = validate_2d_matrix_specs(mesh, matrix_specs)
    solution_row_partition = rhs_specs._partitions[0]
    solution_row_axes = (
        (solution_row_partition,)
        if isinstance(solution_row_partition, str)
        else tuple(solution_row_partition or ())
    )
    solution_row_shards = 1
    for axis in solution_row_axes:
        solution_row_shards *= mesh_axis_size(mesh, axis)
    if int(a.shape[1]) % solution_row_shards:
        raise ValueError(
            f"{caller} cannot restore an N={a.shape[1]} solution with "
            f"RHS row sharding {rhs_specs}. Choose an RHS sharding whose "
            "row-axis extent divides N."
        )
    rank_map = process_rank_map_from_mesh(
        mesh,
        row_axis=row_axis,
        col_axis=col_axis,
        grid=grid,
        caller=caller,
    )
    native_status_specs = status_specs(row_axis, col_axis, grid)
    tile_shape = TileShape(rows=int(tile_size), cols=int(tile_size))
    m, n = map(int, a.shape)
    validate_nonempty_block_cyclic_ownership(
        logical_rows=m,
        logical_cols=n,
        grid=grid,
        tile_shape=tile_shape,
        caller=f"{caller}(A)",
    )
    nrhs = int(b.shape[1])
    validate_nonempty_block_cyclic_ownership(
        logical_rows=m,
        logical_cols=nrhs,
        grid=grid,
        tile_shape=tile_shape,
        caller=f"{caller}(B)",
    )
    b_distribution_cols = rhs_distribution_columns(
        nrhs, process_cols=grid.process_cols, pad=pad
    )
    a_padding = calculate_2d_padding(m, n, grid, tile_shape)
    b_padding = calculate_2d_padding(m, b_distribution_cols, grid, tile_shape)
    for name, padding in (("A", a_padding), ("B", b_padding)):
        if not pad and padding.needs_padding:
            raise ValueError(
                f"{caller}({name}) requires tile-aligned local shards when "
                "pad=False. Use a compatible tile size or set pad=True."
            )
    return (
        mesh,
        matrix_specs,
        rhs_specs,
        native_status_specs,
        grid,
        rank_map,
        a_padding,
        b_padding,
        b_distribution_cols,
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
    mesh: Mesh, matrix_specs: P, *, local_rows: int, local_cols: int
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
def _least_squares_pipeline(
    mesh: Mesh,
    matrix_specs: P,
    native_status_specs: P,
    grid: ProcessGrid,
    rank_map: ProcessRankMap,
    grid_mapping: int,
    a_padding: MatrixPadding2D,
    b_padding: MatrixPadding2D,
    rhs_specs: P,
    *,
    m: int,
    n: int,
    nrhs: int,
    b_distribution_cols: int,
    tile_size: int,
):
    """Build and cache the unjitted JAX-visible least-squares pipeline."""
    process_rows = grid.process_rows
    process_cols = grid.process_cols
    rank_array = standard_grid_rank_map_attr(
        rank_map,
        process_rows=process_rows,
        process_cols=process_cols,
        caller="cusolvermp_gels",
    )
    grid_mapping = cusolvermp_grid_mapping_attr(
        rank_map,
        grid_mapping,
        process_rows=process_rows,
        process_cols=process_cols,
        caller="cusolvermp_gels",
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

    def gels_ffi(_a: Array, _b: Array) -> tuple[Array, Array, Array]:
        """Call fused native redistribution and ``cusolverMpGels``."""
        out_type = (
            jax.ShapeDtypeStruct(_a.shape, _a.dtype),
            jax.ShapeDtypeStruct(_b.shape, _b.dtype),
            jax.ShapeDtypeStruct(
                (_CUSOLVERMP_LEAST_SQUARES_STATUS_SIZE,), jnp.int32
            ),
        )
        ffi_fn = partial(
            jax.ffi.ffi_call(
                "cusolvermp_gels",
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
            m=int(m),
            n=int(n),
            nrhs=int(nrhs),
            b_distribution_cols=int(b_distribution_cols),
            tile_size=int(tile_size),
        )
        return ffi_fn(_a, _b)

    gels_shardmap = jax.shard_map(
        gels_ffi,
        mesh=mesh,
        in_specs=(matrix_specs, matrix_specs),
        out_specs=(matrix_specs, matrix_specs, native_status_specs),
        check_vma=False,
    )

    @use_abstract_mesh_decorator(mesh)
    def impl(_a: Array, _b: Array) -> tuple[Array, Array, Array, Array]:
        """Run padding, fused native least squares, and output restoration."""
        a_padded = pad_a(_a)
        if b_distribution_padding:
            b_distribution = jnp.pad(_b, ((0, 0), (0, b_distribution_padding)))
        else:
            b_distribution = _b
        b_distribution = place_rhs_for_native_work(
            b_distribution, mesh=mesh, matrix_specs=matrix_specs
        )
        b_padded = pad_b(b_distribution)
        a_work, b_solved, native_status = gels_shardmap(a_padded, b_padded)
        b_work = unpad_b(b_solved)
        b_work = restore_rhs_from_native_work(
            b_work, rhs_specs=rhs_specs, mesh=mesh, matrix_specs=matrix_specs
        )
        b_work = b_work[:, :nrhs]
        return a_work, b_work, b_work[:n], native_status

    return impl


@lru_cache(maxsize=None)
def _least_squares_compiled(
    mesh: Mesh,
    matrix_specs: P,
    native_status_specs: P,
    grid: ProcessGrid,
    rank_map: ProcessRankMap,
    grid_mapping: int,
    a_padding: MatrixPadding2D,
    b_padding: MatrixPadding2D,
    rhs_specs: P,
    *,
    m: int,
    n: int,
    nrhs: int,
    b_distribution_cols: int,
    tile_size: int,
    donate: bool,
):
    """Build and cache the internally jitted least-squares pipeline."""
    pipeline = _least_squares_pipeline(
        mesh,
        matrix_specs,
        native_status_specs,
        grid,
        rank_map,
        grid_mapping,
        a_padding,
        b_padding,
        rhs_specs,
        m=m,
        n=n,
        nrhs=nrhs,
        b_distribution_cols=b_distribution_cols,
        tile_size=tile_size,
    )

    @partial(jax.jit, donate_argnums=(0, 1) if donate else ())
    def impl(_a: Array, _b: Array) -> tuple[Array, Array, Array, Array]:
        return pipeline(_a, _b)

    return impl
