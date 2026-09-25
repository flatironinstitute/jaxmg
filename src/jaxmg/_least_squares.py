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
import numpy as np
from jax import Array
from jax.sharding import AbstractMesh, Mesh, PartitionSpec as P

from ._cusolvermp_layout import (
    infer_rhs_specs,
    make_local_pad_fn,
    make_local_unpad_fn,
    mesh_axis_size,
    place_rhs_for_native_work,
    prepare_input_matrix_layout,
    prepare_matrix_padding,
    restore_rhs_from_native_work,
    rhs_distribution_columns,
    use_abstract_mesh_decorator,
)
from ._cusolvermp_status import _CUSOLVERMP_LEAST_SQUARES_STATUS_SIZE
from ._layout_types import (
    MatrixPadding2D,
    ProcessGrid,
    validate_nonempty_block_cyclic_ownership,
)
from ._setup import ensure_init_jaxmg_backend


def least_squares(
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
        mesh (Mesh or AbstractMesh, optional): JAX mesh used by
            ``jax.shard_map``. If omitted, read from the sharding of ``a``
            (its type inside ``jax.jit``), or taken from the context mesh.
        matrix_specs (PartitionSpec or tuple/list[PartitionSpec], optional):
            Rank-2 matrix sharding. If omitted, read from the sharding of
            ``a``, defaulting to the mesh axes in order.
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
    b, vector_rhs, layout, rhs_specs, b_padding, b_distribution_cols = (
        _prepare_least_squares_call(
            a,
            b,
            T_A,
            mesh,
            matrix_specs,
            in_specs=in_specs,
            pad=pad,
            caller="least_squares",
        )
    )

    ensure_init_jaxmg_backend()
    impl = _least_squares_compiled(
        layout.mesh,
        layout.matrix_specs,
        layout.native_status_specs,
        layout.grid,
        layout.partition_slots,
        layout.padding,
        b_padding,
        rhs_specs,
        m=int(a.shape[0]),
        n=int(a.shape[1]),
        nrhs=int(b.shape[1]),
        b_distribution_cols=b_distribution_cols,
        tile_size=layout.tile_shape.rows,
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
    mesh: Mesh | AbstractMesh | None = None,
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
        mesh (Mesh or AbstractMesh, optional): JAX mesh used by
            ``jax.shard_map``. If omitted, read from the sharding of ``a``
            (its type inside ``jax.jit``), or taken from the context mesh.
        matrix_specs (PartitionSpec or tuple/list[PartitionSpec], optional):
            Rank-2 matrix sharding. If omitted, read from the sharding of
            ``a``, defaulting to the mesh axes in order.
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
    b, vector_rhs, layout, rhs_specs, b_padding, b_distribution_cols = (
        _prepare_least_squares_call(
            a,
            b,
            T_A,
            mesh,
            matrix_specs,
            in_specs=in_specs,
            pad=pad,
            caller="least_squares_shardmap_ctx",
        )
    )

    ensure_init_jaxmg_backend()
    impl = _least_squares_pipeline(
        layout.mesh,
        layout.matrix_specs,
        layout.native_status_specs,
        layout.grid,
        layout.partition_slots,
        layout.padding,
        b_padding,
        rhs_specs,
        m=int(a.shape[0]),
        n=int(a.shape[1]),
        nrhs=int(b.shape[1]),
        b_distribution_cols=b_distribution_cols,
        tile_size=layout.tile_shape.rows,
    )
    a_work, b_work, out, native_status = impl(a, b)
    if vector_rhs:
        b_work = b_work[:, 0]
        out = out[:, 0]
    return a_work, b_work, out, native_status


def _prepare_least_squares_call(
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
    """Validate a least-squares solve and derive its matrix and RHS layouts.

    Preparation proceeds as follows:

    1. Validate A and normalize a vector B to a one-column matrix.
    2. Require matching supported dtypes, compatible leading dimensions,
       ``M >= N``, and a positive tile size.
    3. Resolve the mesh, process grid, rank map, and tile-aligned layout of A.
    4. Infer the B sharding and verify that it can represent the N-row solution.
    5. Add any required B routing columns and derive its native work layout.
    """
    if a.ndim != 2:
        raise ValueError(f"{caller} expects a rank-2 matrix A.")
    vector_rhs = b.ndim == 1
    if vector_rhs:
        b = jnp.expand_dims(b, axis=1)
    if b.ndim != 2:
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
    solution_row_partition = (
        rhs_specs._partitions[0] if rhs_specs._partitions else None
    )
    solution_row_axes = (
        (solution_row_partition,)
        if isinstance(solution_row_partition, str)
        else tuple(solution_row_partition or ())
    )
    solution_row_shards = 1
    for axis in solution_row_axes:
        solution_row_shards *= mesh_axis_size(layout.mesh, axis)
    if int(a.shape[1]) % solution_row_shards:
        raise ValueError(
            f"{caller} cannot restore an N={a.shape[1]} solution with "
            f"RHS row sharding {rhs_specs}. Choose an RHS sharding whose "
            "row-axis extent divides N."
        )
    m = int(a.shape[0])
    nrhs = int(b.shape[1])
    validate_nonempty_block_cyclic_ownership(
        logical_rows=m,
        logical_cols=nrhs,
        grid=layout.grid,
        tile_shape=layout.tile_shape,
        caller=f"{caller}(B)",
    )
    b_distribution_cols = rhs_distribution_columns(
        nrhs, process_cols=layout.grid.process_cols, pad=pad
    )
    b_padding = prepare_matrix_padding(
        m,
        b_distribution_cols,
        layout.grid,
        layout.tile_shape,
        pad=pad,
        caller=f"{caller}(B)",
    )
    return b, vector_rhs, layout, rhs_specs, b_padding, b_distribution_cols


_ROW_MAJOR_JAX_LAYOUT = (0, 1)


@lru_cache(maxsize=None)
def _least_squares_pipeline(
    mesh: Mesh | AbstractMesh,
    matrix_specs: P,
    native_status_specs: P,
    grid: ProcessGrid,
    partition_slots: tuple[int, ...],
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
    slots_attr = np.asarray(partition_slots, dtype=np.int64)
    b_distribution_padding = int(b_distribution_cols) - int(nrhs)
    pad_a = make_local_pad_fn(mesh, matrix_specs, a_padding)
    pad_b = make_local_pad_fn(mesh, matrix_specs, b_padding)
    unpad_b = make_local_unpad_fn(mesh, matrix_specs, b_padding)

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
            partition_slots=slots_attr,
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
    mesh: Mesh | AbstractMesh,
    matrix_specs: P,
    native_status_specs: P,
    grid: ProcessGrid,
    partition_slots: tuple[int, ...],
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
        partition_slots,
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
