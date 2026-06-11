from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.sharding import Mesh, PartitionSpec as P

from ._setup import ensure_init_jaxmg_backend


def xla_comm_collective_probe(token: Array) -> Array:
    """Probe XLA collective contexts and borrowed communicator visibility.

    Returns ``[status, cuda_device, local_device_id, global_device_id,
    local_device_count, has_platform_handle, token_nbytes, cuda_device_count]``.
    A zero status with ``has_platform_handle == 1`` means the execute handler
    reached the XLA-owned GPU communicator for the requested clique.
    """
    ensure_init_jaxmg_backend()

    out_type = (jax.ShapeDtypeStruct((8,), jnp.int32),)
    ffi_fn = jax.ffi.ffi_call(
        "xla_comm_collective_probe",
        out_type,
        input_layouts=((0,),),
        output_layouts=((0,),),
    )
    (out,) = ffi_fn(token)
    return out


def xla_comm_collective_probe_shardmap(token: Array, mesh: Mesh, in_specs: P) -> Array:
    """Run :func:`xla_comm_collective_probe` once per shard over ``mesh``."""
    if not isinstance(in_specs, P):
        raise TypeError("in_specs must be a PartitionSpec.")
    if len(in_specs._partitions) != 1 or in_specs._partitions[0] is None:
        raise ValueError("token must be sharded with PartitionSpec P(<axis_name>).")

    axis_name = in_specs._partitions[0]

    @partial(jax.jit)
    @partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=P(axis_name),
        check_vma=False,
    )
    def impl(_token):
        return xla_comm_collective_probe(_token)

    return impl(token)


def xla_comm_allreduce_probe(token: Array) -> Array:
    """Run a tiny SUM all-reduce through the XLA-owned GPU communicator.

    ``token`` must be a rank-1 ``uint32`` array. This is an investigation helper
    for verifying that the communicator borrowed through XLA collective FFI
    contexts is usable, not just visible.
    """
    ensure_init_jaxmg_backend()

    if token.dtype != jnp.uint32:
        raise TypeError("xla_comm_allreduce_probe expects a uint32 token.")
    if token.ndim != 1:
        raise ValueError("xla_comm_allreduce_probe expects a rank-1 token.")

    out_type = (jax.ShapeDtypeStruct(token.shape, token.dtype),)
    ffi_fn = jax.ffi.ffi_call(
        "xla_comm_allreduce_probe",
        out_type,
        input_layouts=((0,),),
        output_layouts=((0,),),
    )
    (out,) = ffi_fn(token)
    return out


def xla_comm_allreduce_probe_shardmap(token: Array, mesh: Mesh, in_specs: P) -> Array:
    """Run :func:`xla_comm_allreduce_probe` once per shard over ``mesh``."""
    if not isinstance(in_specs, P):
        raise TypeError("in_specs must be a PartitionSpec.")
    if len(in_specs._partitions) != 1 or in_specs._partitions[0] is None:
        raise ValueError("token must be sharded with PartitionSpec P(<axis_name>).")

    axis_name = in_specs._partitions[0]

    @partial(jax.jit)
    @partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=P(axis_name),
        check_vma=False,
    )
    def impl(_token):
        return xla_comm_allreduce_probe(_token)

    return impl(token)


def xla_comm_ring_permute_probe(token: Array) -> Array:
    """Move a rank-1 ``uint32`` shard to the next rank in a ring.

    Rank ``i`` sends to ``(i + 1) % n`` and receives from
    ``(i - 1) % n`` using XLA's collective-permute communicator path.
    """
    ensure_init_jaxmg_backend()

    if token.dtype != jnp.uint32:
        raise TypeError("xla_comm_ring_permute_probe expects a uint32 token.")
    if token.ndim != 1:
        raise ValueError("xla_comm_ring_permute_probe expects a rank-1 token.")

    out_type = (jax.ShapeDtypeStruct(token.shape, token.dtype),)
    ffi_fn = jax.ffi.ffi_call(
        "xla_comm_ring_permute_probe",
        out_type,
        input_layouts=((0,),),
        output_layouts=((0,),),
    )
    (out,) = ffi_fn(token)
    return out


def xla_comm_ring_permute_probe_shardmap(
    token: Array, mesh: Mesh, in_specs: P
) -> Array:
    """Run :func:`xla_comm_ring_permute_probe` once per shard over ``mesh``."""
    if not isinstance(in_specs, P):
        raise TypeError("in_specs must be a PartitionSpec.")
    if len(in_specs._partitions) != 1 or in_specs._partitions[0] is None:
        raise ValueError("token must be sharded with PartitionSpec P(<axis_name>).")

    axis_name = in_specs._partitions[0]

    @partial(jax.jit)
    @partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=P(axis_name),
        check_vma=False,
    )
    def impl(_token):
        return xla_comm_ring_permute_probe(_token)

    return impl(token)


def xla_comm_shift_permute_probe(token: Array, *, shift: int) -> Array:
    """Move a rank-1 ``uint32`` shard by a static rank shift.

    Rank ``i`` sends to ``(i + shift) % n`` and receives from
    ``(i - shift) % n`` using XLA's collective-permute communicator path.
    ``shift`` is passed as a static FFI attribute.
    """
    ensure_init_jaxmg_backend()

    if token.dtype != jnp.uint32:
        raise TypeError("xla_comm_shift_permute_probe expects a uint32 token.")
    if token.ndim != 1:
        raise ValueError("xla_comm_shift_permute_probe expects a rank-1 token.")

    out_type = (jax.ShapeDtypeStruct(token.shape, token.dtype),)
    ffi_fn = partial(
        jax.ffi.ffi_call(
            "xla_comm_shift_permute_probe",
            out_type,
            input_layouts=((0,),),
            output_layouts=((0,),),
        ),
        shift=int(shift),
    )
    (out,) = ffi_fn(token)
    return out


def xla_comm_shift_permute_probe_shardmap(
    token: Array, mesh: Mesh, in_specs: P, *, shift: int
) -> Array:
    """Run :func:`xla_comm_shift_permute_probe` once per shard over ``mesh``."""
    if not isinstance(in_specs, P):
        raise TypeError("in_specs must be a PartitionSpec.")
    if len(in_specs._partitions) != 1 or in_specs._partitions[0] is None:
        raise ValueError("token must be sharded with PartitionSpec P(<axis_name>).")

    axis_name = in_specs._partitions[0]

    @partial(jax.jit)
    @partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=P(axis_name),
        check_vma=False,
    )
    def impl(_token):
        return xla_comm_shift_permute_probe(_token, shift=shift)

    return impl(token)


def xla_comm_permute_probe(token: Array, *, targets) -> Array:
    """Move a rank-1 ``uint32`` shard using a static target-rank plan.

    ``targets[src_rank] = dst_rank`` describes a complete one-to-one
    permutation over the participating XLA communicator ranks. This is an
    investigation helper for moving from simple ring/shift probes toward the
    tile movement plans needed by the cuSOLVERMp redistribution.
    """
    ensure_init_jaxmg_backend()

    if token.dtype != jnp.uint32:
        raise TypeError("xla_comm_permute_probe expects a uint32 token.")
    if token.ndim != 1:
        raise ValueError("xla_comm_permute_probe expects a rank-1 token.")

    target_array = np.ascontiguousarray(np.asarray(targets, dtype=np.int64))
    if target_array.ndim != 1:
        raise ValueError("xla_comm_permute_probe expects a rank-1 targets plan.")

    out_type = (jax.ShapeDtypeStruct(token.shape, token.dtype),)
    ffi_fn = partial(
        jax.ffi.ffi_call(
            "xla_comm_permute_probe",
            out_type,
            input_layouts=((0,),),
            output_layouts=((0,),),
        ),
        targets=target_array,
    )
    (out,) = ffi_fn(token)
    return out


def xla_comm_permute_probe_shardmap(
    token: Array, mesh: Mesh, in_specs: P, *, targets
) -> Array:
    """Run :func:`xla_comm_permute_probe` once per shard over ``mesh``."""
    if not isinstance(in_specs, P):
        raise TypeError("in_specs must be a PartitionSpec.")
    if len(in_specs._partitions) != 1 or in_specs._partitions[0] is None:
        raise ValueError("token must be sharded with PartitionSpec P(<axis_name>).")

    axis_name = in_specs._partitions[0]

    @partial(jax.jit)
    @partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=P(axis_name),
        check_vma=False,
    )
    def impl(_token):
        return xla_comm_permute_probe(_token, targets=targets)

    return impl(token)


def _as_i64_attr(name: str, value) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.int64))
    if array.ndim != 1:
        raise ValueError(f"{name} must be a rank-1 array-like value.")
    return array


_RECT_LAYOUT_CODES = {
    "row_major": 0,
    "column_major": 1,
}

_RECT_JAX_LAYOUTS = {
    0: (0, 1),
    1: (1, 0),
}


def _rect_layout_code(layout) -> int:
    if isinstance(layout, str):
        try:
            return _RECT_LAYOUT_CODES[layout]
        except KeyError as err:
            raise ValueError(
                "layout must be 'row_major', 'column_major', 0, or 1."
            ) from err
    layout_code = int(layout)
    if layout_code not in _RECT_JAX_LAYOUTS:
        raise ValueError("layout must be 'row_major', 'column_major', 0, or 1.")
    return layout_code


def xla_comm_chunk_permute_probe(
    token: Array, *, targets, src_offsets, dst_offsets, count: int
) -> Array:
    """Move one contiguous ``uint32`` chunk per rank using a static plan.

    ``targets[src_rank] = dst_rank`` sends from ``src_offsets[src_rank]`` in
    the source shard to ``dst_offsets[src_rank]`` in the target shard. A target
    of ``-1`` means that source rank does not send in this round. The output is
    initialized as a copy of ``token`` before planned chunks are applied, which
    makes this probe useful for validating partial overwrites.
    """
    ensure_init_jaxmg_backend()

    if token.dtype != jnp.uint32:
        raise TypeError("xla_comm_chunk_permute_probe expects a uint32 token.")
    if token.ndim != 1:
        raise ValueError("xla_comm_chunk_permute_probe expects a rank-1 token.")
    if count < 0:
        raise ValueError("count must be non-negative.")

    target_array = _as_i64_attr("targets", targets)
    src_offset_array = _as_i64_attr("src_offsets", src_offsets)
    dst_offset_array = _as_i64_attr("dst_offsets", dst_offsets)
    if not (
        target_array.shape
        == src_offset_array.shape
        == dst_offset_array.shape
    ):
        raise ValueError("targets, src_offsets, and dst_offsets must match in shape.")

    out_type = (jax.ShapeDtypeStruct(token.shape, token.dtype),)
    ffi_fn = partial(
        jax.ffi.ffi_call(
            "xla_comm_chunk_permute_probe",
            out_type,
            input_layouts=((0,),),
            output_layouts=((0,),),
        ),
        targets=target_array,
        src_offsets=src_offset_array,
        dst_offsets=dst_offset_array,
        count=int(count),
    )
    (out,) = ffi_fn(token)
    return out


def xla_comm_chunk_permute_probe_shardmap(
    token: Array,
    mesh: Mesh,
    in_specs: P,
    *,
    targets,
    src_offsets,
    dst_offsets,
    count: int,
) -> Array:
    """Run :func:`xla_comm_chunk_permute_probe` once per shard over ``mesh``."""
    if not isinstance(in_specs, P):
        raise TypeError("in_specs must be a PartitionSpec.")
    if len(in_specs._partitions) != 1 or in_specs._partitions[0] is None:
        raise ValueError("token must be sharded with PartitionSpec P(<axis_name>).")

    axis_name = in_specs._partitions[0]

    @partial(jax.jit)
    @partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=P(axis_name),
        check_vma=False,
    )
    def impl(_token):
        return xla_comm_chunk_permute_probe(
            _token,
            targets=targets,
            src_offsets=src_offsets,
            dst_offsets=dst_offsets,
            count=count,
        )

    return impl(token)


def xla_comm_matrix_column_step(
    matrix: Array,
    scratch: Array,
    *,
    kind: int,
    source_rank: int,
    target_rank: int,
    source_col: int,
    target_col: int,
) -> tuple[Array, Array]:
    """Apply one experimental communicator-backed matrix-column transfer step.

    ``matrix`` is a rank-2 local shard with layout ``(0, 1)``. The first
    dimension indexes local cyclic slots and the second dimension is the
    contiguous column payload moved by the native handler. ``scratch`` is one
    column payload per participating rank.

    This is a low-level investigation helper. Production 1D reshuffling is
    exposed through :func:`jaxmg.cyclic_1d`.
    """
    ensure_init_jaxmg_backend()

    if matrix.ndim != 2:
        raise ValueError("xla_comm_matrix_column_step expects a rank-2 matrix.")
    if scratch.ndim != 1:
        raise ValueError("xla_comm_matrix_column_step expects a rank-1 scratch.")
    if matrix.dtype != scratch.dtype:
        raise TypeError("matrix and scratch dtypes must match.")
    if scratch.shape[0] != matrix.shape[1]:
        raise ValueError("scratch length must equal matrix.shape[1].")

    out_type = (
        jax.ShapeDtypeStruct(matrix.shape, matrix.dtype),
        jax.ShapeDtypeStruct(scratch.shape, scratch.dtype),
    )
    ffi_fn = partial(
        jax.ffi.ffi_call(
            "xla_comm_matrix_column_step",
            out_type,
            input_layouts=((0, 1), (0,)),
            output_layouts=((0, 1), (0,)),
            input_output_aliases={0: 0, 1: 1},
        ),
        kind=int(kind),
        source_rank=int(source_rank),
        target_rank=int(target_rank),
        source_col=int(source_col),
        target_col=int(target_col),
    )
    return ffi_fn(matrix, scratch)


def xla_comm_matrix_column_step_shardmap(
    matrix: Array,
    scratch: Array,
    mesh: Mesh,
    matrix_specs: P,
    scratch_specs: P,
    *,
    kind: int,
    source_rank: int,
    target_rank: int,
    source_col: int,
    target_col: int,
) -> tuple[Array, Array]:
    """Run :func:`xla_comm_matrix_column_step` over a 1D device mesh."""
    if not isinstance(matrix_specs, P) or not isinstance(scratch_specs, P):
        raise TypeError("matrix_specs and scratch_specs must be PartitionSpec values.")

    @partial(jax.jit, donate_argnums=(0, 1))
    @partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(matrix_specs, scratch_specs),
        out_specs=(matrix_specs, scratch_specs),
        check_vma=False,
    )
    def impl(_matrix, _scratch):
        return xla_comm_matrix_column_step(
            _matrix,
            _scratch,
            kind=kind,
            source_rank=source_rank,
            target_rank=target_rank,
            source_col=source_col,
            target_col=target_col,
        )

    return impl(matrix, scratch)


def xla_comm_matrix_column_batch(
    matrix: Array,
    scratch: Array,
    *,
    kinds,
    source_ranks,
    target_ranks,
    source_cols,
    target_cols,
    scratch_slots,
) -> tuple[Array, Array]:
    """Apply a static batch of communicator-backed matrix-column steps.

    This is the batched form of :func:`xla_comm_matrix_column_step`. ``scratch``
    may contain multiple contiguous one-column scratch slots, each of length
    ``matrix.shape[1]``. The native handler executes the supplied steps in
    order within one FFI call.
    """
    ensure_init_jaxmg_backend()

    if matrix.ndim != 2:
        raise ValueError("xla_comm_matrix_column_batch expects a rank-2 matrix.")
    if scratch.ndim != 1:
        raise ValueError("xla_comm_matrix_column_batch expects a rank-1 scratch.")
    if matrix.dtype != scratch.dtype:
        raise TypeError("matrix and scratch dtypes must match.")
    if scratch.shape[0] % matrix.shape[1] != 0:
        raise ValueError("scratch length must be a multiple of matrix.shape[1].")

    kind_array = _as_i64_attr("kinds", kinds)
    source_rank_array = _as_i64_attr("source_ranks", source_ranks)
    target_rank_array = _as_i64_attr("target_ranks", target_ranks)
    source_col_array = _as_i64_attr("source_cols", source_cols)
    target_col_array = _as_i64_attr("target_cols", target_cols)
    scratch_slot_array = _as_i64_attr("scratch_slots", scratch_slots)
    if not (
        kind_array.shape
        == source_rank_array.shape
        == target_rank_array.shape
        == source_col_array.shape
        == target_col_array.shape
        == scratch_slot_array.shape
    ):
        raise ValueError(
            "kinds, source_ranks, target_ranks, source_cols, target_cols, "
            "and scratch_slots must match in shape."
        )

    out_type = (
        jax.ShapeDtypeStruct(matrix.shape, matrix.dtype),
        jax.ShapeDtypeStruct(scratch.shape, scratch.dtype),
    )
    ffi_fn = partial(
        jax.ffi.ffi_call(
            "xla_comm_matrix_column_batch",
            out_type,
            input_layouts=((0, 1), (0,)),
            output_layouts=((0, 1), (0,)),
            input_output_aliases={0: 0, 1: 1},
        ),
        kinds=kind_array,
        source_ranks=source_rank_array,
        target_ranks=target_rank_array,
        source_cols=source_col_array,
        target_cols=target_col_array,
        scratch_slots=scratch_slot_array,
    )
    return ffi_fn(matrix, scratch)


def xla_comm_matrix_column_batch_shardmap(
    matrix: Array,
    scratch: Array,
    mesh: Mesh,
    matrix_specs: P,
    scratch_specs: P,
    *,
    kinds,
    source_ranks,
    target_ranks,
    source_cols,
    target_cols,
    scratch_slots,
) -> tuple[Array, Array]:
    """Run :func:`xla_comm_matrix_column_batch` over a 1D device mesh."""
    if not isinstance(matrix_specs, P) or not isinstance(scratch_specs, P):
        raise TypeError("matrix_specs and scratch_specs must be PartitionSpec values.")

    @partial(jax.jit, donate_argnums=(0, 1))
    @partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(matrix_specs, scratch_specs),
        out_specs=(matrix_specs, scratch_specs),
        check_vma=False,
    )
    def impl(_matrix, _scratch):
        return xla_comm_matrix_column_batch(
            _matrix,
            _scratch,
            kinds=kinds,
            source_ranks=source_ranks,
            target_ranks=target_ranks,
            source_cols=source_cols,
            target_cols=target_cols,
            scratch_slots=scratch_slots,
        )

    return impl(matrix, scratch)


def xla_comm_matrix_column_native_plan(
    matrix: Array, scratch: Array, *, tile_size: int
) -> tuple[Array, Array]:
    """Apply a native-planned communicator-backed 1D cyclic reshuffle.

    The C++ FFI handler builds the same padded block-to-cyclic movement cycles
    used by :func:`jaxmg.cyclic_1d`, then executes them internally through the
    XLA communicator. Python supplies only the tile size and buffers.
    """
    ensure_init_jaxmg_backend()

    if matrix.ndim != 2:
        raise ValueError("xla_comm_matrix_column_native_plan expects a rank-2 matrix.")
    if scratch.ndim != 1:
        raise ValueError("xla_comm_matrix_column_native_plan expects a rank-1 scratch.")
    if matrix.dtype != scratch.dtype:
        raise TypeError("matrix and scratch dtypes must match.")
    if scratch.shape[0] % matrix.shape[1] != 0:
        raise ValueError("scratch length must be a multiple of matrix.shape[1].")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive.")

    out_type = (
        jax.ShapeDtypeStruct(matrix.shape, matrix.dtype),
        jax.ShapeDtypeStruct(scratch.shape, scratch.dtype),
    )
    ffi_fn = partial(
        jax.ffi.ffi_call(
            "xla_comm_matrix_column_native_plan",
            out_type,
            input_layouts=((0, 1), (0,)),
            output_layouts=((0, 1), (0,)),
            input_output_aliases={0: 0, 1: 1},
        ),
        tile_size=int(tile_size),
    )
    return ffi_fn(matrix, scratch)


def xla_comm_matrix_column_native_plan_shardmap(
    matrix: Array,
    scratch: Array,
    mesh: Mesh,
    matrix_specs: P,
    scratch_specs: P,
    *,
    tile_size: int,
) -> tuple[Array, Array]:
    """Run :func:`xla_comm_matrix_column_native_plan` over a 1D device mesh."""
    if not isinstance(matrix_specs, P) or not isinstance(scratch_specs, P):
        raise TypeError("matrix_specs and scratch_specs must be PartitionSpec values.")

    @partial(jax.jit, donate_argnums=(0, 1))
    @partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(matrix_specs, scratch_specs),
        out_specs=(matrix_specs, scratch_specs),
        check_vma=False,
    )
    def impl(_matrix, _scratch):
        return xla_comm_matrix_column_native_plan(
            _matrix, _scratch, tile_size=tile_size
        )

    return impl(matrix, scratch)


def _validate_rect_pack_args(
    matrix: Array,
    scratch: Array,
    *,
    layout="row_major",
    row_start: int,
    col_start: int,
    row_count: int,
    col_count: int,
    target_row: int,
    target_col: int,
) -> tuple[int, int, int, int, int, int]:
    if matrix.ndim != 2:
        raise ValueError("xla_rect_pack_unpack_probe expects a rank-2 matrix.")
    if scratch.ndim != 1:
        raise ValueError("xla_rect_pack_unpack_probe expects a rank-1 scratch.")
    if matrix.dtype != scratch.dtype:
        raise TypeError("matrix and scratch dtypes must match.")

    layout_code = _rect_layout_code(layout)
    row_start = int(row_start)
    col_start = int(col_start)
    row_count = int(row_count)
    col_count = int(col_count)
    target_row = int(target_row)
    target_col = int(target_col)

    if row_count <= 0 or col_count <= 0:
        raise ValueError("row_count and col_count must be positive.")
    if (
        row_start < 0
        or col_start < 0
        or target_row < 0
        or target_col < 0
        or row_start + row_count > matrix.shape[0]
        or col_start + col_count > matrix.shape[1]
        or target_row + row_count > matrix.shape[0]
        or target_col + col_count > matrix.shape[1]
    ):
        raise ValueError("source and target rectangles must fit inside matrix.")
    if scratch.shape[0] < row_count * col_count:
        raise ValueError("scratch length must be at least row_count * col_count.")

    return (
        layout_code,
        row_start,
        col_start,
        row_count,
        col_count,
        target_row,
        target_col,
    )


def _validate_rect_transfer_args(
    matrix: Array,
    scratch: Array,
    *,
    layout="row_major",
    targets,
    src_row_starts,
    src_col_starts,
    dst_row_starts,
    dst_col_starts,
    row_count: int,
    col_count: int,
):
    if matrix.ndim != 2:
        raise ValueError("xla_rect_transfer_probe expects a rank-2 matrix.")
    if scratch.ndim != 1:
        raise ValueError("xla_rect_transfer_probe expects a rank-1 scratch.")
    if matrix.dtype != scratch.dtype:
        raise TypeError("matrix and scratch dtypes must match.")

    layout_code = _rect_layout_code(layout)
    target_array = _as_i64_attr("targets", targets)
    src_row_array = _as_i64_attr("src_row_starts", src_row_starts)
    src_col_array = _as_i64_attr("src_col_starts", src_col_starts)
    dst_row_array = _as_i64_attr("dst_row_starts", dst_row_starts)
    dst_col_array = _as_i64_attr("dst_col_starts", dst_col_starts)
    if not (
        target_array.shape
        == src_row_array.shape
        == src_col_array.shape
        == dst_row_array.shape
        == dst_col_array.shape
    ):
        raise ValueError(
            "targets and all rectangle offset arrays must match in shape."
        )
    row_count = int(row_count)
    col_count = int(col_count)
    if row_count <= 0 or col_count <= 0:
        raise ValueError("row_count and col_count must be positive.")
    if scratch.shape[0] < 2 * row_count * col_count:
        raise ValueError(
            "scratch length must be at least 2 * row_count * col_count."
        )

    return (
        layout_code,
        target_array,
        src_row_array,
        src_col_array,
        dst_row_array,
        dst_col_array,
        row_count,
        col_count,
    )


def _validate_rect_2d_native_plan_args(
    matrix: Array,
    scratch: Array,
    *,
    layout="row_major",
    process_rows: int,
    process_cols: int,
    tile_rows: int,
    tile_cols: int,
) -> tuple[int, int, int, int, int]:
    if matrix.ndim != 2:
        raise ValueError("xla_rect_2d_native_plan expects a rank-2 matrix.")
    if scratch.ndim != 1:
        raise ValueError("xla_rect_2d_native_plan expects a rank-1 scratch.")
    if matrix.dtype != scratch.dtype:
        raise TypeError("matrix and scratch dtypes must match.")

    layout_code = _rect_layout_code(layout)
    process_rows = int(process_rows)
    process_cols = int(process_cols)
    tile_rows = int(tile_rows)
    tile_cols = int(tile_cols)
    if process_rows <= 0 or process_cols <= 0:
        raise ValueError("process_rows and process_cols must be positive.")
    if tile_rows <= 0 or tile_cols <= 0:
        raise ValueError("tile_rows and tile_cols must be positive.")
    if matrix.shape[0] % tile_rows != 0 or matrix.shape[1] % tile_cols != 0:
        raise ValueError(
            "xla_rect_2d_native_plan currently requires tile-aligned local "
            "matrix shards."
        )
    required_scratch = 3 * max(matrix.shape[0] * tile_cols, tile_rows * matrix.shape[1])
    if scratch.shape[0] < required_scratch:
        raise ValueError(
            "scratch length must be at least "
            "3 * max(local_rows * tile_cols, tile_rows * local_cols)."
        )

    return layout_code, process_rows, process_cols, tile_rows, tile_cols


def _axis_edge_padding_max_extent(
    *,
    block_count: int,
    logical_per_block: int,
    physical_per_block: int,
) -> int:
    total = block_count * physical_per_block
    logical_total = block_count * logical_per_block
    is_real = [False] * total
    for block in range(block_count):
        start = block * physical_per_block
        for offset in range(logical_per_block):
            is_real[start + offset] = True

    max_extent = 0
    while True:
        target_start = next(
            (index for index in range(logical_total) if not is_real[index]),
            None,
        )
        if target_start is None:
            break

        target_stop = target_start
        while target_stop < total and not is_real[target_stop]:
            target_stop += 1

        source_start = next(
            (index for index in range(target_stop, total) if is_real[index]),
            None,
        )
        if source_start is None:
            break

        source_stop = source_start
        while source_stop < total and is_real[source_stop]:
            source_stop += 1

        target_block_stop = (
            target_start // physical_per_block + 1
        ) * physical_per_block
        source_block_stop = (
            source_start // physical_per_block + 1
        ) * physical_per_block
        extent = min(
            target_stop - target_start,
            source_stop - source_start,
            target_block_stop - target_start,
            source_block_stop - source_start,
        )
        max_extent = max(max_extent, extent)
        for offset in range(extent):
            is_real[target_start + offset] = True
            is_real[source_start + offset] = False

    return max_extent


def _validate_rect_padded_2d_native_plan_args(
    matrix: Array,
    scratch: Array,
    *,
    layout="row_major",
    process_rows: int,
    process_cols: int,
    tile_rows: int,
    tile_cols: int,
    logical_rows: int,
    logical_cols: int,
) -> tuple[int, int, int, int, int, int, int]:
    if matrix.ndim != 2:
        raise ValueError("xla_rect_padded_2d_native_plan expects a rank-2 matrix.")
    if scratch.ndim != 1:
        raise ValueError("xla_rect_padded_2d_native_plan expects a rank-1 scratch.")
    if matrix.dtype != scratch.dtype:
        raise TypeError("matrix and scratch dtypes must match.")

    layout_code = _rect_layout_code(layout)
    process_rows = int(process_rows)
    process_cols = int(process_cols)
    tile_rows = int(tile_rows)
    tile_cols = int(tile_cols)
    logical_rows = int(logical_rows)
    logical_cols = int(logical_cols)
    if process_rows <= 0 or process_cols <= 0:
        raise ValueError("process_rows and process_cols must be positive.")
    if tile_rows <= 0 or tile_cols <= 0:
        raise ValueError("tile_rows and tile_cols must be positive.")
    if logical_rows <= 0 or logical_cols <= 0:
        raise ValueError("logical_rows and logical_cols must be positive.")
    if logical_rows % process_rows or logical_cols % process_cols:
        raise ValueError("logical matrix shape must divide over the process grid.")

    local_logical_rows = logical_rows // process_rows
    local_logical_cols = logical_cols // process_cols
    expected_rows = local_logical_rows + (-local_logical_rows) % tile_rows
    expected_cols = local_logical_cols + (-local_logical_cols) % tile_cols
    if matrix.shape != (expected_rows, expected_cols):
        raise ValueError(
            "xla_rect_padded_2d_native_plan expects the local physical "
            f"padded shape {(expected_rows, expected_cols)}, got {matrix.shape}."
        )

    horizontal_extent = _axis_edge_padding_max_extent(
        block_count=process_cols,
        logical_per_block=local_logical_cols,
        physical_per_block=expected_cols,
    )
    vertical_extent = _axis_edge_padding_max_extent(
        block_count=process_rows,
        logical_per_block=local_logical_rows,
        physical_per_block=expected_rows,
    )
    required_step_elements = max(
        expected_rows * tile_cols,
        tile_rows * expected_cols,
        expected_rows * horizontal_extent,
        vertical_extent * expected_cols,
    )
    if required_step_elements and scratch.shape[0] < 3 * required_step_elements:
        raise ValueError(
            "scratch length must be at least "
            "3 * max(native padded 2D step elements)."
        )

    return (
        layout_code,
        process_rows,
        process_cols,
        tile_rows,
        tile_cols,
        logical_rows,
        logical_cols,
    )


def xla_rect_pack_unpack_probe(
    matrix: Array,
    scratch: Array,
    *,
    layout="row_major",
    row_start: int,
    col_start: int,
    row_count: int,
    col_count: int,
    target_row: int,
    target_col: int,
) -> tuple[Array, Array]:
    """Pack a local strided rectangle to scratch and unpack it elsewhere.

    This diagnostic exercises the local CUDA primitive needed by the future
    cuSOLVERMp 2D redistribution. It does not use the XLA communicator: it only
    verifies that a rectangular fragment can be copied from a rank-2 local
    buffer into contiguous rank-1 scratch and back into a rank-2 output buffer
    on XLA's CUDA stream.

    ``layout`` controls the physical local matrix layout. ``"row_major"`` packs
    scratch as row-contiguous rows. ``"column_major"`` packs scratch as
    column-contiguous columns, which is the layout cuSOLVERMp expects for local
    matrices.
    """
    (
        layout_code,
        row_start,
        col_start,
        row_count,
        col_count,
        target_row,
        target_col,
    ) = _validate_rect_pack_args(
        matrix,
        scratch,
        layout=layout,
        row_start=row_start,
        col_start=col_start,
        row_count=row_count,
        col_count=col_count,
        target_row=target_row,
        target_col=target_col,
    )
    ensure_init_jaxmg_backend()

    out_type = (
        jax.ShapeDtypeStruct(matrix.shape, matrix.dtype),
        jax.ShapeDtypeStruct(scratch.shape, scratch.dtype),
    )
    matrix_layout = _RECT_JAX_LAYOUTS[layout_code]
    ffi_fn = partial(
        jax.ffi.ffi_call(
            "xla_rect_pack_unpack_probe",
            out_type,
            input_layouts=(matrix_layout, (0,)),
            output_layouts=(matrix_layout, (0,)),
            input_output_aliases={0: 0, 1: 1},
        ),
        layout=layout_code,
        row_start=row_start,
        col_start=col_start,
        row_count=row_count,
        col_count=col_count,
        target_row=target_row,
        target_col=target_col,
    )
    return ffi_fn(matrix, scratch)


def xla_rect_pack_unpack_probe_shardmap(
    matrix: Array,
    scratch: Array,
    mesh: Mesh,
    matrix_specs: P,
    scratch_specs: P,
    *,
    layout="row_major",
    row_start: int,
    col_start: int,
    row_count: int,
    col_count: int,
    target_row: int,
    target_col: int,
) -> tuple[Array, Array]:
    """Run :func:`xla_rect_pack_unpack_probe` over a sharded local matrix."""
    if not isinstance(matrix_specs, P) or not isinstance(scratch_specs, P):
        raise TypeError("matrix_specs and scratch_specs must be PartitionSpec values.")

    @partial(jax.jit, donate_argnums=(0, 1))
    @partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(matrix_specs, scratch_specs),
        out_specs=(matrix_specs, scratch_specs),
        check_vma=False,
    )
    def impl(_matrix, _scratch):
        return xla_rect_pack_unpack_probe(
            _matrix,
            _scratch,
            layout=layout,
            row_start=row_start,
            col_start=col_start,
            row_count=row_count,
            col_count=col_count,
            target_row=target_row,
            target_col=target_col,
        )

    return impl(matrix, scratch)


def _xla_rect_transfer_probe_impl(
    matrix: Array,
    scratch: Array,
    *,
    ffi_target: str,
    layout="row_major",
    targets,
    src_row_starts,
    src_col_starts,
    dst_row_starts,
    dst_col_starts,
    row_count: int,
    col_count: int,
) -> tuple[Array, Array]:
    (
        layout_code,
        target_array,
        src_row_array,
        src_col_array,
        dst_row_array,
        dst_col_array,
        row_count,
        col_count,
    ) = _validate_rect_transfer_args(
        matrix,
        scratch,
        layout=layout,
        targets=targets,
        src_row_starts=src_row_starts,
        src_col_starts=src_col_starts,
        dst_row_starts=dst_row_starts,
        dst_col_starts=dst_col_starts,
        row_count=row_count,
        col_count=col_count,
    )
    ensure_init_jaxmg_backend()

    out_type = (
        jax.ShapeDtypeStruct(matrix.shape, matrix.dtype),
        jax.ShapeDtypeStruct(scratch.shape, scratch.dtype),
    )
    matrix_layout = _RECT_JAX_LAYOUTS[layout_code]
    ffi_fn = partial(
        jax.ffi.ffi_call(
            ffi_target,
            out_type,
            input_layouts=(matrix_layout, (0,)),
            output_layouts=(matrix_layout, (0,)),
            input_output_aliases={0: 0, 1: 1},
        ),
        layout=layout_code,
        targets=target_array,
        src_row_starts=src_row_array,
        src_col_starts=src_col_array,
        dst_row_starts=dst_row_array,
        dst_col_starts=dst_col_array,
        row_count=row_count,
        col_count=col_count,
    )
    return ffi_fn(matrix, scratch)


def xla_rect_transfer_probe(
    matrix: Array,
    scratch: Array,
    *,
    layout="row_major",
    targets,
    src_row_starts,
    src_col_starts,
    dst_row_starts,
    dst_col_starts,
    row_count: int,
    col_count: int,
) -> tuple[Array, Array]:
    """Move one packed rectangular fragment per rank through XLA communication.

    ``targets[source_rank] = target_rank`` describes a one-to-one transfer
    round. Each participating source rank packs its local source rectangle into
    the first scratch slot, sends that dense payload with XLA
    ``CollectivePermute``, and each receiver unpacks the received payload from
    the second scratch slot into its local destination rectangle.
    """
    return _xla_rect_transfer_probe_impl(
        matrix,
        scratch,
        ffi_target="xla_rect_transfer_probe",
        layout=layout,
        targets=targets,
        src_row_starts=src_row_starts,
        src_col_starts=src_col_starts,
        dst_row_starts=dst_row_starts,
        dst_col_starts=dst_col_starts,
        row_count=row_count,
        col_count=col_count,
    )


def xla_rect_transfer_nccl_probe(
    matrix: Array,
    scratch: Array,
    *,
    layout="row_major",
    targets,
    src_row_starts,
    src_col_starts,
    dst_row_starts,
    dst_col_starts,
    row_count: int,
    col_count: int,
) -> tuple[Array, Array]:
    """Move one packed rectangular fragment per rank through raw NCCL.

    This uses the same XLA-owned communicator as ``xla_rect_transfer_probe`` but
    borrows its underlying ``ncclComm_t`` and submits ``ncclSend``/``ncclRecv``
    directly. It is diagnostic code for validating the future cuSOLVERMp
    transport path.
    """
    return _xla_rect_transfer_probe_impl(
        matrix,
        scratch,
        ffi_target="xla_rect_transfer_nccl_probe",
        layout=layout,
        targets=targets,
        src_row_starts=src_row_starts,
        src_col_starts=src_col_starts,
        dst_row_starts=dst_row_starts,
        dst_col_starts=dst_col_starts,
        row_count=row_count,
        col_count=col_count,
    )


def xla_rect_transfer_probe_shardmap(
    matrix: Array,
    scratch: Array,
    mesh: Mesh,
    matrix_specs: P,
    scratch_specs: P,
    *,
    layout="row_major",
    targets,
    src_row_starts,
    src_col_starts,
    dst_row_starts,
    dst_col_starts,
    row_count: int,
    col_count: int,
) -> tuple[Array, Array]:
    """Run :func:`xla_rect_transfer_probe` over sharded local matrices."""
    if not isinstance(matrix_specs, P) or not isinstance(scratch_specs, P):
        raise TypeError("matrix_specs and scratch_specs must be PartitionSpec values.")

    @partial(jax.jit, donate_argnums=(0, 1))
    @partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(matrix_specs, scratch_specs),
        out_specs=(matrix_specs, scratch_specs),
        check_vma=False,
    )
    def impl(_matrix, _scratch):
        return xla_rect_transfer_probe(
            _matrix,
            _scratch,
            layout=layout,
            targets=targets,
            src_row_starts=src_row_starts,
            src_col_starts=src_col_starts,
            dst_row_starts=dst_row_starts,
            dst_col_starts=dst_col_starts,
            row_count=row_count,
            col_count=col_count,
        )

    return impl(matrix, scratch)


def xla_rect_transfer_nccl_probe_shardmap(
    matrix: Array,
    scratch: Array,
    mesh: Mesh,
    matrix_specs: P,
    scratch_specs: P,
    *,
    layout="row_major",
    targets,
    src_row_starts,
    src_col_starts,
    dst_row_starts,
    dst_col_starts,
    row_count: int,
    col_count: int,
) -> tuple[Array, Array]:
    """Run :func:`xla_rect_transfer_nccl_probe` over sharded local matrices."""
    if not isinstance(matrix_specs, P) or not isinstance(scratch_specs, P):
        raise TypeError("matrix_specs and scratch_specs must be PartitionSpec values.")

    @partial(jax.jit, donate_argnums=(0, 1))
    @partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(matrix_specs, scratch_specs),
        out_specs=(matrix_specs, scratch_specs),
        check_vma=False,
    )
    def impl(_matrix, _scratch):
        return xla_rect_transfer_nccl_probe(
            _matrix,
            _scratch,
            layout=layout,
            targets=targets,
            src_row_starts=src_row_starts,
            src_col_starts=src_col_starts,
            dst_row_starts=dst_row_starts,
            dst_col_starts=dst_col_starts,
            row_count=row_count,
            col_count=col_count,
        )

    return impl(matrix, scratch)


def _xla_rect_2d_native_plan_impl(
    matrix: Array,
    scratch: Array,
    *,
    ffi_target: str,
    layout="row_major",
    process_rows: int,
    process_cols: int,
    tile_rows: int,
    tile_cols: int,
) -> tuple[Array, Array]:
    (
        layout_code,
        process_rows,
        process_cols,
        tile_rows,
        tile_cols,
    ) = _validate_rect_2d_native_plan_args(
        matrix,
        scratch,
        layout=layout,
        process_rows=process_rows,
        process_cols=process_cols,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
    )
    ensure_init_jaxmg_backend()

    out_type = (
        jax.ShapeDtypeStruct(matrix.shape, matrix.dtype),
        jax.ShapeDtypeStruct(scratch.shape, scratch.dtype),
    )
    matrix_layout = _RECT_JAX_LAYOUTS[layout_code]
    ffi_fn = partial(
        jax.ffi.ffi_call(
            ffi_target,
            out_type,
            input_layouts=(matrix_layout, (0,)),
            output_layouts=(matrix_layout, (0,)),
            input_output_aliases={0: 0, 1: 1},
        ),
        layout=layout_code,
        process_rows=process_rows,
        process_cols=process_cols,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
    )
    return ffi_fn(matrix, scratch)


def xla_rect_2d_native_plan(
    matrix: Array,
    scratch: Array,
    *,
    layout="row_major",
    process_rows: int,
    process_cols: int,
    tile_rows: int,
    tile_cols: int,
) -> tuple[Array, Array]:
    """Apply a native-planned 2D tile redistribution on local shards.

    The C++ handler constructs the slab schedule, batches conflict-free cycle
    steps, and executes those batches through XLA ``CollectivePermute``.
    """
    return _xla_rect_2d_native_plan_impl(
        matrix,
        scratch,
        ffi_target="xla_rect_2d_native_plan",
        layout=layout,
        process_rows=process_rows,
        process_cols=process_cols,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
    )


def xla_rect_2d_native_plan_nccl(
    matrix: Array,
    scratch: Array,
    *,
    layout="row_major",
    process_rows: int,
    process_cols: int,
    tile_rows: int,
    tile_cols: int,
) -> tuple[Array, Array]:
    """Apply the native 2D tile redistribution through raw NCCL send/recv."""
    return _xla_rect_2d_native_plan_impl(
        matrix,
        scratch,
        ffi_target="xla_rect_2d_native_plan_nccl",
        layout=layout,
        process_rows=process_rows,
        process_cols=process_cols,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
    )


def xla_rect_2d_native_plan_shardmap(
    matrix: Array,
    scratch: Array,
    mesh: Mesh,
    matrix_specs: P,
    scratch_specs: P,
    *,
    layout="row_major",
    process_rows: int,
    process_cols: int,
    tile_rows: int,
    tile_cols: int,
) -> tuple[Array, Array]:
    """Run :func:`xla_rect_2d_native_plan` over a 2D sharded matrix."""
    if not isinstance(matrix_specs, P) or not isinstance(scratch_specs, P):
        raise TypeError("matrix_specs and scratch_specs must be PartitionSpec values.")

    @partial(jax.jit, donate_argnums=(0, 1))
    @partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(matrix_specs, scratch_specs),
        out_specs=(matrix_specs, scratch_specs),
        check_vma=False,
    )
    def impl(_matrix, _scratch):
        return xla_rect_2d_native_plan(
            _matrix,
            _scratch,
            layout=layout,
            process_rows=process_rows,
            process_cols=process_cols,
            tile_rows=tile_rows,
            tile_cols=tile_cols,
        )

    return impl(matrix, scratch)


def xla_rect_2d_native_plan_nccl_shardmap(
    matrix: Array,
    scratch: Array,
    mesh: Mesh,
    matrix_specs: P,
    scratch_specs: P,
    *,
    layout="row_major",
    process_rows: int,
    process_cols: int,
    tile_rows: int,
    tile_cols: int,
) -> tuple[Array, Array]:
    """Run :func:`xla_rect_2d_native_plan_nccl` over a 2D sharded matrix."""
    if not isinstance(matrix_specs, P) or not isinstance(scratch_specs, P):
        raise TypeError("matrix_specs and scratch_specs must be PartitionSpec values.")

    @partial(jax.jit, donate_argnums=(0, 1))
    @partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(matrix_specs, scratch_specs),
        out_specs=(matrix_specs, scratch_specs),
        check_vma=False,
    )
    def impl(_matrix, _scratch):
        return xla_rect_2d_native_plan_nccl(
            _matrix,
            _scratch,
            layout=layout,
            process_rows=process_rows,
            process_cols=process_cols,
            tile_rows=tile_rows,
            tile_cols=tile_cols,
        )

    return impl(matrix, scratch)


def _xla_rect_padded_2d_native_plan_impl(
    matrix: Array,
    scratch: Array,
    *,
    ffi_target: str,
    layout="row_major",
    process_rows: int,
    process_cols: int,
    tile_rows: int,
    tile_cols: int,
    logical_rows: int,
    logical_cols: int,
) -> tuple[Array, Array]:
    (
        layout_code,
        process_rows,
        process_cols,
        tile_rows,
        tile_cols,
        logical_rows,
        logical_cols,
    ) = _validate_rect_padded_2d_native_plan_args(
        matrix,
        scratch,
        layout=layout,
        process_rows=process_rows,
        process_cols=process_cols,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        logical_rows=logical_rows,
        logical_cols=logical_cols,
    )
    ensure_init_jaxmg_backend()

    out_type = (
        jax.ShapeDtypeStruct(matrix.shape, matrix.dtype),
        jax.ShapeDtypeStruct(scratch.shape, scratch.dtype),
    )
    matrix_layout = _RECT_JAX_LAYOUTS[layout_code]
    ffi_fn = partial(
        jax.ffi.ffi_call(
            ffi_target,
            out_type,
            input_layouts=(matrix_layout, (0,)),
            output_layouts=(matrix_layout, (0,)),
            input_output_aliases={0: 0, 1: 1},
        ),
        layout=layout_code,
        process_rows=process_rows,
        process_cols=process_cols,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        logical_rows=logical_rows,
        logical_cols=logical_cols,
    )
    return ffi_fn(matrix, scratch)


def xla_rect_padded_2d_native_plan(
    matrix: Array,
    scratch: Array,
    *,
    layout="row_major",
    process_rows: int,
    process_cols: int,
    tile_rows: int,
    tile_cols: int,
    logical_rows: int,
    logical_cols: int,
) -> tuple[Array, Array]:
    """Apply the native padded 2D redistribution schedule through XLA."""
    return _xla_rect_padded_2d_native_plan_impl(
        matrix,
        scratch,
        ffi_target="xla_rect_padded_2d_native_plan",
        layout=layout,
        process_rows=process_rows,
        process_cols=process_cols,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        logical_rows=logical_rows,
        logical_cols=logical_cols,
    )


def xla_rect_padded_2d_native_plan_nccl(
    matrix: Array,
    scratch: Array,
    *,
    layout="row_major",
    process_rows: int,
    process_cols: int,
    tile_rows: int,
    tile_cols: int,
    logical_rows: int,
    logical_cols: int,
) -> tuple[Array, Array]:
    """Apply the native padded 2D redistribution schedule through raw NCCL."""
    return _xla_rect_padded_2d_native_plan_impl(
        matrix,
        scratch,
        ffi_target="xla_rect_padded_2d_native_plan_nccl",
        layout=layout,
        process_rows=process_rows,
        process_cols=process_cols,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        logical_rows=logical_rows,
        logical_cols=logical_cols,
    )


def xla_rect_padded_2d_native_plan_shardmap(
    matrix: Array,
    scratch: Array,
    mesh: Mesh,
    matrix_specs: P,
    scratch_specs: P,
    *,
    layout="row_major",
    process_rows: int,
    process_cols: int,
    tile_rows: int,
    tile_cols: int,
    logical_rows: int,
    logical_cols: int,
) -> tuple[Array, Array]:
    """Run :func:`xla_rect_padded_2d_native_plan` over a 2D sharded matrix."""
    if not isinstance(matrix_specs, P) or not isinstance(scratch_specs, P):
        raise TypeError("matrix_specs and scratch_specs must be PartitionSpec values.")

    @partial(jax.jit, donate_argnums=(0, 1))
    @partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(matrix_specs, scratch_specs),
        out_specs=(matrix_specs, scratch_specs),
        check_vma=False,
    )
    def impl(_matrix, _scratch):
        return xla_rect_padded_2d_native_plan(
            _matrix,
            _scratch,
            layout=layout,
            process_rows=process_rows,
            process_cols=process_cols,
            tile_rows=tile_rows,
            tile_cols=tile_cols,
            logical_rows=logical_rows,
            logical_cols=logical_cols,
        )

    return impl(matrix, scratch)


def xla_rect_padded_2d_native_plan_nccl_shardmap(
    matrix: Array,
    scratch: Array,
    mesh: Mesh,
    matrix_specs: P,
    scratch_specs: P,
    *,
    layout="row_major",
    process_rows: int,
    process_cols: int,
    tile_rows: int,
    tile_cols: int,
    logical_rows: int,
    logical_cols: int,
) -> tuple[Array, Array]:
    """Run :func:`xla_rect_padded_2d_native_plan_nccl` over a 2D sharded matrix."""
    if not isinstance(matrix_specs, P) or not isinstance(scratch_specs, P):
        raise TypeError("matrix_specs and scratch_specs must be PartitionSpec values.")

    @partial(jax.jit, donate_argnums=(0, 1))
    @partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(matrix_specs, scratch_specs),
        out_specs=(matrix_specs, scratch_specs),
        check_vma=False,
    )
    def impl(_matrix, _scratch):
        return xla_rect_padded_2d_native_plan_nccl(
            _matrix,
            _scratch,
            layout=layout,
            process_rows=process_rows,
            process_cols=process_cols,
            tile_rows=tile_rows,
            tile_cols=tile_cols,
            logical_rows=logical_rows,
            logical_cols=logical_cols,
        )

    return impl(matrix, scratch)
