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
