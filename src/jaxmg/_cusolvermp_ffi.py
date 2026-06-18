"""Private JAX FFI adapter for the fused cuSOLVERMp backend.

The numerical work is not implemented in this file.  The functions below build
the JAX call boundary for native C++/CUDA handlers:

1. validate local buffer ranks, dtypes, and scalar attributes;
2. declare the native FFI symbol names and buffer layouts;
3. pass the process-grid and communicator-rank metadata as static attributes;
   and
4. wrap each local FFI call in ``jax.shard_map`` over the user-provided mesh.

After JAX traces and compiles the call, runtime execution enters the fused
native handler.  The C++ code allocates scratch, redistributes the padded JAX
layout into cuSOLVERMp's 2D block-cyclic layout, calls cuSOLVERMp, and
redistributes outputs back.  Keeping this adapter private prevents public
solver wrappers from accumulating low-level FFI details such as status-vector
sizes, output layouts, and input/output aliasing.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.sharding import Mesh, PartitionSpec as P

from ._setup import ensure_init_jaxmg_backend

_ROW_MAJOR_JAX_LAYOUT = (0, 1)

# These schemas must match the private C++ status vector arrays in
# src/cuda/cusolvermp_routines/cusolvermp_potrs.cc and
# src/cuda/cusolvermp_routines/cusolvermp_syevd.cc. They are intentionally
# private: status vectors are low-level backend error detail and should not
# become part of the public API.
_CUSOLVERMP_POTRS_STATUS_FIELDS = (
    "status_code",
    "cuda_device",
    "nccl_rank",
    "nccl_rank_count",
    "process_rows",
    "process_cols",
    "cusolvermp_version",
    "cusolvermp_runtime_available",
    "handle_created",
    "grid_created",
    "a_descriptor_created",
    "raw_cusolver_status",
    "a_size_bytes",
    "n",
    "tile_size",
    "a_local_rows",
    "a_local_cols",
    "b_local_rows",
    "a_numroc_rows",
    "a_numroc_cols",
    "b_numroc_rows",
    "b_numroc_cols",
    "potrf_device_workspace_kib",
    "potrf_host_workspace_kib",
    "potrs_device_workspace_kib",
    "potrs_host_workspace_kib",
    "potrf_called",
    "potrf_info",
    "potrs_called",
    "potrs_info",
    "a_native_redist",
    "b_native_redist",
    "b_reverse_redist",
    "residual_scaled_1e6",
    "dtype_code",
    "b_local_cols",
    "nrhs",
    "a_native_redist",
    "b_native_redist",
    "reserved",
)
_CUSOLVERMP_SYEVD_STATUS_FIELDS = (
    "status_code",
    "cuda_device",
    "nccl_rank",
    "nccl_rank_count",
    "process_rows",
    "process_cols",
    "cusolvermp_version",
    "cusolvermp_runtime_available",
    "handle_created",
    "grid_created",
    "a_descriptor_created",
    "raw_cusolver_status",
    "a_size_bytes",
    "n",
    "tile_size",
    "a_local_rows",
    "a_local_cols",
    "a_numroc_rows",
    "a_numroc_cols",
    "eigenvalues_size_bytes",
    "compute_eigenvectors",
    "syevd_device_workspace_kib",
    "syevd_host_workspace_kib",
    "syevd_called",
    "syevd_info",
    "dtype_code",
    "grid_mapping",
    "q_descriptor_created",
    "a_native_redist",
    "reserved_0",
    "reserved_1",
    "reserved_2",
    "reserved_3",
    "reserved_4",
    "reserved_5",
    "reserved_6",
)
_CUSOLVERMP_POTRS_STATUS_SIZE = len(_CUSOLVERMP_POTRS_STATUS_FIELDS)
_CUSOLVERMP_SYEVD_STATUS_SIZE = len(_CUSOLVERMP_SYEVD_STATUS_FIELDS)


def _standard_grid_rank_map_attr(
    rank_map,
    *,
    process_rows: int,
    process_cols: int,
    caller: str,
) -> np.ndarray:
    """Return a dense cuSOLVERMp-compatible process-slot -> rank map."""
    process_rows = int(process_rows)
    process_cols = int(process_cols)
    num_ranks = process_rows * process_cols
    if rank_map is None:
        rank_array = np.arange(num_ranks, dtype=np.int64)
    elif hasattr(rank_map, "ranks"):
        rank_array = np.asarray(rank_map.ranks, dtype=np.int64)
    else:
        rank_array = np.asarray(rank_map, dtype=np.int64)
    if rank_array.shape != (num_ranks,):
        raise ValueError(
            f"{caller} rank_map must have shape ({num_ranks},), got "
            f"{rank_array.shape}."
        )
    if sorted(rank_array.tolist()) != list(range(num_ranks)):
        raise ValueError(
            f"{caller} rank_map must be a permutation of [0, {num_ranks})."
        )

    row_major = np.arange(num_ranks, dtype=np.int64)
    column_major = np.asarray(
        [
            process_col * process_rows + process_row
            for process_row in range(process_rows)
            for process_col in range(process_cols)
        ],
        dtype=np.int64,
    )
    if not np.array_equal(rank_array, row_major) and not np.array_equal(
        rank_array, column_major
    ):
        raise NotImplementedError(
            f"{caller} supports only row-major or column-major cuSOLVERMp "
            f"rank maps. Got {rank_array.tolist()}."
        )
    return np.ascontiguousarray(rank_array)


def _cusolvermp_grid_mapping_attr(
    rank_map,
    grid_mapping,
    *,
    process_rows: int,
    process_cols: int,
    caller: str,
) -> int:
    """Return cuSOLVERMp's grid-mapping enum for the validated rank map."""
    rank_array = _standard_grid_rank_map_attr(
        rank_map,
        process_rows=process_rows,
        process_cols=process_cols,
        caller=caller,
    )
    if grid_mapping is None and hasattr(rank_map, "cusolvermp_grid_mapping"):
        grid_mapping = rank_map.cusolvermp_grid_mapping
    if grid_mapping is None:
        row_major = np.arange(int(process_rows) * int(process_cols), dtype=np.int64)
        grid_mapping = 1 if np.array_equal(rank_array, row_major) else 0
    grid_mapping = int(grid_mapping)
    if grid_mapping not in (0, 1):
        raise ValueError(
            f"{caller} grid_mapping must be 0 (column-major) or 1 (row-major), "
            f"got {grid_mapping}."
        )

    expected = (
        np.arange(int(process_rows) * int(process_cols), dtype=np.int64)
        if grid_mapping == 1
        else np.asarray(
            [
                process_col * int(process_rows) + process_row
                for process_row in range(int(process_rows))
                for process_col in range(int(process_cols))
            ],
            dtype=np.int64,
        )
    )
    if not np.array_equal(rank_array, expected):
        raise ValueError(
            f"{caller} rank_map does not match grid_mapping={grid_mapping}."
        )
    return grid_mapping


def _real_dtype_for_eigenvalues(dtype):
    if dtype == jnp.float32 or dtype == jnp.complex64:
        return jnp.float32
    if dtype == jnp.float64 or dtype == jnp.complex128:
        return jnp.float64
    raise TypeError(
        "cuSOLVERMp eigensolvers support float32, float64, complex64, "
        "and complex128."
    )


def cusolvermp_potrs(
    a: Array,
    b: Array,
    *,
    process_rows: int,
    process_cols: int,
    rank_map=None,
    grid_mapping=None,
    n: int,
    nrhs: int,
    b_distribution_cols: int,
    tile_size: int,
) -> tuple[Array, Array]:
    """Call fused native redistribution + ``potrf/potrs`` on local buffers."""
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("cusolvermp_potrs expects rank-2 A and B buffers.")
    if a.dtype != b.dtype:
        raise TypeError("cusolvermp_potrs requires matching A/B dtypes.")
    if a.dtype not in (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128):
        raise TypeError(
            "cusolvermp_potrs supports float32, float64, complex64, "
            "and complex128."
        )
    if a.shape[0] != b.shape[0]:
        raise ValueError("A and B must have matching local row capacity.")

    process_rows = int(process_rows)
    process_cols = int(process_cols)
    n = int(n)
    nrhs = int(nrhs)
    b_distribution_cols = int(b_distribution_cols)
    tile_size = int(tile_size)
    if process_rows <= 0 or process_cols <= 0:
        raise ValueError("process_rows and process_cols must be positive.")
    if n <= 0 or nrhs <= 0 or b_distribution_cols <= 0 or tile_size <= 0:
        raise ValueError(
            "n, nrhs, b_distribution_cols, and tile_size must be positive."
        )
    if b_distribution_cols < nrhs:
        raise ValueError("b_distribution_cols must be at least nrhs.")
    rank_array = _standard_grid_rank_map_attr(
        rank_map,
        process_rows=process_rows,
        process_cols=process_cols,
        caller="cusolvermp_potrs",
    )
    grid_mapping = _cusolvermp_grid_mapping_attr(
        rank_map,
        grid_mapping,
        process_rows=process_rows,
        process_cols=process_cols,
        caller="cusolvermp_potrs",
    )

    ensure_init_jaxmg_backend()

    out_type = (
        jax.ShapeDtypeStruct(a.shape, a.dtype),
        jax.ShapeDtypeStruct(b.shape, b.dtype),
        jax.ShapeDtypeStruct((_CUSOLVERMP_POTRS_STATUS_SIZE,), jnp.int32),
    )
    ffi_fn = partial(
        jax.ffi.ffi_call(
            "cusolvermp_potrs",
            out_type,
            input_layouts=(_ROW_MAJOR_JAX_LAYOUT, _ROW_MAJOR_JAX_LAYOUT),
            output_layouts=(_ROW_MAJOR_JAX_LAYOUT, _ROW_MAJOR_JAX_LAYOUT, (0,)),
            input_output_aliases={0: 0, 1: 1},
        ),
        process_rows=process_rows,
        process_cols=process_cols,
        grid_mapping=grid_mapping,
        rank_map=rank_array,
        n=n,
        nrhs=nrhs,
        b_distribution_cols=b_distribution_cols,
        tile_size=tile_size,
    )
    _, b_out, status = ffi_fn(a, b)
    return b_out, status


def cusolvermp_potrs_shardmap(
    a: Array,
    b: Array,
    mesh: Mesh,
    matrix_specs: P,
    status_specs: P,
    *,
    process_rows: int,
    process_cols: int,
    rank_map=None,
    grid_mapping=None,
    n: int,
    nrhs: int,
    b_distribution_cols: int,
    tile_size: int,
) -> tuple[Array, Array]:
    """Wrap the fused ``potrs`` FFI target in a JAX mesh execution context.

    This function uses ``jax.shard_map`` to coordinate the native FFI calls
    across all participating GPUs in the mesh. It ensures that every rank
    receives the correct static metadata (process-grid dimensions, tile size,
    and rank mapping) needed to enter the native cuSOLVERMp routine. The
    native backend borrows XLA's communicator during the FFI call.

    Args:
        a: Local padded shard of the coefficient matrix.
        b: Local padded shard of the right-hand side.
        mesh: JAX ``Mesh`` describing the device grid.
        matrix_specs: 2D ``PartitionSpec`` for the input and output matrices.
        status_specs: ``PartitionSpec`` for the replicated status vector.
        process_rows: Number of rows in the cuSOLVERMp process grid.
        process_cols: Number of columns in the cuSOLVERMp process grid.
        rank_map: Mapping from process-grid coordinates to communicator ranks.
        grid_mapping: cuSOLVERMp grid-mapping enum (row-major or column-major).
        n: Global dimension of the square matrix A.
        nrhs: Number of right-hand sides in B.
        b_distribution_cols: JAX-visible padded RHS width used for native
            redistribution. cuSOLVERMp still receives ``nrhs``.
        tile_size: Square tile dimension (``T_A``).

    Returns:
        A tuple ``(b_solved, status)`` where ``b_solved`` is the solved RHS in
        the padded JAX-facing layout and ``status`` is the native solver
        status.
    """
    if not isinstance(matrix_specs, P) or not isinstance(status_specs, P):
        raise TypeError("matrix_specs and status_specs must be PartitionSpec values.")

    @partial(jax.jit, donate_argnums=(0, 1))
    @partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(matrix_specs, matrix_specs),
        out_specs=(matrix_specs, status_specs),
        check_vma=False,
    )
    def impl(_a, _b):
        return cusolvermp_potrs(
            _a,
            _b,
            process_rows=process_rows,
            process_cols=process_cols,
            rank_map=rank_map,
            grid_mapping=grid_mapping,
            n=n,
            nrhs=nrhs,
            b_distribution_cols=b_distribution_cols,
            tile_size=tile_size,
        )

    return impl(a, b)


def cusolvermp_syevd(
    a: Array,
    *,
    process_rows: int,
    process_cols: int,
    rank_map=None,
    grid_mapping=None,
    n: int,
    tile_size: int,
) -> tuple[Array, Array, Array]:
    """Call fused native redistribution + vector-producing ``syevd``."""
    if a.ndim != 2:
        raise ValueError("cusolvermp_syevd expects a rank-2 matrix buffer.")
    if a.dtype not in (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128):
        raise TypeError(
            "cusolvermp_syevd supports float32, float64, complex64, "
            "and complex128."
        )

    process_rows = int(process_rows)
    process_cols = int(process_cols)
    n = int(n)
    tile_size = int(tile_size)
    if process_rows <= 0 or process_cols <= 0:
        raise ValueError("process_rows and process_cols must be positive.")
    if n <= 0 or tile_size <= 0:
        raise ValueError("n and tile_size must be positive.")
    rank_array = _standard_grid_rank_map_attr(
        rank_map,
        process_rows=process_rows,
        process_cols=process_cols,
        caller="cusolvermp_syevd",
    )
    grid_mapping = _cusolvermp_grid_mapping_attr(
        rank_map,
        grid_mapping,
        process_rows=process_rows,
        process_cols=process_cols,
        caller="cusolvermp_syevd",
    )

    ensure_init_jaxmg_backend()

    out_type = (
        jax.ShapeDtypeStruct((n,), _real_dtype_for_eigenvalues(a.dtype)),
        jax.ShapeDtypeStruct(a.shape, a.dtype),
        jax.ShapeDtypeStruct(a.shape, a.dtype),
        jax.ShapeDtypeStruct((_CUSOLVERMP_SYEVD_STATUS_SIZE,), jnp.int32),
    )
    ffi_fn = partial(
        jax.ffi.ffi_call(
            "cusolvermp_syevd",
            out_type,
            input_layouts=(_ROW_MAJOR_JAX_LAYOUT,),
            output_layouts=(
                (0,),
                _ROW_MAJOR_JAX_LAYOUT,
                _ROW_MAJOR_JAX_LAYOUT,
                (0,),
            ),
            input_output_aliases={0: 1},
        ),
        process_rows=process_rows,
        process_cols=process_cols,
        grid_mapping=grid_mapping,
        rank_map=rank_array,
        n=n,
        tile_size=tile_size,
    )
    eigenvalues, _, eigenvectors, status = ffi_fn(a)
    return eigenvalues, eigenvectors, status


def cusolvermp_syevd_shardmap(
    a: Array,
    mesh: Mesh,
    matrix_specs: P,
    status_specs: P,
    *,
    process_rows: int,
    process_cols: int,
    rank_map=None,
    grid_mapping=None,
    n: int,
    tile_size: int,
) -> tuple[Array, Array, Array]:
    """Wrap the fused vector ``syevd`` FFI target in a JAX mesh context.

    This function uses ``jax.shard_map`` to coordinate the native FFI calls
    across all participating GPUs in the mesh. It ensures that every rank
    receives the correct static metadata (process-grid dimensions, tile size,
    and rank mapping) needed to enter the native cuSOLVERMp routine. The
    native backend borrows XLA's communicator during the FFI call.

    Args:
        a: Local padded shard of the matrix to diagonalize.
        mesh: JAX ``Mesh`` describing the device grid.
        matrix_specs: 2D ``PartitionSpec`` for the matrix and eigenvectors.
        status_specs: ``PartitionSpec`` for the replicated status vector.
        process_rows: Number of rows in the cuSOLVERMp process grid.
        process_cols: Number of columns in the cuSOLVERMp process grid.
        rank_map: Mapping from process-grid coordinates to communicator ranks.
        grid_mapping: cuSOLVERMp grid-mapping enum (row-major or column-major).
        n: Global dimension of the square matrix A.
        tile_size: Square tile dimension (``T_A``).

    Returns:
        A tuple ``(eigenvalues, eigenvectors, status)`` where eigenvalues are
        replicated across the mesh, eigenvectors are in the padded JAX-facing
        layout, and status is the native solver status.
    """
    if not isinstance(matrix_specs, P) or not isinstance(status_specs, P):
        raise TypeError("matrix_specs and status_specs must be PartitionSpec values.")

    @partial(jax.jit, donate_argnums=(0,))
    @partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=matrix_specs,
        out_specs=(P(None), matrix_specs, status_specs),
        check_vma=False,
    )
    def impl(_a):
        return cusolvermp_syevd(
            _a,
            process_rows=process_rows,
            process_cols=process_cols,
            rank_map=rank_map,
            grid_mapping=grid_mapping,
            n=n,
            tile_size=tile_size,
        )

    return impl(a)
