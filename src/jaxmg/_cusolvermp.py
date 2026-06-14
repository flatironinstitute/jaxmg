"""Production cuSOLVERMp FFI wrappers.

This module is intentionally small: public wrappers such as ``potrs_mp`` and
``syevd_mp`` own user-facing validation, padding, and redistribution policy.
The functions here only call native cuSOLVERMp FFI targets on buffers that are
already in cuSOLVERMp's 2D block-cyclic layout.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.sharding import Mesh, PartitionSpec as P

from ._setup import ensure_init_jaxmg_backend

_RECT_JAX_LAYOUT = (1, 0)
_CUSOLVERMP_POTRS_STATUS_SIZE = 40
_CUSOLVERMP_SYEVD_STATUS_SIZE = 36


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
    """Return the cuSOLVERMp grid-mapping enum for the validated rank map."""
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
    tile_size: int,
) -> tuple[Array, Array, Array]:
    """Run cuSOLVERMp ``potrf``/``potrs`` on 2D block-cyclic device buffers."""
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
    tile_size = int(tile_size)
    if process_rows <= 0 or process_cols <= 0:
        raise ValueError("process_rows and process_cols must be positive.")
    if n <= 0 or nrhs <= 0 or tile_size <= 0:
        raise ValueError("n, nrhs, and tile_size must be positive.")
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
            input_layouts=(_RECT_JAX_LAYOUT, _RECT_JAX_LAYOUT),
            output_layouts=(_RECT_JAX_LAYOUT, _RECT_JAX_LAYOUT, (0,)),
            input_output_aliases={0: 0, 1: 1},
        ),
        process_rows=process_rows,
        process_cols=process_cols,
        grid_mapping=grid_mapping,
        rank_map=rank_array,
        n=n,
        nrhs=nrhs,
        tile_size=tile_size,
    )
    return ffi_fn(a, b)


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
    tile_size: int,
) -> tuple[Array, Array, Array]:
    """Run production cuSOLVERMp ``potrs`` over a 2D process grid."""
    if not isinstance(matrix_specs, P) or not isinstance(status_specs, P):
        raise TypeError("matrix_specs and status_specs must be PartitionSpec values.")

    @partial(jax.jit, donate_argnums=(0, 1))
    @partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(matrix_specs, matrix_specs),
        out_specs=(matrix_specs, matrix_specs, status_specs),
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
    compute_vectors: bool,
) -> tuple[Array, Array, Array]:
    """Run cuSOLVERMp ``syevd`` on a 2D block-cyclic device buffer."""
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
            input_layouts=(_RECT_JAX_LAYOUT,),
            output_layouts=((0,), _RECT_JAX_LAYOUT, _RECT_JAX_LAYOUT, (0,)),
            input_output_aliases={0: 1},
        ),
        process_rows=process_rows,
        process_cols=process_cols,
        grid_mapping=grid_mapping,
        rank_map=rank_array,
        n=n,
        tile_size=tile_size,
        compute_vectors=1 if compute_vectors else 0,
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
    compute_vectors: bool,
) -> tuple[Array, Array, Array]:
    """Run production cuSOLVERMp ``syevd`` over a 2D process grid."""
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
            compute_vectors=compute_vectors,
        )

    return impl(a)
