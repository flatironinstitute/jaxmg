"""Public cuSOLVERMp symmetric/Hermitian eigensolver wrapper.

Only eigenvector-producing SYEVD is exposed.  The Python layer validates JAX
array metadata, applies per-shard tile padding, and constructs the compiled FFI
call.  The native backend converts local storage to cuSOLVERMp's column-major
layout, redistributes into 2D block-cyclic form, calls ``cusolverMpSyevd``, and
returns eigenvalues plus eigenvectors in the original JAX-facing layout.
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
    process_rank_map_from_mesh,
    standard_grid_rank_map_attr,
    status_specs,
    validate_2d_matrix_specs,
)
from ._cusolvermp_status import _CUSOLVERMP_SYEVD_STATUS_SIZE
from ._layout_types import MatrixPadding2D, ProcessGrid, ProcessRankMap, TileShape
from ._layout_types import calculate_2d_padding
from ._setup import ensure_init_jaxmg_backend


def syevd(
    a: Array,
    T_A: int,
    mesh: Mesh | None = None,
    matrix_specs: P | Tuple[P] | List[P] | None = None,
    *,
    in_specs: P | Tuple[P] | List[P] | None = None,
    return_status: bool = False,
    pad: bool = True,
) -> Tuple[Array, Array] | Tuple[Array, Array, Array]:
    """Compute eigenvalues and eigenvectors using the multi-GPU syevd kernel.

    This is the high-level JAXMg symmetric/Hermitian eigensolver entry point.
    It accepts a block-sharded JAX matrix, prepares the tile-aligned local
    capacity required by cuSOLVERMp, calls the fused native backend, and returns
    eigenvalues together with eigenvectors in the same JAX-facing matrix layout
    as the input.

    Tip:
        If the shards of the matrix cannot be padded with tiles of size `T_A`
        we have to add padding to fit the last tile. This requires copying the
        matrix, which we want to avoid at all costs for large ``N``. Make sure
        you pick ``T_A`` large enough (>=128) and such that it can evenly cover
        the shards. In principle, increasing ``T_A`` will increase performance
        at the cost of memory, but depending on ``N``, the performance will
        saturate.

    Args:
        a (Array): A 2D symmetric/Hermitian matrix. Expected to be sharded
            across a 2D mesh with a matrix ``PartitionSpec`` such as
            ``P(<row_axis>, <col_axis>)``.
        T_A (int): Square tile width used by cuSOLVERMp. Each local shard
            dimension must be a multiple of ``T_A`` after padding.
        mesh (Mesh, optional): JAX mesh used for ``jax.shard_map``. If omitted,
            inferred from ``a.sharding.mesh``.
        matrix_specs (PartitionSpec or tuple/list[PartitionSpec], optional):
            PartitionSpec describing the matrix sharding. If omitted, inferred
            from ``a.sharding.spec``.
        in_specs: Backwards-compatible alias for ``matrix_specs``.
        return_status (bool, optional): If True append a native per-rank
            diagnostic status vector to the return values. Default is False.
        pad (bool, optional): If True (default) apply per-device padding to
            meet ``T_A`` requirements; if False the caller must supply already
            correct shapes.

    Returns:
        (eigenvalues, eigenvectors) or (eigenvalues, eigenvectors, status).

    Raises:
        TypeError: If dtypes or ``PartitionSpec`` inputs are unsupported.
        ValueError: If shapes, tile sizes, or mesh layouts are incompatible.

    Notes:
        - Only the eigenvector-producing mode is currently exposed.
        - The FFI call can donate the input buffer to enable zero-copy
          interaction with the native library.
        - Native code converts row-major JAX local storage to cuSOLVERMp's
          column-major local layout, redistributes to 2D block-cyclic layout,
          calls ``cusolverMpSyevd``, and redistributes the eigenvectors back.
        - If the native solver fails the outputs may contain NaNs and the
          status, when requested, will be non-zero.
    """
    if a.ndim != 2:
        raise ValueError("syevd expects a rank-2 matrix A.")
    _check_supported_syevd_dtype(a.dtype)
    if a.shape[0] != a.shape[1]:
        raise ValueError("syevd expects A to be square.")
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
        caller="syevd",
    )
    native_status_specs = status_specs(row_axis, col_axis, grid)
    tile_shape = TileShape(rows=int(T_A), cols=int(T_A))
    a_padding = calculate_2d_padding(
        logical_rows=a.shape[0],
        logical_cols=a.shape[1],
        grid=grid,
        tile_shape=tile_shape,
    )
    _check_padding_allowed(a_padding, pad=pad, caller="syevd")

    ensure_init_jaxmg_backend()

    impl = _syevd_compiled(
        mesh,
        matrix_specs,
        native_status_specs,
        grid,
        rank_map,
        rank_map.cusolvermp_grid_mapping,
        a_padding,
        n=a.shape[0],
        tile_size=tile_shape.rows,
        dtype=a.dtype,
    )
    eigenvalues, vectors, native_status = impl(a)
    if return_status:
        return eigenvalues, vectors, native_status
    return eigenvalues, vectors


_ROW_MAJOR_JAX_LAYOUT = (0, 1)


def _check_supported_syevd_dtype(dtype) -> None:
    """Validate that ``dtype`` maps to a cuSOLVERMp SYEVD entry point.

    cuSOLVERMp provides real and complex Hermitian/symmetric eigensolver entry
    points for single and double precision.  JAXMg rejects unsupported dtypes
    before tracing so the user sees a direct Python error instead of a native
    FFI failure.
    """
    if dtype not in (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128):
        raise TypeError("syevd supports float32, float64, complex64, and complex128.")


def _real_dtype_for_eigenvalues(dtype):
    """Return the real dtype used by cuSOLVERMp for eigenvalue outputs.

    Complex Hermitian eigensolvers still produce real eigenvalues.  This helper
    keeps the output ``ShapeDtypeStruct`` construction consistent with the
    native dispatch table.
    """
    if dtype == jnp.float32 or dtype == jnp.complex64:
        return jnp.float32
    if dtype == jnp.float64 or dtype == jnp.complex128:
        return jnp.float64
    raise TypeError(
        "cuSOLVERMp eigensolvers support float32, float64, complex64, "
        "and complex128."
    )


def _check_padding_allowed(
    padding: MatrixPadding2D,
    *,
    pad: bool,
    caller: str,
) -> None:
    """Enforce the public ``pad`` policy for a padded matrix argument.

    SYEVD uses a square matrix, so there is only one padding record to check.
    When ``pad=False`` the user is promising that the local shards are already
    tile-aligned; if that is not true, the native backend would receive invalid
    local capacity and the wrapper raises before tracing.
    """
    if not pad and padding.needs_padding:
        raise ValueError(
            f"{caller} requires tile-aligned local shards when pad=False. "
            "Use a tile size that divides each local shard or set pad=True."
        )


def _make_local_pad_fn(mesh: Mesh, matrix_specs: P, padding: MatrixPadding2D):
    """Build the shard-local bottom/right padding transform.

    The returned callable has the same sharding contract as the input matrix.
    If padding is unnecessary it is an identity function; otherwise it applies
    the local padding through ``jax.shard_map`` so the visible global mesh layout
    remains unchanged.
    """
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
    """Build the shard-local slice transform for eigenvector output.

    cuSOLVERMp operates on padded local capacity, while users expect
    eigenvectors shaped like the original matrix.  This helper creates the
    reverse local slice used after native code has moved the eigenvectors back
    into the JAX-facing layout.
    """
    return jax.shard_map(
        partial(_unpad_local_2d, local_rows=local_rows, local_cols=local_cols),
        mesh=mesh,
        in_specs=matrix_specs,
        out_specs=matrix_specs,
        check_vma=True,
    )


@lru_cache(maxsize=None)
def _syevd_compiled(
    mesh: Mesh,
    matrix_specs: P,
    native_status_specs: P,
    grid: ProcessGrid,
    rank_map: ProcessRankMap,
    grid_mapping: int,
    a_padding: MatrixPadding2D,
    *,
    n: int,
    tile_size: int,
    dtype,
):
    """Build and cache the full JAX-visible SYEVD execution pipeline.

    The cache key is the static solver configuration: mesh, sharding spec,
    process-grid shape, rank mapping, padded local shape, matrix size, tile
    size, and dtype.  Reusing this factory avoids rebuilding the same
    ``jax.shard_map`` and ``jax.jit`` wrapper for repeated eigensolves with the
    same layout metadata.
    """
    process_rows = grid.process_rows
    process_cols = grid.process_cols
    rank_array = standard_grid_rank_map_attr(
        rank_map,
        process_rows=process_rows,
        process_cols=process_cols,
        caller="cusolvermp_syevd",
    )
    grid_mapping = cusolvermp_grid_mapping_attr(
        rank_map,
        grid_mapping,
        process_rows=process_rows,
        process_cols=process_cols,
        caller="cusolvermp_syevd",
    )
    pad_a = _make_local_pad_fn(mesh, matrix_specs, a_padding)
    unpad_vectors = _make_local_unpad_fn(
        mesh,
        matrix_specs,
        local_rows=a_padding.local_logical_rows,
        local_cols=a_padding.local_logical_cols,
    )

    def syevd_ffi(_a: Array) -> tuple[Array, Array, Array]:
        """Call fused native redistribution and vector-producing ``syevd``.

        This function is mapped over local shards.  It declares the native FFI
        symbol, buffer layouts, result shapes, and static process-grid metadata;
        all heavy work is then performed by the C++/CUDA handler.
        """
        if _a.ndim != 2:
            raise ValueError("cusolvermp_syevd expects a rank-2 matrix buffer.")
        _check_supported_syevd_dtype(_a.dtype)

        out_type = (
            jax.ShapeDtypeStruct((int(n),), _real_dtype_for_eigenvalues(dtype)),
            jax.ShapeDtypeStruct(_a.shape, _a.dtype),
            jax.ShapeDtypeStruct(_a.shape, _a.dtype),
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
            n=int(n),
            tile_size=int(tile_size),
        )
        eigenvalues, _, eigenvectors, status = ffi_fn(_a)
        return eigenvalues, eigenvectors, status

    syevd_shardmap = jax.shard_map(
        syevd_ffi,
        mesh=mesh,
        in_specs=matrix_specs,
        out_specs=(P(None), matrix_specs, native_status_specs),
        check_vma=False,
    )

    @partial(jax.jit, donate_argnums=(0,))
    def impl(_a: Array) -> tuple[Array, Array, Array]:
        """Run padding, fused native SYEVD, and unpadding as one compiled path."""
        a_padded = pad_a(_a)
        eigenvalues, vectors_padded, native_status = syevd_shardmap(a_padded)
        vectors = unpad_vectors(vectors_padded)
        return eigenvalues, vectors, native_status

    return impl
