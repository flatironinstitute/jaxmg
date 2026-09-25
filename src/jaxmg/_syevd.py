"""Public cuSOLVERMp symmetric/Hermitian eigensolver wrapper.

The Python layer validates JAX array metadata, applies per-shard tile padding,
and constructs the compiled FFI call for eigenvalues-only or vector-producing
SYEVD. The native backend converts local storage to cuSOLVERMp's column-major
layout, redistributes into 2D block-cyclic form, and calls
``cusolverMpSyevd``. Eigenvectors are restored to the original JAX-facing
layout only when requested.
"""

from __future__ import annotations

from functools import lru_cache, partial
from typing import List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.sharding import AbstractMesh, Mesh, PartitionSpec as P

from ._cusolvermp_layout import (
    make_local_pad_fn,
    make_local_unpad_fn,
    prepare_input_matrix_layout,
    use_abstract_mesh_decorator,
)
from ._cusolvermp_status import _CUSOLVERMP_SYEVD_STATUS_SIZE
from ._layout_types import MatrixPadding2D, ProcessGrid
from ._setup import ensure_init_jaxmg_backend


def syevd(
    a: Array,
    T_A: int,
    mesh: Mesh | AbstractMesh | None = None,
    matrix_specs: P | Tuple[P] | List[P] | None = None,
    *,
    in_specs: P | Tuple[P] | List[P] | None = None,
    return_eigenvectors: bool = True,
    return_status: bool = False,
    pad: bool = True,
    donate: bool = True,
) -> Array | Tuple[Array, Array] | Tuple[Array, Array, Array]:
    """Compute eigenvalues and optionally eigenvectors using multi-GPU SYEVD.

    This is the high-level JAXMg symmetric/Hermitian eigensolver entry point.
    It accepts a block-sharded JAX matrix, prepares the tile-aligned local
    capacity required by cuSOLVERMp, and calls the fused native backend. By
    default it returns eigenvalues together with eigenvectors in the same
    JAX-facing matrix layout as the input. Setting ``return_eigenvectors=False``
    selects cuSOLVERMp's eigenvalues-only mode and avoids allocating or
    restoring a matrix-sized eigenvector result.

    Note:
        If a local shard dimension is not divisible by ``T_A``, ``pad=True``
        allocates additional tile-aligned capacity before the native call.
        Choosing a tile size that divides the local dimensions avoids this
        allocation. Performance depends on the matrix size, process grid, and
        tile size.

    Args:
        a (Array): A 2D symmetric/Hermitian matrix sharded over a one- or
            two-axis device mesh, for example with ``P(<row_axis>)`` or
            ``P(<row_axis>, <col_axis>)``.
        T_A (int): Square tile width used by cuSOLVERMp. Each local shard
            dimension must be a multiple of ``T_A`` after padding.
        mesh (Mesh, optional): JAX mesh used for ``jax.shard_map``. If omitted,
            inferred from ``a.sharding.mesh``.
        matrix_specs (PartitionSpec or tuple/list[PartitionSpec], optional):
            PartitionSpec describing the matrix sharding. If omitted, inferred
            from ``a.sharding.spec``.
        in_specs: Backwards-compatible alias for ``matrix_specs``.
        return_eigenvectors (bool, optional): If True (default), compute and
            return eigenvectors as well as eigenvalues. If False, return only
            eigenvalues and use the shorter native values-only workflow. This
            must be a Python ``bool`` fixed while tracing.
        return_status (bool, optional): If True append a native per-rank
            diagnostic status vector to the return values. Default is False.
        pad (bool, optional): If True (default) apply per-device padding to
            meet ``T_A`` requirements; if False the caller must supply already
            correct shapes.
        donate (bool, optional): If True (default) the input buffers may be
            donated to the native call for zero-copy execution, which means
            they are deleted and cannot be used again. Pass False to preserve
            them, at the cost of keeping the original and working buffers in
            memory simultaneously.

    Returns:
        If ``return_eigenvectors=True``, ``(eigenvalues, eigenvectors)`` or
        ``(eigenvalues, eigenvectors, status)``. If False, ``eigenvalues`` or
        ``(eigenvalues, status)``.

    Raises:
        TypeError: If dtypes or ``PartitionSpec`` inputs are unsupported.
        ValueError: If shapes, tile sizes, or mesh layouts are incompatible.

    Notes:
        - Unless ``donate=False``, the input buffer is donated for zero-copy
          interaction with the native library.
        - Native code converts row-major JAX local storage to cuSOLVERMp's
          column-major local layout and redistributes to 2D block-cyclic
          layout. The eigenvector mode also redistributes its matrix result
          back to the original JAX layout.
        - If the native solver fails the outputs may contain NaNs and the
          status, when requested, will be non-zero.
    """
    layout = _prepare_syevd_call(
        a,
        T_A,
        mesh,
        matrix_specs,
        in_specs=in_specs,
        return_eigenvectors=return_eigenvectors,
        pad=pad,
        caller="syevd",
    )

    ensure_init_jaxmg_backend()

    impl = _syevd_compiled(
        layout.mesh,
        layout.matrix_specs,
        layout.native_status_specs,
        layout.grid,
        layout.partition_slots,
        layout.padding,
        n=a.shape[0],
        tile_size=layout.tile_shape.rows,
        dtype=a.dtype,
        return_eigenvectors=return_eigenvectors,
        donate=donate,
    )
    outputs = impl(a)
    if not return_eigenvectors:
        eigenvalues, _, native_status = outputs
        if return_status:
            return eigenvalues, native_status
        return eigenvalues

    eigenvalues, _, vectors, native_status = outputs
    if return_status:
        return eigenvalues, vectors, native_status
    return eigenvalues, vectors


def syevd_shardmap_ctx(
    a: Array,
    T_A: int,
    mesh: Mesh | AbstractMesh | None = None,
    matrix_specs: P | Tuple[P] | List[P] | None = None,
    *,
    in_specs: P | Tuple[P] | List[P] | None = None,
    return_eigenvectors: bool = True,
    pad: bool = True,
) -> Tuple[Array, Array, Array] | Tuple[Array, Array, Array, Array]:
    """Compute eigenvalues and optional eigenvectors while exposing work storage.

    This helper is the lower-level variant of :func:`jaxmg.syevd` intended for
    contexts where the caller controls the outer ``jax.jit`` boundary. It
    performs the same validation, local padding, shard-map construction, and
    fused cuSOLVERMp FFI call as the public eigensolver, but does not introduce
    an internal ``jax.jit``. Instead, it returns the native matrix work buffer
    so an outer JIT can donate ``a`` into an ``A``-sized output.

    cuSOLVERMp uses separate distributed matrices for its overwritten input
    ``d_A`` and eigenvector output ``d_Q``. The context interface therefore
    returns ``a_work`` to provide the outer compiled function with an
    ``A``-sized alias target for the donated input. When eigenvectors are
    requested, they occupy a separate matrix-sized allocation. Values-only
    execution omits that allocation and the associated reverse redistribution.

    Note:
        If a local shard dimension is not divisible by ``T_A``, ``pad=True``
        allocates additional tile-aligned capacity before the native call.
        Choosing a tile size that divides the local dimensions avoids this
        allocation. Performance depends on the matrix size, process grid, and
        tile size.

    Args:
        a (Array): A 2D symmetric/Hermitian matrix sharded over a one- or
            two-axis device mesh, for example with ``P(<row_axis>)`` or
            ``P(<row_axis>, <col_axis>)``.
        T_A (int): Square tile width used by cuSOLVERMp. Each local shard
            dimension must be a multiple of ``T_A`` after padding.
        mesh (Mesh, optional): JAX mesh used for ``jax.shard_map``. If omitted,
            inferred from ``a.sharding.mesh``.
        matrix_specs (PartitionSpec or tuple/list[PartitionSpec], optional):
            PartitionSpec describing the matrix sharding. If omitted, inferred
            from ``a.sharding.spec``.
        in_specs: Backwards-compatible alias for ``matrix_specs``.
        return_eigenvectors (bool, optional): If True (default), compute and
            return eigenvectors as well as eigenvalues. If False, use the
            eigenvalues-only native workflow. This must be a Python ``bool``
            fixed while tracing.
        pad (bool, optional): If True (default) apply per-device padding to
            meet ``T_A`` requirements; if False the caller must supply already
            correct shapes.

    Returns:
        tuple: ``(a_work, eigenvalues, eigenvectors, status)`` when
        ``return_eigenvectors=True``, otherwise
        ``(a_work, eigenvalues, status)``. ``a_work`` preserves the donated
        input alias and includes any padding applied by the normal pipeline.
        ``eigenvalues`` is replicated and real-valued, and ``status`` is the
        native per-rank diagnostic vector.

    Raises:
        TypeError: If dtypes or ``PartitionSpec`` inputs are unsupported.
        ValueError: If shapes, tile sizes, or mesh layouts are incompatible.

    Notes:
        - Callers using ``jax.jit(..., donate_argnums=(0,))`` must keep
          ``a_work`` in the outer function's returned pytree so the donated
          matrix has an ``A``-sized output alias.
        - Returning ``a_work`` does not remove cuSOLVERMp's separate
          eigenvector allocation when eigenvectors are requested.
        - Native code converts row-major JAX local storage to cuSOLVERMp's
          column-major local layout, redistributes to 2D block-cyclic layout,
          calls ``cusolverMpSyevd``, and redistributes eigenvectors back when
          requested.
    """
    layout = _prepare_syevd_call(
        a,
        T_A,
        mesh,
        matrix_specs,
        in_specs=in_specs,
        return_eigenvectors=return_eigenvectors,
        pad=pad,
        caller="syevd_shardmap_ctx",
    )

    ensure_init_jaxmg_backend()

    impl = _syevd_pipeline(
        layout.mesh,
        layout.matrix_specs,
        layout.native_status_specs,
        layout.grid,
        layout.partition_slots,
        layout.padding,
        n=a.shape[0],
        tile_size=layout.tile_shape.rows,
        dtype=a.dtype,
        return_eigenvectors=return_eigenvectors,
    )
    outputs = impl(a)
    if not return_eigenvectors:
        eigenvalues, a_work, native_status = outputs
        return a_work, eigenvalues, native_status

    eigenvalues, a_work, eigenvectors, native_status = outputs
    return a_work, eigenvalues, eigenvectors, native_status


def _prepare_syevd_call(
    a: Array,
    tile_size: int,
    mesh: Mesh | AbstractMesh | None,
    matrix_specs: P | Tuple[P] | List[P] | None,
    *,
    in_specs: P | Tuple[P] | List[P] | None,
    return_eigenvectors: bool,
    pad: bool,
    caller: str,
):
    """Validate an eigensolver call and derive its distributed layout.

    Preparation proceeds as follows:

    1. Validate that A is a square matrix with a supported dtype.
    2. Validate the tile size and static ``return_eigenvectors`` mode.
    3. Resolve the mesh, process grid, rank map, and tile-aligned layout shared
       by A and the optional eigenvector output.
    """
    if a.ndim != 2:
        raise ValueError(f"{caller} expects a rank-2 matrix A.")
    _check_supported_syevd_dtype(a.dtype)
    if a.shape[0] != a.shape[1]:
        raise ValueError(f"{caller} expects A to be square.")
    if int(tile_size) <= 0:
        raise ValueError("T_A must be positive.")
    if not isinstance(return_eigenvectors, bool):
        raise TypeError("return_eigenvectors must be a Python bool.")
    return prepare_input_matrix_layout(
        a,
        tile_size,
        mesh=mesh,
        matrix_specs=matrix_specs,
        in_specs=in_specs,
        pad=pad,
        caller=caller,
    )


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

    Complex Hermitian inputs produce real eigenvalues in the corresponding
    component precision.
    """
    if dtype == jnp.float32 or dtype == jnp.complex64:
        return jnp.float32
    if dtype == jnp.float64 or dtype == jnp.complex128:
        return jnp.float64
    raise TypeError(
        "cuSOLVERMp eigensolvers support float32, float64, complex64, "
        "and complex128."
    )


_ROW_MAJOR_JAX_LAYOUT = (0, 1)


@lru_cache(maxsize=None)
def _syevd_pipeline(
    mesh: Mesh | AbstractMesh,
    matrix_specs: P,
    native_status_specs: P,
    grid: ProcessGrid,
    partition_slots: tuple[int, ...],
    a_padding: MatrixPadding2D,
    *,
    n: int,
    tile_size: int,
    dtype,
    return_eigenvectors: bool,
):
    """Build and cache the unjitted JAX-visible SYEVD execution pipeline.

    The cache key is the static solver configuration: mesh, sharding spec,
    process-grid shape, partition mapping, padded local shape, matrix size, tile
    size, dtype, and output mode. Reusing this factory avoids rebuilding the
    same ``jax.shard_map`` structure for repeated eigensolves with identical
    layout metadata.
    """
    process_rows = grid.process_rows
    process_cols = grid.process_cols
    slots_attr = np.asarray(partition_slots, dtype=np.int64)
    pad_a = make_local_pad_fn(mesh, matrix_specs, a_padding)
    if return_eigenvectors:
        unpad_vectors = make_local_unpad_fn(mesh, matrix_specs, a_padding)
        ffi_target = "cusolvermp_syevd"
        out_specs = (
            P(None),
            matrix_specs,
            matrix_specs,
            native_status_specs,
        )
    else:
        unpad_vectors = None
        ffi_target = "cusolvermp_syevd_values"
        out_specs = (P(None), matrix_specs, native_status_specs)

    def syevd_ffi(_a: Array):
        """Call fused native redistribution and the selected SYEVD mode.

        This function is mapped over local shards.  It declares the native FFI
        symbol, buffer layouts, result shapes, and static process-grid metadata;
        all heavy work is then performed by the C++/CUDA handler.
        """
        if _a.ndim != 2:
            raise ValueError("cusolvermp_syevd expects a rank-2 matrix buffer.")
        _check_supported_syevd_dtype(_a.dtype)

        eigenvalue_type = jax.ShapeDtypeStruct(
            (int(n),), _real_dtype_for_eigenvalues(dtype)
        )
        matrix_type = jax.ShapeDtypeStruct(_a.shape, _a.dtype)
        status_type = jax.ShapeDtypeStruct(
            (_CUSOLVERMP_SYEVD_STATUS_SIZE,), jnp.int32
        )
        if return_eigenvectors:
            out_type = (eigenvalue_type, matrix_type, matrix_type, status_type)
            output_layouts = (
                (0,),
                _ROW_MAJOR_JAX_LAYOUT,
                _ROW_MAJOR_JAX_LAYOUT,
                (0,),
            )
        else:
            out_type = (eigenvalue_type, matrix_type, status_type)
            output_layouts = ((0,), _ROW_MAJOR_JAX_LAYOUT, (0,))

        ffi_fn = partial(
            jax.ffi.ffi_call(
                ffi_target,
                out_type,
                input_layouts=(_ROW_MAJOR_JAX_LAYOUT,),
                output_layouts=output_layouts,
                input_output_aliases={0: 1},
            ),
            process_rows=process_rows,
            process_cols=process_cols,
            partition_slots=slots_attr,
            n=int(n),
            tile_size=int(tile_size),
        )
        return ffi_fn(_a)

    syevd_shardmap = jax.shard_map(
        syevd_ffi,
        mesh=mesh,
        in_specs=matrix_specs,
        out_specs=out_specs,
        check_vma=False,
    )

    @use_abstract_mesh_decorator(mesh)
    def impl(_a: Array):
        """Run padding, fused native SYEVD, and optional vector unpadding."""
        a_padded = pad_a(_a)
        outputs = syevd_shardmap(a_padded)
        if not return_eigenvectors:
            return outputs

        eigenvalues, work_padded, vectors_padded, native_status = outputs
        vectors = unpad_vectors(vectors_padded)
        return eigenvalues, work_padded, vectors, native_status

    return impl


@lru_cache(maxsize=None)
def _syevd_compiled(
    mesh: Mesh | AbstractMesh,
    matrix_specs: P,
    native_status_specs: P,
    grid: ProcessGrid,
    partition_slots: tuple[int, ...],
    a_padding: MatrixPadding2D,
    *,
    n: int,
    tile_size: int,
    dtype,
    return_eigenvectors: bool,
    donate: bool,
):
    """Build and cache the internally jitted public SYEVD pipeline."""
    pipeline = _syevd_pipeline(
        mesh,
        matrix_specs,
        native_status_specs,
        grid,
        partition_slots,
        a_padding,
        n=n,
        tile_size=tile_size,
        dtype=dtype,
        return_eigenvectors=return_eigenvectors,
    )

    @partial(jax.jit, donate_argnums=(0,) if donate else ())
    def impl(_a: Array):
        """Run the cached SYEVD pipeline behind the public convenience API."""
        return pipeline(_a)

    return impl
