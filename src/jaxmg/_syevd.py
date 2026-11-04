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
    mark_varying,
    pad_local_2d,
    unpad_local_2d,
    place_for_native_work,
    prepare_input_matrix_layout,
    prepare_local_matrix_layout,
    restore_matrix_result,
)
from ._cusolvermp_status import _CUSOLVERMP_SYEVD_STATUS_SIZE
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
    default it returns eigenvalues together with eigenvectors. Eigenvectors are
    restored to replicated sharding when ``a`` is replicated; otherwise they
    use ``matrix_specs``. Setting ``return_eigenvectors=False`` selects
    cuSOLVERMp's eigenvalues-only mode and avoids allocating or restoring a
    matrix-sized eigenvector result.

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
        mesh (Mesh or AbstractMesh, optional): JAX mesh used for ``jax.shard_map``.
            If omitted, read off the sharding of ``a`` (its type inside
            ``jax.jit``), or taken from the context mesh.
        matrix_specs (PartitionSpec or tuple/list[PartitionSpec], optional):
            PartitionSpec describing the matrix sharding. If omitted, read
            off the sharding of ``a``, defaulting to the mesh axes in order.
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
        ``(eigenvalues, status)``. Eigenvalues are replicated.

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
        - If the native solver fails, the eigenvalues and eigenvectors are
          filled with NaN and the status, when requested, is non-zero.
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

    impl = _syevd_compiled(
        layout, int(T_A), return_eigenvectors=return_eigenvectors, pad=pad, donate=donate,
    )
    outputs = impl(a)
    if not return_eigenvectors:
        _, eigenvalues, native_status = outputs
        if return_status:
            return eigenvalues, native_status
        return eigenvalues

    _, eigenvalues, vectors, native_status = outputs
    if return_status:
        return eigenvalues, vectors, native_status
    return eigenvalues, vectors


def syevd_jit_ctx(
    a: Array,
    T_A: int,
    mesh: Mesh | AbstractMesh | None = None,
    matrix_specs: P | Tuple[P] | List[P] | None = None,
    *,
    in_specs: P | Tuple[P] | List[P] | None = None,
    return_eigenvectors: bool = True,
    pad: bool = True,
    _prepared=None,
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
        mesh (Mesh or AbstractMesh, optional): JAX mesh used for ``jax.shard_map``.
            If omitted, read off the sharding of ``a`` (its type inside
            ``jax.jit``), or taken from the context mesh.
        matrix_specs (PartitionSpec or tuple/list[PartitionSpec], optional):
            PartitionSpec describing the matrix sharding. If omitted, read
            off the sharding of ``a``, defaulting to the mesh axes in order.
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
        input alias in ``matrix_specs`` and includes any padding applied by the
        normal pipeline. It is opaque and is not restored to replicated
        sharding. Eigenvectors are restored when ``a`` is replicated,
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
    # Preserve concrete-array layout metadata across the public JIT boundary.
    layout = _prepared if _prepared is not None else _prepare_syevd_call(
        a,
        T_A,
        mesh,
        matrix_specs,
        in_specs=in_specs,
        return_eigenvectors=return_eigenvectors,
        pad=pad,
        caller="syevd_jit_ctx",
    )

    specs = layout.matrix_specs
    out_specs = (specs, P()) + ((specs,) if return_eigenvectors else ())
    with jax.sharding.use_abstract_mesh(layout.mesh.abstract_mesh):
        a = place_for_native_work(a, mesh=layout.mesh, matrix_specs=specs)
        outputs = jax.shard_map(
            partial(syevd_shardmap_ctx, T_A=T_A, matrix_specs=specs,
                    return_eigenvectors=return_eigenvectors, pad=pad),
            mesh=layout.mesh,
            in_specs=specs,
            out_specs=out_specs + (layout.native_status_specs,),
        )(a)
        restore = partial(
            restore_matrix_result, result_specs=layout.matrix_result_specs,
            mesh=layout.mesh, matrix_specs=specs,
        )
        if not return_eigenvectors:
            return outputs
        work, values, vectors, status = outputs
        return work, values, restore(vectors), status


def syevd_shardmap_ctx(
    a: Array,
    T_A: int,
    matrix_specs: P | Tuple[P] | List[P] | None = None,
    *,
    in_specs: P | Tuple[P] | List[P] | None = None,
    return_eigenvectors: bool = True,
    pad: bool = True,
) -> Union[Tuple[Array, Array, Array], Tuple[Array, Array, Array, Array]]:
    """Compute a distributed eigendecomposition inside a caller's shard map.

    The input and eigenvectors are local matrix blocks. Eigenvalues are
    replicated by the native backend and use ``P()`` in the output specification.

    Args:
        a: Local block of a symmetric or Hermitian global matrix.
        T_A: Square tile width used by cuSOLVERMp.
        matrix_specs: A's partition specification in the enclosing shard map;
            defaults to the context mesh axes in order.
        in_specs: Backwards-compatible alias for ``matrix_specs``.
        return_eigenvectors: Static Python bool selecting whether to compute vectors.
        pad: Allow tile-capacity padding; otherwise require aligned blocks.

    Returns:
        tuple: ``(a_work, eigenvalues, eigenvectors, status)``, or
        ``(a_work, eigenvalues, status)`` without vectors. ``a_work`` is padded
        local storage, eigenvectors are unpadded local blocks, and status is
        this rank's diagnostic vector. Return ``a_work`` from the outer jitted
        function when donating A; eigenvectors use a separate allocation.

    Raises:
        TypeError: If the dtype, specification or vector mode is unsupported.
        ValueError: If called outside shard_map or the global matrix is invalid.
    """
    layout = _prepare_syevd_call(
        a, T_A, None, matrix_specs, in_specs=in_specs,
        return_eigenvectors=return_eigenvectors, pad=pad,
        caller="syevd_shardmap_ctx", local=True,
    )

    ensure_init_jaxmg_backend()
    a_padding = layout.padding
    matrix_specs, grid = layout.matrix_specs, layout.grid
    partition_slots = layout.partition_slots
    n = a_padding.logical_rows
    tile_size, dtype = int(T_A), a.dtype
    process_rows = grid.process_rows
    process_cols = grid.process_cols
    slots_attr = np.asarray(partition_slots, dtype=np.int64)
    if return_eigenvectors:
        ffi_target = "cusolvermp_syevd"
    else:
        ffi_target = "cusolvermp_syevd_values"

    # Stage 1: provide tile-aligned capacity for the local input.
    a = pad_local_2d(a, row_padding=a_padding.row_padding_per_process,
                      col_padding=a_padding.col_padding_per_process)

    # Stage 2: allocate only the outputs requested by the selected mode.
    eigenvalue_type = jax.ShapeDtypeStruct(
        (int(n),), _real_dtype_for_eigenvalues(dtype)
    )
    matrix_type = jax.ShapeDtypeStruct(a.shape, a.dtype)
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
    values, work, *rest = ffi_fn(a)
    # Stage 3: restore local vector blocks; eigenvalues stay replicated.
    axes = tuple(axis for axis, size in zip(matrix_specs, (process_rows, process_cols))
                 if axis is not None and size > 1)
    work = mark_varying(work, axes)
    status = mark_varying(rest[-1], axes)
    if not return_eigenvectors:
        return work, values, status
    vectors = unpad_local_2d(rest[0], local_rows=a_padding.local_logical_rows,
                            local_cols=a_padding.local_logical_cols)
    return work, values, mark_varying(vectors, axes), status


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
    local: bool = False,
):
    """Validate an eigensolver call and derive its distributed layout.

    Preparation proceeds as follows:

    1. Validate that A is a square matrix with a supported dtype.
    2. Validate the tile size and static ``return_eigenvectors`` mode.
    3. Resolve the mesh, process grid, rank map, and tile-aligned layout shared
       by A and the optional eigenvector output. Local blocks use the enclosing
       mesh to reconstruct the global dimensions.
    """
    if a.ndim != 2:
        raise ValueError(f"{caller} expects a rank-2 matrix A.")
    _check_supported_syevd_dtype(a.dtype)
    if not local and a.shape[0] != a.shape[1]:
        raise ValueError(f"{caller} expects A to be square.")
    if int(tile_size) <= 0:
        raise ValueError("T_A must be positive.")
    if not isinstance(return_eigenvectors, bool):
        raise TypeError("return_eigenvectors must be a Python bool.")
    if local:
        layout = prepare_local_matrix_layout(
            a, tile_size, matrix_specs=matrix_specs, in_specs=in_specs,
            pad=pad, caller=caller,
        )
        if layout.padding.logical_rows != layout.padding.logical_cols:
            raise ValueError(f"{caller} expects A to be square.")
        return layout
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
def _syevd_compiled(prepared, tile_size: int, *, return_eigenvectors: bool, pad: bool, donate: bool):
    """Cache the public JIT wrapper around the global context interface."""
    return jax.jit(
        partial(syevd_jit_ctx, T_A=tile_size, return_eigenvectors=return_eigenvectors,
                pad=pad, _prepared=prepared),
        donate_argnums=(0,) if donate else (),
    )
