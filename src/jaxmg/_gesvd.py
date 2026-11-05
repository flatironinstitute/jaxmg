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
    prepare_rectangular_matrix_layout,
    restore_matrix_result,
)
from ._cusolvermp_status import _CUSOLVERMP_GESVD_STATUS_SIZE
from ._setup import ensure_init_jaxmg_backend


def gesvd(
    a: Array,
    T_A: int,
    mesh: Mesh | AbstractMesh | None = None,
    matrix_specs: P | Tuple[P] | List[P] | None = None,
    *,
    in_specs: P | Tuple[P] | List[P] | None = None,
    compute_u: bool = True,
    compute_vh: bool = True,
    full_matrices: bool = False,
    return_status: bool = False,
    pad: bool = True,
    donate: bool = True,
) -> Array | tuple[Array, ...]:
    """Compute a distributed singular-value decomposition with cuSOLVERMp.

    This is the high-level JAXMg interface for rectangular or square matrices.
    For an ``m x n`` input, let ``k = min(m, n)``. By default the routine
    returns the reduced decomposition ``(U, s, Vh)`` with shapes ``(m, k)``,
    ``(k,)``, and ``(k, n)``. ``full_matrices=True`` instead requests U with
    shape ``(m, m)`` and Vh with shape ``(n, n)``. Either vector matrix can be
    disabled independently so JAX does not allocate or redistribute an output
    that the application does not require.

    Requested matrix outputs are restored to replicated sharding when ``a`` is
    replicated; otherwise they use ``matrix_specs``. Internally, A, U, and Vh
    use the same solver ``PartitionSpec``, so their logical dimensions must be
    divisible by the corresponding process-grid dimensions. Tile padding is
    applied separately when their local dimensions are not divisible by
    ``T_A``.

    Args:
        a (Array): A rank-2 real or complex matrix sharded over a one- or
            two-axis device mesh.
        T_A (int): Square cuSOLVERMp tile width. GESVD supports rectangular
            matrices but requires equal row and column tile dimensions.
        mesh (Mesh or AbstractMesh, optional): JAX mesh used by ``jax.shard_map``.
            If omitted, read off the sharding of ``a`` (its type inside
            ``jax.jit``), or taken from the context mesh.
        matrix_specs (PartitionSpec or tuple/list[PartitionSpec], optional):
            Rank-2 matrix sharding. If omitted, read off the sharding of
            ``a``, defaulting to the mesh axes in order.
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
        donate (bool, optional): If True (default) the input buffers may be
            donated to the native call for zero-copy execution, which means
            they are deleted and cannot be used again. Pass False to preserve
            them, at the cost of keeping the original and working buffers in
            memory simultaneously.

    Returns:
        The selected numerical outputs follow NumPy/JAX SVD order:
        ``(U, s, Vh)`` when both vectors are requested, ``(U, s)`` for U only,
        ``(s, Vh)`` for Vh only, and ``s`` for values only. If
        ``return_status=True``, the status vector is appended to that result.
        Singular values are replicated.

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
        - If the native solver fails, the singular values and vectors are
          filled with NaN; use
          ``return_status=True`` when per-rank diagnostics are required.
    """
    layout, u_padding, vh_padding = _prepare_gesvd_call(
        a,
        T_A,
        mesh,
        matrix_specs,
        in_specs=in_specs,
        compute_u=compute_u,
        compute_vh=compute_vh,
        full_matrices=full_matrices,
        pad=pad,
        caller="gesvd",
    )

    impl = _gesvd_compiled(
        (layout, u_padding, vh_padding), int(T_A),
        compute_u=compute_u, compute_vh=compute_vh, full_matrices=full_matrices,
        pad=pad, donate=donate,
    )
    _, *result, native_status = impl(a)
    if not compute_u and not compute_vh:
        return (result[0], native_status) if return_status else result[0]
    return (*result, native_status) if return_status else tuple(result)


def gesvd_jit_ctx(
    a: Array,
    T_A: int,
    mesh: Mesh | AbstractMesh | None = None,
    matrix_specs: P | Tuple[P] | List[P] | None = None,
    *,
    in_specs: P | Tuple[P] | List[P] | None = None,
    compute_u: bool = True,
    compute_vh: bool = True,
    full_matrices: bool = False,
    pad: bool = True,
    _prepared=None,
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
        mesh (Mesh or AbstractMesh, optional): JAX mesh used by ``jax.shard_map``.
            If omitted, read off the sharding of ``a`` (its type inside
            ``jax.jit``), or taken from the context mesh.
        matrix_specs (PartitionSpec or tuple/list[PartitionSpec], optional):
            Rank-2 matrix sharding. If omitted, read off the sharding of
            ``a``, defaulting to the mesh axes in order.
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
        selected vector outputs. ``a_work`` is opaque, remains in
        ``matrix_specs``, and includes any local padding. Requested vector
        matrices are restored when ``a`` is replicated. ``status`` is always
        returned so an enclosing compiled function can propagate diagnostics.

    Raises:
        TypeError: If the dtype, static mode flags, or sharding specification is
            unsupported.
        ValueError: If a shape, tile size, process grid, or requested output
            layout is incompatible with cuSOLVERMp.
    """
    # Preserve concrete-array layout metadata across the public JIT boundary.
    if _prepared is None:
        _prepared = _prepare_gesvd_call(
            a, T_A, mesh, matrix_specs, in_specs=in_specs,
            compute_u=compute_u, compute_vh=compute_vh,
            full_matrices=full_matrices, pad=pad, caller="gesvd_jit_ctx",
        )
    layout, _, _ = _prepared

    specs = layout.matrix_specs
    out_specs = (specs,) + (specs,) * compute_u + (P(),) + (specs,) * compute_vh
    with jax.sharding.use_abstract_mesh(layout.mesh.abstract_mesh):
        a = place_for_native_work(a, mesh=layout.mesh, matrix_specs=specs)
        outputs = jax.shard_map(
            partial(gesvd_shardmap_ctx, T_A=T_A, matrix_specs=specs,
                    compute_u=compute_u, compute_vh=compute_vh,
                    full_matrices=full_matrices, pad=pad),
            mesh=layout.mesh,
            in_specs=specs,
            out_specs=out_specs + (layout.native_status_specs,),
        )(a)
        restore = partial(
            restore_matrix_result, result_specs=layout.matrix_result_specs,
            mesh=layout.mesh, matrix_specs=specs,
        )
        work, *results, status = outputs
        if compute_u:
            results[0] = restore(results[0])
        if compute_vh:
            results[-1] = restore(results[-1])
        return (work, *results, status)


def gesvd_shardmap_ctx(
    a: Array,
    T_A: int,
    matrix_specs: P | Tuple[P] | List[P] | None = None,
    *,
    in_specs: P | Tuple[P] | List[P] | None = None,
    compute_u: bool = True,
    compute_vh: bool = True,
    full_matrices: bool = False,
    pad: bool = True,
) -> tuple[Array, ...]:
    """Compute an SVD inside a caller's ``jax.shard_map``.

    The enclosing mesh determines the global input and vector dimensions.
    Singular values are replicated; requested singular vectors are local blocks.

    Args:
        a: Local block of the global input matrix.
        T_A: Square tile width used by cuSOLVERMp.
        matrix_specs: A's partition specification in the enclosing shard map;
            defaults to the context mesh axes in order.
        in_specs: Backwards-compatible alias for ``matrix_specs``.
        compute_u: Static Python bool selecting left singular vectors.
        compute_vh: Static Python bool selecting conjugate-transposed right vectors.
        full_matrices: Request full rather than reduced vector matrices.
        pad: Allow tile-capacity padding; otherwise require aligned blocks.

    Returns:
        tuple: ``(a_work, U, s, Vh, status)``, omitting U or Vh when disabled.
            The opaque local ``a_work`` includes padding and exposes the input
            donation alias. It cannot share storage with either vector output.
            Status contains this rank's diagnostics.

    Raises:
        TypeError: If the dtype, modes, or partition specification is unsupported.
        ValueError: If called outside shard_map or input/output layouts are invalid.
    """
    layout, u_padding, vh_padding = _prepare_gesvd_call(
        a, T_A, None, matrix_specs, in_specs=in_specs, compute_u=compute_u,
        compute_vh=compute_vh, full_matrices=full_matrices, pad=pad,
        caller="gesvd_shardmap_ctx", local=True,
    )

    ensure_init_jaxmg_backend()
    a_padding = layout.padding
    matrix_specs, grid = layout.matrix_specs, layout.grid
    partition_slots = layout.partition_slots
    m, n = a_padding.logical_rows, a_padding.logical_cols
    tile_size, dtype = int(T_A), a.dtype
    process_rows = grid.process_rows
    process_cols = grid.process_cols
    slots_attr = np.asarray(partition_slots, dtype=np.int64)
    if compute_u and compute_vh:
        ffi_target = "cusolvermp_gesvd_uv"
    elif compute_u:
        ffi_target = "cusolvermp_gesvd_u"
    elif compute_vh:
        ffi_target = "cusolvermp_gesvd_vh"
    else:
        ffi_target = "cusolvermp_gesvd_values"

    # Stage 1: provide tile-aligned capacity for the local input.
    a = pad_local_2d(a, row_padding=a_padding.row_padding_per_process,
                      col_padding=a_padding.col_padding_per_process)

    # Stage 2: allocate only the requested vectors alongside the work alias.
    singular_type = jax.ShapeDtypeStruct(
        (min(int(m), int(n)),), _real_dtype_for_singular_values(dtype)
    )
    work_type = jax.ShapeDtypeStruct(a.shape, a.dtype)
    status_type = jax.ShapeDtypeStruct(
        (_CUSOLVERMP_GESVD_STATUS_SIZE,), jnp.int32
    )
    output_types = [singular_type, work_type]
    output_layouts = [(0,), _ROW_MAJOR_JAX_LAYOUT]
    if compute_u:
        output_types.append(
            jax.ShapeDtypeStruct(
                (u_padding.local_physical_rows, u_padding.local_physical_cols),
                a.dtype,
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
                a.dtype,
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
        partition_slots=slots_attr,
        m=int(m),
        n=int(n),
        tile_size=int(tile_size),
        full_matrices=int(full_matrices),
    )
    singular_values, a_work, *vectors, status = ffi_fn(a)
    # Stage 3: restore vector shapes; keep opaque A work padded.
    axes = tuple(axis for axis, size in zip(matrix_specs, (process_rows, process_cols))
                 if axis is not None and size > 1)
    restored_vectors = ()
    requested_padding = tuple(p for p in (u_padding, vh_padding) if p is not None)
    for vector, padding in zip(vectors, requested_padding):
        vector = unpad_local_2d(vector, local_rows=padding.local_logical_rows,
                               local_cols=padding.local_logical_cols)
        restored_vectors += (mark_varying(vector, axes),)
    result = (mark_varying(a_work, axes),)
    if compute_u:
        result += (restored_vectors[0],)
    result += (singular_values,)
    if compute_vh:
        result += (restored_vectors[-1],)
    return result + (mark_varying(status, axes),)


def _prepare_gesvd_call(
    a: Array,
    tile_size: int,
    mesh: Mesh | AbstractMesh | None,
    matrix_specs: P | Tuple[P] | List[P] | None,
    *,
    in_specs: P | Tuple[P] | List[P] | None,
    compute_u: bool,
    compute_vh: bool,
    full_matrices: bool,
    pad: bool,
    caller: str,
    local: bool = False,
):
    """Validate an SVD call and derive its input and output layouts.

    Preparation proceeds as follows:

    1. Validate the input rank, dtype, and tile size.
    2. Require static Python booleans for the requested vector-output modes.
    3. Resolve the mesh, process grid, rank map, and tile-aligned layout of A;
       local blocks use the enclosing mesh to reconstruct the global shape.
    4. Derive the reduced or full U and Vh shapes and prepare only the
       requested distributed output layouts.
    """
    if a.ndim != 2:
        raise ValueError(f"{caller} expects a rank-2 matrix A.")
    _check_supported_gesvd_dtype(a.dtype)
    if int(tile_size) <= 0:
        raise ValueError("T_A must be positive.")
    for name, value in (
        ("compute_u", compute_u),
        ("compute_vh", compute_vh),
        ("full_matrices", full_matrices),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be a Python bool.")

    if local:
        layout = prepare_local_matrix_layout(
            a, tile_size, matrix_specs=matrix_specs, in_specs=in_specs,
            pad=pad, caller=caller,
        )
    else:
        layout = prepare_input_matrix_layout(
            a, tile_size, mesh=mesh, matrix_specs=matrix_specs,
            in_specs=in_specs, pad=pad, caller=caller,
        )
    m, n = layout.padding.logical_rows, layout.padding.logical_cols
    k = min(m, n)
    u_padding = (
        prepare_rectangular_matrix_layout(
            m,
            m if full_matrices else k,
            layout.grid,
            layout.tile_shape,
            pad=pad,
            caller=f"{caller}(U)",
        )
        if compute_u
        else None
    )
    vh_padding = (
        prepare_rectangular_matrix_layout(
            n if full_matrices else k,
            n,
            layout.grid,
            layout.tile_shape,
            pad=pad,
            caller=f"{caller}(Vh)",
        )
        if compute_vh
        else None
    )
    return layout, u_padding, vh_padding


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


_ROW_MAJOR_JAX_LAYOUT = (0, 1)


@lru_cache(maxsize=None)
def _gesvd_compiled(
    prepared, tile_size: int, *, compute_u: bool, compute_vh: bool,
    full_matrices: bool, pad: bool, donate: bool,
):
    """Cache the public JIT wrapper around the global context interface."""
    return jax.jit(
        partial(gesvd_jit_ctx, T_A=tile_size, compute_u=compute_u,
                compute_vh=compute_vh, full_matrices=full_matrices,
                pad=pad, _prepared=prepared),
        donate_argnums=(0,) if donate else (),
    )
