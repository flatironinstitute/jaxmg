import os
import ctypes
import jax
import jax.numpy as jnp

from jax import Array
from jax.sharding import PartitionSpec as P, Mesh

from functools import partial
from typing import Tuple, Union

from ._cyclic_1d import calculate_padding, pad_rows, unpad_rows


def potri(
    a: Array,
    T_A: int,
    mesh: Mesh,
    in_specs: Tuple[P] | P,
    return_status: bool = False,
    pad=True,
) -> Union[Array, Tuple[Array, int]]:
    """Compute the inverse of a symmetric matrix using the multi-GPU potri native kernel.

    Prepares inputs for the native ``potri_mg`` kernel and executes it via
    ``jax.ffi.ffi_call`` under ``jax.jit`` and ``jax.shard_map``. Handles
    per-device padding driven by ``T_A`` and symmetrizes the result before
    returning.

    Tip:
        If the shards of the matrix cannot be padded with tiles of size `T_A`
        (``N / num_gpus % T_A != 0``) we have to add padding to fit the last tile.
        This requires copying the matrix, which we want to avoid at all costs for 
        large ``N``. Make sure you pick ``T_A`` large enough (>=128) and such that it
        can evenly cover the shards. In principle, increasing ``T_A`` will increase 
        performance at the cost of memory, but depending on ``N``, the performance
          will saturate.

    Args:
        a (Array): A 2D JAX array of shape ``(N_rows, N)``. Must be symmetric and
            is expected to be sharded across the mesh along the first (row)
            axis using ``P(<axis_name>, None)``.
        T_A (int): Tile width used by the native solver. Each 
            local shard length must be a multiple of ``T_A``. If the user provides a 
            ``T_A`` that is incompatible with the shard size we pad the matrix
            accordingly. For small tile sizes (``T_A``< 128), the solver can 
            be extremely slow, so ensure that ``T_A`` is large enough. In principle,
            the larger ``T_A`` the faster the solver runs.
        mesh (Mesh): JAX device mesh used for ``jax.shard_map``.
        in_specs (PartitionSpec or tuple/list[PartitionSpec]): PartitionSpec
            describing the input sharding (row sharding). May be provided as a
            single ``PartitionSpec`` or a single-element container containing one.
        return_status (bool, optional): If True return ``(A_inv, status)`` where
            ``status`` is a host-replicated int32 from the native solver. If
            False return ``A_inv`` only. Default is False.
        pad (bool, optional): If True (default) apply per-device padding to meet
            ``T_A`` requirements; if False the caller must supply already-
            padded shapes.

    Returns:
        Array or (Array, int): The inverted matrix (row-sharded). If
            ``return_status=True`` also return the native solver status code.

    Raises:
        TypeError: If ``in_specs`` is not a ``PartitionSpec`` or a single-
            element container.
        ValueError: If ``in_specs`` does not indicate row sharding
            (``P(<axis_name>, None)``).
        AssertionError: If ``a`` is not 2D or if required shapes do not match
            when ``pad=False``.

    Notes:
        - The FFI call is executed with ``donate_argnums=0`` enabling zero-copy
          buffer sharing with the native library.
        - If the native solver fails the output may contain NaNs and ``status``
          will be non-zero.
    """

    ndev = int(os.environ["JAXMG_NUMBER_OF_DEVICES"])
    # Normalize in_specs so it's a single PartitionSpec instance (not an iterable)
    if isinstance(in_specs, (list, tuple)):
        if len(in_specs) != 1:
            raise ValueError(
                "in_specs must be a single PartitionSpec or a 1-element list/tuple."
            )
        in_specs = in_specs[0]
    if not isinstance(in_specs, P):
        raise TypeError(
            "in_specs must be a PartitionSpec or a 1-element list/tuple containing one."
        )
    if (in_specs._partitions[1] != None) or (in_specs._partitions[0] == None):
        raise ValueError(
            "A must be sharded along the rows with PartitionSpec P(str, None)."
        )
    assert a.ndim == 2, "a must be a 2D array."
    axis_name = in_specs._partitions[0]
    N_rows, N = a.shape

    shard_size = N_rows // ndev

    input_layouts = ((0, 1),)

    # Calculate padding
    padding = calculate_padding(shard_size, T_A)

    if not pad or padding == 0 or T_A >= N // ndev:
        if T_A < N // ndev:
            assert (
                N_rows == N + ndev * padding
            ), f"pad=False, but with T_A={T_A}, we need padding of {padding} rows per device."
            f"Expected {N + ndev * padding} rows, but received {N_rows}"

        pad_fn = lambda _a: _a
        unpad_fn = lambda _a: _a
        padding = 0

    else:
        # Make padding fns
        pad_fn = jax.shard_map(
            partial(pad_rows, padding=padding),
            mesh=mesh,
            in_specs=P(axis_name, None),
            out_specs=P(axis_name, None),
            check_vma=True,
        )
        unpad_fn = jax.shard_map(
            partial(unpad_rows, padding=padding),
            mesh=mesh,
            in_specs=P(axis_name, None),
            out_specs=P(axis_name, None),
            check_vma=True,
        )

    out_type = (
        jax.ShapeDtypeStruct((shard_size + padding, N), a.dtype),
        jax.ShapeDtypeStruct((1,), jnp.int32),
    )
    output_layouts = ((0, 1), (0,))
    out_specs = (
        P(axis_name, None),
        P(None),
    )
    # Prepare ffi call
    ffi_fn = partial(
        jax.ffi.ffi_call(
            "potri_mg",
            out_type,
            input_layouts=input_layouts,
            output_layouts=output_layouts,
            input_output_aliases={0: 0},  # Crucial for buffer sharing
        ),
        T_A=T_A,
    )

    # Jit with donate_argnums=0 is crucial for buffer sharing
    @partial(jax.jit, donate_argnums=0)
    @partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=P(axis_name, None),
        out_specs=out_specs,
        check_vma=False,
    )
    def impl(_a):
        return ffi_fn(_a)

    def fn(_a):
        _a = pad_fn(_a)
        _a, _status = impl(_a)
        return unpad_fn(_a), _status

    out, status = fn(a)
    out = potri_symmetrize(out)
    if return_status:
        return out, status[0]
    else:
        return out


def potri_shardmap_ctx(a: Array, T_A: int, pad=True) -> Union[Array, Tuple[Array, int]]:
    """Compute the inverse of a symmetric matrix for already-sharded inputs.

    This helper is a lower-level variant of :func:`potri` intended for
    environments where the caller already manages sharding/device placement
    (for example inside a custom ``shard_map`` or other placement context).
    It performs the same per-device padding logic driven by ``T_A`` and calls
    the native ``potri_mg`` FFI target directly via ``jax.ffi.ffi_call``.

    Warning:
        On exit, we return the upper triangular part of ``A_inv``.
        To achieve the full inverse, call ``jaxmg.potri_symmetrize`` outside of
        the shardmap_context.

    Warning:
        Multi-process multiple device (MPMD) mode is not supported right now but will
        be in a future release.

    Tip:
        If the shards of the matrix cannot be padded with tiles of size `T_A`
        (``N / num_gpus % T_A != 0``) we have to add padding to fit the last tile.
        This requires copying the matrix, which we want to avoid at all costs for 
        large ``N``. Make sure you pick ``T_A`` large enough (>=128) and such that it
        can evenly cover the shards. In principle, increasing ``T_A`` will increase 
        performance at the cost of memory, but depending on ``N``, the performance
          will saturate.

    Args:
        a (Array): Local, row-sharded slice of the global matrix with shape
            ``(shard_size, N)`` where ``shard_size`` is the per-device local
            row count and ``N`` is the global matrix dimension. The matrix
            should be symmetric.
        T_A (int): Tile width used by the native solver. Each 
            local shard length must be a multiple of ``T_A``. If the user provides a 
            ``T_A`` that is incompatible with the shard size we pad the matrix
            accordingly. For small tile sizes (``T_A``< 128), the solver can 
            be extremely slow, so ensure that ``T_A`` is large enough. In principle,
            the larger ``T_A`` the faster the solver runs.
        pad (bool, optional): If True (default) apply per-device padding to
            ``a`` to satisfy ``T_A``. If False the caller must ensure the
            provided local shape already meets kernel requirements.

    Returns:
        tuple: ``(A_inv, status)`` where ``A_inv`` contains the the upper triangular part
            of the inverted matrix (in  the same local/sharded layout as the input,
            with padding removed if applied) and ``status`` is
            the int32 status value returned by the native kernel
            (shape ``(1,)`` device array).

    Raises:
        AssertionError: If ``a`` is not 2D.
        ValueError: If ``T_A`` is too large or shape requirements are violated
            when ``pad=False``.

    Notes:
        - This function does not create an outer ``jax.shard_map`` or apply
          ``donate_argnums``; it is intended for use when the caller already
          controls sharding and device placement.
        - Padding is handled with :func:`calculate_padding`, :func:`pad_rows`,
          and :func:`unpad_rows`.
        - If the native solver fails the output may contain NaNs and the
          returned ``status`` will be non-zero.
    """

    ndev = int(os.environ["JAXMG_NUMBER_OF_DEVICES"])
    # Normalize in_specs so it's a single PartitionSpec instance (not an iterable)

    assert a.ndim == 2, "a must be a 2D array."
    shard_size, N = a.shape

    input_layouts = ((0, 1),)

    # Calculate padding
    padding = calculate_padding(shard_size, T_A)

    if not pad or padding == 0 or T_A >= N // ndev:
        if T_A < N // ndev:
            assert (
                shard_size == (N + ndev * padding) // ndev
            ), f"pad=False, but with T_A={T_A}, we need padding of {padding} rows per device."
            f"Expected {N + ndev * padding} rows, but received {shard_size}"

        pad_fn = lambda _a: _a
        unpad_fn = lambda _a: _a
        padding = 0

    else:
        # Make padding fns
        pad_fn = partial(pad_rows, padding=padding)
        unpad_fn = partial(unpad_rows, padding=padding)

    out_type = (
        jax.ShapeDtypeStruct((shard_size + padding, N), a.dtype),
        jax.ShapeDtypeStruct((1,), jnp.int32),
    )
    output_layouts = ((0, 1), (0,))

    # Prepare ffi call
    ffi_fn = partial(
        jax.ffi.ffi_call(
            "potri_mg",
            out_type,
            input_layouts=input_layouts,
            output_layouts=output_layouts,
            input_output_aliases={0: 0},  # Crucial for buffer sharing
        ),
        T_A=T_A,
    )

    def fn(_a):
        _a = pad_fn(_a)
        _a, _status = ffi_fn(_a)
        return unpad_fn(_a), _status

    out, status = fn(a)
    return out, status


@partial(jax.jit, donate_argnums=0)
def potri_symmetrize(_a):
    _a = jnp.triu(_a)
    return _a + _a.T.conj() - jnp.diag(jnp.diag(_a))
