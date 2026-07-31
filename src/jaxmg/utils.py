import socket
import os
import hashlib
import warnings
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax import Array
from jax.experimental import multihost_utils as mh


def random_psd(n, dtype, seed):
    """
    Generate a random regularized positive-definite matrix.
    """
    key = jax.random.key(seed)
    A = jax.random.normal(key, (n, n), dtype=dtype) / jnp.sqrt(n)
    return A @ A.T.conj() + jnp.eye(n, dtype=dtype) * 1e-5


def get_mesh_and_spec_from_array(a: Array):
    """Extract ``(mesh, spec)`` from a JAX array with ``NamedSharding``."""
    sharding = a.sharding
    if isinstance(sharding, NamedSharding):
        return sharding.mesh, sharding.spec
    else:
        raise ValueError(
            "Array is not sharded with a NamedSharding, cannot extract mesh and spec."
        )


def maybe_real_dtype_from_complex(dtype):
    """Map complex dtypes to their real component dtype."""
    return (
        jnp.float32
        if dtype == jnp.complex64
        else (jnp.float64 if dtype == jnp.complex128 else dtype)
    )

class JaxMgWarning(UserWarning):
    """Warnings emitted by JaxMg."""

def numeric_machine_key():
    """Return a stable uint64 pair identifying the current host."""
    # 128-bit hash of hostname.
    h = hashlib.blake2b(socket.gethostname().encode(), digest_size=16).digest()
    hi = int.from_bytes(h[:8], "big")
    lo = int.from_bytes(h[8:], "big")
    return jnp.array([hi, lo], dtype=jnp.uint64)
