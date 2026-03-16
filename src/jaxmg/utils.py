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
    Generate a random n x n positive semidefinite matrix.
    """
    key = jax.random.key(seed)
    A = jax.random.normal(key, (n, n), dtype=dtype) / jnp.sqrt(n)
    return A @ A.T.conj() + jnp.eye(n, dtype=dtype) * 1e-5  # symmetric PSD


def get_mesh_and_spec_from_array(a: Array):
    sharding = a.sharding
    if isinstance(sharding, NamedSharding):
        return sharding.mesh, sharding.spec
    else:
        raise ValueError(
            "Array is not sharded with a NamedSharding, cannot extract mesh and spec."
        )


def maybe_real_dtype_from_complex(dtype):
    return (
        jnp.float32
        if dtype == jnp.complex64
        else (jnp.float64 if dtype == jnp.complex128 else dtype)
    )

class JaxMgWarning(UserWarning):
    """Warnings emitted by JaxMg."""

def numeric_machine_key():
    # 128-bit hash of hostname
    h = hashlib.blake2b(socket.gethostname().encode(), digest_size=16).digest()
    hi = int.from_bytes(h[:8], "big")
    lo = int.from_bytes(h[8:], "big")
    return jnp.array([hi, lo], dtype=jnp.uint64)