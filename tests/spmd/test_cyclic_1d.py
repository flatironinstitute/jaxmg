import sys
import os

this_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(this_dir, "..")

sys.path.append(src_path)
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, NamedSharding
import pytest
from jaxmg import cyclic_1d, verify_cyclic
from jaxmg.utils import random_psd
from functools import partial

platforms = set(d.platform for d in jax.devices())
if "gpu" not in platforms:
    pytest.skip("No GPUs found. Skipping", allow_module_level=True)
    
ndev = jax.device_count()
mesh = jax.make_mesh((ndev,), ("x",))
# Test cases
N_list = list(i * ndev for i in [2, 3, 4, 10])
T_A_list = [1, 2, 3, 5]


def cusolver_solve_arange(N, T_A, dtype):
    A = jnp.diag(jnp.arange(N, dtype=dtype) + 1)
    A_before = A.copy()
    # Make mesh and place data
    A = jax.device_put(A, NamedSharding(mesh, P("x", None)))

    out = jax.jit(
        partial(cyclic_1d, mesh=mesh, in_specs=(P("x", None),)), static_argnums=1
    )(A, T_A=T_A)
    out.block_until_ready()
    verify_cyclic(A_before, out, T_A=T_A)


def cusolver_solve_psd(N, T_A, dtype):
    A = random_psd(N, dtype=dtype, seed=1234)
    A_before = A.copy()
    # Make mesh and place data
    _A = jax.device_put(A, NamedSharding(mesh, P("x", None)))
    out = jax.jit(
        partial(cyclic_1d, mesh=mesh, in_specs=(P("x", None),)), static_argnums=1
    )(_A, T_A=T_A)
    out.block_until_ready()
    verify_cyclic(A_before, out, T_A=T_A)


@pytest.mark.parametrize(
    "dtype", (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128)
)
@pytest.mark.parametrize("T_A", T_A_list)
@pytest.mark.parametrize("N", N_list)
def test_cusolver_solve_arange(N, T_A, dtype):
    cusolver_solve_arange(N, T_A, dtype)


@pytest.mark.parametrize(
    "dtype", (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128)
)
@pytest.mark.parametrize("T_A", T_A_list)
@pytest.mark.parametrize("N", N_list)
def test_cusolver_solve_psd(N, T_A, dtype):
    cusolver_solve_psd(N, T_A, dtype)
