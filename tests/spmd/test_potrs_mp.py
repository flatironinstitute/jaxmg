import os
import sys

import numpy as np

this_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(this_dir, "..")
sys.path.append(src_path)

from jax import config

config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from jaxmg import potrs_mp

platforms = {d.platform for d in jax.devices()}
if "gpu" not in platforms:
    pytest.skip("No GPUs found. Skipping", allow_module_level=True)
if len(jax.devices("gpu")) < 4:
    pytest.skip("At least four GPUs are required. Skipping", allow_module_level=True)


def _spd(n: int, dtype) -> np.ndarray:
    base = np.arange(1, n * n + 1, dtype=np.float64).reshape(n, n)
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.complexfloating):
        values = base + 0.25j * (base.T - base)
        matrix = values @ values.conj().T
    else:
        matrix = base @ base.T
    matrix += np.eye(n, dtype=matrix.dtype) * (n * n)
    return matrix.astype(dtype)


def _rhs(n: int, nrhs: int, dtype) -> np.ndarray:
    values = np.arange(1, n * nrhs + 1, dtype=np.float64).reshape(n, nrhs)
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.complexfloating):
        values = values - 0.5j * values[::-1]
    return values.astype(dtype)


@pytest.mark.parametrize(
    "dtype,rtol,atol",
    [
        (np.float64, 1e-10, 1e-10),
        (np.complex128, 1e-10, 1e-10),
    ],
)
def test_potrs_mp_solves_and_restores_block_sharded_rhs(dtype, rtol, atol):
    n = 8
    nrhs = 8
    devices = np.asarray(jax.devices("gpu")[:4], dtype=object).reshape(2, 2)
    mesh = Mesh(devices, ("pr", "pc"))
    sharding = NamedSharding(mesh, P("pr", "pc"))

    a_host = _spd(n, dtype)
    b_host = _rhs(n, nrhs, dtype)
    a = jax.device_put(jnp.asarray(a_host), sharding)
    b = jax.device_put(jnp.asarray(b_host), sharding)

    out, status = potrs_mp(
        a,
        b,
        T_A=4,
        mesh=mesh,
        matrix_specs=P("pr", "pc"),
        return_status=True,
    )
    out.block_until_ready()
    status.block_until_ready()

    statuses = np.asarray(status).reshape(4, 40)
    status_codes = set(statuses[:, 0].tolist())
    if status_codes == {1}:
        pytest.skip("libcusolverMp is not available on the loader path.")
    if status_codes == {21}:
        pytest.skip("cuSOLVERMp potrf/potrs symbols are not available.")

    assert status_codes == {0}, statuses
    np.testing.assert_array_equal(statuses[:, 2], np.arange(4))
    np.testing.assert_array_equal(statuses[:, 3], np.full(4, 4))
    np.testing.assert_array_equal(statuses[:, 32], np.zeros(4))

    expected = np.linalg.solve(a_host, b_host)
    np.testing.assert_allclose(np.asarray(out), expected, rtol=rtol, atol=atol)
