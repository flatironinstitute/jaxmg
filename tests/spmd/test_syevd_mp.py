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

from jaxmg import syevd_mp

platforms = {d.platform for d in jax.devices()}
if "gpu" not in platforms:
    pytest.skip("No GPUs found. Skipping", allow_module_level=True)
if len(jax.devices("gpu")) < 4:
    pytest.skip("At least four GPUs are required. Skipping", allow_module_level=True)


def _hermitian(n: int, dtype) -> np.ndarray:
    dtype = np.dtype(dtype)
    rng = np.random.default_rng(9700 + n)
    if np.issubdtype(dtype, np.complexfloating):
        x = rng.standard_normal((n, n)) + 0.5j * rng.standard_normal((n, n))
        a = x @ x.conj().T
        a += np.eye(n, dtype=a.dtype) * float(n)
        return a.astype(dtype)

    x = rng.standard_normal((n, n))
    a = x @ x.T
    a += np.eye(n, dtype=a.dtype) * float(n)
    return a.astype(dtype)


def _validate_status(status, *, process_rows: int, process_cols: int, eigvecs: bool):
    status.block_until_ready()
    num_processes = process_rows * process_cols
    rows = np.asarray(status).reshape(num_processes, 36)
    status_codes = set(rows[:, 0].tolist())
    if status_codes == {1}:
        pytest.skip("libcusolverMp is not available on the loader path.")
    if status_codes == {21}:
        pytest.skip("cuSOLVERMp syevd symbols are not available.")

    assert status_codes == {0}, rows
    np.testing.assert_array_equal(rows[:, 2], np.arange(num_processes))
    np.testing.assert_array_equal(rows[:, 3], np.full(num_processes, num_processes))
    np.testing.assert_array_equal(rows[:, 4], np.full(num_processes, process_rows))
    np.testing.assert_array_equal(rows[:, 5], np.full(num_processes, process_cols))
    np.testing.assert_array_equal(rows[:, 20], np.full(num_processes, int(eigvecs)))
    np.testing.assert_array_equal(rows[:, 23], np.ones(num_processes))
    np.testing.assert_array_equal(rows[:, 24], np.zeros(num_processes))


def _tolerances(dtype) -> tuple[float, float]:
    dtype = np.dtype(dtype)
    if dtype == np.dtype("float32") or dtype == np.dtype("complex64"):
        return 2e-4, 2e-4
    return 1e-9, 1e-9


@pytest.mark.parametrize(
    "process_rows,process_cols,n,tile,dtype,eigvecs",
    [
        (2, 2, 10, 4, np.float64, True),
        (2, 2, 8, 4, np.complex128, True),
        (1, 4, 12, 4, np.float32, False),
        (4, 1, 12, 4, np.complex64, False),
    ],
)
def test_syevd_mp_solves_and_restores_block_sharded_eigenvectors(
    process_rows,
    process_cols,
    n,
    tile,
    dtype,
    eigvecs,
):
    devices = np.asarray(jax.devices("gpu")[:4], dtype=object).reshape(
        process_rows,
        process_cols,
    )
    mesh = Mesh(devices, ("pr", "pc"))
    sharding = NamedSharding(mesh, P("pr", "pc"))
    a_host = _hermitian(n, dtype)
    a = jax.device_put(jnp.asarray(a_host), sharding)
    rtol, atol = _tolerances(dtype)

    if eigvecs:
        eigenvalues, eigenvectors, status = syevd_mp(
            a,
            T_A=tile,
            eigvecs=True,
            return_status=True,
        )
        eigenvalues.block_until_ready()
        eigenvectors.block_until_ready()
        _validate_status(
            status,
            process_rows=process_rows,
            process_cols=process_cols,
            eigvecs=True,
        )

        expected_values, _ = np.linalg.eigh(a_host)
        values = np.asarray(eigenvalues)
        vectors = np.asarray(eigenvectors)
        np.testing.assert_allclose(values, expected_values, rtol=rtol, atol=atol)

        residual = np.linalg.norm(a_host @ vectors - vectors * values[None, :])
        scale = max(1.0, np.linalg.norm(a_host))
        assert residual / scale < max(10 * rtol, 1e-10)

        eye = np.eye(n, dtype=vectors.dtype)
        np.testing.assert_allclose(
            vectors.conj().T @ vectors,
            eye,
            rtol=max(10 * rtol, 1e-6),
            atol=max(10 * atol, 1e-6),
        )
    else:
        eigenvalues, status = syevd_mp(
            a,
            T_A=tile,
            eigvecs=False,
            return_status=True,
        )
        eigenvalues.block_until_ready()
        _validate_status(
            status,
            process_rows=process_rows,
            process_cols=process_cols,
            eigvecs=False,
        )
        expected_values = np.linalg.eigvalsh(a_host)
        np.testing.assert_allclose(
            np.asarray(eigenvalues),
            expected_values,
            rtol=rtol,
            atol=atol,
        )
