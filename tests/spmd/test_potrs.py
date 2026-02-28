import sys
import os
import pytest

this_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(this_dir, "..")
sys.path.append(src_path)
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, NamedSharding
from jaxmg import potrs, potrs_shardmap_ctx
from jaxmg.utils import random_psd

from functools import partial

platforms = set(d.platform for d in jax.devices())
if "gpu" not in platforms:
    pytest.skip("No GPUs found. Skipping", allow_module_level=True)
else:
    ndev = jax.device_count()
    mesh = jax.make_mesh((ndev,), ("x",))

    # Test cases
    N_list = list(i * ndev for i in [2, 3, 4, 10])
    T_A_list = [1, 2, 3, 5]


    @partial(jax.jit, static_argnames=("_T_A",))
    def jitted_potrs(_a, _b, _T_A):
        out = partial(potrs, mesh=mesh, in_specs=(P("x", None),), pad=True)(_a, _b, _T_A)
        return out


    @partial(jax.jit, static_argnames=("_T_A",))
    def jitted_potrs_status(_a, _b, _T_A):
        out = partial(
            potrs, mesh=mesh, in_specs=(P("x", None),), pad=True, return_status=True
        )(_a, _b, _T_A)
        return out


    @partial(jax.jit, static_argnames=("_T_A",))
    def jitted_potrs_no_shardmap(_a, _b, _T_A):
        out = jax.shard_map(
            partial(potrs_shardmap_ctx, T_A=_T_A),
            mesh=mesh,
            in_specs=(P("x", None), P(None, None)),
            out_specs=(P(None, None), P(None)),
            check_vma=False,
        )(_a, _b)
        return out


    def cusolver_solve_arange(N, T_A, dtype):
        A = jnp.diag(jnp.arange(N, dtype=dtype) + 1)
        b = jnp.ones((N, 1), dtype=dtype)
        # Make mesh and place data
        _A = jax.device_put(A, NamedSharding(mesh, P("x", None)))
        _b = jax.device_put(b, NamedSharding(mesh, P(None, None)))
        out = jitted_potrs(_A.copy(), _b.copy(), T_A)
        out.block_until_ready()
        expected_out = 1.0 / (jnp.arange(N, dtype=dtype) + 1)
        assert jnp.allclose(out.flatten(), expected_out)
        out_no_shm, _ = jitted_potrs_no_shardmap(_A.copy(), _b.copy(), T_A)
        assert jnp.allclose(out_no_shm.flatten(), expected_out)


    def cusolver_solve_psd(N, T_A, dtype):
        A = random_psd(N, dtype=dtype, seed=1234)
        b = jnp.ones((N, 1), dtype=dtype)
        cfac = jax.scipy.linalg.cho_factor(A)
        expected_out = jax.scipy.linalg.cho_solve(cfac, b)
        # Make mesh and place data
        _A = jax.device_put(A, NamedSharding(mesh, P("x", None)))
        _b = jax.device_put(b, NamedSharding(mesh, P(None, None)))

        out = jitted_potrs(_A.copy(), _b.copy().squeeze(), T_A)
        out.block_until_ready()
        norm_scipy = jnp.linalg.norm(b - A @ expected_out)
        norm_potrf = jnp.linalg.norm(b - A @ out)
        assert jnp.isclose(norm_scipy, norm_potrf, atol=1e-4)
        out_no_shm, _ = jitted_potrs_no_shardmap(_A.copy(), _b.copy(), T_A)
        norm_scipy = jnp.linalg.norm(b - A @ expected_out)
        norm_potrf = jnp.linalg.norm(b - A @ out_no_shm)
        assert jnp.allclose(out_no_shm.flatten(), out.flatten())


    def cusolver_solve_non_psd(N, T_A, dtype):
        A = jnp.diag(jnp.arange(N, dtype=dtype) - 1)
        b = jnp.ones((N, 1), dtype=dtype)
        # Make mesh and place data
        _A = jax.device_put(A, NamedSharding(mesh, P("x", None)))
        _b = jax.device_put(b, NamedSharding(mesh, P(None, None)))

        out, status = jitted_potrs_status(_A.copy(), _b.copy(), T_A)
        status.block_until_ready()
        out.block_until_ready()
        assert status == 7
        assert jnp.all(jnp.isnan(out))


    def cusolver_solve_non_symm(N, T_A, dtype):
        A = jnp.diag(jnp.arange(N, dtype=dtype) + 1)
        # TODO: For some reason the solver does not fail when we set this to 1.0.
        A = A.at[0, 1].set(2.0)
        b = jnp.ones((N, 1), dtype=dtype)
        # Make mesh and place data
        A = jax.device_put(A, NamedSharding(mesh, P("x", None)))
        b = jax.device_put(b, NamedSharding(mesh, P(None, None)))

        out, status = jitted_potrs_status(A, b, T_A)
        status.block_until_ready()
        out.block_until_ready()
        assert status == 7
        assert jnp.all(jnp.isnan(out))


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


    @pytest.mark.parametrize(
        "dtype", (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128)
    )
    @pytest.mark.parametrize("T_A", T_A_list)
    @pytest.mark.parametrize("N", N_list)
    def test_cusolver_solve_non_psd(N, T_A, dtype):
        cusolver_solve_non_psd(N, T_A, dtype)


    @pytest.mark.parametrize(
        "dtype", (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128)
    )
    @pytest.mark.parametrize("T_A", T_A_list)
    @pytest.mark.parametrize("N", N_list)
    def test_cusolver_solve_non_symm(N, T_A, dtype):
        cusolver_solve_non_symm(N, T_A, dtype)


    @pytest.mark.parametrize(
        "dtype", (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128)
    )
    def test_cusolver_inplace_check(dtype):
        N = ndev * 2
        T_A = 1
        A = random_psd(N, dtype=dtype, seed=1234)
        b = jnp.ones((N, 1), dtype=dtype)
        cfac = jax.scipy.linalg.cho_factor(A)
        expected_out = jax.scipy.linalg.cho_solve(cfac, b)
        # Make mesh and place data
        _A = jax.device_put(A, NamedSharding(mesh, P("x", None)))
        _b = jax.device_put(b, NamedSharding(mesh, P(None, None)))

        out = jitted_potrs(_A, _b, T_A)
        assert jnp.allclose(out, expected_out, atol=1e-4)
        out = jitted_potrs(_A, _b, T_A)
        assert jnp.allclose(out, expected_out, atol=1e-4)


    def test_potrs_multiple_rhs():
        N = ndev * 2
        T_A = 1
        dtype = jnp.float64
        A = random_psd(N, dtype=dtype, seed=5678)
        b = jnp.arange(N * 3, dtype=dtype).reshape(N, 3) + 1
        cfac = jax.scipy.linalg.cho_factor(A)
        expected_out = jax.scipy.linalg.cho_solve(cfac, b)
        _A = jax.device_put(A, NamedSharding(mesh, P("x", None)))
        _b = jax.device_put(b, NamedSharding(mesh, P(None, None)))
        out = jitted_potrs(_A, _b, T_A)
        assert jnp.allclose(out, expected_out, atol=1e-4)


    def test_potrs_loop_shm():
        N = ndev * 2
        T_A = 1
        dtype = jnp.float64
        A = random_psd(N, dtype=dtype, seed=5678)
        b = jnp.arange(N, dtype=dtype)

        _A = jax.device_put(A, NamedSharding(mesh, P("x", None)))
        _b = jax.device_put(b, NamedSharding(mesh, P(None)))
        fd_start = len(os.listdir("/proc/self/fd"))
        for i in range(100):
            out = jitted_potrs(_A, _b, T_A)
            out.block_until_ready()
        fd_end = len(os.listdir("/proc/self/fd"))
        assert fd_end - fd_start < 100
