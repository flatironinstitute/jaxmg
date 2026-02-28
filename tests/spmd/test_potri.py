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
from jaxmg import potri, potri_shardmap_ctx, potri_symmetrize
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
    def jitted_potri(_a, _T_A):
        out = partial(potri, mesh=mesh, in_specs=(P("x", None),), pad=True)(_a, _T_A)
        return out


    @partial(jax.jit, static_argnames=("_T_A",))
    def jitted_potri_status(_a, _T_A):
        out, status = partial(
            potri, mesh=mesh, in_specs=(P("x", None),), pad=True, return_status=True
        )(_a, _T_A)
        return out, status


    @partial(jax.jit, static_argnames=("_T_A",))
    def jitted_potri_no_shardmap(_a, _T_A):
        out, status = jax.shard_map(
            partial(potri_shardmap_ctx, T_A=_T_A),
            mesh=mesh,
            in_specs=(P("x", None),),
            out_specs=(P("x", None), P(None)),
            check_vma=False,
        )(_a)
        return out, status


    def cusolver_solve_arange(N, T_A, dtype):
        A = jnp.diag(jnp.arange(N, dtype=dtype) + 1)
        # Make mesh and place data
        _A = jax.device_put(A, NamedSharding(mesh, P("x", None)))
        out = jitted_potri(_A.copy(), T_A)
        out.block_until_ready()
        expected_out = jnp.diag(1.0 / (jnp.arange(N, dtype=dtype) + 1))
        assert jnp.allclose(out, expected_out)
        expected_out_no_shm, _ = jitted_potri_no_shardmap(_A, T_A)
        expected_out_no_shm = potri_symmetrize(expected_out_no_shm)
        assert jnp.allclose(out, expected_out_no_shm)


    def cusolver_solve_psd(N, T_A, dtype):
        A = random_psd(N, dtype=dtype, seed=1234)
        expected_out = jnp.linalg.inv(A)
        # Make mesh and place data
        _A = jax.device_put(A, NamedSharding(mesh, P("x", None)))
        out = jitted_potri(_A, T_A)
        out.block_until_ready()
        assert jnp.allclose(A.conj().T, A)
        norm_potri = jnp.linalg.norm(A @ out - jnp.eye(N, dtype=dtype))
        norm_lax = jnp.linalg.norm(A @ expected_out - jnp.eye(N, dtype=dtype))
        assert jnp.isclose(norm_potri, norm_lax, rtol=10, atol=1e-8)
        expected_out_no_shm, _ = jitted_potri_no_shardmap(_A, T_A)
        expected_out_no_shm = potri_symmetrize(expected_out_no_shm)
        norm_potri_no_shm = jnp.linalg.norm(
            A @ expected_out_no_shm - jnp.eye(N, dtype=dtype)
        )
        assert jnp.allclose(norm_potri_no_shm, norm_lax, rtol=10, atol=1e-8)


    def cusolver_solve_non_psd(N, T_A, dtype):
        A = jnp.diag(jnp.arange(N, dtype=dtype) - 1)
        # Make mesh and place data
        _A = jax.device_put(A, NamedSharding(mesh, P("x", None)))
        out, status = jitted_potri_status(_A, T_A)
        status.block_until_ready()
        out.block_until_ready()
        assert status == 7
        assert jnp.all(jnp.isnan(out))


    def cusolver_solve_non_symm(N, T_A, dtype):
        A = jnp.diag(jnp.arange(N, dtype=dtype) + 1)
        # TODO: For some reason the solver does not fail when we set this to 1.0.
        A = A.at[0, 1].set(2.0)
        # Make mesh and place data
        _A = jax.device_put(A, NamedSharding(mesh, P("x", None)))
        out, status = jitted_potri_status(_A, T_A)
        status.block_until_ready()
        out.block_until_ready()
        assert status == 7
        assert jnp.all(jnp.isnan(out))


    @pytest.mark.parametrize(
        "dtype", (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128)
    )
    @pytest.mark.parametrize("T_A", T_A_list)
    @pytest.mark.parametrize("N", N_list)
    def test_cusolver_solve_arange_dev_1(N, T_A, dtype):
        cusolver_solve_arange(N, T_A, dtype)


    @pytest.mark.parametrize(
        "dtype", (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128)
    )
    @pytest.mark.parametrize("T_A", T_A_list)
    @pytest.mark.parametrize("N", N_list)
    def test_cusolver_solve_psd_dev_1(N, T_A, dtype):
        cusolver_solve_psd(N, T_A, dtype)


    @pytest.mark.parametrize(
        "dtype", (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128)
    )
    @pytest.mark.parametrize("T_A", T_A_list)
    @pytest.mark.parametrize("N", N_list)
    def test_cusolver_solve_non_psd_dev_1(N, T_A, dtype):
        cusolver_solve_non_psd(N, T_A, dtype)


    @pytest.mark.parametrize(
        "dtype", (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128)
    )
    @pytest.mark.parametrize("T_A", T_A_list)
    @pytest.mark.parametrize("N", N_list)
    def test_cusolver_solve_non_symm_dev_1(N, T_A, dtype):
        cusolver_solve_non_symm(N, T_A, dtype)


    @pytest.mark.parametrize(
        "dtype", (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128)
    )
    def test_cusolver_inplace_check(dtype):
        N = ndev * 2
        T_A = 1
        A = random_psd(N, dtype=dtype, seed=1234)
        expected_out = jnp.linalg.inv(A)
        # Make mesh and place data
        _A = jax.device_put(A, NamedSharding(mesh, P("x", None)))
        A_inv = jitted_potri(_A, T_A)
        norm_potri = jnp.linalg.norm(A @ A_inv - jnp.eye(N, dtype=dtype))
        norm_lax = jnp.linalg.norm(A @ expected_out - jnp.eye(N, dtype=dtype))
        assert jnp.isclose(norm_potri, norm_lax, rtol=10, atol=1e-8)
        A_inv = jitted_potri(_A, T_A)
        norm_potri = jnp.linalg.norm(A @ A_inv - jnp.eye(N, dtype=dtype))
        assert jnp.isclose(norm_potri, norm_lax, rtol=10, atol=1e-8)

    def test_potri_loop_shm():
        N = ndev * 2
        T_A = 1
        dtype = jnp.float64
        A = random_psd(N, dtype=dtype, seed=5678)
        _A = jax.device_put(A, NamedSharding(mesh, P("x", None)))
        fd_start = len(os.listdir("/proc/self/fd"))
        for i in range(100):
            out = jitted_potri(_A,  T_A)
            out.block_until_ready()
        fd_end = len(os.listdir("/proc/self/fd"))
        assert fd_end - fd_start < 100