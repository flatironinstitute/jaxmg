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

from functools import partial

from jaxmg import syevd, syevd_shardmap_ctx
from jaxmg.utils import random_psd

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
    def jitted_syevd(_a, _T_A):
        out = partial(syevd, mesh=mesh, in_specs=(P("x", None),), pad=True)(_a, _T_A)
        return out


    @partial(jax.jit, static_argnames=("_T_A",))
    def jitted_syevd_no_shardmap(_a, _T_A):
        out = jax.shard_map(
            partial(syevd_shardmap_ctx, T_A=_T_A),
            mesh=mesh,
            in_specs=(P("x", None),),
            out_specs=(P(None), P(None, None), P(None)),
            check_vma=False,
        )(_a)
        return out


    @partial(jax.jit, static_argnames=("_T_A",))
    def jitted_syevd_no_V(_a, _T_A):
        out = partial(
            syevd, mesh=mesh, in_specs=(P("x", None),), return_eigenvectors=False, pad=True
        )(_a, _T_A)
        return out


    @partial(jax.jit, static_argnames=("_T_A",))
    def jitted_syevd_no_V_no_shardmap(_a, _T_A):
        out = jax.shard_map(
            partial(syevd_shardmap_ctx, T_A=_T_A, return_eigenvectors=False),
            mesh=mesh,
            in_specs=(P("x", None),),
            out_specs=(P(None), P(None)),
            check_vma=False,
        )(_a)
        return out


    def cusolver_solve_arange(N, T_A, dtype):
        A = jnp.diag(jnp.arange(N, dtype=dtype) + 1)
        eigenvalues_expected = jnp.diag(A)
        # Make mesh and place data
        _A = jax.device_put(A, NamedSharding(mesh, P("x", None)))
        eigenvalues, V = jitted_syevd(_A.copy(), T_A)
        eigenvalues.block_until_ready()
        V.block_until_ready()
        assert jnp.allclose(eigenvalues_expected, eigenvalues)
        eigenvalus_VtAV = jnp.diag(V @ A @ V.T)
        assert jnp.allclose(eigenvalus_VtAV, eigenvalues_expected)
        eigenvalues_no_shm, V, _ = jitted_syevd_no_shardmap(_A.copy(), T_A)
        assert jnp.allclose(eigenvalues_expected, eigenvalues_no_shm)


    def cusolver_solve_psd(N, T_A, dtype):
        A = random_psd(N, dtype=dtype, seed=1234)
        eigenvalues_expected, V_expected = jnp.linalg.eigh(A)
        # Make mesh and place data
        _A = jax.device_put(A, NamedSharding(mesh, P("x", None)))
        eigenvalues, V = jitted_syevd(_A.copy(), T_A)
        eigenvalues.block_until_ready()
        V.block_until_ready()
        norm_syevd = jnp.linalg.norm(V @ A - jnp.diag(eigenvalues) @ V.T)
        norm_lax = jnp.linalg.norm(
            V_expected @ A - jnp.diag(eigenvalues_expected) @ V_expected.T
        )
        assert jnp.isclose(norm_syevd, norm_lax, rtol=10, atol=1e-8)
        eigenvalues_no_shm, V, _ = jitted_syevd_no_shardmap(_A.copy(), T_A)
        assert jnp.allclose(eigenvalues_expected, eigenvalues_no_shm, rtol=10, atol=1e-10)


    def cusolver_solve_arange_no_V(N, T_A, dtype):
        A = jnp.diag(jnp.arange(N, dtype=dtype) + 1)
        eigenvalues_expected = jnp.diag(A)
        # Make mesh and place data
        _A = jax.device_put(A, NamedSharding(mesh, P("x", None)))
        eigenvalues = jitted_syevd_no_V(_A.copy(), T_A)
        eigenvalues.block_until_ready()
        assert jnp.allclose(eigenvalues_expected, eigenvalues)
        eigenvalues_no_shm, _ = jitted_syevd_no_V_no_shardmap(_A.copy(), T_A)
        assert jnp.allclose(eigenvalues_expected, eigenvalues_no_shm)


    def cusolver_solve_psd_no_V(N, T_A, dtype):
        A = random_psd(N, dtype=dtype, seed=1234)
        eigenvalues_expected = jnp.linalg.eigvalsh(A)
        # Make mesh and place data
        _A = jax.device_put(A, NamedSharding(mesh, P("x", None)))
        eigenvalues = jitted_syevd_no_V(A.copy(), T_A)
        eigenvalues.block_until_ready()
        assert jnp.allclose(eigenvalues, eigenvalues_expected, rtol=10, atol=0.0)
        eigenvalues_no_shm, _ = jitted_syevd_no_V_no_shardmap(_A.copy(), T_A)
        assert jnp.allclose(eigenvalues_expected, eigenvalues_no_shm, rtol=10, atol=0.0)


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
    def test_cusolver_solve_arange_no_v(N, T_A, dtype):
        cusolver_solve_arange_no_V(N, T_A, dtype)


    @pytest.mark.parametrize(
        "dtype", (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128)
    )
    @pytest.mark.parametrize("T_A", T_A_list)
    @pytest.mark.parametrize("N", N_list)
    def test_cusolver_solve_psd_no_v(N, T_A, dtype):
        cusolver_solve_psd_no_V(N, T_A, dtype)

    @pytest.mark.parametrize(
        "dtype", (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128)
    )
    def test_cusolver_inplace_check(dtype):
        N = ndev * 2
        T_A = 1
        A = random_psd(N, dtype=dtype, seed=1234)
        # Make mesh and place data
        _A = jax.device_put(A, NamedSharding(mesh, P("x", None)))
        expected = jnp.linalg.eigvalsh(A)
        eigenvalues, _ = jitted_syevd(_A, T_A)
        assert jnp.allclose(expected, eigenvalues, atol=1e-4)
        eigenvalues, _ = jitted_syevd(_A, T_A)
        assert jnp.allclose(expected, eigenvalues, atol=1e-4)


    def test_syevd_loop_shm():
        N = ndev * 2
        T_A = 1
        dtype = jnp.float64
        A = random_psd(N, dtype=dtype, seed=5678)
        _A = jax.device_put(A, NamedSharding(mesh, P("x", None)))
        fd_start = len(os.listdir("/proc/self/fd"))
        for i in range(100):
            out,_ = jitted_syevd(_A,  T_A)
            out.block_until_ready()
        fd_end = len(os.listdir("/proc/self/fd"))
        assert fd_end - fd_start < 100

    def test_syevd_no_V_loop_shm():
        N = ndev * 2
        T_A = 1
        dtype = jnp.float64
        A = random_psd(N, dtype=dtype, seed=5678)
        _A = jax.device_put(A, NamedSharding(mesh, P("x", None)))
        fd_start = len(os.listdir("/proc/self/fd"))
        for i in range(100):
            out = jitted_syevd_no_V(_A,  T_A)
            out.block_until_ready()
        fd_end = len(os.listdir("/proc/self/fd"))
        assert fd_end - fd_start < 100