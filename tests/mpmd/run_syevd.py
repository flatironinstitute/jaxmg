import os
import sys
import json
import traceback
from typing import Callable, Dict, List

import jax

coord_addr = sys.argv[1]
proc_id = int(sys.argv[2])
num_procs = int(sys.argv[3])

# Initialize the GPU machines.
jax.distributed.initialize(
    coordinator_address=coord_addr,
    num_processes=num_procs,
    process_id=proc_id,
    local_device_ids=proc_id,
    coordinator_bind_address=coord_addr,
)

# Basic diagnostics for debugging
print("process id =", jax.process_index(), flush=True)
print("global devices =", jax.devices(), flush=True)
print("local devices =", jax.local_devices(), flush=True)
print("visible devices", os.environ.get("CUDA_VISIBLE_DEVICES", ""), flush=True)


def _println(prefix: str, payload: dict):
    """Print a single-line JSON payload with a stable prefix for log parsing."""
    print(f"{prefix} {json.dumps(payload, sort_keys=True)}", flush=True)


import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from functools import partial


# These will be initialized after jax.distributed.initialize()
devices = [d for d in jax.devices() if d.platform == "gpu"]
mesh = jax.make_mesh((jax.device_count(),), ("x",))

from jaxmg import syevd, syevd_shardmap_ctx
from jaxmg.utils import random_psd


@partial(jax.jit, static_argnames=("_T_A",))
def jitted_syevd(_a, _T_A):
    eigenvalues, V = partial(syevd, mesh=mesh, in_specs=(P("x", None),), pad=True)(
        _a, _T_A
    )
    return eigenvalues, V


@partial(jax.jit, static_argnames=("_T_A",))
def jitted_syevd_no_shardmap(_a, _T_A):
    eigenvalues, V, status = jax.shard_map(
        partial(syevd_shardmap_ctx, T_A=_T_A),
        mesh=mesh,
        in_specs=(P("x", None),),
        out_specs=(P(None), P(None, None), P(None)),
        check_vma=False,
    )(_a)
    return eigenvalues, V, status


@partial(jax.jit, static_argnames=("_T_A",))
def jitted_syevd_no_V(_a, _T_A):
    eigenvalues = partial(
        syevd, mesh=mesh, in_specs=(P("x", None),), return_eigenvectors=False, pad=True
    )(_a, _T_A)
    return eigenvalues


@partial(jax.jit, static_argnames=("_T_A",))
def jitted_syevd_no_V_no_shardmap(_a, _T_A):
    eigenvalues, status = jax.shard_map(
        partial(syevd_shardmap_ctx, T_A=_T_A, return_eigenvectors=False),
        mesh=mesh,
        in_specs=(P("x", None),),
        out_specs=(P(None), P(None)),
        check_vma=False,
    )(_a)
    return eigenvalues, status


def cusolver_solve_arange(N, T_A, dtype):
    A = jnp.diag(jnp.arange(N, dtype=dtype) + 1)
    eigenvalues_expected = jnp.diag(A)
    # Make mesh and place data
    _A = jax.device_put(A, NamedSharding(mesh, P("x", None)))
    eigenvalues, V = jitted_syevd(_A.copy(), T_A)
    assert jnp.allclose(eigenvalues_expected, eigenvalues)
    eigenvalues_VtAV = jnp.diag(V @ A @ V.T)
    assert jnp.allclose(eigenvalues_VtAV, eigenvalues_expected)
    eigenvalues_no_shm, V_no_shm, _ = jitted_syevd_no_shardmap(_A.copy(), T_A)
    assert jnp.allclose(eigenvalues_expected, eigenvalues_no_shm)


def cusolver_solve_psd(N, T_A, dtype):
    A = random_psd(N, dtype=dtype, seed=1234)
    eigenvalues_expected, V_expected = jnp.linalg.eigh(A)
    # Make mesh and place data
    _A = jax.device_put(A, NamedSharding(mesh, P("x", None)))
    eigenvalues, V = jitted_syevd(_A.copy(), T_A)
    norm_syevd = jnp.linalg.norm(V @ A - jnp.diag(eigenvalues) @ V.T)
    norm_lax = jnp.linalg.norm(
        V_expected @ A - jnp.diag(eigenvalues_expected) @ V_expected.T
    )
    assert jnp.isclose(norm_syevd, norm_lax, rtol=10, atol=1e-8)
    eigenvalues_no_shm, V_no_shm, _ = jitted_syevd_no_shardmap(_A.copy(), T_A)
    assert jnp.allclose(eigenvalues_expected, eigenvalues_no_shm, rtol=10, atol=1e-10)


def cusolver_solve_arange_no_V(N, T_A, dtype):
    A = jnp.diag(jnp.arange(N, dtype=dtype) + 1)
    eigenvalues_expected = jnp.diag(A)
    # Make mesh and place data
    _A = jax.device_put(A, NamedSharding(mesh, P("x", None)))
    eigenvalues = jitted_syevd_no_V(_A.copy(), T_A)
    assert jnp.allclose(eigenvalues_expected, eigenvalues)
    eigenvalues_no_shm, _ = jitted_syevd_no_V_no_shardmap(_A.copy(), T_A)
    assert jnp.allclose(eigenvalues_expected, eigenvalues_no_shm)


def cusolver_solve_psd_no_V(N, T_A, dtype):
    A = random_psd(N, dtype=dtype, seed=1234)
    eigenvalues_expected = jnp.linalg.eigvalsh(A)
    # Make mesh and place data
    _A = jax.device_put(A, NamedSharding(mesh, P("x", None)))
    eigenvalues = jitted_syevd_no_V(_A.copy(), T_A)
    assert jnp.allclose(eigenvalues, eigenvalues_expected, rtol=10, atol=0.0)
    eigenvalues_no_shm, _ = jitted_syevd_no_V_no_shardmap(_A.copy(), T_A)
    assert jnp.allclose(eigenvalues_expected, eigenvalues_no_shm, rtol=10, atol=0.0)


def _build_registry() -> Dict[str, Callable]:
    # Map test names to callables that accept (N, T_A, dtype)
    return {
        "arange": cusolver_solve_arange,
        "psd": cusolver_solve_psd,
        "arange_no_V": cusolver_solve_arange_no_V,
        "psd_no_V": cusolver_solve_psd_no_V,
    }


def main(argv: List[str]):
    # Single-task only: expect arguments
    # run_syevd.py <coord_addr> <proc_id> <num_procs> <test_name> <dtype_name>

    task_name = argv[4]
    task_dtype_name = argv[5]
    registry = _build_registry()

    # Parameter grid metadata (for discover message)
    ndev = len(devices)
    dtypes = [jnp.float32, jnp.float64, jnp.complex64, jnp.complex128]
    N_list = list(i * ndev for i in [2, 3, 4, 10])
    T_A_list = [1, 2, 3, 5]
    for task_N in N_list:
        for task_T_A in T_A_list:
            params_summary = {
                "N": [task_N],
                "T_A": [task_T_A],
                "dtype": [task_dtype_name],
            }

            # Announce discovery for this single task
            _println(
                "MPTEST_DISCOVER",
                {
                    "proc": proc_id,
                    "available": sorted(registry.keys()),
                    "selected": [task_name],
                    "params": params_summary,
                },
            )

            fn = registry[task_name]
            # Map dtype name to jnp dtype
            dt = next(dt for dt in dtypes if jnp.dtype(dt).name == task_dtype_name)
            n_ok = 0
            n_fail = 0
            try:
                fn(task_N, task_T_A, dt)
                _println(
                    "MPTEST_RESULT",
                    {
                        "proc": proc_id,
                        "name": task_name,
                        "status": "ok",
                        "params": {
                            "N": task_N,
                            "T_A": task_T_A,
                            "dtype": task_dtype_name,
                        },
                    },
                )
                n_ok += 1
            except Exception:
                tb = traceback.format_exc(limit=40)
                _println(
                    "MPTEST_RESULT",
                    {
                        "proc": proc_id,
                        "name": task_name,
                        "status": "fail",
                        "params": {
                            "N": task_N,
                            "T_A": task_T_A,
                            "dtype": task_dtype_name,
                        },
                        "traceback": tb,
                    },
                )
                n_fail += 1

            _println(
                "MPTEST_SUMMARY",
                {"proc": proc_id, "ok": n_ok, "fail": n_fail, "total": n_ok + n_fail},
            )
            return 0


main(sys.argv)
