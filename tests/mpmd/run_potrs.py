import os
import sys
import json
import traceback
from typing import Callable, Dict, List

import jax

coord_addr = sys.argv[1]
proc_id = int(sys.argv[2])
num_procs = int(sys.argv[3])
import time
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
from jaxmg import potrs, potrs_shardmap_ctx
from functools import partial
from jaxmg.utils import random_psd
from jax.experimental import multihost_utils as mu
from itertools import product


# These will be initialized after jax.distributed.initialize()
devices = [d for d in jax.devices() if d.platform == "gpu"]
mesh = jax.make_mesh((jax.device_count(),), ("x",))

@partial(jax.jit, static_argnames=("_T_A",))
def jitted_potrs(_a, _b, _T_A):
    out = partial(potrs, mesh=mesh, in_specs=(P("x", None), P(None, None)), pad=True)(
        _a, _b, _T_A
    )
    return out


@partial(jax.jit, static_argnames=("_T_A",))
def jitted_potrs_status(_a, _b, _T_A):
    out = partial(
        potrs, mesh=mesh, in_specs=(P("x", None), P(None, None)), pad=True, return_status=True
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

    out = jitted_potrs(_A.copy(), _b.copy(), T_A)
    norm_scipy = jnp.linalg.norm(b - A @ expected_out)
    norm_potrf = jnp.linalg.norm(b - A @ out)
    print(norm_scipy, norm_potrf)
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


def _build_registry() -> Dict[str, Callable]:
    # Map test names to callables that accept (N, T_A, dtype)
    return {
        "arange": cusolver_solve_arange,
        "non_psd": cusolver_solve_non_psd,
        "non_symm": cusolver_solve_non_symm,
        "psd": cusolver_solve_psd,
    }


def main(argv: List[str]):
    # Single-task only: expect arguments
    # run_potrs.py <coord_addr> <proc_id> <num_procs> <test_name> <dtype_name>
    
    task_name = argv[4]
    task_dtype_name = argv[5]
    registry = _build_registry()

    # Parameter grid metadata (for discover message)
    ndev= len(devices)
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
                        "params": {"N": task_N, "T_A": task_T_A, "dtype": task_dtype_name},
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
                        "params": {"N": task_N, "T_A": task_T_A, "dtype": task_dtype_name},
                        "traceback": tb,
                    },
                )
                n_fail += 1

            _println("MPTEST_SUMMARY", {"proc": proc_id, "ok": n_ok, "fail": n_fail, "total": n_ok + n_fail})
            return 0


main(sys.argv)
