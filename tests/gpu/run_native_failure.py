"""Rank-per-GPU check that a failed native solve returns NaNs on every rank.

``A = diag(2, ..., 2, 0)`` is singular, so the distributed Cholesky
factorization fails. The native call reports it in the status vector but can
leave finite, wrong values in the solution buffer.
"""

import sys
import traceback
from functools import partial

import jax

if not jax.config.jax_enable_x64:
    jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from cusolvermp_case_utils import (
    emit,
    global_array_to_numpy,
    local_device_id_for_process,
    native_status_words,
    select_gpu_allocator,
)


coord_addr = sys.argv[1]
proc_id = int(sys.argv[2])
num_procs = int(sys.argv[3])
case_name = sys.argv[4]
dtype_name = sys.argv[5]

select_gpu_allocator(proc_id)

jax.distributed.initialize(
    coordinator_address=coord_addr,
    num_processes=num_procs,
    process_id=proc_id,
    local_device_ids=[local_device_id_for_process(proc_id)],
    coordinator_bind_address=coord_addr if proc_id == 0 else None,
)

from jaxmg import potrs, potrs_shardmap_ctx
from jaxmg._cusolvermp_status import _CUSOLVERMP_POTRS_STATUS_SIZE


N = 128
T_A = 64


def run_case() -> None:
    singular = case_name == "potrs_singular"
    diagonal = np.full(N, 2.0)
    rhs = np.ones(N)
    if singular:
        diagonal[-1] = 0.0
        rhs[-1] = 0.0
    a_host = np.diag(diagonal)

    mesh = Mesh(np.asarray(jax.devices(), dtype=object), ("x",))
    with jax.set_mesh(mesh):
        a = jax.device_put(a_host, NamedSharding(mesh, P("x", None)))
        b = jax.device_put(rhs, NamedSharding(mesh, P()))

        solves = {
            "public": partial(potrs, T_A=T_A, donate=False, return_status=True),
            "public_jit": jax.jit(
                partial(potrs, T_A=T_A, donate=False, return_status=True)
            ),
            "context_jit": jax.jit(
                lambda a, b: potrs_shardmap_ctx(a, b, T_A=T_A)[1:]
            ),
        }
        for interface, solve in solves.items():
            x, status = solve(a, b)
            x = global_array_to_numpy(x)
            codes = native_status_words(status)[::_CUSOLVERMP_POTRS_STATUS_SIZE]
            if singular:
                assert np.all(codes != 0), (interface, codes)
                assert np.all(np.isnan(x)), (interface, x)
            else:
                assert np.all(codes == 0), (interface, codes)
                np.testing.assert_allclose(x, rhs / diagonal, rtol=1e-12)

        # The status is not needed for the solution to be NaN.
        x = global_array_to_numpy(potrs(a, b, T_A=T_A, donate=False))
        assert np.all(np.isnan(x)) == singular

    emit(
        "GPU_TEST_RESULT",
        {
            "proc": proc_id,
            "name": case_name,
            "dtype": dtype_name,
            "status": "ok",
            "interface": "public",
        },
    )
    multihost_utils.sync_global_devices(f"native_failure_{case_name}_complete")


def main() -> None:
    try:
        run_case()
    except Exception:
        emit(
            "GPU_TEST_RESULT",
            {
                "proc": proc_id,
                "name": case_name,
                "dtype": dtype_name,
                "status": "fail",
                "traceback": traceback.format_exc(),
            },
        )
    finally:
        emit(
            "GPU_TEST_SUMMARY",
            {"proc": proc_id, "name": case_name, "dtype": dtype_name},
        )


if __name__ == "__main__":
    main()
