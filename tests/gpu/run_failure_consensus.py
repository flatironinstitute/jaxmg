import os
import sys
import traceback

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from cusolvermp_case_utils import (
    assert_rank_failure_with_nan,
    emit,
    local_device_id_for_process,
    native_status_words,
    select_gpu_allocator,
)


coord_addr, proc_id, num_procs, routine, dtype_name = sys.argv[1:]
proc_id, num_procs = int(proc_id), int(num_procs)
select_gpu_allocator(proc_id)
jax.distributed.initialize(
    coordinator_address=coord_addr,
    num_processes=num_procs,
    process_id=proc_id,
    local_device_ids=[local_device_id_for_process(proc_id)],
    coordinator_bind_address=coord_addr if proc_id == 0 else None,
)

import jaxmg
from jaxmg import _cusolvermp_status


def main():
    """Force a post-solver failure on one rank after an otherwise valid solve."""
    record = {
        "proc": proc_id,
        "name": routine,
        "dtype": dtype_name,
        "interface": "public",
    }
    try:
        mesh = Mesh(
            np.asarray(jax.devices(), dtype=object).reshape(num_procs, 1), ("pr", "pc")
        )
        tile_size = 64
        n = tile_size * num_procs
        a = jax.device_put(
            jnp.diag(jnp.arange(n, dtype=jnp.float32) + 1),
            NamedSharding(mesh, P("pr", "pc")),
        )
        # Only the last rank reports an error, after releasing its real handle.
        os.environ["JAXMG_TEST_FAIL_DESTROY"] = str(int(proc_id == num_procs - 1))
        if routine == "potrs":
            b = jax.device_put(
                jnp.ones(n, dtype=jnp.float32), NamedSharding(mesh, P("pr"))
            )
            *results, status = jaxmg.potrs(
                a, b, tile_size, mesh=mesh, return_logdet=True, return_status=True
            )
        else:
            *results, status = jaxmg.syevd(
                a, tile_size, mesh=mesh, return_status=True
            )
        status.block_until_ready()
        os.environ["JAXMG_TEST_FAIL_DESTROY"] = "0"

        status_size = getattr(
            _cusolvermp_status, f"_CUSOLVERMP_{routine.upper()}_STATUS_SIZE"
        )
        words = native_status_words(status).reshape(num_procs, status_size)
        # kDestroyHandleFailed is 16. Other diagnostics remain rank-local:
        # only the last rank saw CUSOLVER_STATUS_INTERNAL_ERROR (field 11).
        np.testing.assert_array_equal(words[:, 11], [0] * (num_procs - 1) + [7])
        assert_rank_failure_with_nan(words.ravel(), status_size, (16,), *results)
        multihost_utils.sync_global_devices(f"{routine}_failure_consensus_complete")
        record["status"] = "ok"
    except Exception:
        record.update(status="fail", traceback=traceback.format_exc())
    finally:
        os.environ["JAXMG_TEST_FAIL_DESTROY"] = "0"
        emit("GPU_TEST_RESULT", record)


if __name__ == "__main__":
    main()
