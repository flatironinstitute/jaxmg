"""Inject an info-readback error and check native cleanup on one GPU."""

import ctypes
import os
import sys
import traceback

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from cusolvermp_case_utils import (
    emit,
    local_device_id_for_process,
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


def main():
    record = dict(proc=proc_id, name=routine, dtype=dtype_name, interface="public")
    try:
        assert num_procs == 1
        mesh = Mesh(np.asarray(jax.devices()).reshape(1, 1), ("pr", "pc"))
        a = jax.device_put(
            jnp.eye(64, dtype=jnp.float32), NamedSharding(mesh, P("pr", "pc"))
        )
        b = jax.device_put(
            jnp.ones(64, dtype=jnp.float32), NamedSharding(mesh, P("pr"))
        )
        jax.block_until_ready((a, b))
        hooks = ctypes.CDLL(None)
        before = hooks.jaxmg_test_destroyed_descriptors()
        # Solves read factorization info first, then solve info. SYEVD reads once.
        os.environ["JAXMG_TEST_FAIL_INFO_COPY"] = "1" if routine == "syevd" else "2"
        try:
            args = (a,) if routine == "syevd" else (a, b)
            result = getattr(jaxmg, routine)(*args, 64, mesh=mesh)
            jax.block_until_ready(result)
        except Exception as error:
            assert "CUDA error 1 " in str(error), str(error)
            assert f"cusolvermp_{routine}.cc" in str(error), str(error)
        else:
            raise AssertionError("Injected CUDA error did not reach the caller")
        assert hooks.jaxmg_test_copy_failure_injected() == 1
        assert hooks.jaxmg_test_destroyed_descriptors() - before == 2
        record["status"] = "ok"
    except Exception:
        record.update(status="fail", traceback=traceback.format_exc())
    finally:
        os.environ.pop("JAXMG_TEST_FAIL_INFO_COPY", None)
        emit("GPU_TEST_RESULT", record)


if __name__ == "__main__":
    main()
