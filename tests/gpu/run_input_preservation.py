import sys
import traceback

import jax
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
    """Check input preservation and numerical results with donation disabled."""
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
        sharding = NamedSharding(mesh, P("pr", "pc"))
        for n in (128, 150):  # Complete tiles, then padding on both matrix axes.
            diagonal = np.linspace(1, 2, n, dtype=dtype_name)
            a_host = np.diag(diagonal) + np.outer(diagonal, diagonal) / n
            a = jax.device_put(a_host, sharding)
            options = dict(mesh=mesh, donate=False, return_status=True)
            if routine in ("potrs", "lu_solve", "least_squares"):
                expected = np.column_stack((np.ones_like(diagonal), diagonal))
                b_host = a_host @ expected
                b = jax.device_put(b_host, sharding)
                x, status = getattr(jaxmg, routine)(a, b, 64, **options)
                np.testing.assert_allclose(
                    global_array_to_numpy(x), expected, rtol=5e-4, atol=5e-4
                )
                np.testing.assert_array_equal(global_array_to_numpy(b), b_host)
            elif routine == "syevd":
                values, vectors, status = jaxmg.syevd(a, 64, **options)
                values = global_array_to_numpy(values)
                vectors = global_array_to_numpy(vectors)
                np.testing.assert_allclose(
                    (vectors * values) @ vectors.T, a_host, rtol=5e-4, atol=5e-4
                )
            else:
                u, values, vh, status = jaxmg.gesvd(a, 64, **options)
                u, values, vh = map(global_array_to_numpy, (u, values, vh))
                np.testing.assert_allclose(
                    (u * values) @ vh, a_host, rtol=5e-4, atol=5e-4
                )
            np.testing.assert_array_equal(global_array_to_numpy(a), a_host)
            native_name = "GELS" if routine == "least_squares" else routine.upper()
            status_size = getattr(
                _cusolvermp_status, f"_CUSOLVERMP_{native_name}_STATUS_SIZE"
            )
            words = native_status_words(status).reshape(num_procs, status_size)
            np.testing.assert_array_equal(words[:, 0], 0)
        multihost_utils.sync_global_devices(f"{routine}_input_preservation_complete")
        record["status"] = "ok"
    except Exception:
        record.update(status="fail", traceback=traceback.format_exc())
    emit("GPU_TEST_RESULT", record)


if __name__ == "__main__":
    main()
