import ctypes
import os
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


coord_addr, proc_id, num_procs, name, dtype_name = sys.argv[1:]
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
    """Check numerical results and native creation counts in one long-lived process."""
    record = {"proc": proc_id, "name": name, "dtype": dtype_name,
              "interface": "public"}
    try:
        layouts = [(num_procs, 1), (1, num_procs), (num_procs, 1)]
        for layout in layouts:
            mesh = Mesh(np.asarray(jax.devices(), dtype=object).reshape(layout),
                        ("pr", "pc"))
            sharding = NamedSharding(mesh, P("pr", "pc"))
            for n in (128, 192):
                diagonal = np.linspace(1, 2, n, dtype=dtype_name)
                a_host = np.diag(diagonal) + np.outer(diagonal, diagonal) / n
                expected = np.ones((n, 128), dtype=dtype_name)
                a = jax.device_put(a_host, sharding)
                b = jax.device_put(a_host @ expected, sharding)
                for routine in ("potrs", "lu_solve"):
                    status_size = getattr(
                        _cusolvermp_status, f"_CUSOLVERMP_{routine.upper()}_STATUS_SIZE"
                    )
                    for _ in range(2):
                        x, status = getattr(jaxmg, routine)(
                            a, b, 64, mesh=mesh, donate=False, return_status=True
                        )
                        np.testing.assert_allclose(
                            global_array_to_numpy(x), expected, rtol=5e-4, atol=5e-4
                        )
                        words = native_status_words(status).reshape(num_procs, status_size)
                        np.testing.assert_array_equal(words[:, 0], 0)

        # A rank-local descriptor error must not poison the cached handle/grid.
        for fail in (True, False):
            os.environ["JAXMG_TEST_FAIL_DESCRIPTOR_DESTROY"] = str(
                int(fail and proc_id == num_procs - 1)
            )
            x, status = jaxmg.potrs(
                a, b, 64, mesh=mesh, donate=False, return_status=True
            )
            actual = global_array_to_numpy(x)
            words = native_status_words(status).reshape(
                num_procs, _cusolvermp_status._CUSOLVERMP_POTRS_STATUS_SIZE
            )
            np.testing.assert_array_equal(words[:, 0], 14 if fail else 0)
            if fail:
                assert np.isnan(actual).all()
            else:
                np.testing.assert_allclose(actual, expected, rtol=5e-4, atol=5e-4)

        shim = ctypes.CDLL(None)
        assert shim.jaxmg_test_created_handles() == 1
        assert shim.jaxmg_test_created_grids() == len(set(layouts))
        assert shim.jaxmg_test_destroyed_handles() == 0
        assert shim.jaxmg_test_destroyed_grids() == 0
        multihost_utils.sync_global_devices("cache_reuse_complete")
        record["status"] = "ok"
    except Exception:
        record.update(status="fail", traceback=traceback.format_exc())
    finally:
        os.environ["JAXMG_TEST_FAIL_DESCRIPTOR_DESTROY"] = "0"
        emit("GPU_TEST_RESULT", record)


if __name__ == "__main__":
    main()
