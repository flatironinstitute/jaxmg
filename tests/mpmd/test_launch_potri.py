import pytest
pytestmark = pytest.mark.mpmd
from pathlib import Path

import pytest
import jax

from mpmd_helper import run_mpmd_test

HERE = Path(__file__).parent
MP_TEST = HERE / "run_potri.py"

platforms = set(d.platform for d in jax.devices())
if "gpu" not in platforms:
    pytest.skip("No GPUs found. Skipping", allow_module_level=True)
else:
    gpu_count = jax.device_count()

    # Only run for the currently visible GPU count; the original test enumerated
    # requested_procs=(1,2,3,4) and skipped when mismatched. Here we parametrize
    # only for the local visible gpu count to keep collection stable.
    requested_procs_list = (gpu_count,)


    dtypes = ["float32", "float64", "complex64", "complex128"]
    test_names = ["arange", "non_psd", "non_symm", "psd"]


    tasks = []
    task_ids = []
    for requested_procs in requested_procs_list:
        for name in test_names:
            for dtype_name in dtypes:
                tasks.append((requested_procs, name, dtype_name))
                task_ids.append(f"{name}-{dtype_name}-p{requested_procs}")


    @pytest.mark.parametrize(
        "requested_procs,name, dtype_name",
        tasks,
        ids=task_ids,
    )
    def test_task_mpmd_potri(requested_procs, name, dtype_name):
        """Run a single distributed potri task as an individual pytest test."""

        run_mpmd_test(MP_TEST, requested_procs, name, dtype_name)
