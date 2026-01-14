from pathlib import Path

import pytest
import jax

from mpmd_helper import run_mpmd_test

HERE = Path(__file__).parent
MP_TEST = HERE / "run_syevd.py"

if len(jax.devices("gpu")) == 0:
    pytest.skip("No GPUs found. Skipping")


# Build the parameter grid once at collection time and parametrize each
# (requested_procs, test_name, N, T_A, dtype) as a separate pytest test so
# each task appears individually in pytest's summary.
GPU_COUNT = jax.device_count("gpu")
if GPU_COUNT == 0:
    pytest.skip("No GPUs found. Skipping")

# Only run for the currently visible GPU count; the original test enumerated
# requested_procs=(1,2,3,4) and skipped when mismatched. Here we parametrize
# only for the local visible gpu count to keep collection stable.
REQUESTED_PROCS_LIST = (GPU_COUNT,)

DTYPES = ["float32", "float64", "complex64", "complex128"]
TEST_NAMES = ["arange", "psd", "arange_no_V", "psd_no_V"]

TASKS = []
TASK_IDS = []
for requested_procs in REQUESTED_PROCS_LIST:
    for name in TEST_NAMES:
        for dtype_name in DTYPES:
            TASKS.append((requested_procs, name, dtype_name))
            TASK_IDS.append(f"{name}-{dtype_name}-p{requested_procs}")


@pytest.mark.parametrize(
    "requested_procs,name, dtype_name",
    TASKS,
    ids=TASK_IDS,
)
def test_task_mpmd_syevd(requested_procs, name, dtype_name):
    """Run a single distributed syevd task as an individual pytest test."""

    run_mpmd_test(MP_TEST, requested_procs, name, dtype_name)
