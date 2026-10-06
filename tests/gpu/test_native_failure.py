from pathlib import Path

import pytest

from gpu_test_helper import run_gpu_test


pytestmark = pytest.mark.gpu

HERE = Path(__file__).resolve().parent
GPU_TEST = HERE / "run_native_failure.py"


@pytest.mark.parametrize(
    "requested_procs",
    [
        pytest.param(1, marks=pytest.mark.single_gpu),
        pytest.param(2, marks=pytest.mark.multi_gpu),
    ],
)
@pytest.mark.parametrize("case_name", ["potrs_singular", "potrs_spd"])
def test_potrs_native_failure_returns_nans(requested_procs, case_name):
    """A singular matrix makes POTRS fail natively: the solution must be NaN."""
    run_gpu_test(GPU_TEST, requested_procs, case_name, "float64")
