import os
from pathlib import Path

import pytest

from gpu_test_helper import run_gpu_test


pytestmark = pytest.mark.gpu

HERE = Path(__file__).resolve().parent
GPU_TEST = HERE / "run_getrf.py"
DTYPES = ("float32", "float64", "complex64", "complex128")
SMOKE_CASES = (
    pytest.param(1, "row_major_no_padding", marks=pytest.mark.single_gpu),
    pytest.param(1, "column_major_padding", marks=pytest.mark.single_gpu),
    pytest.param(2, "row_major_no_padding", marks=pytest.mark.multi_gpu),
    pytest.param(2, "column_grid_padding", marks=pytest.mark.multi_gpu),
    pytest.param(2, "column_major_padding", marks=pytest.mark.multi_gpu),
    pytest.param(4, "column_major_padding", marks=pytest.mark.multi_gpu),
)
INTERFACE_CASES = (
    pytest.param(1, "row_major_no_padding", "context", marks=pytest.mark.single_gpu),
    pytest.param(1, "row_major_no_padding", "inferred", marks=pytest.mark.single_gpu),
    pytest.param(1, "replicated_input", "public", marks=pytest.mark.single_gpu),
    pytest.param(2, "column_grid_no_padding", "context", marks=pytest.mark.multi_gpu),
    pytest.param(2, "column_major_padding", "inferred", marks=pytest.mark.multi_gpu),
    pytest.param(2, "replicated_input", "context", marks=pytest.mark.multi_gpu),
)
SINGULAR_PROCESS_COUNTS = (
    pytest.param(1, marks=pytest.mark.single_gpu),
    pytest.param(2, marks=pytest.mark.multi_gpu),
    pytest.param(4, marks=pytest.mark.multi_gpu),
)
COMPREHENSIVE_PROCESS_COUNTS = (
    pytest.param(1, marks=pytest.mark.single_gpu),
    *(
        pytest.param(count, marks=pytest.mark.multi_gpu)
        for count in range(2, 9)
    ),
)
COMPREHENSIVE_CASES = (
    "row_major_no_padding",
    "column_major_padding",
    "column_grid_padding",
    "column_grid_no_padding",
)


@pytest.mark.parametrize("requested_procs,case_name", SMOKE_CASES)
@pytest.mark.parametrize("dtype_name", DTYPES)
def test_getrf_rank_per_gpu_smoke(requested_procs, case_name, dtype_name):
    """Run representative GETRF process grids through the public API."""
    run_gpu_test(GPU_TEST, requested_procs, case_name, dtype_name)


@pytest.mark.parametrize("requested_procs,case_name,interface", INTERFACE_CASES)
def test_getrf_interfaces(requested_procs, case_name, interface):
    """Cover the context variant, inferred layouts, and replicated input."""
    run_gpu_test(GPU_TEST, requested_procs, case_name, "float64", interface=interface)


@pytest.mark.parametrize("requested_procs", SINGULAR_PROCESS_COUNTS)
@pytest.mark.parametrize("dtype_name", DTYPES)
def test_getrf_failure_returns_nan(requested_procs, dtype_name):
    """A failed factorization fills every rank's factor with NaN."""
    run_gpu_test(
        GPU_TEST, requested_procs, "singular_diagonal", dtype_name, interface="singular"
    )


@pytest.mark.slow
@pytest.mark.parametrize("requested_procs", COMPREHENSIVE_PROCESS_COUNTS)
@pytest.mark.parametrize("case_name", COMPREHENSIVE_CASES)
@pytest.mark.parametrize("dtype_name", DTYPES)
def test_getrf_rank_per_gpu_comprehensive(requested_procs, case_name, dtype_name):
    """Exercise many process-grid shapes when explicitly requested."""
    if os.environ.get("JAXMG_RUN_COMPREHENSIVE_GPU_TESTS") != "1":
        pytest.skip(
            "set JAXMG_RUN_COMPREHENSIVE_GPU_TESTS=1 for the full GPU suite"
        )
    run_gpu_test(GPU_TEST, requested_procs, case_name, dtype_name)
