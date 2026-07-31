import os
from pathlib import Path

import pytest

from gpu_test_helper import run_gpu_test


pytestmark = pytest.mark.gpu

HERE = Path(__file__).resolve().parent
GPU_TEST = HERE / "run_lu_solve.py"
DTYPES = ("float32", "float64", "complex64", "complex128")
SMOKE_CASES = (
    pytest.param(1, "row_major_no_padding", marks=pytest.mark.single_gpu),
    pytest.param(2, "row_major_no_padding", marks=pytest.mark.multi_gpu),
    pytest.param(2, "column_major_padding", marks=pytest.mark.multi_gpu),
    pytest.param(4, "skinny_rhs", marks=pytest.mark.multi_gpu),
)
RHS_PLACEMENT_CASES = (
    (2, "skinny_rhs_explicit_mesh", "float32"),
    (4, "vector_rhs_replicated", "float32"),
    (4, "single_column_rhs_replicated", "float32"),
    (4, "single_column_rhs_row_sharded", "float32"),
    (4, "documented_degenerate_rhs", "float32"),
    (4, "row_major_no_padding", "float32"),
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
    "skinny_rhs",
)


@pytest.mark.parametrize("requested_procs,case_name", SMOKE_CASES)
@pytest.mark.parametrize("dtype_name", DTYPES)
def test_lu_solve_rank_per_gpu_smoke(requested_procs, case_name, dtype_name):
    """Run representative LU rank-per-GPU cases through public API only."""
    run_gpu_test(GPU_TEST, requested_procs, case_name, dtype_name)


@pytest.mark.multi_gpu
@pytest.mark.parametrize("requested_procs,case_name,dtype_name", RHS_PLACEMENT_CASES)
def test_lu_solve_rhs_placement_modes(requested_procs, case_name, dtype_name):
    """Check supported user-facing sharding choices for the RHS input."""
    run_gpu_test(GPU_TEST, requested_procs, case_name, dtype_name)


@pytest.mark.multi_gpu
def test_lu_solve_shardmap_ctx_two_gpu():
    """Run the caller-jitted LU context interface on two GPUs."""
    run_gpu_test(
        GPU_TEST,
        2,
        "column_grid_no_padding",
        "float32",
        interface="context",
    )


@pytest.mark.slow
@pytest.mark.parametrize("requested_procs", COMPREHENSIVE_PROCESS_COUNTS)
@pytest.mark.parametrize("case_name", COMPREHENSIVE_CASES)
@pytest.mark.parametrize("dtype_name", DTYPES)
def test_lu_solve_rank_per_gpu_comprehensive(requested_procs, case_name, dtype_name):
    """Exercise many process-grid shapes when explicitly requested."""
    if os.environ.get("JAXMG_RUN_COMPREHENSIVE_GPU_TESTS") != "1":
        pytest.skip(
            "set JAXMG_RUN_COMPREHENSIVE_GPU_TESTS=1 for the full GPU suite"
        )
    run_gpu_test(GPU_TEST, requested_procs, case_name, dtype_name)
