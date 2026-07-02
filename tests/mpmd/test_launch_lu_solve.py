import os
from pathlib import Path

import pytest

from mpmd_helper import run_mpmd_test


pytestmark = pytest.mark.mpmd

HERE = Path(__file__).resolve().parent
MP_TEST = HERE / "run_lu_solve.py"
DTYPES = ("float32", "float64", "complex64", "complex128")
SMOKE_CASES = (
    (1, "row_major_no_padding"),
    (2, "row_major_no_padding"),
    (2, "column_major_padding"),
    (4, "skinny_rhs"),
)
RHS_PLACEMENT_CASES = (
    (4, "vector_rhs_replicated", "float32"),
    (4, "single_column_rhs_replicated", "float32"),
    (4, "single_column_rhs_row_sharded", "float32"),
    (4, "row_major_no_padding", "float32"),
)
COMPREHENSIVE_PROCESS_COUNTS = (1, 2, 3, 4, 5, 6, 7, 8)
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
    run_mpmd_test(MP_TEST, requested_procs, case_name, dtype_name)


@pytest.mark.parametrize("requested_procs,case_name,dtype_name", RHS_PLACEMENT_CASES)
def test_lu_solve_rhs_placement_modes(requested_procs, case_name, dtype_name):
    """Check supported user-facing sharding choices for the RHS input."""
    run_mpmd_test(MP_TEST, requested_procs, case_name, dtype_name)


@pytest.mark.slow
@pytest.mark.parametrize("requested_procs", COMPREHENSIVE_PROCESS_COUNTS)
@pytest.mark.parametrize("case_name", COMPREHENSIVE_CASES)
@pytest.mark.parametrize("dtype_name", DTYPES)
def test_lu_solve_rank_per_gpu_comprehensive(requested_procs, case_name, dtype_name):
    """Exercise many process-grid shapes when explicitly requested."""
    if os.environ.get("JAXMG_RUN_COMPREHENSIVE_MPMD") != "1":
        pytest.skip("set JAXMG_RUN_COMPREHENSIVE_MPMD=1 for the full MPMD suite")
    run_mpmd_test(MP_TEST, requested_procs, case_name, dtype_name)
