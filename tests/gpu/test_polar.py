from pathlib import Path

import pytest

from gpu_test_helper import run_gpu_test


pytestmark = pytest.mark.gpu

HERE = Path(__file__).resolve().parent
GPU_TEST = HERE / "run_polar.py"
DTYPES = ("float32", "float64", "complex64", "complex128")


@pytest.mark.parametrize("dtype_name", DTYPES)
@pytest.mark.multi_gpu
def test_polar_padded_factors(dtype_name):
    """Validate padded Up and H outputs for every supported scalar type."""
    run_gpu_test(GPU_TEST, 2, "padded_uh", dtype_name)


@pytest.mark.multi_gpu
def test_polar_can_omit_h():
    """Exercise the shorter Up-only native handler."""
    run_gpu_test(GPU_TEST, 2, "padded_u", "float32")


@pytest.mark.multi_gpu
def test_polar_jit_ctx():
    """Run polar under a caller-owned JIT with donated A storage."""
    run_gpu_test(GPU_TEST, 2, "padded_uh", "float32", interface="context")


@pytest.mark.multi_gpu
def test_polar_column_grid():
    """Exercise padded output redistribution over process columns."""
    run_gpu_test(GPU_TEST, 2, "column_grid", "float32")


@pytest.mark.single_gpu
def test_polar_square_aligned():
    """Exercise the square boundary without local padding."""
    run_gpu_test(GPU_TEST, 1, "square_aligned", "float32")


@pytest.mark.parametrize("num_procs,case", [
    pytest.param(1, "square_aligned", marks=pytest.mark.single_gpu),
    pytest.param(2, "padded_uh", marks=pytest.mark.multi_gpu),
    pytest.param(2, "padded_u", marks=pytest.mark.multi_gpu),
    pytest.param(2, "column_grid", marks=pytest.mark.multi_gpu),
])
def test_polar_shardmap_ctx(num_procs, case):
    """Validate local polar factors with either output mode and grid axis."""
    run_gpu_test(GPU_TEST, num_procs, case, "float64", interface="shardmap")
