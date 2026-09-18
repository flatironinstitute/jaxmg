from pathlib import Path

import pytest

from gpu_test_helper import run_gpu_test


pytestmark = [pytest.mark.gpu, pytest.mark.multi_gpu]

HERE = Path(__file__).resolve().parent
GPU_TEST = HERE / "run_polar.py"
DTYPES = ("float32", "float64", "complex64", "complex128")


@pytest.mark.parametrize("dtype_name", DTYPES)
def test_polar_padded_factors(dtype_name):
    """Validate padded Up and H outputs for every supported scalar type."""
    run_gpu_test(GPU_TEST, 2, "padded_uh", dtype_name)


def test_polar_can_omit_h():
    """Exercise the shorter Up-only native handler."""
    run_gpu_test(GPU_TEST, 2, "padded_u", "float32")


def test_polar_shardmap_ctx():
    """Run polar under a caller-owned JIT with donated A storage."""
    run_gpu_test(GPU_TEST, 2, "padded_uh", "float32", interface="context")


def test_polar_column_grid():
    """Exercise padded output redistribution over process columns."""
    run_gpu_test(GPU_TEST, 2, "column_grid", "float32")


def test_polar_square_aligned():
    """Exercise the square boundary without local padding."""
    run_gpu_test(GPU_TEST, 1, "square_aligned", "float32")
