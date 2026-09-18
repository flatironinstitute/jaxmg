from pathlib import Path

import pytest

from gpu_test_helper import run_gpu_test


pytestmark = [pytest.mark.gpu, pytest.mark.multi_gpu]

HERE = Path(__file__).resolve().parent
GPU_TEST = HERE / "run_qr.py"
DTYPES = ("float32", "float64", "complex64", "complex128")


@pytest.mark.parametrize("dtype_name", DTYPES)
def test_qr_padded_reduced_factors(dtype_name):
    """Validate padded Q and R outputs for every supported scalar type."""
    run_gpu_test(GPU_TEST, 2, "padded_qr", dtype_name)


def test_qr_shardmap_ctx():
    """Run reduced QR under a caller-owned JIT with donated A storage."""
    run_gpu_test(GPU_TEST, 2, "padded_qr", "float32", interface="context")


def test_qr_column_grid():
    """Exercise distributed R extraction over process columns."""
    run_gpu_test(GPU_TEST, 2, "column_grid", "float32")


def test_qr_square_aligned():
    """Exercise the square boundary without local padding."""
    run_gpu_test(GPU_TEST, 1, "square_aligned", "float32")
