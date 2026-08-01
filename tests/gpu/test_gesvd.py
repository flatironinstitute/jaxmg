import os
from pathlib import Path

import pytest

from gpu_test_helper import run_gpu_test


pytestmark = pytest.mark.gpu

HERE = Path(__file__).resolve().parent
GPU_TEST = HERE / "run_gesvd.py"
DTYPES = ("float32", "float64", "complex64", "complex128")


@pytest.mark.multi_gpu
@pytest.mark.parametrize("dtype_name", DTYPES)
def test_gesvd_rectangular_thin_smoke(dtype_name):
    """Validate thin tall-matrix GESVD for every supported scalar type."""
    run_gpu_test(GPU_TEST, 2, "tall_uv", dtype_name)


@pytest.mark.multi_gpu
@pytest.mark.parametrize(
    "case_name",
    ("wide_padding_u", "wide_padding_vh", "wide_padding_values"),
)
def test_gesvd_independent_output_modes(case_name):
    """Exercise padded U-only, Vh-only, and values-only native handlers."""
    run_gpu_test(GPU_TEST, 2, case_name, "float32")


@pytest.mark.multi_gpu
def test_gesvd_full_matrices():
    """Validate full U and Vh output shapes on two GPUs."""
    run_gpu_test(GPU_TEST, 2, "tall_full_uv", "float32")


@pytest.mark.multi_gpu
def test_gesvd_shardmap_ctx():
    """Run GESVD under a caller-owned JIT with donated A storage."""
    run_gpu_test(
        GPU_TEST,
        2,
        "wide_padding_uv",
        "float32",
        interface="context",
    )


@pytest.mark.slow
@pytest.mark.multi_gpu
def test_gesvd_non_degenerate_2d_grid():
    """Exercise rectangular redistribution on a 2x2 process grid."""
    if os.environ.get("JAXMG_RUN_COMPREHENSIVE_GPU_TESTS") != "1":
        pytest.skip("set JAXMG_RUN_COMPREHENSIVE_GPU_TESTS=1 for the 2x2 case")
    run_gpu_test(GPU_TEST, 4, "balanced_padding_uv", "float32")
