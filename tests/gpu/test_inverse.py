from pathlib import Path

import pytest

from gpu_test_helper import run_gpu_test


pytestmark = pytest.mark.gpu
GPU_TEST = Path(__file__).with_name("run_inverse.py")


@pytest.mark.parametrize("routine", ["potri", "getri"])
@pytest.mark.parametrize("dtype", ["float32", "float64", "complex64", "complex128"])
@pytest.mark.parametrize("ranks,case", [
    pytest.param(1, "column_major_padding", marks=pytest.mark.single_gpu),
    pytest.param(2, "column_grid_padding", marks=pytest.mark.multi_gpu),
])
def test_inverse_dtypes_and_padding(routine, dtype, ranks, case):
    """Compare full inverses and both identity products with NumPy."""
    run_gpu_test(GPU_TEST, ranks, f"{routine}:{case}", dtype)


@pytest.mark.parametrize("routine", ["potri", "getri"])
@pytest.mark.parametrize("ranks,case,interface", [
    pytest.param(1, "row_major_no_padding", "context", marks=pytest.mark.single_gpu),
    pytest.param(2, "column_major_padding", "shardmap", marks=pytest.mark.multi_gpu),
    pytest.param(2, "row_major_no_padding", "context", marks=pytest.mark.multi_gpu),
    pytest.param(2, "replicated_input", "public", marks=pytest.mark.multi_gpu),
    pytest.param(4, "column_major_padding", "public", marks=pytest.mark.multi_gpu),
])
def test_inverse_layouts_and_contexts(routine, ranks, case, interface):
    """Exercise row/column grids, a square grid, and replicated result restoration."""
    run_gpu_test(GPU_TEST, ranks, f"{routine}:{case}", "complex128", interface=interface)


@pytest.mark.parametrize("routine", ["potri", "getri"])
@pytest.mark.parametrize("ranks", [
    pytest.param(1, marks=pytest.mark.single_gpu),
    pytest.param(2, marks=pytest.mark.multi_gpu),
])
def test_inverse_singular_returns_nan(routine, ranks):
    """Failed factorization must skip inversion and invalidate every rank."""
    run_gpu_test(
        GPU_TEST, ranks, f"{routine}:column_grid_no_padding", "float64",
        interface="singular",
    )
