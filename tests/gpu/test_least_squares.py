from pathlib import Path

import pytest

from gpu_test_helper import run_gpu_test


pytestmark = pytest.mark.gpu

GPU_TEST = Path(__file__).resolve().parent / "run_least_squares.py"


@pytest.mark.parametrize("dtype_name", ("float32", "float64", "complex64", "complex128"))
@pytest.mark.multi_gpu
def test_least_squares_padded_two_gpu(dtype_name):
    """Validate padded distributed GELS for every supported dtype."""
    run_gpu_test(GPU_TEST, 2, "matrix", dtype_name)


@pytest.mark.multi_gpu
def test_least_squares_vector_input():
    """Preserve the rank of a vector solve input."""
    run_gpu_test(GPU_TEST, 2, "vector", "float32")


@pytest.mark.parametrize(
    "case_name", ("matrix_replicated", "matrix_2d_sharded", "vector_replicated")
)
@pytest.mark.multi_gpu
def test_least_squares_rhs_placement_modes(case_name):
    """Preserve supported user-facing solve-input shardings."""
    run_gpu_test(GPU_TEST, 2, case_name, "float32")


@pytest.mark.multi_gpu
def test_least_squares_shardmap_ctx():
    """Run the least-squares context interface inside an external JIT."""
    run_gpu_test(
        GPU_TEST, 2, "matrix", "float32", interface="context"
    )


@pytest.mark.multi_gpu
def test_least_squares_column_grid():
    """Exercise padded A and B redistribution over process columns."""
    run_gpu_test(GPU_TEST, 2, "column_grid", "float32")


@pytest.mark.single_gpu
def test_least_squares_square_aligned():
    """Exercise the square boundary without local padding."""
    run_gpu_test(GPU_TEST, 1, "square_aligned", "float32")
