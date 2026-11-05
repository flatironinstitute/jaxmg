from pathlib import Path

import pytest

from gpu_test_helper import run_gpu_test


pytestmark = [pytest.mark.gpu, pytest.mark.multi_gpu]
HERE = Path(__file__).resolve().parent


@pytest.mark.parametrize(
    "routine", ("potrs", "lu_solve", "least_squares", "syevd", "gesvd")
)
def test_non_donated_inputs_are_preserved(routine):
    """Padded and unpadded solves must leave non-donated inputs unchanged."""
    run_gpu_test(HERE / "run_input_preservation.py", 2, routine, "float32")
