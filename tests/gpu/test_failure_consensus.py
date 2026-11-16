import os
from pathlib import Path
import subprocess
import sys

import pytest

from gpu_test_helper import run_gpu_test


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(sys.platform != "linux", reason="Requires Linux LD_PRELOAD"),
]
HERE = Path(__file__).resolve().parent


@pytest.fixture(scope="module")
def failed_cleanup_library(tmp_path_factory):
    """Interpose cuSOLVERMp cleanup without changing the production backend."""
    library = tmp_path_factory.mktemp("failed_cleanup") / "failed_cleanup.so"
    subprocess.run(
        ["cc", "-shared", "-fPIC", str(HERE / "fail_cusolvermp_destroy.c"),
         "-o", str(library), "-ldl"],
        check=True,
        capture_output=True,
        text=True,
    )
    return library


@pytest.mark.parametrize(
    "routine",
    ("potrs", "lu_solve", "potrf", "getrf", "syevd", "gesvd", "polar", "qr",
     "least_squares"),
)
@pytest.mark.multi_gpu
def test_rank_local_failure_returns_nan(routine, failed_cleanup_library, monkeypatch):
    """One rank's cleanup failure invalidates distributed and scalar outputs."""
    preload = str(failed_cleanup_library)
    if os.environ.get("LD_PRELOAD"):
        preload += ":" + os.environ["LD_PRELOAD"]
    monkeypatch.setenv("LD_PRELOAD", preload)
    monkeypatch.setenv("JAXMG_CUSOLVERMP_CACHE", "1")
    run_gpu_test(HERE / "run_failure_consensus.py", 2, routine, "float32")


@pytest.mark.multi_gpu
def test_uncached_handle_failure_returns_nan(failed_cleanup_library, monkeypatch):
    """The uncached path still reports rank-local handle destruction failures."""
    preload = str(failed_cleanup_library)
    if os.environ.get("LD_PRELOAD"):
        preload += ":" + os.environ["LD_PRELOAD"]
    monkeypatch.setenv("LD_PRELOAD", preload)
    monkeypatch.setenv("JAXMG_CUSOLVERMP_CACHE", "0")
    run_gpu_test(HERE / "run_failure_consensus.py", 2, "potrs", "float32")


@pytest.mark.parametrize("num_procs", [
    pytest.param(1, marks=pytest.mark.single_gpu),
    pytest.param(2, marks=pytest.mark.multi_gpu),
])
def test_solver_setup_is_reused(num_procs, failed_cleanup_library, monkeypatch):
    """Reuse setup across routines, sizes, layouts, and a recoverable failure."""
    preload = str(failed_cleanup_library)
    if os.environ.get("LD_PRELOAD"):
        preload += ":" + os.environ["LD_PRELOAD"]
    monkeypatch.setenv("LD_PRELOAD", preload)
    monkeypatch.setenv("JAXMG_CUSOLVERMP_CACHE", "1")
    run_gpu_test(HERE / "run_cache_reuse.py", num_procs, "cache", "float32")


@pytest.mark.parametrize("routine", ("potrs", "lu_solve", "syevd"))
@pytest.mark.single_gpu
def test_cuda_error_releases_descriptors(routine, failed_cleanup_library, monkeypatch):
    """A CUDA error after computation must still release both descriptors."""
    preload = str(failed_cleanup_library)
    if os.environ.get("LD_PRELOAD"):
        preload += ":" + os.environ["LD_PRELOAD"]
    monkeypatch.setenv("LD_PRELOAD", preload)
    run_gpu_test(HERE / "run_cuda_error_cleanup.py", 1, routine, "float32")
