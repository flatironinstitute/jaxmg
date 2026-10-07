import os
from pathlib import Path
import subprocess
import sys

import pytest

from gpu_test_helper import run_gpu_test


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.multi_gpu,
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


@pytest.mark.parametrize("routine", ("potrs", "syevd"))
def test_rank_local_failure_returns_nan(routine, failed_cleanup_library, monkeypatch):
    """One rank's cleanup failure invalidates distributed and scalar outputs."""
    preload = str(failed_cleanup_library)
    if os.environ.get("LD_PRELOAD"):
        preload += ":" + os.environ["LD_PRELOAD"]
    monkeypatch.setenv("LD_PRELOAD", preload)
    run_gpu_test(HERE / "run_failure_consensus.py", 2, routine, "float32")
