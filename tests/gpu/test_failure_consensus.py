import os
from pathlib import Path
import subprocess
import sys
import signal

import jax
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
     "least_squares", "potri", "getri"),
)
@pytest.mark.multi_gpu
def test_rank_local_failure_returns_nan(routine, failed_cleanup_library, monkeypatch):
    """One rank's cleanup failure invalidates distributed and scalar outputs."""
    preload = str(failed_cleanup_library)
    if os.environ.get("LD_PRELOAD"):
        preload += ":" + os.environ["LD_PRELOAD"]
    monkeypatch.setenv("LD_PRELOAD", preload)
    run_gpu_test(HERE / "run_failure_consensus.py", 2, routine, "float32")


@pytest.mark.parametrize("routine", ("potrs", "lu_solve", "syevd"))
@pytest.mark.single_gpu
def test_cuda_error_releases_descriptors(routine, failed_cleanup_library, monkeypatch):
    """A CUDA error after computation must still release both descriptors."""
    preload = str(failed_cleanup_library)
    if os.environ.get("LD_PRELOAD"):
        preload += ":" + os.environ["LD_PRELOAD"]
    monkeypatch.setenv("LD_PRELOAD", preload)
    run_gpu_test(HERE / "run_cuda_error_cleanup.py", 1, routine, "float32")


@pytest.mark.parametrize("routine,stage", [
    ("potrf", "factor"), ("getrf", "factor"),
    ("potrs", "factor"), ("lu_solve", "factor"),
    ("potri", "factor"), ("getri", "factor"),
    ("potri", "inverse"), ("getri", "inverse"),
])
@pytest.mark.multi_gpu
def test_workspace_failure_skips_compute(
    routine, stage, failed_cleanup_library, monkeypatch,
):
    """One rank's allocation failure skips the numerical call on every rank."""
    preload = str(failed_cleanup_library)
    if os.environ.get("LD_PRELOAD"):
        preload += ":" + os.environ["LD_PRELOAD"]
    monkeypatch.setenv("LD_PRELOAD", preload)
    monkeypatch.setenv("JAXMG_TEST_WORKSPACE_STAGE", stage)
    run_gpu_test(HERE / "run_failure_consensus.py", 2, routine, "float32")


@pytest.mark.single_gpu
@pytest.mark.parametrize("operation,status", [("CREATE", 7), ("CREATE", 3), ("DESTROY", 7)])
def test_grid_failure_stops_process(operation, status, failed_cleanup_library):
    """Grid failures abort before result restoration can reuse the communicator."""
    try:
        if not jax.devices("gpu"):
            pytest.skip("Requires a GPU")
    except RuntimeError:
        pytest.skip("Requires a GPU")
    env = os.environ.copy()
    env["LD_PRELOAD"] = str(failed_cleanup_library) + (
        ":" + env["LD_PRELOAD"] if env.get("LD_PRELOAD") else ""
    )
    env[f"JAXMG_TEST_FAIL_GRID_{operation}"] = str(status)
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    result = subprocess.run(
        [sys.executable, "-c", """
import resource
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
import jax
import jax.numpy as jnp
import jaxmg
with jax.set_mesh(jax.make_mesh((1, 1), ('pr', 'pc'))):
    jaxmg.potrf(jnp.eye(64), 32).block_until_ready()
raise AssertionError('A failed grid must not return a numerical result')
"""],
        env=env, capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == -signal.SIGABRT, result.stdout + result.stderr
    native_call = "cusolverMpCreateDeviceGrid" if operation == "CREATE" else "cusolverMpDestroyGrid"
    assert f"JAXMg fatal: {native_call}" in result.stderr
    assert "restart the distributed job" in result.stderr
