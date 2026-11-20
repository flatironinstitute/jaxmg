import os
from pathlib import Path
import subprocess
import sys
import signal
import time

import jax
import pytest

from gpu_test_helper import _find_free_port, run_gpu_test


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


@pytest.mark.multi_gpu
@pytest.mark.parametrize("status", [3, 7])
def test_cached_handle_failure_stops_ranks(status, failed_cleanup_library, tmp_path):
    """A rank-local stream-binding error stops both ranks after cache reuse."""
    try:
        if jax.local_device_count("gpu") < 2:
            pytest.skip("Requires two local GPUs")
    except RuntimeError:
        pytest.skip("Requires two local GPUs")
    env = os.environ.copy()
    env["LD_PRELOAD"] = str(failed_cleanup_library) + (
        ":" + env["LD_PRELOAD"] if env.get("LD_PRELOAD") else ""
    )
    env["JAXMG_CUSOLVERMP_CACHE"] = "1"
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    coord = f"127.0.0.1:{_find_free_port()}"
    env["JAX_COORDINATOR_BIND_ADDRESS"] = coord
    devices = env.get("CUDA_VISIBLE_DEVICES", "0,1").split(",")
    code = """
import os, resource, sys
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
import jax
import jax.numpy as jnp
from jax.experimental import multihost_utils
coord, rank, failure = sys.argv[1:]
rank = int(rank)
jax.distributed.initialize(
    coordinator_address=coord, num_processes=2, process_id=rank,
    local_device_ids=[0], coordinator_bind_address=coord if rank == 0 else None,
    heartbeat_timeout_seconds=10,
)
import jaxmg
with jax.set_mesh(jax.make_mesh((2, 1), ('pr', 'pc'))):
    jaxmg.potrf(jnp.eye(128), 64).block_until_ready()
    print("JAXMG_TEST_CACHE_WARM", flush=True)
    multihost_utils.sync_global_devices("cache_warm")
    if rank == 0:
        os.environ["JAXMG_TEST_FAIL_SET_STREAM"] = failure
    jaxmg.potrf(jnp.eye(128), 64).block_until_ready()
raise AssertionError("A failed handle must not return a numerical result")
"""
    processes = []
    paths = [tmp_path / f"rank_{rank}.log" for rank in range(2)]
    try:
        for rank, path in enumerate(paths):
            rank_env = dict(env, CUDA_VISIBLE_DEVICES=devices[rank])
            with path.open("w") as log:
                processes.append(subprocess.Popen(
                    [sys.executable, "-u", "-c", code, coord, str(rank), str(status)],
                    env=rank_env, stdout=log, stderr=subprocess.STDOUT,
                ))
        deadline = time.monotonic() + 90
        for process in processes:
            process.wait(timeout=max(1, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        pytest.fail("Ranks did not stop after handle failure:\n" +
                    "\n".join(path.read_text() for path in paths))
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.wait()
    logs = [path.read_text() for path in paths]
    assert all("JAXMG_TEST_CACHE_WARM" in log for log in logs), "\n".join(logs)
    assert processes[0].returncode == -signal.SIGABRT, logs[0]
    assert "JAXMg fatal: cuSOLVERMp handle setup" in logs[0], logs[0]
    assert f"cuSOLVER status {status}" in logs[0], logs[0]
    assert processes[1].returncode != 0, logs[1]
    assert all("A failed handle must not return" not in log for log in logs), "\n".join(logs)


@pytest.mark.single_gpu
def test_workspace_failure_reports_diagnostics(failed_cleanup_library):
    """Debug logging preserves the original allocation status and NaN result."""
    try:
        if not jax.devices("gpu"):
            pytest.skip("Requires a GPU")
    except RuntimeError:
        pytest.skip("Requires a GPU")
    env = os.environ.copy()
    env["LD_PRELOAD"] = str(failed_cleanup_library) + (
        ":" + env["LD_PRELOAD"] if env.get("LD_PRELOAD") else ""
    )
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["JAXMG_CUSOLVERMP_DEBUG"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", """
import ctypes
import os
import jax
import jax.numpy as jnp
import numpy as np
import jaxmg
with jax.set_mesh(jax.make_mesh((1, 1), ('pr', 'pc'))):
    os.environ['JAXMG_TEST_FAIL_INFO_ALLOC'] = '1'
    inverse, status = jaxmg.getri(jnp.eye(128), 64, return_status=True)
    assert np.asarray(status)[0] == 24, status
    assert np.isnan(np.asarray(inverse)).all()
assert ctypes.CDLL(None).jaxmg_test_alloc_failure_injected() == 1
"""],
        env=env, capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "getri failure status=[24," in result.stderr
    assert "factorization workspace requested_bytes=" in result.stderr
    assert "original_status=24" in result.stderr
    assert "memory_query_status=0" in result.stderr


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
    env["JAXMG_CUSOLVERMP_CACHE"] = "0"
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
