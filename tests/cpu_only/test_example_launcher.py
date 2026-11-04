"""Exercise example diagnostics with real subprocesses, without requiring GPUs."""

import importlib.util
from pathlib import Path
import subprocess

import jax
import pytest


pytestmark = pytest.mark.cpu_only


@pytest.fixture
def launcher(monkeypatch, tmp_path):
    gpu_tests = Path(__file__).resolve().parents[1] / "gpu"
    monkeypatch.syspath_prepend(str(gpu_tests))
    spec = importlib.util.spec_from_file_location(
        "example_launcher_under_test", gpu_tests / "test_examples.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "EXAMPLES_DIR", tmp_path)
    monkeypatch.setattr(jax, "device_count", lambda backend: 2)
    monkeypatch.setenv("JAXMG_GPU_TEST_LAUNCHER", "subprocess")
    monkeypatch.setenv("JAXMG_GPU_TEST_TIMEOUT", "2")
    processes = []
    original_popen = subprocess.Popen

    def record_process(*args, **kwargs):
        proc = original_popen(*args, **kwargs)
        processes.append(proc)
        return proc

    monkeypatch.setattr(subprocess, "Popen", record_process)
    yield module, tmp_path
    assert all(proc.poll() is not None for proc in processes)


def test_example_launcher_sets_bind_address(launcher, monkeypatch):
    module, path = launcher
    monkeypatch.setenv("JAX_COORDINATOR_BIND_ADDRESS", "0.0.0.0:12345")
    (path / "example.py").write_text(
        "import os, sys\n"
        "coord = sys.argv[sys.argv.index('--coordinator') + 1]\n"
        "assert coord.startswith('127.0.0.1:')\n"
        "assert os.environ['JAX_COORDINATOR_BIND_ADDRESS'] == coord\n"
    )
    # Both ranks must exit successfully; a failed assertion on either is reported.
    module._run_example("example.py")


def test_example_launcher_drains_large_output(launcher):
    module, path = launcher
    (path / "example.py").write_text(
        "def main():\n"
        "    print('x' * 200000)\n"
        "    print('Result correct:', True)\n"
        "main()\n"
    )
    output = module._run_example("example.py")
    assert "Result correct: True" in output
    assert "main return" in output
    assert "entering interpreter shutdown" in output


@pytest.mark.parametrize("failed_rank", [0, 1])
def test_example_launcher_reports_failed_rank_and_peer(launcher, failed_rank):
    module, path = launcher
    (path / "example.py").write_text(
        "import sys, time\n"
        "def main():\n"
        "    rank = int(sys.argv[sys.argv.index('--process-id') + 1])\n"
        "    print('rank', rank, flush=True)\n"
        f"    if rank == {failed_rank}:\n"
        "        raise RuntimeError('deliberate failure')\n"
        "    time.sleep(30)\n"
        "main()\n"
    )
    with pytest.raises(pytest.fail.Exception) as error:
        module._run_example("example.py")
    message = str(error.value)
    assert "a rank exited unsuccessfully" in message
    assert "deliberate failure" in message
    assert "--- rank 0:" in message and "--- rank 1:" in message
    assert f"rank {failed_rank}: before cleanup=1" in message


@pytest.mark.parametrize("shutdown", [False, True])
def test_example_launcher_retains_timeout_progress_and_stacks(launcher, shutdown):
    module, path = launcher
    stall = "atexit.register(time.sleep, 30)" if shutdown else "time.sleep(30)"
    (path / "example.py").write_text(
        "import atexit, sys, time\n"
        "def main():\n"
        "    rank = int(sys.argv[sys.argv.index('--process-id') + 1])\n"
        "    print('rank', rank, flush=True)\n"
        "    if rank == 1:\n"
        f"        {stall}\n"
        "main()\n"
    )
    with pytest.raises(pytest.fail.Exception) as error:
        module._run_example("example.py")
    message = str(error.value)
    assert "did not finish within 2s" in message
    assert "rank 0: before cleanup=0" in message
    assert "rank 1: before cleanup=None, after cleanup=" in message
    rank_one = message.split("--- rank 1:")[1]
    assert stall in rank_one
    assert "Timeout (" in rank_one
    assert ("entering interpreter shutdown" in rank_one) == shutdown
