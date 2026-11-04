"""Execute the shipped example scripts and assert they report success."""

import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

from gpu_test_helper import _find_free_port

pytestmark = [pytest.mark.gpu, pytest.mark.examples]

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"

ALL_EXAMPLES = tuple(sorted(p.name for p in EXAMPLES_DIR.glob("run_*_example.py")))

CORRECT_LINE = re.compile(r"correct:\s*(True|False)", re.I)


def _run_example(script_name, num_processes=2):
    """Launch one example with one process per GPU and return rank 0's stdout."""
    launcher = os.environ.get("JAXMG_GPU_TEST_LAUNCHER")
    if launcher is None:
        launcher = "srun" if "SLURM_JOB_ID" in os.environ else "subprocess"
    if launcher != "subprocess":
        pytest.skip(
            "example tests only support the subprocess launcher; "
            "set JAXMG_GPU_TEST_LAUNCHER=subprocess"
        )

    try:
        import jax

        gpu_count = jax.device_count("gpu")
    except RuntimeError:
        gpu_count = 0
    if gpu_count < num_processes:
        pytest.skip(
            f"Need at least {num_processes} GPUs in CUDA_VISIBLE_DEVICES "
            f"to run this example (have {gpu_count})"
        )

    script = EXAMPLES_DIR / script_name
    assert script.is_file(), f"missing example {script}"

    timeout = int(os.environ.get("JAXMG_GPU_TEST_TIMEOUT", "300"))
    coord = f"127.0.0.1:{_find_free_port()}"

    procs, logs = [], []
    failure = None
    exits = []
    # Files avoid blocking a rank on a full pipe while its peer is being read.
    with tempfile.TemporaryDirectory(prefix="jaxmg-example-") as log_dir:
        try:
            deadline = time.monotonic() + timeout
            for rank in range(num_processes):
                rank_env = os.environ.copy()
                rank_env["CUDA_VISIBLE_DEVICES"] = str(rank)
                rank_env["JAXMG_LOCAL_DEVICE_ID"] = "0"
                # Bind on the same interface used to select the free port.
                rank_env["JAX_COORDINATOR_BIND_ADDRESS"] = coord
                log = open(Path(log_dir) / f"rank-{rank}.log", "w+")
                logs.append(log)
                procs.append(subprocess.Popen(
                    [
                        sys.executable, "-u",
                        str(Path(__file__).with_name("run_example_diagnostic.py")),
                        str(script), "--coordinator", coord,
                        "--process-id", str(rank),
                        "--num-processes", str(num_processes),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=rank_env,
                    text=True,
                ))

            while True:
                exits = [proc.poll() for proc in procs]
                if any(code not in (None, 0) for code in exits):
                    failure = "a rank exited unsuccessfully"
                    break
                if all(code is not None for code in exits):
                    break
                if time.monotonic() >= deadline:
                    failure = f"did not finish within {timeout}s"
                    break
                time.sleep(0.05)
        finally:
            for proc in procs:
                if proc.poll() is None:
                    proc.kill()
            for proc in procs:
                proc.wait()
            outputs = []
            for log in logs:
                log.seek(0)
                outputs.append(log.read())
                log.close()

    if failure:
        joined = "\n".join(
            f"--- rank {i}: before cleanup={exits[i]}, "
            f"after cleanup={proc.returncode} ---\n{output}"
            for i, (proc, output) in enumerate(zip(procs, outputs))
        )
        pytest.fail(f"{script_name}: {failure}\n{joined}")
    return outputs[0]


def _assert_reports_success(script_name):
    stdout = _run_example(script_name)
    match = CORRECT_LINE.search(stdout)
    assert match, (
        f"{script_name} printed no 'correct:' verdict.\n--- rank 0 ---\n{stdout}"
    )
    assert match.group(1).lower() == "true", (
        f"{script_name} reported an incorrect result.\n--- rank 0 ---\n{stdout}"
    )


@pytest.mark.multi_gpu
@pytest.mark.parametrize("script_name", ALL_EXAMPLES)
def test_example_runs(script_name):
    """Every shipped example runs on two ranks and reports a correct result."""
    _assert_reports_success(script_name)
