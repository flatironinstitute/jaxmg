"""Execute the shipped example scripts and assert they report success."""

import os
import re
import subprocess
import sys
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

    procs = []
    for rank in range(num_processes):
        rank_env = os.environ.copy()
        rank_env["CUDA_VISIBLE_DEVICES"] = str(rank)
        rank_env["JAXMG_LOCAL_DEVICE_ID"] = "0"
        procs.append(
            subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    str(script),
                    "--coordinator",
                    coord,
                    "--process-id",
                    str(rank),
                    "--num-processes",
                    str(num_processes),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=rank_env,
                text=True,
            )
        )

    outputs = []
    try:
        for proc in procs:
            outputs.append(proc.communicate(timeout=timeout)[0])
    except subprocess.TimeoutExpired:
        for proc in procs:
            proc.kill()
        for proc in procs:
            proc.communicate()
        pytest.fail(f"{script_name} did not finish within {timeout}s")
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.kill()

    # A peer that dies takes the collective down with it, so report every rank
    # rather than only rank 0 -- the real error is often elsewhere.
    for rank, proc in enumerate(procs):
        if proc.returncode != 0:
            joined = "\n".join(
                f"--- rank {i} ---\n{out}" for i, out in enumerate(outputs)
            )
            pytest.fail(
                f"{script_name} rank {rank} exited {proc.returncode}\n{joined}"
            )
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
