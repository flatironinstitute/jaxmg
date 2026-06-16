import pytest
pytestmark = pytest.mark.mpmd
import os
import sys
import subprocess
import tempfile
import time
import pathlib
import pytest


def test_launch_2d_grid_reduce_two_processes_CPU():
    num_processes = 2

    here = pathlib.Path(__file__).resolve().parent
    script = here / "run_2dgrid_reduce.py"
    if not script.exists():
        pytest.skip(f"{script} not found; skipping launcher test")

    tmpdir = tempfile.mkdtemp(prefix="jaxmg_2dgrid_")
    procs = []

    env_base = os.environ.copy()

    for i in range(num_processes):
        out_path = os.path.join(tmpdir, f"toy_{i}.out")
        out_file = open(out_path, "wb")
        env = env_base.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""  # No GPUs
        env["JAX_PLATFORMS"] = "cpu"
        env["JAX_NUM_CPU_DEVICES"] = "2"
        cmd = [sys.executable, "-u", str(script), str(i), str(num_processes)]
        p = subprocess.Popen(
            cmd,
            stdout=out_file,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(here),
        )
        procs.append((p, out_file, cmd, out_path))

    # Wait for completion with a reasonable timeout
    deadline = time.time() + 150
    for p, out_file, cmd, out_path in procs:
        while p.poll() is None and time.time() < deadline:
            time.sleep(0.1)
        # If still running past deadline, terminate
        if p.poll() is None:
            p.terminate()
            p.wait(timeout=5)
        out_file.close()

    # Print outputs and assert success
    for idx, (p, out_file, cmd, out_path) in enumerate(procs):
        print("=================== process {} output ===================".format(idx))
        try:
            with open(out_path, "r") as f:
                data = f.read()
        except Exception as e:
            data = f"(failed to read output file: {e})"
        print(data)
        assert p.returncode == 0, f"process {idx} failed (cmd: {cmd}), see {out_path}"
