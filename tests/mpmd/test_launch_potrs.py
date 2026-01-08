import os
import sys
import socket
import subprocess
import time
import json
from pathlib import Path

import pytest
import jax

HERE = Path(__file__).parent
MP_TEST = HERE / "run_potrs.py"

if len(jax.devices("gpu"))==0:
    pytest.skip("No GPUs found. Skipping")

def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# Build the parameter grid once at collection time and parametrize each
# (requested_procs, test_name, N, T_A, dtype) as a separate pytest test so
# each task appears individually in pytest's summary.
gpu_count = jax.device_count("gpu")
if gpu_count == 0:
    pytest.skip("No GPUs found. Skipping")

# Only run for the currently visible GPU count; the original test enumerated
# requested_procs=(1,2,3,4) and skipped when mismatched. Here we parametrize
# only for the local visible gpu count to keep collection stable.
requested_procs_list = (gpu_count,)

ndev = jax.device_count()
dtypes = ["float32", "float64", "complex64", "complex128"]
test_names = ["arange", "non_psd", "non_symm", "psd"]

tasks = []
task_ids = []
for requested_procs in requested_procs_list:
    for name in test_names:
        for dtype_name in dtypes:
            tasks.append((requested_procs, name, dtype_name))
            task_ids.append(f"{name}-{dtype_name}-p{requested_procs}")

@pytest.mark.parametrize(
    "requested_procs,name, dtype_name",
    tasks,
    ids=task_ids,
)
def test_task_mpmd(requested_procs, name, dtype_name):
    """Run a single distributed task as an individual pytest test.

    Each parametrized invocation will spawn `requested_procs` child
    processes that initialize JAX distributed and run the single-task
    `run_potrs.py` runner. The child runner is expected to emit
    MPTEST_* JSON lines which we parse for pass/fail.
    """

    # Quick guard: ensure visible GPUs still match the requested value.
    gpu_count = jax.device_count("gpu")
    if gpu_count != requested_procs:
        pytest.skip(f"Need {requested_procs} GPUs in CUDA_VISIBLE_DEVICES to run this test (have {gpu_count})")

    port = _find_free_port()
    coord = f"127.0.0.1:{port}"

    env = os.environ.copy()
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    print(f"[launcher] starting task {name}: dtype={dtype_name}, procs={requested_procs}")

    procs = []
    logs = []
    for i in range(requested_procs):
        cmd = [
            sys.executable,
            "-u",
            str(MP_TEST),
            coord,
            str(i),
            str(requested_procs),
            name,
            dtype_name,
        ]
        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(HERE),
            text=True,
            bufsize=1,
        )
        procs.append(p)

    # Collect output with a timeout to avoid hanging the test suite
    deadline = time.time() + 150
    for idx, p in enumerate(procs):
        out_chunks = []
        while p.poll() is None and time.time() < deadline:
            line = p.stdout.readline()
            if line:
                out_chunks.append(line)
        remaining = p.stdout.read() or ""
        if remaining:
            out_chunks.append(remaining)
        logs.append("".join(out_chunks))

    # Ensure all processes exited
    exits = [p.wait(timeout=5) for p in procs]
    for idx, code in enumerate(exits):
        if code != 0:
            print(f"===== mp_test proc {idx} combined output =====")
            print(logs[idx])
        assert code == 0, f"mp_test process {idx} failed with exit code {code}"

    # Parse MPTEST JSON lines and aggregate results
    parsed = []
    per_proc_seen = set()
    for idx, log in enumerate(logs):
        for line in log.splitlines():
            try:
                if line.startswith("MPTEST_RESULT "):
                    payload = json.loads(line.split(" ", 1)[1])
                    parsed.append(payload)
                    per_proc_seen.add(payload.get("proc"))
                elif line.startswith("MPTEST_SUMMARY "):
                    payload = json.loads(line.split(" ", 1)[1])
                    per_proc_seen.add(payload.get("proc"))
            except json.JSONDecodeError:
                pass

    expected_procs = set(range(requested_procs))
    assert expected_procs.issubset(per_proc_seen), (
        f"Missing results from some processes for task {name} dtype={dtype_name}. expected={sorted(expected_procs)} seen={sorted(per_proc_seen)}\n"
        f"Raw logs:\n" + "\n\n".join(f"===== proc {i} =====\n{l}" for i, l in enumerate(logs))
    )

    failures = [r for r in parsed if r.get("status") == "fail"]
    if failures:
        def _short_msg(tb: str) -> str:
            if not tb:
                return ""
            lines = [ln for ln in tb.splitlines() if ln.strip()]
            return lines[-1] if lines else tb.strip()

        summary_lines = [f"Task {name} dtype={dtype_name} failures:"]
        for r in failures:
            summary_lines.append(
                f"- proc {r.get('proc')} :: {r.get('name')}: {_short_msg(r.get('traceback',''))}"
            )
        summary_lines.append("")
        for i, l in enumerate(logs):
            summary_lines.append(f"===== proc {i} =====\n{l}")
        pytest.fail("\n".join(summary_lines))

    ok_count = sum(1 for r in parsed if r.get("status") == "ok")
    assert ok_count > 0, f"Expected at least one ok result for task {name} dtype={dtype_name}; raw logs:\n" + "\n\n".join(logs)

    print(f"[launcher] task {name} dtype={dtype_name} completed successfully")