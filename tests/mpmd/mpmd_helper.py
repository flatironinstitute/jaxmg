import os
import sys
import socket
import subprocess
import time
import json
from pathlib import Path
from typing import List

import pytest
import jax


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run_mpmd_test(mp_test: Path, requested_procs: int, name: str, dtype_name: str) -> None:
    """Launch an MPMD test runner script across processes and assert success.

    The runner script is expected to emit single-line JSON messages prefixed
    with ``MPTEST_RESULT`` and ``MPTEST_SUMMARY`` describing per-process
    outcomes. This helper handles launching, log collection, parsing, and
    basic validation.
    """

    # Quick guard: ensure visible GPUs still match the requested value.
    gpu_count = jax.device_count("gpu")
    if gpu_count != requested_procs:
        pytest.skip(
            f"Need {requested_procs} GPUs in CUDA_VISIBLE_DEVICES to run this test (have {gpu_count})"
        )

    port = _find_free_port()
    coord = f"127.0.0.1:{port}"

    env = os.environ.copy()
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "platform")

    here = mp_test.parent
    print(f"[launcher] starting task {name}: dtype={dtype_name}, procs={requested_procs}")

    procs: List[subprocess.Popen] = []
    logs: List[str] = []
    for i in range(requested_procs):
        cmd = [
            sys.executable,
            "-u",
            str(mp_test),
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
            cwd=str(here),
            text=True,
            bufsize=1,
        )
        procs.append(p)

    # Collect output with a timeout to avoid hanging the test suite
    deadline = time.time() + 150
    for idx, p in enumerate(procs):
        out_chunks: List[str] = []
        while p.poll() is None and time.time() < deadline:
            assert p.stdout is not None
            line = p.stdout.readline()
            if line:
                out_chunks.append(line)
        remaining = p.stdout.read() or ""  # type: ignore[union-attr]
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
                # Ignore non-JSON lines
                pass

    expected_procs = set(range(requested_procs))
    assert expected_procs.issubset(per_proc_seen), (
        f"Missing results from some processes for task {name} dtype={dtype_name}. "
        f"expected={sorted(expected_procs)} seen={sorted(per_proc_seen)}\n"
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
    assert ok_count > 0, (
        f"Expected at least one ok result for task {name} dtype={dtype_name}; raw logs:\n"
        + "\n\n".join(logs)
    )

    print(f"[launcher] task {name} dtype={dtype_name} completed successfully")
