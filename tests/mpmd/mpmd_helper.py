import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import jax
import pytest


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def run_mpmd_script(script: Path, requested_procs: int, name: str) -> None:
    gpu_count = jax.device_count("gpu")
    if gpu_count != requested_procs:
        pytest.skip(
            f"Need {requested_procs} visible GPUs to run MPMD test; saw {gpu_count}."
        )

    coord = f"127.0.0.1:{_find_free_port()}"
    env = os.environ.copy()
    env.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    env["JAXMG_NUMBER_OF_DEVICES"] = str(requested_procs)

    procs = []
    for proc_id in range(requested_procs):
        cmd = [
            sys.executable,
            "-u",
            str(script),
            coord,
            str(proc_id),
            str(requested_procs),
            name,
        ]
        procs.append(
            subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=str(script.parent),
                text=True,
                bufsize=1,
            )
        )

    logs = []
    deadline = time.time() + 180
    for proc in procs:
        chunks = []
        while proc.poll() is None and time.time() < deadline:
            assert proc.stdout is not None
            line = proc.stdout.readline()
            if line:
                chunks.append(line)
        if proc.stdout is not None:
            chunks.append(proc.stdout.read() or "")
        logs.append("".join(chunks))

    exit_codes = [proc.wait(timeout=5) for proc in procs]
    for proc_id, code in enumerate(exit_codes):
        if code != 0:
            print(f"===== MPMD proc {proc_id} output =====")
            print(logs[proc_id])
        assert code == 0

    seen = set()
    failures = []
    for proc_id, log in enumerate(logs):
        for line in log.splitlines():
            if not line.startswith("MPMD_RESULT "):
                continue
            payload = json.loads(line.split(" ", 1)[1])
            seen.add(payload["proc"])
            if payload["status"] != "ok":
                failures.append(payload)

    assert seen == set(range(requested_procs)), (
        f"Missing MPMD result lines. seen={sorted(seen)}\n"
        + "\n\n".join(f"===== proc {i} =====\n{log}" for i, log in enumerate(logs))
    )
    assert not failures, (
        "MPMD failures:\n"
        + "\n".join(json.dumps(failure, sort_keys=True) for failure in failures)
        + "\n\n"
        + "\n\n".join(f"===== proc {i} =====\n{log}" for i, log in enumerate(logs))
    )
