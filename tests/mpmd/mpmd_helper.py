import os
import sys
import math
import socket
import subprocess
import time
import json
import re
import shlex
from pathlib import Path
from typing import List

import pytest
import jax

def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _positive_int_from_env(name: str) -> int | None:
    """Return a positive integer environment override, if one is set."""
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return parsed


def _first_integer(value: str | None) -> int | None:
    """Parse the first integer from a Slurm resource string."""
    if not value:
        return None
    match = re.search(r"\d+", value)
    return int(match.group(0)) if match else None


def _slurm_hostnames() -> list[str]:
    """Return hostnames in the current Slurm allocation."""
    nodelist = os.environ.get("SLURM_JOB_NODELIST")
    if not nodelist:
        return []
    try:
        completed = subprocess.run(
            ["scontrol", "show", "hostnames", nodelist],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def _slurm_node_count(hostnames: list[str]) -> int:
    """Return the available Slurm node count for nested ``srun`` launches."""
    override = _positive_int_from_env("JAXMG_MPMD_SRUN_MAX_NODES")
    if override is None:
        override = _positive_int_from_env("JAXMG_MPMD_SRUN_NODES")
    if override is not None:
        return override
    for name in ("SLURM_NNODES", "SLURM_JOB_NUM_NODES"):
        parsed = _positive_int_from_env(name)
        if parsed is not None:
            return parsed
    return max(1, len(hostnames))


def _gpus_per_node(local_gpu_count: int) -> int:
    """Return the expected GPUs per node in the Slurm allocation."""
    override = _positive_int_from_env("JAXMG_MPMD_GPUS_PER_NODE")
    if override is not None:
        return override
    for name in ("SLURM_GPUS_ON_NODE", "SLURM_GPUS_PER_NODE"):
        parsed = _first_integer(os.environ.get(name))
        if parsed is not None and parsed > 0:
            return parsed
    return max(1, local_gpu_count)


def _srun_node_count_for(requested_procs: int, local_gpu_count: int) -> int:
    """Choose enough Slurm nodes for a rank-per-GPU test case."""
    hostnames = _slurm_hostnames()
    gpus_per_node = _gpus_per_node(local_gpu_count)
    max_nodes = _slurm_node_count(hostnames)
    needed_nodes = math.ceil(requested_procs / gpus_per_node)
    if needed_nodes > max_nodes:
        pytest.skip(
            f"Need {needed_nodes} Slurm nodes for {requested_procs} ranks "
            f"with {gpus_per_node} GPUs per node; allocation allows {max_nodes}"
        )
    return max(1, needed_nodes)


def _srun_coordinator(port: int) -> str:
    """Return a coordinator address reachable by all Slurm ranks."""
    hostnames = _slurm_hostnames()
    hostname = hostnames[0] if hostnames else socket.gethostname()
    return f"{hostname}:{port}"


def _srun_gpu_args() -> list[str]:
    """Return optional Slurm GPU binding arguments for nested ``srun``."""
    value = os.environ.get("JAXMG_MPMD_SRUN_GPU_ARGS", "")
    return shlex.split(value) if value else []


def _launcher_env(name: str, dtype_name: str, requested_procs: int) -> dict[str, str]:
    """Return the common environment used by all ranks in one test case.

    The GPU allocator is chosen per-rank in the runner scripts (see
    ``cusolvermp_case_utils.select_gpu_allocator``), since under multi-node Slurm
    the launcher node may differ from the compute nodes. An explicit
    ``XLA_PYTHON_CLIENT_ALLOCATOR`` in the environment is inherited and honored.
    """
    env = os.environ.copy()
    env.setdefault("JAXMG_BARRIER_NAME", f"{name}_{dtype_name}_{requested_procs}")
    return env


def _check_results(
    logs: List[str],
    *,
    requested_procs: int,
    name: str,
    dtype_name: str,
    interface: str,
) -> None:
    """Parse runner JSON records and fail the pytest case on any bad rank."""
    parsed = []
    per_proc_seen = set()
    for log in logs:
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
                # Ignore non-JSON lines from JAX, NCCL, or Slurm.
                pass

    expected_procs = set(range(requested_procs))
    assert expected_procs.issubset(per_proc_seen), (
        f"Missing results from some processes for task {name} dtype={dtype_name}. "
        f"expected={sorted(expected_procs)} seen={sorted(per_proc_seen)}\n"
        f"Raw logs:\n" + "\n\n".join(f"===== log {i} =====\n{l}" for i, l in enumerate(logs))
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
            summary_lines.append(f"===== log {i} =====\n{l}")
        pytest.fail("\n".join(summary_lines))

    wrong_interfaces = [
        r
        for r in parsed
        if r.get("status") == "ok" and r.get("interface") != interface
    ]
    assert not wrong_interfaces, (
        f"Expected interface={interface!r}, received results={wrong_interfaces}"
    )

    ok_count = sum(1 for r in parsed if r.get("status") == "ok")
    assert ok_count > 0, (
        f"Expected at least one ok result for task {name} dtype={dtype_name}; raw logs:\n"
        + "\n\n".join(logs)
    )


def _run_with_srun(
    mp_test: Path,
    requested_procs: int,
    nodes: int,
    name: str,
    dtype_name: str,
    coord: str,
    env: dict[str, str],
) -> List[str]:
    """Run one distributed case through Slurm, one task per Python rank.

    This is the preferred path on HPC systems.  It lets Slurm create the
    per-rank environment, including ``SLURM_PROCID`` and ``SLURM_LOCALID``,
    instead of starting several distributed JAX ranks as unmanaged subprocesses
    inside one Slurm task.
    """
    timeout = int(os.environ.get("JAXMG_MPMD_TEST_TIMEOUT", "300"))
    gpu_args = _srun_gpu_args()
    env.update(
        {
            "JAXMG_COORD": coord,
            "JAXMG_NUM_PROCS": str(requested_procs),
            "JAXMG_CASE_NAME": name,
            "JAXMG_DTYPE_NAME": dtype_name,
            "JAXMG_RUNNER": mp_test.name,
            "JAXMG_RUNNER_DIR": str(mp_test.parent),
            "JAXMG_PYTHON": sys.executable,
        }
    )
    if any(arg.startswith("--gpus-per-task") for arg in gpu_args):
        env.setdefault("JAXMG_LOCAL_DEVICE_ID", "0")
    rank_command = (
        'cd "$JAXMG_RUNNER_DIR" && '
        '"$JAXMG_PYTHON" -u "$JAXMG_RUNNER" "$JAXMG_COORD" "$SLURM_PROCID" '
        '"$JAXMG_NUM_PROCS" "$JAXMG_CASE_NAME" "$JAXMG_DTYPE_NAME"'
    )
    cmd = [
        "srun",
        f"--nodes={nodes}",
        f"--ntasks={requested_procs}",
        "--kill-on-bad-exit=1",
        *gpu_args,
        "bash",
        "-lc",
        rank_command,
    ]
    try:
        completed = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        pytest.fail(
            f"srun timed out for task {name} dtype={dtype_name} "
            f"procs={requested_procs}\n{output}"
        )
    if completed.returncode != 0:
        pytest.fail(
            f"srun failed for task {name} dtype={dtype_name} "
            f"procs={requested_procs} rc={completed.returncode}\n{completed.stdout}"
        )
    return [completed.stdout]


def _run_with_local_subprocesses(
    mp_test: Path,
    requested_procs: int,
    name: str,
    dtype_name: str,
    coord: str,
    env: dict[str, str],
) -> List[str]:
    """Run one distributed case with local subprocesses for developer machines."""
    here = mp_test.parent
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
        rank_env = env.copy()
        rank_env["CUDA_VISIBLE_DEVICES"] = str(i)
        rank_env["JAXMG_LOCAL_DEVICE_ID"] = "0"
        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=rank_env,
            cwd=str(here),
            text=True,
            bufsize=1,
        )
        procs.append(p)

    deadline = time.time() + int(os.environ.get("JAXMG_MPMD_TEST_TIMEOUT", "300"))
    for p in procs:
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

    exits = [p.wait(timeout=5) for p in procs]
    for idx, code in enumerate(exits):
        if code != 0:
            print(f"===== mp_test proc {idx} combined output =====")
            print(logs[idx])
        assert code == 0, f"mp_test process {idx} failed with exit code {code}"
    return logs


def run_mpmd_test(
    mp_test: Path,
    requested_procs: int,
    name: str,
    dtype_name: str,
    *,
    interface: str = "public",
) -> None:
    """Run one rank-per-GPU solver case and assert success.

    Pytest executes cases sequentially.  A single case still has to start
    ``requested_procs`` ranks concurrently because JAX distributed arrays and
    cuSOLVERMp require one live Python process per participating GPU.  Under
    Slurm, use ``srun`` so that process-to-GPU binding matches the production
    launch model.  Outside Slurm, fall back to local subprocesses with one
    visible GPU per rank. ``interface`` selects either the internally jitted
    public wrapper or the caller-jitted context interface.
    """
    if interface not in ("public", "context"):
        raise ValueError(f"unknown solver interface {interface!r}")

    launcher = os.environ.get("JAXMG_MPMD_LAUNCHER")
    if launcher is None:
        launcher = "srun" if "SLURM_JOB_ID" in os.environ else "subprocess"

    # Quick guard: outside Slurm the subprocess launcher needs all requested
    # GPUs visible to the parent process.  Under Slurm, the parent pytest
    # process only sees the local node; nested ``srun`` can still start ranks
    # across the full multi-node allocation.
    try:
        gpu_count = jax.device_count("gpu")
    except RuntimeError:
        gpu_count = 0
    srun_nodes = None
    if launcher == "srun":
        srun_nodes = _srun_node_count_for(requested_procs, gpu_count)
    elif launcher == "subprocess" and gpu_count < requested_procs:
        pytest.skip(
            f"Need at least {requested_procs} GPUs in CUDA_VISIBLE_DEVICES "
            f"to run this test (have {gpu_count})"
        )
    elif launcher not in ("srun", "subprocess"):
        raise ValueError(f"unknown JAXMG_MPMD_LAUNCHER={launcher!r}")

    port = _find_free_port()
    coord = _srun_coordinator(port) if launcher == "srun" else f"127.0.0.1:{port}"

    env = _launcher_env(name, dtype_name, requested_procs)
    env["JAXMG_TEST_INTERFACE"] = interface
    print(
        f"[launcher] starting task {name}: dtype={dtype_name}, "
        f"procs={requested_procs}, interface={interface}"
    )

    if launcher == "srun":
        assert srun_nodes is not None
        logs = _run_with_srun(
            mp_test,
            requested_procs,
            srun_nodes,
            name,
            dtype_name,
            coord,
            env,
        )
    else:
        logs = _run_with_local_subprocesses(
            mp_test, requested_procs, name, dtype_name, coord, env
        )

    _check_results(
        logs,
        requested_procs=requested_procs,
        name=name,
        dtype_name=dtype_name,
        interface=interface,
    )
    print(
        f"[launcher] task {name} dtype={dtype_name} "
        f"interface={interface} completed successfully"
    )
