"""Distributed launch helpers for the cuSOLVERMp backend.

The cuSOLVERMp path has a stricter launch contract than ordinary local JAXMg
usage. The native solver assumes that the row-major order of the JAX device mesh
is the same row-major process-grid order passed to cuSOLVERMp:

    rank = process_row * process_cols + process_col

This module keeps that contract explicit. In particular, the first multi-node
target is one Python process per node, with that process controlling every GPU
on the node. Some Slurm/Open MPI JAX launch modes default to one visible device
per process unless ``local_device_ids`` is passed to
``jax.distributed.initialize``. ``initialize_node_process`` therefore infers the
local GPU ids from the scheduler environment and passes them explicitly before
any JAX computation is allowed to initialize the backend.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import os
import re

import jax
import numpy as np
from jax.sharding import Mesh


def _positive_int_from_env(name: str, environ: Mapping[str, str]) -> int | None:
    value = environ.get(name)
    if value is None or value == "":
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    if parsed <= 0:
        raise ValueError(f"{name} must be positive when set.")
    return parsed


def _parse_visible_device_count(
    visible_devices: str | None,
) -> int | None:
    if visible_devices is None:
        return None
    visible_devices = visible_devices.strip()
    if visible_devices in {"", "-1", "all", "none", "None", "NoDevFiles"}:
        return None
    devices = [device.strip() for device in visible_devices.split(",") if device.strip()]
    return len(devices) if devices else None


def _parse_slurm_gpu_count(value: str | None) -> int | None:
    """Best-effort parser for common Slurm GPU count encodings."""
    if value is None or value == "":
        return None
    value = value.strip()
    if value.isdigit():
        return int(value)

    if "," in value and all(part.strip().isdigit() for part in value.split(",")):
        return len([part for part in value.split(",") if part.strip()])

    # Examples seen across Slurm installations include "gpu:4",
    # "a100:4", "gpu:a100:4", and sometimes "4(S:0-1)".
    value_before_socket_suffix = value.split("(", 1)[0]
    matches = re.findall(r"\d+", value_before_socket_suffix)
    if not matches:
        return None
    parsed = int(matches[-1])
    return parsed if parsed > 0 else None


def _infer_local_device_count(
    environ: Mapping[str, str] = os.environ,
) -> int | None:
    explicit = _positive_int_from_env("JAXMG_LOCAL_DEVICE_COUNT", environ)
    if explicit is not None:
        return explicit

    for name in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"):
        count = _parse_visible_device_count(environ.get(name))
        if count is not None:
            return count

    for name in ("SLURM_GPUS_ON_NODE", "SLURM_GPUS_PER_NODE", "SLURM_STEP_GPUS"):
        count = _parse_slurm_gpu_count(environ.get(name))
        if count is not None:
            return count

    return None


def _normalize_local_device_ids(
    local_device_ids: Sequence[int] | None,
    *,
    environ: Mapping[str, str] = os.environ,
) -> tuple[int, ...]:
    if local_device_ids is not None:
        ids = tuple(int(device_id) for device_id in local_device_ids)
        if not ids:
            raise ValueError("local_device_ids must not be empty.")
        if len(set(ids)) != len(ids):
            raise ValueError("local_device_ids must not contain duplicates.")
        if any(device_id < 0 for device_id in ids):
            raise ValueError("local_device_ids must be non-negative.")
        return ids

    count = _infer_local_device_count(environ)
    if count is None:
        raise RuntimeError(
            "Could not infer the local GPU count for one-process-per-node "
            "JAXMg initialization. Pass local_device_ids explicitly or set "
            "JAXMG_LOCAL_DEVICE_COUNT."
        )
    return tuple(range(count))


def initialize_node_process(
    *,
    local_device_ids: Sequence[int] | None = None,
    coordinator_address: str | None = None,
    num_processes: int | None = None,
    process_id: int | None = None,
    cluster_detection_method: str | None = None,
    initialization_timeout: int = 300,
    heartbeat_timeout_seconds: int = 100,
    shutdown_timeout_seconds: int = 300,
    coordinator_bind_address: str | None = None,
    partition_index: int | None = None,
) -> tuple[int, ...]:
    """Initialize JAX for the one-Python-process-per-node cuSOLVERMp mode.

    Call this before any operation that may initialize the JAX/XLA backend,
    including ``jax.devices()``, ``jax.device_put()``, array creation on GPU, or
    any JIT compilation. On Slurm and Open MPI, JAX can usually discover the
    coordinator and process ids automatically, but the local GPU ids are passed
    explicitly so that one process sees all GPUs on its node.

    Args:
        local_device_ids: Local GPU ids controlled by this Python process. If
            omitted, JAXMg tries to infer ``range(num_local_gpus)`` from
            ``CUDA_VISIBLE_DEVICES``, ``NVIDIA_VISIBLE_DEVICES``, common Slurm
            GPU variables, or ``JAXMG_LOCAL_DEVICE_COUNT``.
        coordinator_address: Optional coordinator address for manual JAX
            distributed setup.
        num_processes: Optional global number of Python processes.
        process_id: Optional dense process id for this Python process.
        cluster_detection_method: Optional JAX cluster detection method.
        initialization_timeout: JAX distributed initialization timeout, seconds.
        heartbeat_timeout_seconds: JAX distributed heartbeat timeout, seconds.
        shutdown_timeout_seconds: JAX distributed shutdown timeout, seconds.
        coordinator_bind_address: Optional coordinator bind address.
        partition_index: Optional JAX partition index.

    Returns:
        The normalized local device ids passed to JAX.
    """
    ids = _normalize_local_device_ids(local_device_ids)
    os.environ["JAXMG_EXECUTION_MODE"] = "SPMD"
    os.environ["JAXMG_NUMBER_OF_DEVICES"] = str(len(ids))

    if jax.distributed.is_initialized():
        return ids

    jax.distributed.initialize(
        coordinator_address=coordinator_address,
        num_processes=num_processes,
        process_id=process_id,
        local_device_ids=list(ids),
        cluster_detection_method=cluster_detection_method,
        initialization_timeout=initialization_timeout,
        heartbeat_timeout_seconds=heartbeat_timeout_seconds,
        shutdown_timeout_seconds=shutdown_timeout_seconds,
        coordinator_bind_address=coordinator_bind_address,
        partition_index=partition_index,
    )
    return ids


def _row_major_device_grid(devices: Sequence[object], process_rows: int, process_cols: int):
    if process_rows <= 0 or process_cols <= 0:
        raise ValueError("process_rows and process_cols must be positive.")
    required = process_rows * process_cols
    if len(devices) != required:
        raise ValueError(
            "cuSOLVERMp mesh construction requires exactly "
            f"{required} devices, got {len(devices)}."
        )
    return np.asarray(list(devices), dtype=object).reshape(process_rows, process_cols)


def make_cusolvermp_mesh(
    process_rows: int,
    process_cols: int,
    *,
    axis_names: tuple[str, str] = ("pr", "pc"),
    devices: Sequence[object] | None = None,
) -> Mesh:
    """Create the row-major JAX mesh expected by ``jaxmg.potrs_mp``.

    The returned mesh is shaped ``(process_rows, process_cols)`` and uses
    row-major device order. This makes the JAX mesh rank order match the
    cuSOLVERMp process-grid mapping used natively:

        rank = process_row * process_cols + process_col

    Args:
        process_rows: Number of cuSOLVERMp process rows.
        process_cols: Number of cuSOLVERMp process columns.
        axis_names: Two mesh axis names. The default is ``("pr", "pc")``.
        devices: Optional explicit global device sequence. If omitted,
            ``jax.devices()`` is used, so this function must be called after
            distributed initialization in a multi-node run.
    """
    if len(axis_names) != 2:
        raise ValueError("axis_names must contain exactly two names.")
    device_seq = list(jax.devices() if devices is None else devices)
    grid = _row_major_device_grid(device_seq, int(process_rows), int(process_cols))
    return Mesh(grid, axis_names)
