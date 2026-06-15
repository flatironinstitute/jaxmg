"""JAX-only rank-per-GPU launch diagnostic for cuSOLVERMp jobs.

This script deliberately does not import JAXMg.  It checks the cluster launch
contract that cuSOLVERMp uses before the native backend is involved:

  * one Python process per GPU,
  * a working JAX distributed runtime,
  * a global mesh spanning all ranks,
  * host-to-sharded-device placement over that mesh, and
  * one trivial compiled collective-free operation on the resulting array.

If this script fails, a cuSOLVERMp test would fail before reaching JAXMg.  The
``JAXMG_JAX_SMOKE_LOCAL_DEVICE_MODE`` environment variable selects how the
local device id is passed to ``jax.distributed.initialize``:

  * ``auto``: use 0 when only one CUDA device is visible, otherwise
    ``SLURM_LOCALID``;
  * ``zero``: always use local device id 0;
  * ``slurm``: always use ``SLURM_LOCALID``;
  * ``env``: use ``JAXMG_LOCAL_DEVICE_ID``;
  * ``none``: omit ``local_device_ids`` and let JAX choose.

The mode is intentionally explicit because different Slurm installations expose
GPU visibility differently for one-task-per-GPU launches.
"""

from __future__ import annotations

import json
import os
import socket
import traceback

import numpy as np
from jax import config

config.update("jax_enable_x64", True)


def _emit(label: str, **payload) -> None:
    print(
        "JAXMG_JAX_RANK_GPU_SMOKE "
        + json.dumps({"label": label, "host": socket.gethostname(), **payload}),
        flush=True,
    )


def _int_env(name: str, *, default: int | None = None) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        if default is None:
            raise RuntimeError(f"missing required environment variable {name}")
        return default
    return int(value)


def _visible_device_count() -> int | None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or visible in {"", "-1", "all", "none", "NoDevFiles"}:
        return None
    return len([part for part in visible.split(",") if part.strip()])


def _local_device_ids_for_mode(mode: str, *, local_rank: int) -> list[int] | None:
    visible_count = _visible_device_count()
    if mode == "auto":
        return [0 if visible_count == 1 else local_rank]
    if mode == "zero":
        return [0]
    if mode == "slurm":
        return [local_rank]
    if mode == "env":
        return [_int_env("JAXMG_LOCAL_DEVICE_ID")]
    if mode == "none":
        return None
    raise ValueError(
        "JAXMG_JAX_SMOKE_LOCAL_DEVICE_MODE must be one of "
        "'auto', 'zero', 'slurm', 'env', or 'none'"
    )


def _initialize_jax_distributed():
    import jax

    if jax.distributed.is_initialized():
        return None

    coordinator_address = os.environ.get("JAX_COORDINATOR_ADDRESS")
    if not coordinator_address:
        raise RuntimeError("JAX_COORDINATOR_ADDRESS must be set by the job script")

    process_id = _int_env("SLURM_PROCID")
    num_processes = _int_env("SLURM_NTASKS")
    local_rank = _int_env("SLURM_LOCALID", default=0)
    mode = os.environ.get("JAXMG_JAX_SMOKE_LOCAL_DEVICE_MODE", "auto")
    local_device_ids = _local_device_ids_for_mode(mode, local_rank=local_rank)

    kwargs = dict(
        coordinator_address=coordinator_address,
        num_processes=num_processes,
        process_id=process_id,
        coordinator_bind_address=coordinator_address if process_id == 0 else None,
        initialization_timeout=180,
        heartbeat_timeout_seconds=120,
        shutdown_timeout_seconds=180,
    )
    if local_device_ids is not None:
        kwargs["local_device_ids"] = local_device_ids

    _emit(
        "initialize",
        mode=mode,
        process_id=process_id,
        num_processes=num_processes,
        local_rank=local_rank,
        local_device_ids=local_device_ids,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        slurm_step_gpus=os.environ.get("SLURM_STEP_GPUS"),
    )

    jax.distributed.initialize(**kwargs)
    return local_device_ids


def main() -> None:
    try:
        local_device_ids = _initialize_jax_distributed()

        import jax
        import jax.numpy as jnp
        from jax.sharding import NamedSharding, PartitionSpec as P

        process_count = jax.process_count()
        mesh = jax.make_mesh((process_count,), ("rank",))
        sharding = NamedSharding(mesh, P("rank"))

        _emit(
            "runtime",
            process_index=jax.process_index(),
            process_count=process_count,
            local_device_count=jax.local_device_count(),
            global_device_count=len(jax.devices()),
            local_device_ids=local_device_ids,
            slurm_localid=os.environ.get("SLURM_LOCALID"),
            slurm_procid=os.environ.get("SLURM_PROCID"),
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            slurm_step_gpus=os.environ.get("SLURM_STEP_GPUS"),
            local_devices=[str(device) for device in jax.local_devices()],
            global_devices=[str(device) for device in jax.devices()],
            mesh_devices=[str(device) for device in np.asarray(mesh.devices).ravel()],
        )

        host = np.arange(process_count, dtype=np.int32)
        placed = jax.device_put(host, sharding)
        placed.block_until_ready()

        # This is not meant to benchmark anything. It forces a small compiled
        # operation on the sharded array after distributed placement succeeds.
        out = jax.jit(lambda x: x + np.int32(1))(placed)
        out.block_until_ready()
        local_values = [
            np.asarray(shard.data).reshape(-1).tolist()
            for shard in out.addressable_shards
        ]

        gathered = jax.experimental.multihost_utils.process_allgather(
            jnp.asarray([jax.process_index()], dtype=jnp.int32)
        )

        _emit(
            "success",
            process_index=jax.process_index(),
            local_values=local_values,
            gathered=np.asarray(gathered).reshape(-1).tolist(),
        )
    except Exception:
        _emit("failure", traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
