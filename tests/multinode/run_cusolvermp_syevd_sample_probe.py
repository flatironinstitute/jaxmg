"""Rank-per-GPU cuSOLVERMp SYEVD sample-layout diagnostic.

This script tests the native ``cusolvermp_syevd_probe`` FFI entrypoint.  The
probe deliberately avoids JAXMg's 2D redistribution: each rank allocates fresh
cuSOLVERMp local buffers, asks cuSOLVERMp to scatter a deterministic host
matrix into its own block-cyclic format, and then calls ``cusolverMpSyevd``.

That makes the diagnostic useful when the production ``syevd`` wrapper
fails.  If this probe fails too, the failure is likely in the cuSOLVERMp SYEVD
call, communicator ownership, stream/device context, or runtime setup.  If this
probe passes while production fails, the likely target is JAXMg's redistributed
input layout or production descriptor/output wiring.
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
        "JAXMG_CUSOLVERMP_SYEVD_SAMPLE "
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


def _initialize_jax_distributed_rank_process() -> tuple[int, ...]:
    import jax

    if jax.distributed.is_initialized():
        return tuple(range(jax.local_device_count()))

    coordinator_address = os.environ.get("JAX_COORDINATOR_ADDRESS")
    if not coordinator_address:
        raise RuntimeError("JAX_COORDINATOR_ADDRESS must be set by the job script")

    process_id = _int_env("SLURM_PROCID")
    num_processes = _int_env("SLURM_NTASKS")
    local_rank = _int_env("SLURM_LOCALID", default=0)
    visible_count = _visible_device_count()
    local_device_id = 0 if visible_count == 1 else local_rank

    jax.distributed.initialize(
        coordinator_address=coordinator_address,
        num_processes=num_processes,
        process_id=process_id,
        local_device_ids=[local_device_id],
        coordinator_bind_address=coordinator_address if process_id == 0 else None,
        initialization_timeout=180,
        heartbeat_timeout_seconds=120,
        shutdown_timeout_seconds=180,
    )
    return (local_device_id,)


def _status_rows(status, *, num_processes: int) -> np.ndarray:
    status.block_until_ready()
    rows = []
    for shard in status.addressable_shards:
        data = np.asarray(shard.data).reshape(-1)
        if data.size % 36:
            raise AssertionError(f"unexpected status length {data.size}")
        rows.extend(data[offset : offset + 36] for offset in range(0, data.size, 36))
    if len(rows) != 1:
        raise AssertionError(f"expected one local status row, got {len(rows)}")
    row = np.asarray(rows[0], dtype=np.int32)
    if int(row[3]) != num_processes:
        raise AssertionError(f"communicator size mismatch in status row {row}")
    return row


def _run_probe(*, use_private_stream: bool) -> None:
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P

    from jaxmg._xla_comm_probe import cusolvermp_syevd_probe_shardmap

    process_count = jax.device_count()
    process_rows = _int_env("JAXMG_SYEVD_PROCESS_ROWS", default=2)
    process_cols = _int_env("JAXMG_SYEVD_PROCESS_COLS", default=process_count // process_rows)
    if process_rows * process_cols != process_count:
        raise AssertionError(
            f"grid {process_rows}x{process_cols} does not match {process_count} ranks"
        )

    n = _int_env("JAXMG_SYEVD_N", default=12)
    tile = _int_env("JAXMG_SYEVD_TILE", default=4)
    dtype_name = os.environ.get("JAXMG_SYEVD_DTYPE", "float64")
    dtype = getattr(jnp, dtype_name)

    mesh = jax.make_mesh((process_rows, process_cols), ("pr", "pc"))
    token = jax.device_put(
        jnp.zeros((max(tile, 1), max(tile, 1)), dtype=dtype),
        NamedSharding(mesh, P("pr", "pc")),
    )

    _emit(
        "case_start",
        process_index=jax.process_index(),
        process_rows=process_rows,
        process_cols=process_cols,
        n=n,
        tile=tile,
        dtype=dtype_name,
        use_private_stream=use_private_stream,
    )
    status = cusolvermp_syevd_probe_shardmap(
        token,
        mesh,
        P("pr", "pc"),
        P(("pr", "pc")),
        process_rows=process_rows,
        process_cols=process_cols,
        n=n,
        tile_size=tile,
        grid_mapping=1,
        use_private_stream=use_private_stream,
    )
    row = _status_rows(status, num_processes=process_count)
    _emit("status", process_index=jax.process_index(), row=row.tolist())
    if int(row[0]) != 0:
        raise AssertionError(f"non-zero SYEVD sample status row {row}")
    _emit("case_success", process_index=jax.process_index())


def main() -> None:
    try:
        local_ids = _initialize_jax_distributed_rank_process()

        import jax

        expected_device_count = _int_env("JAXMG_EXPECTED_DEVICE_COUNT", default=4)
        _emit(
            "runtime",
            local_ids=list(local_ids),
            process_index=jax.process_index(),
            process_count=jax.process_count(),
            local_device_count=jax.local_device_count(),
            global_device_count=jax.device_count(),
            expected_device_count=expected_device_count,
            slurm_localid=os.environ.get("SLURM_LOCALID"),
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            slurm_step_gpus=os.environ.get("SLURM_STEP_GPUS"),
            local_devices=[str(device) for device in jax.local_devices()],
        )
        if jax.process_count() != expected_device_count:
            raise AssertionError(
                f"expected {expected_device_count} JAX processes, got "
                f"{jax.process_count()}"
            )
        if jax.local_device_count() != 1:
            raise AssertionError(
                f"expected 1 local GPU per process, got {jax.local_device_count()}"
            )

        private_stream = os.environ.get("JAXMG_SYEVD_PRIVATE_STREAM", "0") in {
            "1",
            "true",
            "yes",
        }

        _run_probe(use_private_stream=private_stream)

        _emit("success")
    except Exception:
        _emit("failure", traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
