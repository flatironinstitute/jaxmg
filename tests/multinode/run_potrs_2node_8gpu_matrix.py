"""Two-node cuSOLVERMp validation driver for the public potrs wrapper.

This script is intentionally not a pytest test: it must be launched by a
cluster runner that starts one Python process per GPU and initializes JAX's
distributed runtime before device discovery. The expected topology is:

  * 4 or 8 JAX processes,
  * 1 local GPU per process,
  * one global GPU per process.

Within that fixed topology the script exercises several cuSOLVERMp process
grids. This matters because different grids stress different pieces of the
native redistribution path:

  * 2x4 and 4x2 validate non-degenerate row and column movement,
  * 1x8 validates the column-only degenerate grid,
  * 8x1 validates the row-only degenerate grid,
  * repeated 2x4 solves catch basic donated-buffer and communicator lifetime
    regressions without introducing more JIT compilation shapes.

The Slurm/PBS wrapper that sets paths and library variables should live outside
the package repository. This file should remain cluster-agnostic.
"""

from __future__ import annotations

import json
import os
import socket
import traceback
from dataclasses import dataclass

import numpy as np
from jax import config

config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class Case:
    name: str
    process_rows: int
    process_cols: int
    n: int
    nrhs: int
    tile: int
    dtype: str


def _emit(label: str, **payload) -> None:
    print(
        "JAXMG_MULTINODE_POTRS_MATRIX "
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
    """Initialize JAX using the standard rank-per-GPU Slurm environment."""
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


def _local_status_rows(status):
    status.block_until_ready()
    rows = []
    for shard in status.addressable_shards:
        index = shard.index[0]
        if not isinstance(index, slice):
            raise TypeError(f"unexpected status shard index {shard.index!r}")
        start = 0 if index.start is None else index.start
        data = np.asarray(shard.data).reshape(-1)
        if data.size % 40:
            raise AssertionError(
                f"status shard length is not a multiple of 40: {data.size}"
            )
        for offset in range(0, data.size, 40):
            rows.append((start // 40 + offset // 40, data[offset : offset + 40]))
    return rows


def _validate_status(status, *, label: str, grid, nrhs: int) -> None:
    rows = _local_status_rows(status)
    if not rows:
        raise AssertionError(f"{label}: no local status rows")

    _emit(
        label + "_status",
        rows=[
            {"rank": int(rank), "code": int(row[0]), "raw": row.tolist()}
            for rank, row in rows
        ],
    )

    status_codes = {int(row[0]) for _, row in rows}
    if status_codes != {0}:
        raise AssertionError(f"{label}: non-zero status codes {status_codes}")
    for rank, row in rows:
        if int(row[2]) != rank:
            raise AssertionError(f"{label}: row rank mismatch {row[2]} != {rank}")
        if int(row[3]) != grid.num_processes:
            raise AssertionError(f"{label}: process count mismatch row={row}")
        if int(row[4]) != grid.process_rows or int(row[5]) != grid.process_cols:
            raise AssertionError(f"{label}: process grid mismatch row={row}")
        if int(row[36]) != nrhs:
            raise AssertionError(f"{label}: nrhs mismatch row={row}")


def _assert_addressable_shards_close(
    array, expected: np.ndarray, *, label: str, rtol: float, atol: float
) -> None:
    array.block_until_ready()
    checked = 0
    for shard in array.addressable_shards:
        np.testing.assert_allclose(
            np.asarray(shard.data),
            expected[shard.index],
            rtol=rtol,
            atol=atol,
            err_msg=f"{label} shard {shard.index}",
        )
        checked += 1
    _emit(label + "_checked", shards=checked)


def _spd_matrix(n: int, dtype_name: str) -> np.ndarray:
    dtype = np.dtype(dtype_name)
    rng = np.random.default_rng(1234 + n)
    if np.issubdtype(dtype, np.complexfloating):
        real = rng.standard_normal((n, n))
        imag = rng.standard_normal((n, n))
        x = (real + 1j * imag).astype(dtype)
        a = x @ x.conj().T
        a += np.eye(n, dtype=dtype) * np.array(n * n, dtype=dtype)
        return a.astype(dtype)

    x = rng.standard_normal((n, n)).astype(dtype)
    a = x @ x.T
    a += np.eye(n, dtype=dtype) * np.array(n * n, dtype=dtype)
    return a.astype(dtype)


def _rhs(n: int, nrhs: int, dtype_name: str) -> np.ndarray:
    dtype = np.dtype(dtype_name)
    values = np.arange(1, n * nrhs + 1, dtype=np.float64).reshape(n, nrhs)
    if np.issubdtype(dtype, np.complexfloating):
        values = values + 0.25j * values[::-1]
    return values.astype(dtype)


def _tolerances(dtype_name: str) -> tuple[float, float]:
    dtype = np.dtype(dtype_name)
    if dtype == np.dtype("float32") or dtype == np.dtype("complex64"):
        return 2e-4, 2e-4
    return 1e-9, 1e-9


def _run_case(case: Case) -> None:
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P

    import jaxmg
    from tests.reference_block_cyclic_2d_plan import ProcessGrid

    grid = ProcessGrid(case.process_rows, case.process_cols)
    if jax.device_count() != grid.num_processes:
        raise AssertionError(
            f"{case.name}: expected {grid.num_processes} devices, got "
            f"{jax.device_count()}"
        )

    mesh = jax.make_mesh((grid.process_rows, grid.process_cols), ("pr", "pc"))
    matrix_specs = P("pr", "pc")
    rtol, atol = _tolerances(case.dtype)

    a_host = _spd_matrix(case.n, case.dtype)
    b_host = _rhs(case.n, case.nrhs, case.dtype)
    sharding = NamedSharding(mesh, matrix_specs)
    a = jax.device_put(jnp.asarray(a_host), sharding)
    b = jax.device_put(jnp.asarray(b_host), sharding)

    _emit(
        "case_start",
        name=case.name,
        process_rows=grid.process_rows,
        process_cols=grid.process_cols,
        n=case.n,
        nrhs=case.nrhs,
        tile=case.tile,
        dtype=case.dtype,
        process_index=jax.process_index(),
    )

    out, status = jaxmg.potrs(
        a,
        b,
        T_A=case.tile,
        return_status=True,
    )
    expected = np.linalg.solve(a_host, b_host)
    _validate_status(status, label=case.name, grid=grid, nrhs=case.nrhs)
    _assert_addressable_shards_close(
        out,
        expected,
        label=case.name + "_solution",
        rtol=rtol,
        atol=atol,
    )
    _emit("case_success", name=case.name)


def _cases_for_device_count(device_count: int) -> list[Case]:
    if device_count == 4:
        return [
            Case("grid_2x2_f64_colpad", 2, 2, 12, 4, 4, "float64"),
            Case("grid_1x4_f32_colonly", 1, 4, 16, 4, 4, "float32"),
            Case("grid_4x1_c64_rowonly", 4, 1, 16, 4, 4, "complex64"),
            Case("grid_2x2_c128_bothpad", 2, 2, 16, 4, 4, "complex128"),
        ]
    if device_count == 8:
        return [
            Case("grid_2x4_f64_colpad", 2, 4, 24, 8, 4, "float64"),
            Case("grid_4x2_f32_rowpad", 4, 2, 24, 8, 4, "float32"),
            Case("grid_1x8_f32_colonly", 1, 8, 32, 8, 8, "float32"),
            Case("grid_8x1_c64_rowonly", 8, 1, 32, 8, 8, "complex64"),
            Case("grid_2x4_c128_bothpad", 2, 4, 32, 8, 8, "complex128"),
        ]
    raise AssertionError(
        "rank-per-GPU potrs matrix validation currently supports 4 or 8 "
        f"global devices, got {device_count}."
    )


def main() -> None:
    try:
        local_ids = _initialize_jax_distributed_rank_process()

        import jax

        expected_device_count = _int_env("JAXMG_EXPECTED_DEVICE_COUNT", default=8)
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
        if jax.device_count() != expected_device_count:
            raise AssertionError(
                f"expected {expected_device_count} global GPUs, got "
                f"{jax.device_count()}"
            )

        for case in _cases_for_device_count(expected_device_count):
            _run_case(case)

        repeat_case = (
            Case("repeat_2x2_f64", 2, 2, 12, 4, 4, "float64")
            if expected_device_count == 4
            else Case("repeat_2x4_f64", 2, 4, 24, 8, 4, "float64")
        )
        for repeat in range(3):
            _run_case(
                Case(
                    f"{repeat_case.name}_{repeat}",
                    repeat_case.process_rows,
                    repeat_case.process_cols,
                    repeat_case.n,
                    repeat_case.nrhs,
                    repeat_case.tile,
                    repeat_case.dtype,
                )
            )

        _emit("success")
    except Exception:
        _emit("failure", traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
