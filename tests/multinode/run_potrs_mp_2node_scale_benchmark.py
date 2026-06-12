"""Two-node scale validation and timing driver for ``jaxmg.potrs_mp``.

The small multi-node drivers validate many layouts with tiny matrices.  This
driver is the next checkpoint: it keeps the same public API path but moves to
larger matrices and records warm end-to-end solve times.  It is deliberately a
cluster-launched script, not a pytest test, because it requires:

  * one Python process per node,
  * four visible GPUs per process,
  * two nodes/eight global GPUs total, and
  * the cuSOLVERMp/NVHPC runtime libraries on ``LD_LIBRARY_PATH``.

Each case constructs a dense Hermitian/symmetric positive-definite matrix on
the host, chooses a known solution ``X``, forms ``B = A @ X``, shards ``A`` and
``B`` on an ordinary JAX mesh, and calls ``jaxmg.potrs_mp(A, B, T_A=...)``.
The timed region starts after host-to-device placement and ends after the
native solve and reverse redistribution have completed.  Correctness is checked
against the known ``X`` outside the timed region.
"""

from __future__ import annotations

import json
import socket
import statistics
import time
import traceback
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from jax import config

config.update("jax_enable_x64", True)

import jaxmg


@dataclass(frozen=True)
class ScaleCase:
    name: str
    process_rows: int
    process_cols: int
    n: int
    nrhs: int
    tile: int
    dtype: str
    warmups: int = 1
    repeats: int = 2


def _emit(label: str, **payload) -> None:
    print(
        "JAXMG_MULTINODE_SCALE "
        + json.dumps({"label": label, "host": socket.gethostname(), **payload}),
        flush=True,
    )


def _device_process_index(device) -> int:
    value = getattr(device, "process_index", None)
    if callable(value):
        return int(value())
    if value is None:
        raise AttributeError(f"device {device!r} has no process_index")
    return int(value)


def _device_id(device) -> int:
    value = getattr(device, "id", None)
    if value is None:
        raise AttributeError(f"device {device!r} has no id")
    return int(value)


def _device_local_hardware_id(device) -> int:
    value = getattr(device, "local_hardware_id", None)
    if value is None:
        return _device_id(device)
    return int(value)


def _device_sort_key(device) -> tuple[int, int, int]:
    return (
        _device_process_index(device),
        _device_id(device),
        _device_local_hardware_id(device),
    )


def _select_devices_spanning_processes(
    devices: Sequence[object],
    count: int,
) -> list[object]:
    """Select devices in XLA global-device order, spanning both nodes if used."""
    if count <= 0:
        raise ValueError("device count must be positive")
    if count > len(devices):
        raise ValueError(f"requested {count} devices from only {len(devices)}")

    by_process: dict[int, list[object]] = {}
    for device in sorted(devices, key=_device_sort_key):
        by_process.setdefault(_device_process_index(device), []).append(device)
    process_ids = sorted(by_process)

    selected: list[object] = []
    if count >= len(process_ids):
        for process_id in process_ids:
            selected.append(by_process[process_id][0])

    cursor = {
        process_id: 1 if by_process[process_id][0] in selected else 0
        for process_id in process_ids
    }
    for process_id in process_ids:
        while (
            len(selected) < count
            and cursor[process_id] < len(by_process[process_id])
        ):
            selected.append(by_process[process_id][cursor[process_id]])
            cursor[process_id] += 1

    if len(selected) != count:
        raise RuntimeError(f"selected {len(selected)} devices, expected {count}")
    return sorted(selected, key=_device_sort_key)


def _status_rows(status):
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


def _validate_status(status, *, case: ScaleCase) -> None:
    rows = _status_rows(status)
    if not rows:
        raise AssertionError(f"{case.name}: no local status rows")
    status_codes = {int(row[0]) for _, row in rows}
    if status_codes != {0}:
        raise AssertionError(f"{case.name}: non-zero status codes {status_codes}")

    expected_ranks = case.process_rows * case.process_cols
    for rank, row in rows:
        if int(row[2]) != rank:
            raise AssertionError(f"{case.name}: row rank mismatch {row[2]} != {rank}")
        if int(row[3]) != expected_ranks:
            raise AssertionError(f"{case.name}: process count mismatch row={row}")
        if int(row[4]) != case.process_rows or int(row[5]) != case.process_cols:
            raise AssertionError(f"{case.name}: process grid mismatch row={row}")
        if int(row[36]) != case.nrhs:
            raise AssertionError(f"{case.name}: nrhs mismatch row={row}")


def _dtype_tolerances(dtype_name: str) -> tuple[float, float]:
    dtype = np.dtype(dtype_name)
    if dtype == np.dtype("float32") or dtype == np.dtype("complex64"):
        return 5e-3, 5e-3
    return 1e-7, 1e-7


def _structured_spd(n: int, dtype_name: str) -> np.ndarray:
    """Build a dense, well-conditioned SPD/Hermitian matrix in O(n^2) work."""
    dtype = np.dtype(dtype_name)
    real_dtype = np.float32 if dtype.itemsize <= 8 else np.float64
    x = np.linspace(0.0, 2.0 * np.pi, n, dtype=real_dtype)
    diag = np.linspace(float(n), 2.0 * float(n), n, dtype=real_dtype)

    if np.issubdtype(dtype, np.complexfloating):
        u = (np.sin(x) + 0.25j * np.cos(3.0 * x)).astype(dtype)
        a = (0.02 * np.outer(u, np.conjugate(u))).astype(dtype)
    else:
        u = np.sin(x).astype(dtype)
        a = (0.02 * np.outer(u, u)).astype(dtype)

    a[np.diag_indices(n)] += diag.astype(dtype)
    return a.astype(dtype)


def _known_solution(n: int, nrhs: int, dtype_name: str) -> np.ndarray:
    dtype = np.dtype(dtype_name)
    values = np.linspace(-0.5, 0.5, n * nrhs, dtype=np.float64).reshape(n, nrhs)
    if np.issubdtype(dtype, np.complexfloating):
        values = values + 0.125j * values[::-1]
    return values.astype(dtype)


def _assert_addressable_shards_close(
    array,
    expected: np.ndarray,
    *,
    case: ScaleCase,
) -> None:
    array.block_until_ready()
    rtol, atol = _dtype_tolerances(case.dtype)
    checked = 0
    for shard in array.addressable_shards:
        np.testing.assert_allclose(
            np.asarray(shard.data),
            expected[shard.index],
            rtol=rtol,
            atol=atol,
            err_msg=f"{case.name} shard {shard.index}",
        )
        checked += 1
    _emit("case_solution_checked", name=case.name, shards=checked)


def _run_case(case: ScaleCase, devices: Sequence[object]) -> None:
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P

    device_count = case.process_rows * case.process_cols
    selected = _select_devices_spanning_processes(devices, device_count)
    mesh = jaxmg.make_cusolvermp_mesh(
        case.process_rows,
        case.process_cols,
        devices=selected,
    )
    sharding = NamedSharding(mesh, P("pr", "pc"))

    _emit(
        "case_prepare_start",
        name=case.name,
        process_rows=case.process_rows,
        process_cols=case.process_cols,
        n=case.n,
        nrhs=case.nrhs,
        tile=case.tile,
        dtype=case.dtype,
        warmups=case.warmups,
        repeats=case.repeats,
        selected_devices=[
            {
                "process_index": _device_process_index(device),
                "id": _device_id(device),
                "local_hardware_id": _device_local_hardware_id(device),
            }
            for device in selected
        ],
    )
    prepare_start = time.perf_counter()
    a_host = _structured_spd(case.n, case.dtype)
    x_expected = _known_solution(case.n, case.nrhs, case.dtype)
    b_host = (a_host @ x_expected).astype(case.dtype)
    prepare_seconds = time.perf_counter() - prepare_start
    _emit("case_prepare_done", name=case.name, seconds=prepare_seconds)

    timings: list[float] = []
    last_out = None
    last_status = None
    for iteration in range(case.warmups + case.repeats):
        phase = "warmup" if iteration < case.warmups else "timed"
        a = jax.device_put(jnp.asarray(a_host), sharding)
        b = jax.device_put(jnp.asarray(b_host), sharding)
        a.block_until_ready()
        b.block_until_ready()

        start = time.perf_counter()
        out, status = jaxmg.potrs_mp(a, b, T_A=case.tile, return_status=True)
        out.block_until_ready()
        status.block_until_ready()
        seconds = time.perf_counter() - start

        _validate_status(status, case=case)
        _emit(
            "case_iteration",
            name=case.name,
            phase=phase,
            iteration=iteration,
            seconds=seconds,
        )
        if phase == "timed":
            timings.append(seconds)
        last_out = out
        last_status = status

    if last_out is None or last_status is None:
        raise AssertionError(f"{case.name}: no solve iterations ran")
    _assert_addressable_shards_close(last_out, x_expected, case=case)

    _emit(
        "case_success",
        name=case.name,
        timed_seconds=timings,
        median_seconds=statistics.median(timings),
        min_seconds=min(timings),
        max_seconds=max(timings),
    )


def _cases() -> list[ScaleCase]:
    return [
        ScaleCase("full_2x4_f64_2048_t128", 2, 4, 2048, 128, 128, "float64"),
        ScaleCase("full_4x2_f64_2304_t128_padded", 4, 2, 2304, 128, 128, "float64"),
        ScaleCase("full_1x8_f32_4096_t128", 1, 8, 4096, 128, 128, "float32"),
        ScaleCase("full_8x1_c64_3072_t128", 8, 1, 3072, 64, 128, "complex64"),
        ScaleCase("full_2x4_c128_1536_t128", 2, 4, 1536, 64, 128, "complex128"),
        ScaleCase("subset_2x3_f64_1536_t128", 2, 3, 1536, 96, 128, "float64"),
        ScaleCase("subset_3x2_f64_2112_t128_padded", 3, 2, 2112, 96, 128, "float64"),
    ]


def main() -> None:
    try:
        local_ids = jaxmg.initialize_node_process(initialization_timeout=120)

        import jax

        devices = list(jax.devices())
        _emit(
            "runtime",
            local_ids=list(local_ids),
            process_index=jax.process_index(),
            process_count=jax.process_count(),
            local_device_count=jax.local_device_count(),
            global_device_count=jax.device_count(),
        )
        if jax.process_count() != 2:
            raise AssertionError(f"expected 2 JAX processes, got {jax.process_count()}")
        if jax.local_device_count() != 4:
            raise AssertionError(
                f"expected 4 local GPUs per process, got {jax.local_device_count()}"
            )
        if len(devices) != 8:
            raise AssertionError(f"expected 8 global devices, got {len(devices)}")

        for case in _cases():
            _run_case(case, devices)

        _emit("success")
    except Exception:
        _emit("failure", traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
