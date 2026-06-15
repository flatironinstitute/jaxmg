"""Subset-mesh cuSOLVERMp validation on a two-node JAX runtime.

This driver is for the end-user workflow where the user creates an ordinary
JAX mesh that may use fewer devices than the full distributed job, shards
``A`` and ``B`` on that mesh, and calls ``jaxmg.potrs(A, B, T_A=...)``.

The expected launch topology is still one Python process per node over two
nodes with four local GPUs each. Individual cases deliberately build smaller
meshes from the eight global devices, e.g. 1x3 or 2x3.  That validates two
things at once:

  * the Python wrapper infers mesh/spec from ``A.sharding``; and
  * the native XLA/NCCL communicator path scopes itself to the devices assigned
    to the FFI operation, not blindly to every GPU in the distributed job.
"""

from __future__ import annotations

import json
import math
import socket
import traceback
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from jax import config

config.update("jax_enable_x64", True)

import jaxmg
from tests.distributed_helpers import initialize_node_process


@dataclass(frozen=True)
class GridCase:
    name: str
    process_rows: int
    process_cols: int
    n: int
    nrhs: int
    tile: int
    dtype: str
    padded: bool


def _emit(label: str, **payload) -> None:
    print(
        "JAXMG_MULTINODE_SUBSET_SWEEP "
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
    """Return a stable local device id for diagnostics and deterministic sorting.

    Some PJRT backends expose ``local_hardware_id`` but leave it as ``None`` for
    global devices.  In that case the global device id is still stable and is
    already the order used by XLA's all-assigned clique ranks, so it is a better
    fallback than failing before the actual subset-mesh case starts.
    """
    value = getattr(device, "local_hardware_id", None)
    if value is None:
        return _device_id(device)
    return int(value)


def _device_sort_key(device) -> tuple[int, int, int]:
    process_index = _device_process_index(device)
    device_id = _device_id(device)
    local_id = _device_local_hardware_id(device)
    return process_index, device_id, local_id


def _select_devices_spanning_processes(
    devices: Sequence[object],
    count: int,
) -> list[object]:
    """Choose ``count`` devices, spanning both node processes when possible.

    XLA's all-assigned clique ranks are ordered by global device id.  We
    therefore return the selected devices in that same global order.  For
    two-node tests with ``count >= 2`` this picks at least one device from each
    process, then fills the remaining slots from lower process ids first.
    """
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


def _validate_status(status, *, case: GridCase) -> None:
    rows = _status_rows(status)
    if not rows:
        raise AssertionError(f"{case.name}: no local status rows")
    _emit(
        "case_status",
        name=case.name,
        rows=[
            {"rank": int(rank), "code": int(row[0]), "raw": row.tolist()}
            for rank, row in rows
        ],
    )
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


def _assert_addressable_shards_close(
    array,
    expected: np.ndarray,
    *,
    case: GridCase,
) -> None:
    array.block_until_ready()
    rtol, atol = (
        (2e-4, 2e-4)
        if np.dtype(case.dtype) == np.dtype("float32")
        else (1e-9, 1e-9)
    )
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


def _spd_matrix(n: int, dtype_name: str) -> np.ndarray:
    dtype = np.dtype(dtype_name)
    rng = np.random.default_rng(1000 + n)
    x = rng.standard_normal((n, n)).astype(dtype)
    a = x @ x.T
    a += np.eye(n, dtype=dtype) * np.array(n * n, dtype=dtype)
    return a.astype(dtype)


def _rhs(n: int, nrhs: int, dtype_name: str) -> np.ndarray:
    return (
        np.arange(1, n * nrhs + 1, dtype=np.float64)
        .reshape(n, nrhs)
        .astype(dtype_name)
    )


def _make_cases() -> list[GridCase]:
    grids = [
        (1, 2),
        (1, 3),
        (1, 4),
        (1, 5),
        (1, 6),
        (1, 7),
        (1, 8),
        (2, 1),
        (3, 1),
        (4, 1),
        (5, 1),
        (6, 1),
        (7, 1),
        (8, 1),
        (2, 2),
        (2, 3),
        (2, 4),
        (3, 2),
        (4, 2),
    ]
    tile = 4
    cases: list[GridCase] = []
    for rows, cols in grids:
        factor = math.lcm(rows, cols)
        aligned_n = factor * tile
        padded_n = factor * (tile + 1)
        cases.append(
            GridCase(
                name=f"grid_{rows}x{cols}_aligned",
                process_rows=rows,
                process_cols=cols,
                n=aligned_n,
                nrhs=cols * tile,
                tile=tile,
                dtype="float64",
                padded=False,
            )
        )
        cases.append(
            GridCase(
                name=f"grid_{rows}x{cols}_padded",
                process_rows=rows,
                process_cols=cols,
                n=padded_n,
                nrhs=cols * (tile + 1),
                tile=tile,
                dtype="float64",
                padded=True,
            )
        )
    return cases


def _run_case(case: GridCase, devices: Sequence[object]) -> None:
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    selected = _select_devices_spanning_processes(
        devices,
        case.process_rows * case.process_cols,
    )
    mesh = Mesh(
        np.asarray(selected, dtype=object).reshape(
            case.process_rows,
            case.process_cols,
        ),
        ("pr", "pc"),
    )
    sharding = NamedSharding(mesh, P("pr", "pc"))

    a_host = _spd_matrix(case.n, case.dtype)
    b_host = _rhs(case.n, case.nrhs, case.dtype)
    a = jax.device_put(jnp.asarray(a_host), sharding)
    b = jax.device_put(jnp.asarray(b_host), sharding)

    _emit(
        "case_start",
        name=case.name,
        process_rows=case.process_rows,
        process_cols=case.process_cols,
        n=case.n,
        nrhs=case.nrhs,
        tile=case.tile,
        padded=case.padded,
        selected_devices=[
            {
                "process_index": _device_process_index(device),
                "id": _device_id(device),
                "local_hardware_id": _device_local_hardware_id(device),
            }
            for device in selected
        ],
    )
    out, status = jaxmg.potrs(a, b, T_A=case.tile, return_status=True)
    expected = np.linalg.solve(a_host, b_host)
    _validate_status(status, case=case)
    _assert_addressable_shards_close(out, expected, case=case)
    _emit("case_success", name=case.name)


def main() -> None:
    try:
        local_ids = initialize_node_process(initialization_timeout=120)

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

        for case in _make_cases():
            _run_case(case, devices)

        _emit("success")
    except Exception:
        _emit("failure", traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
