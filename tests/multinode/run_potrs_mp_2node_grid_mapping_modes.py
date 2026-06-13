"""Run ``potrs_mp`` on explicit row-major and column-major JAX meshes.

JAXMg accepts exactly the two process-grid mappings that cuSOLVERMp can encode:

* row-major:    rank = process_row * process_cols + process_col
* column-major: rank = process_col * process_rows + process_row

This two-node driver constructs both layouts with ordinary
``jax.sharding.Mesh`` objects and checks that the full redistribution,
cuSOLVERMp solve, and reverse redistribution all agree with the dense NumPy
reference solution.
"""

from __future__ import annotations

import json
import math
import socket
import traceback
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from jax import config

config.update("jax_enable_x64", True)

import jaxmg


GridMapping = Literal["row_major", "column_major"]


@dataclass(frozen=True)
class MappingCase:
    process_rows: int
    process_cols: int
    grid_mapping: GridMapping
    tile: int = 4

    @property
    def name(self) -> str:
        return f"{self.grid_mapping}_{self.process_rows}x{self.process_cols}"

    @property
    def num_devices(self) -> int:
        return self.process_rows * self.process_cols

    @property
    def n(self) -> int:
        return math.lcm(self.process_rows, self.process_cols) * (self.tile + 1)

    @property
    def nrhs(self) -> int:
        return self.process_cols * (self.tile + 1)


def _emit(label: str, **payload) -> None:
    print(
        "JAXMG_GRID_MAPPING_MODES "
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


def _device_payload(devices: Sequence[object]) -> list[dict[str, int]]:
    return [
        {
            "process_index": _device_process_index(device),
            "id": _device_id(device),
            "local_hardware_id": _device_local_hardware_id(device),
        }
        for device in devices
    ]


def _canonical_devices(devices: Sequence[object]) -> list[object]:
    return sorted(devices, key=_device_sort_key)


def _select_devices_spanning_processes(
    devices: Sequence[object],
    count: int,
) -> list[object]:
    ordered = _canonical_devices(devices)
    by_process: dict[int, list[object]] = {}
    for device in ordered:
        by_process.setdefault(_device_process_index(device), []).append(device)
    process_ids = sorted(by_process)

    selected: list[object] = []
    if count >= len(process_ids):
        selected.extend(by_process[process_id][0] for process_id in process_ids)

    for device in ordered:
        if len(selected) == count:
            break
        if device not in selected:
            selected.append(device)

    if len(selected) != count:
        raise RuntimeError(f"selected {len(selected)} devices, expected {count}")
    return _canonical_devices(selected)


def _mesh_grid(case: MappingCase, devices: Sequence[object]) -> np.ndarray:
    canonical = _select_devices_spanning_processes(devices, case.num_devices)
    if case.grid_mapping == "row_major":
        flat = canonical
    else:
        flat = [
            canonical[process_col * case.process_rows + process_row]
            for process_row in range(case.process_rows)
            for process_col in range(case.process_cols)
        ]
    return np.asarray(flat, dtype=object).reshape(
        case.process_rows,
        case.process_cols,
    )


def _spd_matrix(n: int) -> np.ndarray:
    rng = np.random.default_rng(4100 + n)
    x = rng.standard_normal((n, n))
    a = x @ x.T
    a += np.eye(n) * float(n * n)
    return a.astype(np.float64)


def _rhs(n: int, nrhs: int) -> np.ndarray:
    return np.linspace(-1.0, 1.0, n * nrhs, dtype=np.float64).reshape(n, nrhs)


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


def _validate_status(status, *, case: MappingCase) -> None:
    rows = _status_rows(status)
    _emit(
        "case_status",
        name=case.name,
        rows=[
            {"rank": int(rank), "code": int(row[0]), "raw": row.tolist()}
            for rank, row in rows
        ],
    )
    expected_mapping = 1 if case.grid_mapping == "row_major" else 0
    status_codes = {int(row[0]) for _, row in rows}
    if status_codes != {0}:
        raise AssertionError(f"{case.name}: non-zero status codes {status_codes}")
    for rank, row in rows:
        if int(row[2]) != rank:
            raise AssertionError(f"{case.name}: row rank mismatch {row[2]} != {rank}")
        if int(row[3]) != case.num_devices:
            raise AssertionError(f"{case.name}: rank count mismatch row={row}")
        if int(row[4]) != case.process_rows or int(row[5]) != case.process_cols:
            raise AssertionError(f"{case.name}: process grid mismatch row={row}")
        if int(row[36]) != case.nrhs:
            raise AssertionError(f"{case.name}: nrhs mismatch row={row}")
        if int(row[39]) != expected_mapping:
            raise AssertionError(f"{case.name}: grid mapping mismatch row={row}")


def _run_case(case: MappingCase, devices: Sequence[object]) -> None:
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    device_grid = _mesh_grid(case, devices)
    mesh = Mesh(device_grid, ("x", "y"))
    sharding = NamedSharding(mesh, P("x", "y"))
    a_host = _spd_matrix(case.n)
    x_expected = _rhs(case.n, case.nrhs)
    b_host = a_host @ x_expected
    a = jax.device_put(jnp.asarray(a_host), sharding)
    b = jax.device_put(jnp.asarray(b_host), sharding)

    _emit(
        "case_start",
        name=case.name,
        process_rows=case.process_rows,
        process_cols=case.process_cols,
        grid_mapping=case.grid_mapping,
        n=case.n,
        nrhs=case.nrhs,
        tile=case.tile,
        mesh_devices=_device_payload(list(device_grid.reshape(-1))),
    )
    out, status = jaxmg.potrs_mp(a, b, T_A=case.tile, return_status=True)
    _validate_status(status, case=case)
    out.block_until_ready()
    checked = 0
    for shard in out.addressable_shards:
        np.testing.assert_allclose(
            np.asarray(shard.data),
            x_expected[shard.index],
            rtol=1e-9,
            atol=1e-9,
            err_msg=f"{case.name} shard {shard.index}",
        )
        checked += 1
    _emit("case_success", name=case.name, shards=checked)


def _cases() -> list[MappingCase]:
    shapes = [(2, 2), (2, 3), (3, 2), (2, 4), (4, 2)]
    return [
        MappingCase(rows, cols, mapping)
        for rows, cols in shapes
        for mapping in ("row_major", "column_major")
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
