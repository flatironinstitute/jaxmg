"""Probe ``jax.make_mesh`` layout and solve compatibility on two nodes.

This driver answers a narrow API question: if users build their process grid
with native JAX mesh construction, does the resulting ``mesh.devices`` order
match JAXMg/cuSOLVERMp's row-major process-grid contract?

For grids that consume all eight devices in the two-node test job, the script
calls ``jax.make_mesh(shape, axis_names)`` directly.  For smaller grids, JAX
requires an explicit subset of devices, so the script calls
``jax.make_mesh(shape, axis_names, devices=selected_devices)``.  Both are
native JAX APIs.  No JAXMg mesh helper is used.

Each compatible layout is then used for a small padded ``potrs_mp`` solve.
If JAX chooses a non-row-major mesh order, the script records the layout and
verifies that ``potrs_mp`` rejects it before native redistribution.
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


@dataclass(frozen=True)
class MeshCase:
    process_rows: int
    process_cols: int
    tile: int = 4

    @property
    def name(self) -> str:
        return f"make_mesh_{self.process_rows}x{self.process_cols}"

    @property
    def num_devices(self) -> int:
        return self.process_rows * self.process_cols

    @property
    def n(self) -> int:
        # Divisible by both process axes, but local dimensions are usually not
        # multiples of tile.  That keeps the solve small while exercising the
        # padding path.
        return math.lcm(self.process_rows, self.process_cols) * (self.tile + 1)

    @property
    def nrhs(self) -> int:
        return self.process_cols * (self.tile + 1)


def _emit(label: str, **payload) -> None:
    print(
        "JAXMG_MAKE_MESH_LAYOUT "
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
    """Choose a deterministic row-major subset using native JAX devices.

    For ``count >= process_count`` this picks at least one device from each
    process, then fills remaining slots in process/device order.  The returned
    list is sorted in canonical global rank order so any reordering in
    ``jax.make_mesh`` is visible in the emitted diagnostics.
    """
    if count <= 0:
        raise ValueError("device count must be positive")
    if count > len(devices):
        raise ValueError(f"requested {count} devices from only {len(devices)}")

    by_process: dict[int, list[object]] = {}
    for device in _canonical_devices(devices):
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
    return _canonical_devices(selected)


def _mesh_flat_devices(mesh) -> list[object]:
    return list(np.asarray(mesh.devices, dtype=object).reshape(-1))


def _is_row_major_mesh(mesh, canonical: Sequence[object]) -> bool:
    mesh_keys = [_device_sort_key(device) for device in _mesh_flat_devices(mesh)]
    canonical_keys = [_device_sort_key(device) for device in canonical]
    return mesh_keys == canonical_keys


def _spd_matrix(n: int) -> np.ndarray:
    rng = np.random.default_rng(3100 + n)
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


def _validate_status(status, *, case: MeshCase) -> None:
    rows = _status_rows(status)
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
    for rank, row in rows:
        if int(row[2]) != rank:
            raise AssertionError(f"{case.name}: row rank mismatch {row[2]} != {rank}")
        if int(row[3]) != case.num_devices:
            raise AssertionError(f"{case.name}: rank count mismatch row={row}")
        if int(row[4]) != case.process_rows or int(row[5]) != case.process_cols:
            raise AssertionError(f"{case.name}: process grid mismatch row={row}")
        if int(row[36]) != case.nrhs:
            raise AssertionError(f"{case.name}: nrhs mismatch row={row}")


def _run_case(case: MeshCase, devices: Sequence[object]) -> None:
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P

    if case.num_devices > len(devices):
        _emit("case_skipped", name=case.name, reason="not_enough_devices")
        return

    if case.num_devices == len(devices):
        selected = _canonical_devices(devices)
        mesh = jax.make_mesh((case.process_rows, case.process_cols), ("x", "y"))
        construction = "jax.make_mesh"
    else:
        selected = _select_devices_spanning_processes(devices, case.num_devices)
        mesh = jax.make_mesh(
            (case.process_rows, case.process_cols),
            ("x", "y"),
            devices=np.asarray(selected, dtype=object),
        )
        construction = "jax.make_mesh_devices"

    canonical = _canonical_devices(selected)
    is_row_major = _is_row_major_mesh(mesh, canonical)
    _emit(
        "case_layout",
        name=case.name,
        construction=construction,
        process_rows=case.process_rows,
        process_cols=case.process_cols,
        n=case.n,
        nrhs=case.nrhs,
        tile=case.tile,
        selected_devices=_device_payload(selected),
        mesh_devices=_device_payload(_mesh_flat_devices(mesh)),
        canonical_order=_device_payload(canonical),
        row_major=is_row_major,
    )

    sharding = NamedSharding(mesh, P("x", "y"))
    a_host = _spd_matrix(case.n)
    x_expected = _rhs(case.n, case.nrhs)
    b_host = a_host @ x_expected
    a = jax.device_put(jnp.asarray(a_host), sharding)
    b = jax.device_put(jnp.asarray(b_host), sharding)

    if not is_row_major:
        try:
            jaxmg.potrs_mp(a, b, T_A=case.tile, return_status=True)
        except ValueError as exc:
            if "row-major" not in str(exc):
                raise
            _emit("case_rejected_non_row_major", name=case.name, error=str(exc))
            return
        raise AssertionError(f"{case.name}: potrs_mp accepted non-row-major mesh")

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
    _emit("case_solution_checked", name=case.name, shards=checked)
    _emit("case_success", name=case.name)


def _cases() -> list[MeshCase]:
    return [
        MeshCase(1, 2),
        MeshCase(1, 3),
        MeshCase(1, 4),
        MeshCase(1, 5),
        MeshCase(1, 6),
        MeshCase(1, 7),
        MeshCase(1, 8),
        MeshCase(2, 1),
        MeshCase(3, 1),
        MeshCase(4, 1),
        MeshCase(5, 1),
        MeshCase(6, 1),
        MeshCase(7, 1),
        MeshCase(8, 1),
        MeshCase(2, 2),
        MeshCase(2, 3),
        MeshCase(2, 4),
        MeshCase(3, 2),
        MeshCase(4, 2),
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
