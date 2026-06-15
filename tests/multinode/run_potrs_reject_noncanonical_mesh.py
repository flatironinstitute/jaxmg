"""Validate that ``potrs`` rejects exotic JAX mesh order.

Users should build cuSOLVERMp inputs with ordinary JAX sharding APIs, for
example ``jax.make_mesh`` or ``jax.sharding.Mesh``.  JAXMg intentionally keeps
the production solver contract narrower than arbitrary JAX mesh permutations:
the actual mesh device array must enumerate the process grid in row-major or
column-major communicator order, matching cuSOLVERMp's grid mapping enum.  This
driver checks that other permutations fail before native redistribution or the
cuSOLVERMp solve is launched.
"""

from __future__ import annotations

import json
import socket
import traceback
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from jax import config

config.update("jax_enable_x64", True)

import jaxmg


@dataclass(frozen=True)
class NonCanonicalMeshCase:
    name: str
    process_rows: int
    process_cols: int
    min_global_devices: int
    selector: Callable[[Sequence[object]], list[object]]


def _emit(label: str, **payload) -> None:
    print(
        "JAXMG_REJECT_NONCANONICAL_MESH "
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


def _select_permuted_2x2_4g(devices: Sequence[object]) -> list[object]:
    ordered = _canonical_devices(devices)
    base = [ordered[0], ordered[1], ordered[2], ordered[3]]
    return [base[0], base[3], base[1], base[2]]


def _select_permuted_1x4_4g(devices: Sequence[object]) -> list[object]:
    ordered = _canonical_devices(devices)
    base = [ordered[0], ordered[1], ordered[2], ordered[3]]
    return [base[2], base[0], base[3], base[1]]


def _select_permuted_across_nodes_2x2_8g(
    devices: Sequence[object],
) -> list[object]:
    ordered = _canonical_devices(devices)
    base = [ordered[0], ordered[1], ordered[4], ordered[5]]
    return [base[0], base[3], base[1], base[2]]


def _cases(global_device_count: int) -> list[NonCanonicalMeshCase]:
    cases = [
        NonCanonicalMeshCase(
            "permuted_2x2_4g",
            2,
            2,
            4,
            _select_permuted_2x2_4g,
        ),
        NonCanonicalMeshCase(
            "permuted_1x4_4g",
            1,
            4,
            4,
            _select_permuted_1x4_4g,
        ),
        NonCanonicalMeshCase(
            "permuted_across_nodes_2x2_8g",
            2,
            2,
            8,
            _select_permuted_across_nodes_2x2_8g,
        ),
    ]
    return [
        case for case in cases if case.min_global_devices <= global_device_count
    ]


def _run_case(case: NonCanonicalMeshCase, devices: Sequence[object]) -> None:
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    selected = case.selector(devices)
    device_grid = np.asarray(selected, dtype=object).reshape(
        case.process_rows,
        case.process_cols,
    )
    mesh = Mesh(device_grid, ("rows", "cols"))
    sharding = NamedSharding(mesh, P("rows", "cols"))

    # The test is about rejecting an exotic device order inside ``potrs``.
    # Keep the array shapes divisible by the mesh axes so JAX can place the
    # buffers and the failure is raised by JAXMg rather than by sharding setup.
    nrhs = max(2, case.process_cols)
    a = jax.device_put(jnp.eye(8, dtype=jnp.float64), sharding)
    b = jax.device_put(jnp.ones((8, nrhs), dtype=jnp.float64), sharding)

    _emit(
        "case_start",
        name=case.name,
        selected_devices=_device_payload(selected),
        canonical_order=_device_payload(_canonical_devices(selected)),
    )
    try:
        jaxmg.potrs(a, b, T_A=4)
    except ValueError as exc:
        if "row-major or column-major" not in str(exc):
            raise
        _emit("case_success", name=case.name, error=str(exc))
        return
    raise AssertionError(f"{case.name}: potrs accepted a noncanonical mesh")


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
        if len(devices) < 4:
            raise AssertionError(f"expected at least 4 global devices, got {len(devices)}")

        selected_cases = _cases(len(devices))
        if not selected_cases:
            raise AssertionError(f"no rejection cases for {len(devices)} devices")
        _emit(
            "selected_cases",
            global_device_count=len(devices),
            names=[case.name for case in selected_cases],
        )
        for case in selected_cases:
            _run_case(case, devices)

        _emit("success")
    except Exception:
        _emit("failure", traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
