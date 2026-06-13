"""Two-node validation for arbitrary JAX mesh device order.

The normal cuSOLVERMp helper creates a row-major JAX mesh so its logical
process-grid coordinates match cuSOLVERMp's dense communicator ranks.  Real
JAX users can also construct ``jax.sharding.Mesh`` directly, including meshes
that:

  * use non-contiguous physical/global device ids;
  * list devices in a non-row-major order; and
  * use arbitrary axis names.

This driver exercises that public JAX workflow.  JAXMg should infer the mesh
from ``A.sharding``, translate from the JAX-facing mesh order to the canonical
cuSOLVERMp rank order for the native solve, then translate the result back to
the original JAX mesh order.
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
class ArbitraryMeshCase:
    name: str
    process_rows: int
    process_cols: int
    n: int
    nrhs: int
    tile: int
    dtype: str
    min_global_devices: int
    selector: Callable[[Sequence[object]], list[object]]


def _emit(label: str, **payload) -> None:
    print(
        "JAXMG_MULTINODE_ARBITRARY_MESH "
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
    return [base[0], base[2], base[1], base[3]]


def _select_permuted_1x4_4g(devices: Sequence[object]) -> list[object]:
    ordered = _canonical_devices(devices)
    base = [ordered[0], ordered[1], ordered[2], ordered[3]]
    return [base[2], base[0], base[3], base[1]]


def _select_permuted_4x1_4g(devices: Sequence[object]) -> list[object]:
    ordered = _canonical_devices(devices)
    base = [ordered[0], ordered[1], ordered[2], ordered[3]]
    return [base[3], base[1], base[0], base[2]]


def _select_permuted_2x2_8g(devices: Sequence[object]) -> list[object]:
    ordered = _canonical_devices(devices)
    base = [ordered[0], ordered[1], ordered[4], ordered[5]]
    return [base[0], base[2], base[1], base[3]]


def _select_noncontiguous_2x2(devices: Sequence[object]) -> list[object]:
    ordered = _canonical_devices(devices)
    base = [ordered[0], ordered[1], ordered[3], ordered[4]]
    return [base[0], base[1], base[3], base[2]]


def _select_permuted_1x4_8g(devices: Sequence[object]) -> list[object]:
    ordered = _canonical_devices(devices)
    base = [ordered[0], ordered[2], ordered[4], ordered[6]]
    return [base[2], base[0], base[3], base[1]]


def _select_permuted_4x1_8g(devices: Sequence[object]) -> list[object]:
    ordered = _canonical_devices(devices)
    base = [ordered[1], ordered[3], ordered[5], ordered[7]]
    return [base[3], base[1], base[0], base[2]]


def _structured_spd(n: int, dtype_name: str) -> np.ndarray:
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


def _validate_status(status, *, case: ArbitraryMeshCase) -> None:
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


def _dtype_tolerances(dtype_name: str) -> tuple[float, float]:
    dtype = np.dtype(dtype_name)
    if dtype == np.dtype("float32") or dtype == np.dtype("complex64"):
        return 5e-3, 5e-3
    return 1e-7, 1e-7


def _assert_addressable_shards_close(
    array,
    expected: np.ndarray,
    *,
    case: ArbitraryMeshCase,
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


def _cases(global_device_count: int) -> list[ArbitraryMeshCase]:
    """Return the arbitrary-mesh cases supported by the allocated job.

    The full stress test uses eight global devices so it can cover
    non-contiguous selected device ids as well as permutations.  CSD3 often
    has two-node, two-GPU allocations available sooner, so the first three
    cases are intentionally four-device checks.  They still construct direct
    JAX meshes with non-row-major device order and arbitrary axis names, which
    is the production behavior we need to support.
    """
    cases = [
        ArbitraryMeshCase(
            "permuted_2x2_f64_padded_4g",
            2,
            2,
            20,
            12,
            4,
            "float64",
            4,
            _select_permuted_2x2_4g,
        ),
        ArbitraryMeshCase(
            "permuted_1x4_f32_aligned_4g",
            1,
            4,
            16,
            8,
            4,
            "float32",
            4,
            _select_permuted_1x4_4g,
        ),
        ArbitraryMeshCase(
            "permuted_4x1_f64_aligned_4g",
            4,
            1,
            16,
            4,
            4,
            "float64",
            4,
            _select_permuted_4x1_4g,
        ),
    ]
    cases.extend(
        [
            ArbitraryMeshCase(
                "permuted_2x2_f64_padded_8g",
                2,
                2,
                20,
                12,
                4,
                "float64",
                8,
                _select_permuted_2x2_8g,
            ),
            ArbitraryMeshCase(
                "noncontiguous_2x2_c128_padded_8g",
                2,
                2,
                20,
                8,
                4,
                "complex128",
                8,
                _select_noncontiguous_2x2,
            ),
            ArbitraryMeshCase(
                "permuted_1x4_f32_aligned_8g",
                1,
                4,
                16,
                8,
                4,
                "float32",
                8,
                _select_permuted_1x4_8g,
            ),
            ArbitraryMeshCase(
                "permuted_4x1_f64_aligned_8g",
                4,
                1,
                16,
                4,
                4,
                "float64",
                8,
                _select_permuted_4x1_8g,
            ),
        ]
    )
    return [
        case for case in cases if case.min_global_devices <= global_device_count
    ]


def _run_case(case: ArbitraryMeshCase, devices: Sequence[object]) -> None:
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    selected = case.selector(devices)
    if len(selected) != case.process_rows * case.process_cols:
        raise AssertionError(
            f"{case.name}: selector returned {len(selected)} devices for "
            f"{case.process_rows}x{case.process_cols} grid"
        )
    if len({_device_sort_key(device) for device in selected}) != len(selected):
        raise AssertionError(f"{case.name}: selector returned duplicate devices")

    device_grid = np.asarray(selected, dtype=object).reshape(
        case.process_rows,
        case.process_cols,
    )
    mesh = Mesh(device_grid, ("jax_rows", "jax_cols"))
    sharding = NamedSharding(mesh, P("jax_rows", "jax_cols"))

    a_host = _structured_spd(case.n, case.dtype)
    x_expected = _known_solution(case.n, case.nrhs, case.dtype)
    b_host = (a_host @ x_expected).astype(case.dtype)
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
        dtype=case.dtype,
        selected_devices=_device_payload(selected),
        canonical_order=_device_payload(_canonical_devices(selected)),
    )
    out, status = jaxmg.potrs_mp(a, b, T_A=case.tile, return_status=True)
    _validate_status(status, case=case)
    _assert_addressable_shards_close(out, x_expected, case=case)
    _emit("case_success", name=case.name)


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
        if jax.local_device_count() < 2:
            raise AssertionError(
                f"expected at least 2 local GPUs per process, got "
                f"{jax.local_device_count()}"
            )
        if len(devices) < 4:
            raise AssertionError(f"expected at least 4 global devices, got {len(devices)}")

        selected_cases = _cases(len(devices))
        if not selected_cases:
            raise AssertionError(f"no arbitrary-mesh cases for {len(devices)} devices")
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
