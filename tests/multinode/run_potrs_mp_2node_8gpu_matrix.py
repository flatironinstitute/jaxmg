"""Two-node cuSOLVERMp validation driver for the public potrs_mp wrapper.

This script is intentionally not a pytest test: it must be launched by a
cluster runner that starts one Python process per node and initializes JAX's
distributed runtime before device discovery. The expected topology is:

  * 2 JAX processes,
  * 4 local GPUs per process,
  * 8 global GPUs total.

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
import socket
import traceback
from dataclasses import dataclass

import numpy as np
from jax import config

config.update("jax_enable_x64", True)

import jaxmg


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

    from jaxmg._block_cyclic_2d_plan import ProcessGrid

    grid = ProcessGrid(case.process_rows, case.process_cols)
    if jax.device_count() != grid.num_processes:
        raise AssertionError(
            f"{case.name}: expected {grid.num_processes} devices, got "
            f"{jax.device_count()}"
        )

    mesh = jaxmg.make_cusolvermp_mesh(grid.process_rows, grid.process_cols)
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

    out, status = jaxmg.potrs_mp(
        a,
        b,
        T_A=case.tile,
        mesh=mesh,
        matrix_specs=matrix_specs,
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


def main() -> None:
    try:
        local_ids = jaxmg.initialize_node_process(initialization_timeout=120)

        import jax

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
        if jax.device_count() != 8:
            raise AssertionError(f"expected 8 global GPUs, got {jax.device_count()}")

        cases = [
            Case("grid_2x4_f64_colpad", 2, 4, 24, 8, 4, "float64"),
            Case("grid_4x2_f32_rowpad", 4, 2, 24, 8, 4, "float32"),
            Case("grid_1x8_f32_colonly", 1, 8, 32, 8, 8, "float32"),
            Case("grid_8x1_c64_rowonly", 8, 1, 32, 8, 8, "complex64"),
            Case("grid_2x4_c128_bothpad", 2, 4, 32, 8, 8, "complex128"),
        ]
        for case in cases:
            _run_case(case)

        for repeat in range(3):
            _run_case(Case(f"repeat_2x4_f64_{repeat}", 2, 4, 24, 8, 4, "float64"))

        _emit("success")
    except Exception:
        _emit("failure", traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
