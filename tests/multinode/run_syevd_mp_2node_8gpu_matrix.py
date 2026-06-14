"""Two-node cuSOLVERMp validation driver for the public syevd_mp wrapper.

This script is launched by a cluster runner with one Python process per GPU.
It expects:

  * 8 JAX processes,
  * 1 local GPU per process,
  * 8 global GPUs total.

The cases cover full 2D process grids and degenerate row-only/column-only
grids. Both cuSOLVERMp SYEVD modes are exercised:

  * ``eigvecs=True`` validates reverse redistribution of eigenvectors back to
    the JAX-facing block-sharded layout.
  * ``eigvecs=False`` validates the eigenvalue-only path, where no reverse
    matrix redistribution is needed.

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
    tile: int
    dtype: str
    eigvecs: bool


def _emit(label: str, **payload) -> None:
    print(
        "JAXMG_MULTINODE_SYEVD_MATRIX "
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
    """Initialize JAX using the standard rank-per-GPU Slurm environment.

    JAXMg deliberately does not own distributed initialization. These tests are
    ordinary user code: the cluster wrapper supplies the coordinator address and
    Slurm rank variables, then this script calls ``jax.distributed.initialize``
    directly before importing JAXMg or creating any arrays.
    """
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
        if data.size % 36:
            raise AssertionError(
                f"status shard length is not a multiple of 36: {data.size}"
            )
        for offset in range(0, data.size, 36):
            rows.append((start // 36 + offset // 36, data[offset : offset + 36]))
    return rows


def _validate_status(status, *, label: str, grid, eigvecs: bool) -> None:
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
        if int(row[20]) != int(eigvecs):
            raise AssertionError(f"{label}: eigvec flag mismatch row={row}")
        if int(row[23]) != 1:
            raise AssertionError(f"{label}: cusolverMpSyevd was not called row={row}")
        if int(row[24]) != 0:
            raise AssertionError(f"{label}: syevd info was non-zero row={row}")


def _hermitian_matrix(n: int, dtype_name: str) -> np.ndarray:
    dtype = np.dtype(dtype_name)
    rng = np.random.default_rng(7100 + n)
    if np.issubdtype(dtype, np.complexfloating):
        real = rng.standard_normal((n, n))
        imag = rng.standard_normal((n, n))
        x = (real + 0.5j * imag).astype(dtype)
        a = x @ x.conj().T
        a += np.eye(n, dtype=dtype) * np.array(n, dtype=dtype)
        return a.astype(dtype)

    x = rng.standard_normal((n, n)).astype(dtype)
    a = x @ x.T
    a += np.eye(n, dtype=dtype) * np.array(n, dtype=dtype)
    return a.astype(dtype)


def _tolerances(dtype_name: str) -> tuple[float, float]:
    dtype = np.dtype(dtype_name)
    if dtype == np.dtype("float32") or dtype == np.dtype("complex64"):
        return 3e-4, 3e-4
    return 1e-9, 1e-9


def _validate_eigenvectors(
    *,
    label: str,
    a_host: np.ndarray,
    eigenvalues,
    eigenvectors,
    rtol: float,
    atol: float,
) -> None:
    eigenvalues.block_until_ready()
    eigenvectors.block_until_ready()
    values = np.asarray(eigenvalues)
    vectors = np.asarray(eigenvectors)
    expected_values, _ = np.linalg.eigh(a_host)

    np.testing.assert_allclose(values, expected_values, rtol=rtol, atol=atol)

    residual = np.linalg.norm(a_host @ vectors - vectors * values[None, :])
    scale = max(1.0, np.linalg.norm(a_host))
    if residual / scale >= max(10 * rtol, 1e-10):
        raise AssertionError(
            f"{label}: eigenvector residual too large {residual / scale}"
        )

    eye = np.eye(a_host.shape[0], dtype=vectors.dtype)
    np.testing.assert_allclose(
        vectors.conj().T @ vectors,
        eye,
        rtol=max(10 * rtol, 1e-6),
        atol=max(10 * atol, 1e-6),
        err_msg=label + " eigenvectors are not orthonormal",
    )


def _run_case(case: Case) -> None:
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P

    import jaxmg
    from jaxmg._block_cyclic_2d_plan import ProcessGrid

    grid = ProcessGrid(case.process_rows, case.process_cols)
    if jax.device_count() != grid.num_processes:
        raise AssertionError(
            f"{case.name}: expected {grid.num_processes} devices, got "
            f"{jax.device_count()}"
        )

    mesh = jax.make_mesh((grid.process_rows, grid.process_cols), ("pr", "pc"))
    sharding = NamedSharding(mesh, P("pr", "pc"))
    a_host = _hermitian_matrix(case.n, case.dtype)
    a = jax.device_put(jnp.asarray(a_host), sharding)
    rtol, atol = _tolerances(case.dtype)

    _emit(
        "case_start",
        name=case.name,
        process_rows=grid.process_rows,
        process_cols=grid.process_cols,
        n=case.n,
        tile=case.tile,
        dtype=case.dtype,
        eigvecs=case.eigvecs,
        process_index=jax.process_index(),
    )

    if case.eigvecs:
        eigenvalues, eigenvectors, status = jaxmg.syevd_mp(
            a,
            T_A=case.tile,
            eigvecs=True,
            return_status=True,
        )
        _validate_status(status, label=case.name, grid=grid, eigvecs=True)
        _validate_eigenvectors(
            label=case.name,
            a_host=a_host,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            rtol=rtol,
            atol=atol,
        )
    else:
        eigenvalues, status = jaxmg.syevd_mp(
            a,
            T_A=case.tile,
            eigvecs=False,
            return_status=True,
        )
        eigenvalues.block_until_ready()
        _validate_status(status, label=case.name, grid=grid, eigvecs=False)
        expected_values = np.linalg.eigvalsh(a_host)
        np.testing.assert_allclose(
            np.asarray(eigenvalues),
            expected_values,
            rtol=rtol,
            atol=atol,
        )

    _emit("case_success", name=case.name)


def main() -> None:
    try:
        local_ids = _initialize_jax_distributed_rank_process()

        import jax

        _emit(
            "runtime",
            local_ids=list(local_ids),
            process_index=jax.process_index(),
            process_count=jax.process_count(),
            local_device_count=jax.local_device_count(),
            global_device_count=jax.device_count(),
        )
        if jax.process_count() != 8:
            raise AssertionError(f"expected 8 JAX processes, got {jax.process_count()}")
        if jax.local_device_count() != 1:
            raise AssertionError(
                f"expected 1 local GPU per process, got {jax.local_device_count()}"
            )
        if jax.device_count() != 8:
            raise AssertionError(f"expected 8 global GPUs, got {jax.device_count()}")

        cases = [
            Case("grid_2x4_f64_vectors_colpad", 2, 4, 24, 4, "float64", True),
            Case("grid_4x2_f32_values_rowpad", 4, 2, 24, 4, "float32", False),
            Case("grid_1x8_c64_vectors_colonly", 1, 8, 32, 8, "complex64", True),
            Case("grid_8x1_c128_values_rowonly", 8, 1, 32, 8, "complex128", False),
        ]
        for case in cases:
            _run_case(case)

        _emit("success")
    except Exception:
        _emit("failure", traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
