import json
import math
import os
from dataclasses import dataclass

import jax
if not jax.config.jax_enable_x64:
    jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import Mesh


@dataclass(frozen=True)
class SolverCase:
    """Static matrix, mesh, and tile configuration for one MPMD test."""

    process_rows: int
    process_cols: int
    grid_order: str
    n: int
    tile_size: int
    nrhs: int = 1
    rhs_mode: str = "matrix_2d_sharded"


def emit(prefix: str, payload: dict) -> None:
    """Print one parser-friendly JSON event line."""
    print(f"{prefix} {json.dumps(payload, sort_keys=True)}", flush=True)


def dtype_from_name(dtype_name: str):
    """Map pytest dtype parameters to JAX dtypes."""
    mapping = {
        "float32": jnp.float32,
        "float64": jnp.float64,
        "complex64": jnp.complex64,
        "complex128": jnp.complex128,
    }
    return mapping[dtype_name]


def local_device_id_for_process(process_id: int) -> int:
    """Return the local GPU id used by this rank-per-GPU test process."""
    if "JAXMG_LOCAL_DEVICE_ID" in os.environ:
        return int(os.environ["JAXMG_LOCAL_DEVICE_ID"])
    if "SLURM_LOCALID" in os.environ:
        return int(os.environ["SLURM_LOCALID"])
    return int(process_id)


def balanced_process_grid(num_processes: int) -> tuple[int, int]:
    """Choose a near-square process grid for a rank count."""
    for rows in range(int(math.sqrt(num_processes)), 0, -1):
        if num_processes % rows == 0:
            return rows, num_processes // rows
    return 1, num_processes


def _first_valid_n(process_rows: int, process_cols: int, tile: int, *, padded: bool) -> int:
    """Choose a robust square matrix size for a grid/tile combination."""
    divisor = math.lcm(process_rows, process_cols)
    if not padded:
        no_padding_divisor = math.lcm(divisor, tile)
        return max(
            no_padding_divisor,
            math.ceil(512 / no_padding_divisor) * no_padding_divisor,
        )

    for n in range(512, 4097):
        if n % divisor != 0:
            continue
        local_rows = n // process_rows
        local_cols = n // process_cols
        owns_all_process_rows = math.ceil(n / tile) >= process_rows
        owns_all_process_cols = math.ceil(n / tile) >= process_cols
        needs_padding = local_rows % tile != 0 or local_cols % tile != 0
        if owns_all_process_rows and owns_all_process_cols and needs_padding:
            return n
    raise ValueError("could not find a padded test matrix size")


def solver_case(case_name: str, num_processes: int, *, routine: str) -> SolverCase:
    """Build a named test case for POTRS or SYEVD MPMD runners."""
    rhs_mode = "matrix_2d_sharded"
    if case_name == "row_major_no_padding":
        if routine == "potrs":
            rows, cols = 1, num_processes
            nrhs = max(1, cols)
        else:
            rows, cols = 1, num_processes
            nrhs = 1
        tile, padded = 64, False
        grid_order = "row_major"
    elif case_name == "column_major_padding":
        rows, cols = balanced_process_grid(num_processes)
        tile, padded, nrhs = 96, True, max(2, cols)
        grid_order = "column_major"
    elif case_name == "column_grid_padding":
        rows, cols, tile, padded, nrhs = num_processes, 1, 96, True, 1
        grid_order = "row_major"
    elif case_name == "skinny_rhs":
        if routine != "potrs":
            raise ValueError("skinny_rhs is only meaningful for POTRS")
        rows, cols, tile, padded, nrhs = 1, num_processes, 64, False, 1
        grid_order = "row_major"
        rhs_mode = "matrix_row_sharded"
    elif case_name == "vector_rhs_replicated":
        if routine != "potrs":
            raise ValueError("vector_rhs_replicated is only meaningful for POTRS")
        rows, cols, tile, padded, nrhs = 1, num_processes, 64, False, 1
        grid_order = "row_major"
        rhs_mode = "vector_replicated"
    elif case_name == "single_column_rhs_replicated":
        if routine != "potrs":
            raise ValueError(
                "single_column_rhs_replicated is only meaningful for POTRS"
            )
        rows, cols, tile, padded, nrhs = 1, num_processes, 64, False, 1
        grid_order = "row_major"
        rhs_mode = "matrix_replicated"
    elif case_name == "single_column_rhs_row_sharded":
        if routine != "potrs":
            raise ValueError(
                "single_column_rhs_row_sharded is only meaningful for POTRS"
            )
        rows, cols = balanced_process_grid(num_processes)
        tile, padded, nrhs = 64, False, 1
        grid_order = "row_major"
        rhs_mode = "matrix_row_sharded"
    else:
        raise ValueError(f"unknown MPMD solver case {case_name!r}")

    n = _first_valid_n(rows, cols, tile, padded=padded)
    return SolverCase(
        process_rows=rows,
        process_cols=cols,
        grid_order=grid_order,
        n=n,
        tile_size=tile,
        nrhs=nrhs,
        rhs_mode=rhs_mode,
    )


def make_process_mesh(case: SolverCase) -> Mesh:
    """Create a JAX mesh whose device order matches cuSOLVERMp grid mapping."""
    devices = np.asarray(jax.devices(), dtype=object)
    expected = case.process_rows * case.process_cols
    if devices.size != expected:
        raise ValueError(
            f"expected {expected} JAX devices after distributed initialize, "
            f"got {devices.size}"
        )

    if case.grid_order == "row_major":
        process_grid = devices.reshape(case.process_rows, case.process_cols)
    elif case.grid_order == "column_major":
        process_grid = np.empty((case.process_rows, case.process_cols), dtype=object)
        for process_row in range(case.process_rows):
            for process_col in range(case.process_cols):
                process_grid[process_row, process_col] = devices[
                    process_col * case.process_rows + process_row
                ]
    else:
        raise ValueError(f"unknown grid order {case.grid_order!r}")
    return Mesh(process_grid, ("pr", "pc"))


def make_hermitian_positive_definite(n: int, dtype, *, seed: int):
    """Create a deterministic real SPD or complex Hermitian positive matrix."""
    rng = np.random.default_rng(seed)
    real_dtype = (
        np.float32
        if np.dtype(dtype) in (np.dtype(np.float32), np.dtype(np.complex64))
        else np.float64
    )
    x = rng.normal(size=(n, n)).astype(real_dtype)
    if np.issubdtype(np.dtype(dtype), np.complexfloating):
        y = rng.normal(size=(n, n)).astype(real_dtype)
        x = x + 1j * y
    a = x @ x.conj().T
    a += n * np.eye(n, dtype=a.dtype)
    return jnp.asarray(a, dtype=dtype)


def make_rhs(n: int, nrhs: int, dtype):
    """Create deterministic POTRS right-hand side data."""
    values = np.arange(1, n * nrhs + 1, dtype=np.float64).reshape(n, nrhs)
    if np.issubdtype(np.dtype(dtype), np.complexfloating):
        values = values + 0.25j * values
    return jnp.asarray(values, dtype=dtype)


def global_array_to_numpy(value):
    """Materialize a possibly multi-host JAX array as a host NumPy array.

    In rank-per-GPU tests a result can be a global ``jax.Array`` whose shards
    live on devices owned by other Python processes.  Direct ``np.asarray`` is
    only valid when all shards are addressable from the current process.  For
    distributed arrays, gather the full global value through JAX's multihost
    helper so all ranks run the same validation code.
    """
    if isinstance(value, jax.Array) and not value.is_fully_addressable:
        return np.asarray(multihost_utils.process_allgather(value, tiled=True))
    return np.asarray(value)


def native_status_words(status) -> np.ndarray:
    """Return a flattened native status vector from a JAXMg solver result."""
    return global_array_to_numpy(status).reshape(-1)


def assert_close_scaled(actual, expected, *, atol: float = 5e-4, rtol: float = 5e-4):
    """Assert approximate equality with tolerances suitable for GPU solvers."""
    np.testing.assert_allclose(
        global_array_to_numpy(actual),
        np.asarray(expected),
        atol=atol,
        rtol=rtol,
    )
