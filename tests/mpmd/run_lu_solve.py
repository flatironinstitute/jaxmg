import sys
import traceback
from functools import partial

import jax

if not jax.config.jax_enable_x64:
    jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding, PartitionSpec as P

from cusolvermp_case_utils import (
    assert_close_scaled,
    dtype_from_name,
    emit,
    global_array_to_numpy,
    local_device_id_for_process,
    make_process_mesh,
    make_rhs,
    native_status_words,
    select_gpu_allocator,
    solver_case,
)


coord_addr = sys.argv[1]
proc_id = int(sys.argv[2])
num_procs = int(sys.argv[3])
case_name = sys.argv[4]
dtype_name = sys.argv[5]

# Choose the GPU allocator (vmm vs platform) before the backend is created.
select_gpu_allocator(proc_id)

jax.distributed.initialize(
    coordinator_address=coord_addr,
    num_processes=num_procs,
    process_id=proc_id,
    local_device_ids=[local_device_id_for_process(proc_id)],
    coordinator_bind_address=coord_addr if proc_id == 0 else None,
)

from jaxmg import lu_solve
from jaxmg._cusolvermp_status import _CUSOLVERMP_LU_SOLVE_STATUS_SIZE


def make_nonsingular_matrix(n: int, dtype, *, seed: int):
    """Create a deterministic, diagonally dominant general matrix."""
    rng = np.random.default_rng(seed)
    real_dtype = (
        np.float32
        if np.dtype(dtype) in (np.dtype(np.float32), np.dtype(np.complex64))
        else np.float64
    )
    a = rng.normal(size=(n, n)).astype(real_dtype)
    if np.issubdtype(np.dtype(dtype), np.complexfloating):
        a = a + 0.25j * rng.normal(size=(n, n)).astype(real_dtype)
    a += (2.0 * n) * np.eye(n, dtype=a.dtype)
    return jnp.asarray(a, dtype=dtype)


def run_case() -> None:
    """Run one rank-per-GPU LU solve case and emit parser-friendly results."""
    dtype = dtype_from_name(dtype_name)
    case = solver_case(case_name, num_procs, routine="lu_solve")
    mesh = make_process_mesh(case)
    matrix_specs = P("pr", "pc")

    a = make_nonsingular_matrix(case.n, dtype, seed=4321)
    b = make_rhs(case.n, case.nrhs, dtype)
    if case.rhs_mode == "vector_replicated":
        b = b[:, 0]
    expected = jnp.linalg.solve(a, b)

    a_dev = jax.device_put(a, NamedSharding(mesh, matrix_specs))
    if case.rhs_mode == "vector_replicated":
        rhs_specs = P(None)
    elif case.rhs_mode == "matrix_replicated":
        rhs_specs = P(None, None)
    elif case.rhs_mode == "matrix_row_sharded":
        rhs_specs = P("pr", None)
    elif case.rhs_mode == "matrix_2d_sharded":
        rhs_specs = matrix_specs
    else:
        raise ValueError(f"unknown RHS placement mode {case.rhs_mode!r}")
    b_dev = jax.device_put(b, NamedSharding(mesh, rhs_specs))

    @partial(jax.jit, static_argnames=("tile_size",))
    def solve(_a, _b, *, tile_size):
        return lu_solve(
            _a,
            _b,
            tile_size,
            mesh=mesh,
            matrix_specs=matrix_specs,
            return_status=True,
            pad=True,
        )

    out, status = solve(a_dev, b_dev, tile_size=case.tile_size)
    out.block_until_ready()
    status.block_until_ready()

    status_words = native_status_words(status)
    assert status_words.size % _CUSOLVERMP_LU_SOLVE_STATUS_SIZE == 0, status_words
    assert np.all(status_words[::_CUSOLVERMP_LU_SOLVE_STATUS_SIZE] == 0), status_words
    assert_close_scaled(out, expected)

    out_host = global_array_to_numpy(out)
    a_host = np.asarray(a)
    b_host = np.asarray(b)
    residual = np.linalg.norm(a_host @ out_host - b_host) / np.linalg.norm(b_host)
    assert float(residual) < 1e-3

    emit(
        "MPTEST_RESULT",
        {
            "proc": proc_id,
            "name": case_name,
            "dtype": dtype_name,
            "status": "ok",
            "params": {
                "n": case.n,
                "nrhs": case.nrhs,
                "tile_size": case.tile_size,
                "process_rows": case.process_rows,
                "process_cols": case.process_cols,
                "grid_order": case.grid_order,
                "rhs_mode": case.rhs_mode,
            },
        },
    )
    multihost_utils.sync_global_devices(
        f"lu_solve_{case_name}_{dtype_name}_{num_procs}_complete"
    )


def main() -> None:
    try:
        run_case()
    except Exception:
        emit(
            "MPTEST_RESULT",
            {
                "proc": proc_id,
                "name": case_name,
                "dtype": dtype_name,
                "status": "fail",
                "traceback": traceback.format_exc(),
            },
        )
    finally:
        emit(
            "MPTEST_SUMMARY",
            {
                "proc": proc_id,
                "name": case_name,
                "dtype": dtype_name,
            },
        )


if __name__ == "__main__":
    main()
