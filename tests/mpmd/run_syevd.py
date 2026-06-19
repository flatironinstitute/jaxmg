import sys
import traceback
from functools import partial

import jax
if not jax.config.jax_enable_x64:
    jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from cusolvermp_case_utils import (
    assert_close_scaled,
    dtype_from_name,
    emit,
    local_device_id_for_process,
    make_hermitian_positive_definite,
    make_process_mesh,
    solver_case,
)


coord_addr = sys.argv[1]
proc_id = int(sys.argv[2])
num_procs = int(sys.argv[3])
case_name = sys.argv[4]
dtype_name = sys.argv[5]

jax.distributed.initialize(
    coordinator_address=coord_addr,
    num_processes=num_procs,
    process_id=proc_id,
    local_device_ids=[local_device_id_for_process(proc_id)],
    coordinator_bind_address=coord_addr if proc_id == 0 else None,
)

from jaxmg import syevd
from jaxmg._cusolvermp_status import _CUSOLVERMP_SYEVD_STATUS_SIZE


def run_case() -> None:
    """Run one rank-per-GPU SYEVD case and emit parser-friendly results."""
    dtype = dtype_from_name(dtype_name)
    case = solver_case(case_name, num_procs, routine="syevd")
    mesh = make_process_mesh(case)
    matrix_specs = P("pr", "pc")

    a = make_hermitian_positive_definite(case.n, dtype, seed=5678)
    expected_eigenvalues, _ = jnp.linalg.eigh(a)

    a_dev = jax.device_put(a, NamedSharding(mesh, matrix_specs))

    @partial(jax.jit, static_argnames=("tile_size",))
    def eigensolve(_a, *, tile_size):
        return syevd(
            _a,
            tile_size,
            mesh=mesh,
            matrix_specs=matrix_specs,
            return_status=True,
            pad=True,
        )

    eigenvalues, vectors, status = eigensolve(a_dev, tile_size=case.tile_size)
    eigenvalues.block_until_ready()
    vectors.block_until_ready()
    status.block_until_ready()

    status_words = np.asarray(status).reshape(-1)
    assert status_words.size % _CUSOLVERMP_SYEVD_STATUS_SIZE == 0, status_words
    assert np.all(status_words[::_CUSOLVERMP_SYEVD_STATUS_SIZE] == 0), status_words

    assert_close_scaled(eigenvalues, expected_eigenvalues, atol=1e-3, rtol=1e-3)
    residual = jnp.linalg.norm(a @ vectors - vectors * eigenvalues[None, :])
    residual = residual / jnp.linalg.norm(a)
    assert float(residual) < 5e-3

    identity = jnp.eye(case.n, dtype=vectors.dtype)
    orthogonality = jnp.linalg.norm(vectors.conj().T @ vectors - identity)
    orthogonality = orthogonality / jnp.sqrt(case.n)
    assert float(orthogonality) < 5e-3

    emit(
        "MPTEST_RESULT",
        {
            "proc": proc_id,
            "name": case_name,
            "dtype": dtype_name,
            "status": "ok",
            "params": {
                "n": case.n,
                "tile_size": case.tile_size,
                "process_rows": case.process_rows,
                "process_cols": case.process_cols,
                "grid_order": case.grid_order,
            },
        },
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
