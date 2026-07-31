import os
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
    make_hermitian_positive_definite,
    make_process_mesh,
    native_status_words,
    select_gpu_allocator,
    solver_case,
)


coord_addr = sys.argv[1]
proc_id = int(sys.argv[2])
num_procs = int(sys.argv[3])
case_name = sys.argv[4]
dtype_name = sys.argv[5]
interface = os.environ.get("JAXMG_TEST_INTERFACE", "public")
return_eigenvectors = (
    os.environ.get("JAXMG_SYEVD_RETURN_EIGENVECTORS", "1") == "1"
)

# Choose the GPU allocator (vmm vs platform) before the backend is created.
select_gpu_allocator(proc_id)

jax.distributed.initialize(
    coordinator_address=coord_addr,
    num_processes=num_procs,
    process_id=proc_id,
    local_device_ids=[local_device_id_for_process(proc_id)],
    coordinator_bind_address=coord_addr if proc_id == 0 else None,
)

from jaxmg import syevd, syevd_shardmap_ctx
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

    if interface == "context":

        @partial(
            jax.jit,
            donate_argnums=(0,),
            static_argnames=("tile_size",),
        )
        def eigensolve(_a, *, tile_size):
            return syevd_shardmap_ctx(
                _a,
                tile_size,
                mesh=mesh,
                matrix_specs=matrix_specs,
                return_eigenvectors=return_eigenvectors,
                pad=True,
            )

        outputs = eigensolve(a_dev, tile_size=case.tile_size)
        if return_eigenvectors:
            a_work, eigenvalues, vectors, status = outputs
        else:
            a_work, eigenvalues, status = outputs
            vectors = None
        a_work.block_until_ready()
    else:

        @partial(jax.jit, static_argnames=("tile_size",))
        def eigensolve(_a, *, tile_size):
            return syevd(
                _a,
                tile_size,
                mesh=mesh,
                matrix_specs=matrix_specs,
                return_eigenvectors=return_eigenvectors,
                return_status=True,
                pad=True,
            )

        outputs = eigensolve(a_dev, tile_size=case.tile_size)
        if return_eigenvectors:
            eigenvalues, vectors, status = outputs
        else:
            eigenvalues, status = outputs
            vectors = None
    eigenvalues.block_until_ready()
    if vectors is not None:
        vectors.block_until_ready()
    status.block_until_ready()

    status_words = native_status_words(status)
    assert status_words.size % _CUSOLVERMP_SYEVD_STATUS_SIZE == 0, status_words
    assert np.all(status_words[::_CUSOLVERMP_SYEVD_STATUS_SIZE] == 0), status_words
    assert np.all(
        status_words[20::_CUSOLVERMP_SYEVD_STATUS_SIZE]
        == int(return_eigenvectors)
    ), status_words
    assert np.all(
        status_words[27::_CUSOLVERMP_SYEVD_STATUS_SIZE]
        == int(return_eigenvectors)
    ), status_words

    assert_close_scaled(eigenvalues, expected_eigenvalues, atol=1e-3, rtol=1e-3)

    if vectors is not None:
        a_host = np.asarray(a)
        eigenvalues_host = global_array_to_numpy(eigenvalues)
        vectors_host = global_array_to_numpy(vectors)
        residual = np.linalg.norm(
            a_host @ vectors_host - vectors_host * eigenvalues_host[None, :]
        )
        residual = residual / np.linalg.norm(a_host)
        assert float(residual) < 5e-3

        identity = np.eye(case.n, dtype=vectors_host.dtype)
        orthogonality = np.linalg.norm(
            vectors_host.conj().T @ vectors_host - identity
        )
        orthogonality = orthogonality / np.sqrt(case.n)
        assert float(orthogonality) < 5e-3

    emit(
        "MPTEST_RESULT",
        {
            "proc": proc_id,
            "name": case_name,
            "dtype": dtype_name,
            "status": "ok",
            "interface": interface,
            "return_eigenvectors": return_eigenvectors,
            "params": {
                "n": case.n,
                "tile_size": case.tile_size,
                "process_rows": case.process_rows,
                "process_cols": case.process_cols,
                "grid_order": case.grid_order,
            },
        },
    )
    multihost_utils.sync_global_devices(
        f"syevd_{case_name}_{dtype_name}_{num_procs}_complete"
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
