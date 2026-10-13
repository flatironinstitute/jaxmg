import os
import sys
import traceback
from functools import partial

import jax

if not jax.config.jax_enable_x64:
    jax.config.update("jax_enable_x64", True)

import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from cusolvermp_case_utils import (
    SINGULAR_TILE_SIZE,
    assert_rank_failure_with_nan,
    dtype_from_name,
    emit,
    global_array_to_numpy,
    local_device_id_for_process,
    make_hermitian_positive_definite,
    make_process_mesh,
    make_singular_diagonal_system,
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

# Choose the GPU allocator (vmm vs platform) before the backend is created.
select_gpu_allocator(proc_id)

jax.distributed.initialize(
    coordinator_address=coord_addr,
    num_processes=num_procs,
    process_id=proc_id,
    local_device_ids=[local_device_id_for_process(proc_id)],
    coordinator_bind_address=coord_addr if proc_id == 0 else None,
)

from jaxmg import potrf, potrf_shardmap_ctx
from jaxmg._cusolvermp_status import _CUSOLVERMP_POTRF_STATUS_SIZE


def emit_ok(params=None) -> None:
    emit(
        "GPU_TEST_RESULT",
        {
            "proc": proc_id,
            "name": case_name,
            "dtype": dtype_name,
            "status": "ok",
            "interface": interface,
            "params": params or {},
        },
    )
    multihost_utils.sync_global_devices(
        f"potrf_{case_name}_{interface}_{dtype_name}_{num_procs}_complete"
    )


def run_singular_case(dtype) -> None:
    """Check that a failed Cholesky factorization returns NaN on every rank."""
    mesh = Mesh(np.asarray(jax.devices(), dtype=object), ("x",))
    matrix_specs = P("x", None)
    a, _ = make_singular_diagonal_system(num_procs, dtype)
    a_dev = jax.device_put(a, NamedSharding(mesh, matrix_specs))

    factor, status = jax.jit(
        lambda _a: potrf_shardmap_ctx(
            _a, SINGULAR_TILE_SIZE, mesh=mesh, matrix_specs=matrix_specs
        )
    )(a_dev)
    status.block_until_ready()

    # 26 is kPotrfFailed and 27 is kPotrfInfoNonzero.
    assert_rank_failure_with_nan(
        native_status_words(status), _CUSOLVERMP_POTRF_STATUS_SIZE, (26, 27), factor
    )
    emit_ok()


def run_case() -> None:
    """Run one rank-per-GPU Cholesky factorization and validate the factor."""
    dtype = dtype_from_name(dtype_name)
    if interface == "singular":
        run_singular_case(dtype)
        return
    case = solver_case(case_name, num_procs, routine="potrf")
    mesh = make_process_mesh(case)
    matrix_specs = P("pr", "pc")
    replicated_input = case_name == "replicated_input"

    a = make_hermitian_positive_definite(case.n, dtype, seed=2468)
    a_host = np.asarray(a).copy()
    a_dev = jax.device_put(
        a, NamedSharding(mesh, P() if replicated_input else matrix_specs)
    )

    if interface == "context":

        @partial(jax.jit, donate_argnums=(0,), static_argnames=("tile_size",))
        def factorize(_a, *, tile_size):
            return potrf_shardmap_ctx(
                _a, tile_size, mesh=mesh, matrix_specs=matrix_specs
            )

        factor, status = factorize(a_dev, tile_size=case.tile_size)
    elif interface == "inferred":
        # Neither mesh nor matrix_specs: inside the caller's jit, JAXMg reads
        # them off the type of A and the abstract context mesh.
        @partial(jax.jit, static_argnames=("tile_size",))
        def factorize(_a, *, tile_size):
            return potrf(_a, tile_size, return_status=True)

        with jax.set_mesh(mesh):
            factor, status = factorize(a_dev, tile_size=case.tile_size)
    else:
        factor, status = potrf(
            a_dev,
            case.tile_size,
            mesh=mesh,
            matrix_specs=matrix_specs,
            return_status=True,
        )
    factor.block_until_ready()
    status.block_until_ready()

    words = native_status_words(status)
    size = _CUSOLVERMP_POTRF_STATUS_SIZE
    assert words.size % size == 0, words
    assert np.all(words[0::size] == 0), words
    assert np.all(words[21::size] == 1), words  # potrf_called
    assert np.all(words[22::size] == 0), words  # potrf_info
    assert np.all(words[23::size] == 1), words  # upper_zeroed

    if replicated_input:
        assert factor.sharding.is_fully_replicated
    else:
        assert factor.sharding.is_equivalent_to(
            NamedSharding(mesh, matrix_specs), factor.ndim
        )

    factor_host = global_array_to_numpy(factor)
    assert factor_host.shape == a_host.shape
    assert factor_host.dtype == a_host.dtype
    assert np.all(np.triu(factor_host, 1) == 0)

    single = np.dtype(dtype) in (np.dtype(np.float32), np.dtype(np.complex64))
    tolerance = 1e-4 if single else 1e-10
    reference = np.linalg.cholesky(
        a_host.astype(np.result_type(a_host.dtype, np.float64))
    )
    reconstruction_error = np.linalg.norm(
        factor_host @ factor_host.conj().T - a_host
    ) / np.linalg.norm(a_host)
    factor_error = np.linalg.norm(factor_host - reference) / np.linalg.norm(reference)
    assert float(reconstruction_error) < tolerance, reconstruction_error
    assert float(factor_error) < tolerance, factor_error

    emit_ok(
        {
            "n": case.n,
            "tile_size": case.tile_size,
            "process_rows": case.process_rows,
            "process_cols": case.process_cols,
            "grid_order": case.grid_order,
            "reconstruction_error": float(reconstruction_error),
        }
    )


def main() -> None:
    try:
        run_case()
    except Exception:
        emit(
            "GPU_TEST_RESULT",
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
            "GPU_TEST_SUMMARY",
            {
                "proc": proc_id,
                "name": case_name,
                "dtype": dtype_name,
            },
        )


if __name__ == "__main__":
    main()
