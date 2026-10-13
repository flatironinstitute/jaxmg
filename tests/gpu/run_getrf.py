import os
import sys
import traceback
from functools import partial

import jax

if not jax.config.jax_enable_x64:
    jax.config.update("jax_enable_x64", True)

import numpy as np
import scipy.linalg
from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from cusolvermp_case_utils import (
    SINGULAR_TILE_SIZE,
    assert_rank_failure_with_nan,
    dtype_from_name,
    emit,
    global_array_to_numpy,
    local_device_id_for_process,
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

from jaxmg import getrf, getrf_shardmap_ctx
from jaxmg._cusolvermp_status import _CUSOLVERMP_GETRF_STATUS_SIZE


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
        f"getrf_{case_name}_{interface}_{dtype_name}_{num_procs}_complete"
    )


def make_pivoting_matrix(n: int, dtype, *, seed: int) -> np.ndarray:
    """Create a well-conditioned random matrix that needs row interchanges.

    Unlike the diagonally dominant LU-solve matrix, partial pivoting picks
    off-diagonal rows here, so the gathered pivot vector is non-trivial.
    """
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n))
    if np.issubdtype(np.dtype(dtype), np.complexfloating):
        a = a + 1j * rng.normal(size=(n, n))
    return a.astype(np.dtype(dtype))


def permutation_from_pivots(piv: np.ndarray) -> np.ndarray:
    """Apply LAPACK-style row interchanges to the identity permutation."""
    perm = np.arange(piv.size)
    for i, j in enumerate(piv):
        perm[i], perm[j] = perm[j], perm[i]
    return perm


def run_singular_case(dtype) -> None:
    """Check that a singular LU factorization returns NaN and -1 pivots."""
    mesh = Mesh(np.asarray(jax.devices(), dtype=object), ("x",))
    matrix_specs = P("x", None)
    a, _ = make_singular_diagonal_system(num_procs, dtype)
    a_dev = jax.device_put(a, NamedSharding(mesh, matrix_specs))

    lu, piv, status = jax.jit(
        lambda _a: getrf_shardmap_ctx(
            _a, SINGULAR_TILE_SIZE, mesh=mesh, matrix_specs=matrix_specs
        )
    )(a_dev)
    status.block_until_ready()

    # 38 is kGetrfFailed and 39 is kGetrfInfoNonzero.
    assert_rank_failure_with_nan(
        native_status_words(status), _CUSOLVERMP_GETRF_STATUS_SIZE, (38, 39), lu
    )
    assert np.all(global_array_to_numpy(piv) == -1)
    emit_ok()


def run_case() -> None:
    """Run one rank-per-GPU LU factorization and validate factor and pivots."""
    dtype = dtype_from_name(dtype_name)
    if interface == "singular":
        run_singular_case(dtype)
        return
    case = solver_case(case_name, num_procs, routine="getrf")
    mesh = make_process_mesh(case)
    matrix_specs = P("pr", "pc")
    replicated_input = case_name == "replicated_input"

    a_host = make_pivoting_matrix(case.n, dtype, seed=1357)
    a_dev = jax.device_put(
        a_host, NamedSharding(mesh, P() if replicated_input else matrix_specs)
    )

    if interface == "context":

        @partial(jax.jit, donate_argnums=(0,), static_argnames=("tile_size",))
        def factorize(_a, *, tile_size):
            return getrf_shardmap_ctx(
                _a, tile_size, mesh=mesh, matrix_specs=matrix_specs
            )

        lu, piv, status = factorize(a_dev, tile_size=case.tile_size)
    elif interface == "inferred":
        # Neither mesh nor matrix_specs: inside the caller's jit, JAXMg reads
        # them off the type of A and the abstract context mesh.
        @partial(jax.jit, static_argnames=("tile_size",))
        def factorize(_a, *, tile_size):
            return getrf(_a, tile_size, return_status=True)

        with jax.set_mesh(mesh):
            lu, piv, status = factorize(a_dev, tile_size=case.tile_size)
    else:
        lu, piv, status = getrf(
            a_dev,
            case.tile_size,
            mesh=mesh,
            matrix_specs=matrix_specs,
            return_status=True,
        )
    lu.block_until_ready()
    piv.block_until_ready()
    status.block_until_ready()

    words = native_status_words(status)
    size = _CUSOLVERMP_GETRF_STATUS_SIZE
    assert words.size % size == 0, words
    assert np.all(words[0::size] == 0), words
    assert np.all(words[21::size] == 1), words  # getrf_called
    assert np.all(words[22::size] == 0), words  # getrf_info
    assert np.all(words[25::size] == 1), words  # pivots_gathered

    if replicated_input:
        assert lu.sharding.is_fully_replicated
    else:
        assert lu.sharding.is_equivalent_to(NamedSharding(mesh, matrix_specs), lu.ndim)
    assert piv.sharding.is_fully_replicated

    n = case.n
    lu_host = global_array_to_numpy(lu)
    piv_host = global_array_to_numpy(piv)
    assert lu_host.shape == (n, n)
    assert piv_host.shape == (n,)
    assert piv_host.dtype == np.int32
    assert np.all(piv_host >= np.arange(n)), piv_host
    assert np.all(piv_host < n), piv_host
    assert np.any(piv_host != np.arange(n)), "test matrix produced no pivoting"

    lower = np.tril(lu_host, -1) + np.eye(n, dtype=lu_host.dtype)
    upper = np.triu(lu_host)
    perm = permutation_from_pivots(piv_host)
    single = np.dtype(dtype) in (np.dtype(np.float32), np.dtype(np.complex64))
    tolerance = 1e-4 if single else 1e-10
    reconstruction_error = np.linalg.norm(
        a_host[perm] - lower @ upper
    ) / np.linalg.norm(a_host)
    assert float(reconstruction_error) < tolerance, reconstruction_error

    if not single:
        # Ties between pivot candidates are vanishingly unlikely in double
        # precision, so the pivot sequence must match LAPACK exactly.
        lu_ref, piv_ref = scipy.linalg.lu_factor(a_host)
        np.testing.assert_array_equal(piv_host, piv_ref)
        np.testing.assert_allclose(lu_host, lu_ref, rtol=1e-8, atol=1e-8)

    emit_ok(
        {
            "n": n,
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
