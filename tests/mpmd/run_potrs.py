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
return_logdet = os.environ.get("JAXMG_TEST_POTRS_LOGDET") == "1"
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

from jaxmg import potrs, potrs_shardmap_ctx
from jaxmg._cusolvermp_status import _CUSOLVERMP_POTRS_STATUS_SIZE


def run_case() -> None:
    """Run one rank-per-GPU POTRS case and emit parser-friendly results."""
    dtype = dtype_from_name(dtype_name)
    case = solver_case(case_name, num_procs, routine="potrs")
    mesh = make_process_mesh(case)
    matrix_specs = P("pr", "pc")

    a = make_hermitian_positive_definite(case.n, dtype, seed=1234)
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

    if interface == "context":
        if return_logdet:
            raise ValueError("the context smoke test does not request logdet")

        @partial(
            jax.jit,
            donate_argnums=(0, 1),
            static_argnames=("tile_size",),
        )
        def solve(_a, _b, *, tile_size):
            return potrs_shardmap_ctx(
                _a,
                _b,
                tile_size,
                mesh=mesh,
                matrix_specs=matrix_specs,
                pad=True,
            )

        a_work, out, status = solve(a_dev, b_dev, tile_size=case.tile_size)
        a_work.block_until_ready()
    else:
        @partial(jax.jit, static_argnames=("tile_size",))
        def solve(_a, _b, *, tile_size):
            return potrs(
                _a,
                _b,
                tile_size,
                mesh=mesh,
                matrix_specs=matrix_specs,
                return_status=True,
                return_logdet=return_logdet,
                pad=True,
            )

        result = solve(a_dev, b_dev, tile_size=case.tile_size)
        if return_logdet:
            out, logdet, status = result
            logdet.block_until_ready()
        else:
            out, status = result
    out.block_until_ready()
    status.block_until_ready()

    status_words = native_status_words(status)
    assert status_words.size % _CUSOLVERMP_POTRS_STATUS_SIZE == 0, status_words
    assert np.all(status_words[::_CUSOLVERMP_POTRS_STATUS_SIZE] == 0), status_words
    assert_close_scaled(out, expected)

    out_host = global_array_to_numpy(out)
    a_host = np.asarray(a)
    b_host = np.asarray(b)
    residual = np.linalg.norm(a_host @ out_host - b_host) / np.linalg.norm(b_host)
    assert float(residual) < 1e-3
    if return_logdet:
        expected_logdet_dtype = (
            jnp.float32 if dtype in (jnp.float32, jnp.complex64) else jnp.float64
        )
        assert logdet.dtype == expected_logdet_dtype
        expected_sign, expected_logdet = np.linalg.slogdet(a_host)
        assert float(np.real(expected_sign)) > 0.0
        tolerance = 5e-4 if dtype in (jnp.float32, jnp.complex64) else 1e-10
        assert np.allclose(
            float(logdet), float(expected_logdet), rtol=tolerance, atol=tolerance
        )
        assert np.all(status_words[33::_CUSOLVERMP_POTRS_STATUS_SIZE] == 1)

    emit(
        "MPTEST_RESULT",
        {
            "proc": proc_id,
            "name": case_name,
            "dtype": dtype_name,
            "status": "ok",
            "interface": interface,
            "return_logdet": return_logdet,
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
        f"potrs_{case_name}_{dtype_name}_{num_procs}_complete"
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
