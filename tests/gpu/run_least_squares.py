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
    SolverCase,
    dtype_from_name,
    emit,
    global_array_to_numpy,
    local_device_id_for_process,
    make_process_mesh,
    native_status_words,
    select_gpu_allocator,
)


coord_addr = sys.argv[1]
proc_id = int(sys.argv[2])
num_procs = int(sys.argv[3])
case_name = sys.argv[4]
dtype_name = sys.argv[5]
interface = os.environ.get("JAXMG_TEST_INTERFACE", "public")

select_gpu_allocator(proc_id)
jax.distributed.initialize(
    coordinator_address=coord_addr,
    num_processes=num_procs,
    process_id=proc_id,
    local_device_ids=[local_device_id_for_process(proc_id)],
    coordinator_bind_address=coord_addr if proc_id == 0 else None,
)

from jaxmg import least_squares, least_squares_shardmap_ctx
from jaxmg._cusolvermp_status import _CUSOLVERMP_LEAST_SQUARES_STATUS_SIZE


def run_case() -> None:
    """Run one distributed least-squares case and validate its solution."""
    dtype = dtype_from_name(dtype_name)
    if case_name in (
        "matrix",
        "matrix_replicated",
        "vector",
        "vector_replicated",
    ):
        process_rows, process_cols = 2, 1
        m, n, tile_size, nrhs = 192, 128, 64, 3
    elif case_name == "matrix_2d_sharded":
        process_rows, process_cols = 1, 2
        m, n, tile_size, nrhs = 192, 128, 64, 128
    elif case_name == "column_grid":
        process_rows, process_cols = 1, 2
        m, n, tile_size, nrhs = 192, 96, 64, 65
    elif case_name == "square_aligned":
        process_rows, process_cols = 1, 1
        m, n, tile_size, nrhs = 128, 128, 64, 64
    else:
        raise ValueError(f"unknown least-squares test case {case_name!r}")
    if process_rows * process_cols != num_procs:
        raise ValueError(
            f"{case_name} requires {process_rows * process_cols} processes"
        )
    rhs_mode = {
        "matrix_replicated": "matrix_replicated",
        "matrix_2d_sharded": "matrix_2d_sharded",
        "vector_replicated": "vector_replicated",
    }.get(case_name, "matrix_row_sharded")
    case = SolverCase(
        process_rows=process_rows,
        process_cols=process_cols,
        grid_order="row_major",
        n=m,
        tile_size=tile_size,
        nrhs=nrhs,
        rhs_mode=rhs_mode,
    )
    mesh = make_process_mesh(case)
    matrix_specs = P("pr", "pc")

    rng = np.random.default_rng(90210)
    real_dtype = (
        np.float32
        if np.dtype(dtype) in (np.dtype(np.float32), np.dtype(np.complex64))
        else np.float64
    )
    a_host = rng.normal(size=(m, n)).astype(real_dtype)
    b_host = rng.normal(size=(m, nrhs)).astype(real_dtype)
    if np.issubdtype(np.dtype(dtype), np.complexfloating):
        a_host = a_host + 0.25j * rng.normal(size=a_host.shape).astype(real_dtype)
        b_host = b_host + 0.25j * rng.normal(size=b_host.shape).astype(real_dtype)
    a_host = a_host.astype(np.dtype(dtype))
    b_host = b_host.astype(np.dtype(dtype))
    expected = np.linalg.lstsq(a_host, b_host, rcond=None)[0]

    if case_name in ("vector", "vector_replicated"):
        b_host = b_host[:, 0]
        expected = expected[:, 0]
    rhs_specs = {
        "matrix_replicated": P(None, None),
        "matrix_2d_sharded": matrix_specs,
        "vector": P("pr"),
        "vector_replicated": P(None),
    }.get(case_name, P("pr", None))
    a = jax.device_put(jnp.asarray(a_host), NamedSharding(mesh, matrix_specs))
    b = jax.device_put(jnp.asarray(b_host), NamedSharding(mesh, rhs_specs))
    expected_output_sharding = b.sharding

    if interface == "context":

        @partial(jax.jit, donate_argnums=(0, 1))
        def solve(_a, _b):
            return least_squares_shardmap_ctx(
                _a, _b, case.tile_size, mesh=mesh, matrix_specs=matrix_specs
            )

        a_work, b_work, out, status = solve(a, b)
        a_work.block_until_ready()
        b_work.block_until_ready()
        assert b_work.sharding.is_equivalent_to(
            expected_output_sharding, b_work.ndim
        )
    else:
        out, status = least_squares(
            a,
            b,
            case.tile_size,
            mesh=mesh,
            matrix_specs=matrix_specs,
            return_status=True,
        )

    out.block_until_ready()
    status.block_until_ready()
    words = native_status_words(status)
    assert words.size % _CUSOLVERMP_LEAST_SQUARES_STATUS_SIZE == 0, words
    assert np.all(
        words[::_CUSOLVERMP_LEAST_SQUARES_STATUS_SIZE] == 0
    ), words
    tolerance = (
        2e-3
        if np.dtype(dtype) in (np.dtype(np.float32), np.dtype(np.complex64))
        else 1e-9
    )
    np.testing.assert_allclose(
        global_array_to_numpy(out), expected, rtol=tolerance, atol=tolerance
    )
    assert out.sharding.is_equivalent_to(expected_output_sharding, out.ndim)

    emit(
        "GPU_TEST_RESULT",
        {
            "proc": proc_id,
            "name": case_name,
            "dtype": dtype_name,
            "status": "ok",
            "interface": interface,
        },
    )
    multihost_utils.sync_global_devices(
        f"least_squares_{case_name}_{dtype_name}_{interface}_complete"
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
        emit("GPU_TEST_SUMMARY", {"proc": proc_id, "name": case_name})


if __name__ == "__main__":
    main()
