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

from jaxmg import qr, qr_shardmap_ctx
from jaxmg._cusolvermp_status import _CUSOLVERMP_QR_STATUS_SIZE


def run_case() -> None:
    """Run one padded reduced QR decomposition and validate both factors."""
    dtype = dtype_from_name(dtype_name)
    if case_name == "padded_qr":
        process_rows, process_cols = num_procs, 1
        m, n, tile_size = 384, 192, 128
    elif case_name == "column_grid":
        process_rows, process_cols = 1, num_procs
        m, n, tile_size = 384, 192, 128
    elif case_name == "square_aligned":
        process_rows, process_cols = 1, 1
        m, n, tile_size = 256, 256, 128
    else:
        raise ValueError(f"unknown QR test case {case_name!r}")
    if process_rows * process_cols != num_procs:
        raise ValueError(
            f"{case_name} requires {process_rows * process_cols} processes"
        )
    case = SolverCase(
        process_rows=process_rows,
        process_cols=process_cols,
        grid_order="row_major",
        n=m,
        tile_size=tile_size,
    )
    mesh = make_process_mesh(case)
    matrix_specs = P("pr", "pc")

    rng = np.random.default_rng(7)
    if np.issubdtype(np.dtype(dtype), np.complexfloating):
        a_host = rng.normal(size=(m, n)) + 1j * rng.normal(size=(m, n))
    else:
        a_host = rng.normal(size=(m, n))
    a_host = a_host.astype(np.dtype(dtype))
    a_host[np.arange(n), np.arange(n)] += np.asarray(3, dtype=np.dtype(dtype))
    a_dev = jax.device_put(a_host, NamedSharding(mesh, matrix_specs))

    if interface == "context":

        @partial(jax.jit, donate_argnums=(0,), static_argnames=("tile_size",))
        def decomposition(_a, *, tile_size):
            return qr_shardmap_ctx(
                _a,
                tile_size,
                mesh=mesh,
                matrix_specs=matrix_specs,
            )

        q, r, status = decomposition(a_dev, tile_size=tile_size)
    else:
        q, r, status = qr(
            a_dev,
            tile_size,
            mesh=mesh,
            matrix_specs=matrix_specs,
            return_status=True,
        )

    q.block_until_ready()
    r.block_until_ready()
    status.block_until_ready()

    words = native_status_words(status)
    assert words.size % _CUSOLVERMP_QR_STATUS_SIZE == 0, words
    assert np.all(words[::_CUSOLVERMP_QR_STATUS_SIZE] == 0), words
    assert np.all(words[28::_CUSOLVERMP_QR_STATUS_SIZE] == 1), words
    assert np.all(words[29::_CUSOLVERMP_QR_STATUS_SIZE] == 0), words
    assert np.all(words[30::_CUSOLVERMP_QR_STATUS_SIZE] == 1), words
    assert np.all(words[31::_CUSOLVERMP_QR_STATUS_SIZE] == 1), words
    assert np.all(words[32::_CUSOLVERMP_QR_STATUS_SIZE] == 0), words

    q_host = global_array_to_numpy(q)
    r_host = global_array_to_numpy(r)
    tolerance = (
        2e-3
        if np.dtype(dtype) in (np.dtype(np.float32), np.dtype(np.complex64))
        else 2e-10
    )
    np.testing.assert_allclose(
        q_host.conj().T @ q_host,
        np.eye(n, dtype=q_host.dtype),
        rtol=tolerance,
        atol=tolerance,
    )
    np.testing.assert_allclose(
        q_host @ r_host, a_host, rtol=tolerance, atol=tolerance
    )
    np.testing.assert_allclose(
        np.tril(r_host, -1), 0, rtol=tolerance, atol=tolerance
    )

    emit(
        "GPU_TEST_RESULT",
        {
            "proc": proc_id,
            "name": case_name,
            "dtype": dtype_name,
            "status": "ok",
            "interface": interface,
            "params": {
                "m": m,
                "n": n,
                "tile_size": tile_size,
                "process_rows": process_rows,
                "process_cols": process_cols,
            },
        },
    )
    multihost_utils.sync_global_devices(
        f"qr_{case_name}_{dtype_name}_{num_procs}_complete"
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
            {"proc": proc_id, "name": case_name, "dtype": dtype_name},
        )


if __name__ == "__main__":
    main()
