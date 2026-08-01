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
    balanced_process_grid,
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

from jaxmg import gesvd, gesvd_shardmap_ctx
from jaxmg._cusolvermp_status import _CUSOLVERMP_GESVD_STATUS_SIZE


def run_case() -> None:
    """Run one distributed GESVD case and validate the selected factorization."""
    dtype = dtype_from_name(dtype_name)
    compute_u = not case_name.endswith("_vh") and not case_name.endswith("_values")
    compute_vh = not case_name.endswith("_u") and not case_name.endswith("_values")
    full_matrices = "_full" in case_name

    if case_name.startswith("tall"):
        process_rows, process_cols = num_procs, 1
        m, n, tile_size = 512, 256, 64
        grid_order = "row_major"
    elif case_name.startswith("wide_padding"):
        process_rows, process_cols = 1, num_procs
        m, n, tile_size = 256, 384, 128
        grid_order = "row_major"
    elif case_name.startswith("balanced_padding"):
        process_rows, process_cols = balanced_process_grid(num_procs)
        m = process_rows * 192
        n = process_cols * 192
        tile_size = 128
        grid_order = "column_major"
    else:
        raise ValueError(f"unknown GESVD case {case_name!r}")

    case = SolverCase(
        process_rows=process_rows,
        process_cols=process_cols,
        grid_order=grid_order,
        n=max(m, n),
        tile_size=tile_size,
    )
    mesh = make_process_mesh(case)
    matrix_specs = P("pr", "pc")
    k = min(m, n)

    # Distinct positive diagonal entries give known singular values while the
    # rectangular zero regions exercise U and Vh output geometry.
    a_host = np.zeros((m, n), dtype=np.dtype(dtype))
    expected_singular_values = np.linspace(2.0, 1.0, k).astype(
        np.float32 if dtype in (jnp.float32, jnp.complex64) else np.float64
    )
    a_host[np.arange(k), np.arange(k)] = expected_singular_values
    a_dev = jax.device_put(a_host, NamedSharding(mesh, matrix_specs))

    if interface == "context":

        @partial(jax.jit, donate_argnums=(0,), static_argnames=("tile_size",))
        def decomposition(_a, *, tile_size):
            return gesvd_shardmap_ctx(
                _a,
                tile_size,
                mesh=mesh,
                matrix_specs=matrix_specs,
                compute_u=compute_u,
                compute_vh=compute_vh,
                full_matrices=full_matrices,
            )

        outputs = decomposition(a_dev, tile_size=tile_size)
        a_work, *numerical_outputs, status = outputs
        a_work.block_until_ready()
    else:
        outputs = gesvd(
            a_dev,
            tile_size,
            mesh=mesh,
            matrix_specs=matrix_specs,
            compute_u=compute_u,
            compute_vh=compute_vh,
            full_matrices=full_matrices,
            return_status=True,
        )
        *numerical_outputs, status = outputs

    if compute_u and compute_vh:
        u, singular_values, vh = numerical_outputs
    elif compute_u:
        u, singular_values = numerical_outputs
        vh = None
    elif compute_vh:
        singular_values, vh = numerical_outputs
        u = None
    else:
        (singular_values,) = numerical_outputs
        u = None
        vh = None

    singular_values.block_until_ready()
    if u is not None:
        u.block_until_ready()
    if vh is not None:
        vh.block_until_ready()
    status.block_until_ready()

    status_words = native_status_words(status)
    assert status_words.size % _CUSOLVERMP_GESVD_STATUS_SIZE == 0, status_words
    assert np.all(status_words[::_CUSOLVERMP_GESVD_STATUS_SIZE] == 0), status_words
    assert np.all(
        status_words[21::_CUSOLVERMP_GESVD_STATUS_SIZE] == int(compute_u)
    ), status_words
    assert np.all(
        status_words[22::_CUSOLVERMP_GESVD_STATUS_SIZE] == int(compute_vh)
    ), status_words
    assert np.all(
        status_words[23::_CUSOLVERMP_GESVD_STATUS_SIZE] == int(full_matrices)
    ), status_words
    assert np.all(
        status_words[35::_CUSOLVERMP_GESVD_STATUS_SIZE] == k
    ), status_words

    singular_values_host = global_array_to_numpy(singular_values)
    np.testing.assert_allclose(
        singular_values_host, expected_singular_values, rtol=5e-3, atol=5e-3
    )
    if u is not None:
        u_host = global_array_to_numpy(u)
        expected_u_cols = m if full_matrices else k
        assert u_host.shape == (m, expected_u_cols)
        np.testing.assert_allclose(
            u_host.conj().T @ u_host,
            np.eye(expected_u_cols, dtype=u_host.dtype),
            rtol=5e-3,
            atol=5e-3,
        )
    else:
        u_host = None
    if vh is not None:
        vh_host = global_array_to_numpy(vh)
        expected_vh_rows = n if full_matrices else k
        assert vh_host.shape == (expected_vh_rows, n)
        np.testing.assert_allclose(
            vh_host @ vh_host.conj().T,
            np.eye(expected_vh_rows, dtype=vh_host.dtype),
            rtol=5e-3,
            atol=5e-3,
        )

    if u_host is not None and vh is not None:
        reconstructed = (
            u_host[:, :k] * singular_values_host[None, :]
        ) @ vh_host[:k, :]
        residual = np.linalg.norm(reconstructed - a_host) / np.linalg.norm(a_host)
        assert float(residual) < 5e-3

    emit(
        "GPU_TEST_RESULT",
        {
            "proc": proc_id,
            "name": case_name,
            "dtype": dtype_name,
            "status": "ok",
            "interface": interface,
            "compute_u": compute_u,
            "compute_vh": compute_vh,
            "full_matrices": full_matrices,
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
        f"gesvd_{case_name}_{dtype_name}_{num_procs}_complete"
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
