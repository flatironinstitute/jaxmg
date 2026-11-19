"""Validate a fused inverse against NumPy on one rank per GPU."""

import os
import sys
import traceback
from functools import partial

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding, PartitionSpec as P

from cusolvermp_case_utils import (
    assert_rank_failure_with_nan,
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


coord_addr, proc_id, num_procs, case_name, dtype_name = sys.argv[1:6]
proc_id, num_procs = int(proc_id), int(num_procs)
routine, layout_case = case_name.split(":", 1)
interface = os.environ.get("JAXMG_TEST_INTERFACE", "public")
select_gpu_allocator(proc_id)
jax.distributed.initialize(
    coordinator_address=coord_addr,
    num_processes=num_procs,
    process_id=proc_id,
    local_device_ids=[local_device_id_for_process(proc_id)],
    coordinator_bind_address=coord_addr if proc_id == 0 else None,
)

import jaxmg
from jaxmg._cusolvermp_status import _CUSOLVERMP_INVERSE_STATUS_SIZE


def run_case():
    dtype = dtype_from_name(dtype_name)
    case = solver_case(layout_case, num_procs, routine=routine)
    mesh = make_process_mesh(case)
    specs = P("pr", "pc")
    if routine == "potri":
        a = np.asarray(make_hermitian_positive_definite(case.n, dtype, seed=514)).copy()
    else:
        rng = np.random.default_rng(514)
        a = rng.normal(size=(case.n, case.n))
        if np.issubdtype(np.dtype(dtype), np.complexfloating):
            a = a + 1j * rng.normal(size=a.shape)
        a = a / np.sqrt(case.n) + 4 * np.eye(case.n)
        # Force row interchanges across tiles without making A ill-conditioned.
        a = np.roll(a, case.n // 2, axis=0).astype(np.dtype(dtype))
    if interface == "singular":
        a[:] = 0
        np.fill_diagonal(a, 2)
        a[-1, -1] = 0
    replicated = layout_case == "replicated_input"
    a_dev = jax.device_put(a, NamedSharding(mesh, P() if replicated else specs))
    if interface == "shardmap":
        invert = jax.jit(jax.shard_map(
            partial(getattr(jaxmg, routine + "_shardmap_ctx"),
                    T_A=case.tile_size, matrix_specs=specs),
            mesh=mesh, in_specs=specs, out_specs=(specs, P(("pr", "pc"))),
        ), donate_argnums=(0,))
    elif interface == "context":
        invert = jax.jit(partial(
            getattr(jaxmg, routine + "_jit_ctx"), T_A=case.tile_size,
            mesh=mesh, matrix_specs=specs,
        ), donate_argnums=(0,))
    else:
        invert = partial(
            getattr(jaxmg, routine), T_A=case.tile_size, mesh=mesh,
            matrix_specs=specs, return_status=True, donate=False,
        )
    inverse, status = invert(a_dev)
    inverse.block_until_ready()
    words = native_status_words(status)
    size = _CUSOLVERMP_INVERSE_STATUS_SIZE
    if interface == "singular":
        assert_rank_failure_with_nan(
            words, size, (26, 27) if routine == "potri" else (38, 39), inverse,
        )
        assert np.all(words[25::size] == 0), words  # No inversion after failed factorization.
    else:
        rows = words.reshape(-1, size)
        assert np.all(rows[:, 0] == 0), rows
        assert np.all(rows[:, [21, 25, 31]] == 1), rows
        assert np.all(rows[:, [22, 26]] == 0), rows
        assert np.all(rows[:, [19, 20, 23, 24]] >= 0), rows
        assert np.all(rows[:, 30] == (routine == "potri")), rows
        got = global_array_to_numpy(inverse)
        single = np.dtype(dtype) in (np.dtype(np.float32), np.dtype(np.complex64))
        tol = 2e-4 if single else 1e-10
        reference = np.linalg.inv(a.astype(np.result_type(a.dtype, np.float64)))
        assert got.shape == a.shape and got.dtype == a.dtype
        assert np.linalg.norm(got - reference) / np.linalg.norm(reference) < tol
        for product in (a @ got, got @ a):
            assert np.linalg.norm(product - np.eye(case.n)) / np.sqrt(case.n) < tol
        if routine == "potri":
            np.testing.assert_allclose(got, got.conj().T, atol=tol, rtol=tol)
        if replicated:
            assert inverse.sharding.is_fully_replicated
        else:
            assert inverse.sharding.is_equivalent_to(NamedSharding(mesh, specs), 2)
    if interface in ("public", "singular"):
        np.testing.assert_array_equal(global_array_to_numpy(a_dev), a)
    multihost_utils.sync_global_devices(f"inverse_{case_name}_{interface}_complete")


try:
    run_case()
    emit("GPU_TEST_RESULT", {
        "proc": proc_id, "name": case_name, "dtype": dtype_name, "status": "ok",
    })
except Exception:
    emit("GPU_TEST_RESULT", {
        "proc": proc_id, "name": case_name, "dtype": dtype_name, "status": "fail",
        "traceback": traceback.format_exc(),
    })
finally:
    emit("GPU_TEST_SUMMARY", {"proc": proc_id, "name": case_name, "dtype": dtype_name})
