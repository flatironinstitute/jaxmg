import os
import sys

import numpy as np

this_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(this_dir, "..")
sys.path.append(src_path)

import jax
import jax.numpy as jnp
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from tests.diagnostics.xla_comm_probe import cusolvermp_init_probe_shardmap

platforms = {d.platform for d in jax.devices()}
if "gpu" not in platforms:
    pytest.skip("No GPUs found. Skipping", allow_module_level=True)
if len(jax.devices("gpu")) < 2:
    pytest.skip("At least two GPUs are required. Skipping", allow_module_level=True)


def test_cusolvermp_init_probe_reports_library_or_success():
    devices = np.asarray(jax.devices("gpu")[:2], dtype=object)
    mesh = Mesh(devices, ("x",))
    token = jax.device_put(
        jnp.arange(2, dtype=jnp.uint32),
        NamedSharding(mesh, P("x")),
    )

    out = cusolvermp_init_probe_shardmap(
        token,
        mesh,
        P("x"),
        process_rows=1,
        process_cols=2,
        matrix_rows=8,
        matrix_cols=8,
        tile_rows=4,
        tile_cols=4,
    )
    out.block_until_ready()

    # Sharding the rank-local (16,) status vector over P("x") yields one
    # contiguous 16-entry segment per participating rank.
    statuses = np.asarray(out).reshape(2, 16)
    status_codes = set(statuses[:, 0].tolist())

    # Ordinary CUDA/cuSolver installs do not always provide cuSOLVERMp, so
    # status 1 is a valid diagnostic result. A system with libcusolverMp on the
    # loader path should return 0 on every rank.
    assert status_codes in ({0}, {1})

    if status_codes == {0}:
        np.testing.assert_array_equal(statuses[:, 2], np.arange(2))
        np.testing.assert_array_equal(statuses[:, 3], np.full(2, 2))
        np.testing.assert_array_equal(statuses[:, 4], np.full(2, 1))
        np.testing.assert_array_equal(statuses[:, 5], np.full(2, 2))
        np.testing.assert_array_equal(statuses[:, 7], np.full(2, 1))
        np.testing.assert_array_equal(statuses[:, 8], np.full(2, 1))
        np.testing.assert_array_equal(statuses[:, 9], np.full(2, 1))
        np.testing.assert_array_equal(statuses[:, 10], np.full(2, 1))
