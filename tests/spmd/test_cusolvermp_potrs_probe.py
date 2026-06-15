import os
import sys

import numpy as np

this_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(this_dir, "..")
sys.path.append(src_path)

from jax import config

config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from tests.reference_block_cyclic_2d_plan import ProcessGrid
from tests.diagnostics.xla_comm_probe import cusolvermp_potrs_probe_shardmap

platforms = {d.platform for d in jax.devices()}
if "gpu" not in platforms:
    pytest.skip("No GPUs found. Skipping", allow_module_level=True)
if len(jax.devices("gpu")) < 2:
    pytest.skip("At least two GPUs are required. Skipping", allow_module_level=True)


def _status_specs(grid: ProcessGrid) -> P:
    if grid.process_rows == 1:
        return P("pc")
    if grid.process_cols == 1:
        return P("pr")
    return P(("pr", "pc"))


def _local_numroc(n: int, block_size: int, process: int, process_count: int) -> int:
    owned = 0
    for tile in range((n + block_size - 1) // block_size):
        if tile % process_count != process:
            continue
        start = tile * block_size
        owned += min(block_size, n - start)
    return owned


def _run_potrs_probe_case(*, grid: ProcessGrid, n: int, tile_size: int, dtype):
    if len(jax.devices("gpu")) < grid.num_processes:
        pytest.skip(f"{grid.num_processes} GPUs are required. Skipping")

    # These probe cases deliberately use dimensions where every process row and
    # process column receives the same local capacity. That keeps the JAX array
    # sharding regular while still exercising nontrivial cuSOLVERMp grids.
    local_rows = _local_numroc(n, tile_size, 0, grid.process_rows)
    local_cols_a = _local_numroc(n, tile_size, 0, grid.process_cols)
    a_shape = (
        local_rows * grid.process_rows,
        local_cols_a * grid.process_cols,
    )
    b_shape = (
        local_rows * grid.process_rows,
        grid.process_cols,
    )

    devices = np.asarray(
        jax.devices("gpu")[: grid.num_processes],
        dtype=object,
    ).reshape(grid.process_rows, grid.process_cols)
    mesh = Mesh(devices, ("pr", "pc"))
    matrix_specs = P("pr", "pc")
    status_specs = _status_specs(grid)

    a = jax.device_put(
        jnp.zeros(a_shape, dtype=dtype),
        NamedSharding(mesh, matrix_specs),
    )
    b = jax.device_put(
        jnp.zeros(b_shape, dtype=dtype),
        NamedSharding(mesh, matrix_specs),
    )

    a_out, b_out, status = cusolvermp_potrs_probe_shardmap(
        a,
        b,
        mesh,
        matrix_specs,
        status_specs,
        process_rows=grid.process_rows,
        process_cols=grid.process_cols,
        n=n,
        tile_size=tile_size,
    )
    a_out.block_until_ready()
    b_out.block_until_ready()
    status.block_until_ready()

    statuses = np.asarray(status).reshape(grid.num_processes, 40)
    status_codes = set(statuses[:, 0].tolist())
    if status_codes == {1}:
        pytest.skip("libcusolverMp is not available on the loader path.")
    if status_codes == {17}:
        pytest.skip("cusolverMpMatrixScatterH2D is not available in this SDK.")
    if status_codes == {21}:
        pytest.skip("cuSOLVERMp potrf/potrs symbols are not available.")
    if status_codes == {30}:
        pytest.skip("cusolverMpMatrixGatherD2H is not available in this SDK.")

    assert status_codes == {0}, statuses
    np.testing.assert_array_equal(statuses[:, 2], np.arange(grid.num_processes))
    np.testing.assert_array_equal(
        statuses[:, 3],
        np.full(grid.num_processes, grid.num_processes),
    )
    np.testing.assert_array_equal(
        statuses[:, 4],
        np.full(grid.num_processes, grid.process_rows),
    )
    np.testing.assert_array_equal(
        statuses[:, 5],
        np.full(grid.num_processes, grid.process_cols),
    )
    np.testing.assert_array_equal(statuses[:, 26], np.full(grid.num_processes, 1))
    np.testing.assert_array_equal(statuses[:, 27], np.zeros(grid.num_processes))
    np.testing.assert_array_equal(statuses[:, 28], np.full(grid.num_processes, 1))
    np.testing.assert_array_equal(statuses[:, 29], np.zeros(grid.num_processes))
    np.testing.assert_array_equal(statuses[:, 30], np.full(grid.num_processes, 1))
    np.testing.assert_array_equal(statuses[:, 31], np.full(grid.num_processes, 1))
    np.testing.assert_array_equal(statuses[:, 32], np.full(grid.num_processes, 1))
    assert statuses[0, 33] <= 10, statuses


@pytest.mark.parametrize(
    "grid,n,tile_size,dtype",
    [
        (ProcessGrid(1, 2), 8, 4, jnp.float64),
        (ProcessGrid(2, 1), 8, 4, jnp.float32),
        (ProcessGrid(2, 2), 8, 4, jnp.float64),
        (ProcessGrid(1, 4), 16, 4, jnp.float32),
    ],
)
def test_cusolvermp_potrs_probe_runs_with_borrowed_xla_communicator(
    grid,
    n,
    tile_size,
    dtype,
):
    _run_potrs_probe_case(
        grid=grid,
        n=n,
        tile_size=tile_size,
        dtype=dtype,
    )
