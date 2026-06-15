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

from tests.redistribution_helpers import (
    execute_padded_block_cyclic_2d_shardmap,
    required_padded_block_cyclic_2d_scratch_size,
)
from jaxmg._block_cyclic_2d_plan import (
    ProcessGrid,
    TileShape,
    build_edge_padding_compaction_plan,
)
from jaxmg._xla_comm_probe import cusolvermp_distributed_potrs_probe_shardmap

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


def _initial_edge_padded_host(host: np.ndarray, plan) -> np.ndarray:
    padding = plan.padding
    physical = np.zeros(
        (padding.physical_rows, padding.physical_cols),
        dtype=host.dtype,
    )
    for process_row in range(plan.grid.process_rows):
        for process_col in range(plan.grid.process_cols):
            row_start = process_row * padding.local_physical_rows
            col_start = process_col * padding.local_physical_cols
            physical[
                row_start : row_start + padding.local_logical_rows,
                col_start : col_start + padding.local_logical_cols,
            ] = host[
                process_row
                * padding.local_logical_rows : (process_row + 1)
                * padding.local_logical_rows,
                process_col
                * padding.local_logical_cols : (process_col + 1)
                * padding.local_logical_cols,
            ]
    return physical


def _diagonal_spd(n: int, dtype) -> np.ndarray:
    dtype = np.dtype(dtype)
    values = np.zeros((n, n), dtype=dtype)
    diagonal = np.arange(n + 1, 2 * n + 1, dtype=np.float64)
    if np.issubdtype(dtype, np.complexfloating):
        diagonal = diagonal.astype(dtype)
    values[np.diag_indices(n)] = diagonal
    return values


def _rhs(n: int, nrhs: int, dtype) -> np.ndarray:
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.complexfloating):
        value = np.array(1.0 - 1.0j, dtype=dtype)
    else:
        value = np.array(1.0, dtype=dtype)
    return np.full((n, nrhs), value, dtype=dtype)


def _redistribute(
    host: np.ndarray,
    *,
    mesh: Mesh,
    matrix_specs: P,
    scratch_specs: P,
    grid: ProcessGrid,
    tile_shape: TileShape,
):
    plan = build_edge_padding_compaction_plan(
        logical_rows=host.shape[0],
        logical_cols=host.shape[1],
        tile_shape=tile_shape,
        grid=grid,
    )
    physical = _initial_edge_padded_host(host, plan)
    scratch_per_rank = required_padded_block_cyclic_2d_scratch_size(
        logical_rows=host.shape[0],
        logical_cols=host.shape[1],
        grid=grid,
        tile_rows=tile_shape.rows,
        tile_cols=tile_shape.cols,
    )
    matrix = jax.device_put(
        jnp.asarray(physical),
        NamedSharding(mesh, matrix_specs),
    )
    scratch = jax.device_put(
        jnp.zeros((grid.num_processes * scratch_per_rank,), dtype=matrix.dtype),
        NamedSharding(mesh, scratch_specs),
    )
    redistributed, scratch_out = execute_padded_block_cyclic_2d_shardmap(
        matrix,
        scratch,
        mesh,
        matrix_specs,
        scratch_specs,
        logical_rows=host.shape[0],
        logical_cols=host.shape[1],
        grid=grid,
        tile_rows=tile_shape.rows,
        tile_cols=tile_shape.cols,
    )
    redistributed.block_until_ready()
    scratch_out.block_until_ready()
    return redistributed


def _run_distributed_potrs_case(
    *,
    grid: ProcessGrid,
    n: int,
    nrhs: int,
    tile_size: int,
    dtype,
):
    if len(jax.devices("gpu")) < grid.num_processes:
        pytest.skip(f"{grid.num_processes} GPUs are required. Skipping")

    devices = np.asarray(
        jax.devices("gpu")[: grid.num_processes],
        dtype=object,
    ).reshape(grid.process_rows, grid.process_cols)
    mesh = Mesh(devices, ("pr", "pc"))
    matrix_specs = P("pr", "pc")
    status_specs = _status_specs(grid)
    tile_shape = TileShape(tile_size, tile_size)

    a = _redistribute(
        _diagonal_spd(n, dtype),
        mesh=mesh,
        matrix_specs=matrix_specs,
        scratch_specs=status_specs,
        grid=grid,
        tile_shape=tile_shape,
    )
    b = _redistribute(
        _rhs(n, nrhs, dtype),
        mesh=mesh,
        matrix_specs=matrix_specs,
        scratch_specs=status_specs,
        grid=grid,
        tile_shape=tile_shape,
    )

    a_out, b_out, status = cusolvermp_distributed_potrs_probe_shardmap(
        a,
        b,
        mesh,
        matrix_specs,
        status_specs,
        process_rows=grid.process_rows,
        process_cols=grid.process_cols,
        n=n,
        nrhs=nrhs,
        tile_size=tile_size,
    )
    a_out.block_until_ready()
    b_out.block_until_ready()
    status.block_until_ready()

    statuses = np.asarray(status).reshape(grid.num_processes, 40)
    status_codes = set(statuses[:, 0].tolist())
    if status_codes == {1}:
        pytest.skip("libcusolverMp is not available on the loader path.")
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
    np.testing.assert_array_equal(statuses[:, 30], np.zeros(grid.num_processes))
    np.testing.assert_array_equal(statuses[:, 31], np.zeros(grid.num_processes))
    np.testing.assert_array_equal(statuses[:, 32], np.full(grid.num_processes, 1))
    np.testing.assert_array_equal(statuses[:, 36], np.full(grid.num_processes, nrhs))
    np.testing.assert_array_equal(statuses[:, 37], np.full(grid.num_processes, 1))
    np.testing.assert_array_equal(statuses[:, 38], np.full(grid.num_processes, 1))
    assert statuses[0, 33] <= 10, statuses


@pytest.mark.parametrize(
    "grid,n,nrhs,tile_size,dtype",
    [
        (ProcessGrid(1, 2), 8, 8, 4, jnp.float32),
        (ProcessGrid(1, 2), 16, 16, 4, jnp.float32),
        (ProcessGrid(2, 1), 8, 4, 4, jnp.float64),
        (ProcessGrid(2, 1), 16, 8, 4, jnp.float64),
        (ProcessGrid(2, 2), 8, 8, 4, jnp.float64),
        (ProcessGrid(2, 2), 16, 16, 4, jnp.float64),
        (ProcessGrid(1, 4), 16, 16, 4, jnp.complex64),
    ],
)
def test_cusolvermp_potrs_consumes_native_redistributed_buffers(
    grid,
    n,
    nrhs,
    tile_size,
    dtype,
):
    _run_distributed_potrs_case(
        grid=grid,
        n=n,
        nrhs=nrhs,
        tile_size=tile_size,
        dtype=dtype,
    )
