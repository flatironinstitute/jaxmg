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

from jaxmg._block_cyclic_2d_execute import (
    execute_padded_block_cyclic_2d_shardmap,
    required_padded_block_cyclic_2d_scratch_size,
)
from jaxmg._block_cyclic_2d_plan import (
    ProcessGrid,
    TileShape,
    build_edge_padding_compaction_plan,
)
from jaxmg._xla_comm_probe import cusolvermp_scatter_layout_probe_shardmap

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


def _source_host(rows: int, cols: int, dtype) -> np.ndarray:
    real = np.arange(1, rows * cols + 1, dtype=np.float64).reshape(rows, cols)
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.complexfloating):
        values = real - 1j * real
    else:
        values = real
    return values.astype(dtype)


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


def _logical_block_cyclic_mask(
    logical_rows: int,
    logical_cols: int,
    physical_shape: tuple[int, int],
    grid: ProcessGrid,
    tile_shape: TileShape,
) -> np.ndarray:
    local_rows = physical_shape[0] // grid.process_rows
    local_cols = physical_shape[1] // grid.process_cols
    mask = np.zeros(physical_shape, dtype=bool)

    for global_row in range(logical_rows):
        tile_row = global_row // tile_shape.rows
        row_in_tile = global_row % tile_shape.rows
        process_row = tile_row % grid.process_rows
        local_row = (tile_row // grid.process_rows) * tile_shape.rows + row_in_tile

        for global_col in range(logical_cols):
            tile_col = global_col // tile_shape.cols
            col_in_tile = global_col % tile_shape.cols
            process_col = tile_col % grid.process_cols
            local_col = (
                (tile_col // grid.process_cols) * tile_shape.cols + col_in_tile
            )
            mask[
                process_row * local_rows + local_row,
                process_col * local_cols + local_col,
            ] = True

    return mask


def _assert_scatter_status(status, grid: ProcessGrid):
    statuses = np.asarray(status).reshape(grid.num_processes, 24)
    status_codes = set(statuses[:, 0].tolist())
    if status_codes == {1}:
        pytest.skip("libcusolverMp is not available on the loader path.")
    if status_codes == {17}:
        pytest.skip("cusolverMpMatrixScatterH2D is not available in this SDK.")

    assert status_codes == {0}, statuses
    np.testing.assert_array_equal(
        statuses[:, 2],
        np.arange(grid.num_processes),
    )
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
    np.testing.assert_array_equal(statuses[:, 22], np.full(grid.num_processes, 1))


def _run_scatter_oracle_case(
    *,
    grid: ProcessGrid,
    logical_shape: tuple[int, int],
    tile_shape: TileShape,
    dtype,
):
    if len(jax.devices("gpu")) < grid.num_processes:
        pytest.skip(f"{grid.num_processes} GPUs are required. Skipping")

    devices = np.asarray(
        jax.devices("gpu")[: grid.num_processes],
        dtype=object,
    ).reshape(grid.process_rows, grid.process_cols)
    mesh = Mesh(devices, ("pr", "pc"))
    host = _source_host(*logical_shape, dtype=dtype)
    compaction_plan = build_edge_padding_compaction_plan(
        logical_rows=host.shape[0],
        logical_cols=host.shape[1],
        tile_shape=tile_shape,
        grid=grid,
    )
    physical = _initial_edge_padded_host(host, compaction_plan)

    scratch_per_rank = required_padded_block_cyclic_2d_scratch_size(
        logical_rows=host.shape[0],
        logical_cols=host.shape[1],
        grid=grid,
        tile_rows=tile_shape.rows,
        tile_cols=tile_shape.cols,
    )
    matrix_specs = P("pr", "pc")
    status_specs = _status_specs(grid)
    matrix = jax.device_put(
        jnp.asarray(physical),
        NamedSharding(mesh, matrix_specs),
    )
    scratch = jax.device_put(
        jnp.zeros((grid.num_processes * scratch_per_rank,), dtype=matrix.dtype),
        NamedSharding(mesh, status_specs),
    )

    redistributed, scratch_out = execute_padded_block_cyclic_2d_shardmap(
        matrix,
        scratch,
        mesh,
        matrix_specs,
        status_specs,
        logical_rows=host.shape[0],
        logical_cols=host.shape[1],
        grid=grid,
        tile_rows=tile_shape.rows,
        tile_cols=tile_shape.cols,
    )
    redistributed.block_until_ready()
    scratch_out.block_until_ready()

    oracle_input = jax.device_put(
        jnp.zeros_like(jnp.asarray(physical)),
        NamedSharding(mesh, matrix_specs),
    )
    oracle, status = cusolvermp_scatter_layout_probe_shardmap(
        oracle_input,
        mesh,
        matrix_specs,
        status_specs,
        process_rows=grid.process_rows,
        process_cols=grid.process_cols,
        logical_rows=host.shape[0],
        logical_cols=host.shape[1],
        tile_rows=tile_shape.rows,
        tile_cols=tile_shape.cols,
    )
    oracle.block_until_ready()
    status.block_until_ready()
    _assert_scatter_status(status, grid)

    mask = _logical_block_cyclic_mask(
        logical_rows=host.shape[0],
        logical_cols=host.shape[1],
        physical_shape=physical.shape,
        grid=grid,
        tile_shape=tile_shape,
    )
    np.testing.assert_array_equal(
        np.asarray(redistributed)[mask],
        np.asarray(oracle)[mask],
    )


@pytest.mark.parametrize(
    "grid,logical_shape,tile_shape,dtype",
    [
        (ProcessGrid(1, 2), (10, 10), TileShape(4, 4), np.float32),
        (ProcessGrid(2, 1), (10, 10), TileShape(4, 4), np.float64),
        (ProcessGrid(2, 2), (10, 10), TileShape(4, 4), np.complex64),
        (ProcessGrid(2, 2), (16, 16), TileShape(4, 4), np.float32),
        (ProcessGrid(1, 3), (12, 15), TileShape(4, 4), np.float32),
        (ProcessGrid(3, 1), (15, 12), TileShape(4, 4), np.float64),
        (ProcessGrid(1, 4), (12, 12), TileShape(4, 4), np.complex128),
        (ProcessGrid(4, 1), (12, 12), TileShape(4, 4), np.float32),
    ],
)
def test_padded_redistribution_matches_cusolvermp_scatter_oracle(
    grid,
    logical_shape,
    tile_shape,
    dtype,
):
    _run_scatter_oracle_case(
        grid=grid,
        logical_shape=logical_shape,
        tile_shape=tile_shape,
        dtype=dtype,
    )
