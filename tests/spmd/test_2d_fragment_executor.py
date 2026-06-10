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
    execute_tile_aligned_native_2d_plan_shardmap,
    execute_fragment_transfer_batches_shardmap,
    required_native_2d_plan_scratch_size,
    required_rect_transfer_scratch_size,
)
from jaxmg._block_cyclic_2d_plan import (
    ProcessGrid,
    TileShape,
    batch_executable_fragment_transfers,
    build_executable_fragment_transfer_schedule,
    build_two_phase_2d_plan,
)

platforms = {d.platform for d in jax.devices()}
if "gpu" not in platforms:
    pytest.skip("No GPUs found. Skipping", allow_module_level=True)
if len(jax.devices("gpu")) < 2:
    pytest.skip("At least two GPUs are required. Skipping", allow_module_level=True)


def _expected_from_batches(host, grid, batches):
    local_rows = host.shape[0] // grid.process_rows
    local_cols = host.shape[1] // grid.process_cols
    buffers = []
    for process_row in range(grid.process_rows):
        for process_col in range(grid.process_cols):
            buffers.append(
                host[
                    process_row * local_rows : (process_row + 1) * local_rows,
                    process_col * local_cols : (process_col + 1) * local_cols,
                ].copy()
            )

    for batch in batches:
        payloads = []
        for transfer in batch.transfers:
            source = transfer.source_rect
            payloads.append(
                (
                    transfer,
                    buffers[transfer.source_rank][
                        source.row_start : source.row_stop,
                        source.col_start : source.col_stop,
                    ].copy(),
                )
            )
        for transfer, payload in payloads:
            target = transfer.target_rect
            buffers[transfer.target_rank][
                target.row_start : target.row_stop,
                target.col_start : target.col_stop,
            ] = payload

    expected = np.empty_like(host)
    for process_row in range(grid.process_rows):
        for process_col in range(grid.process_cols):
            rank = grid.rank(process_row, process_col)
            expected[
                process_row * local_rows : (process_row + 1) * local_rows,
                process_col * local_cols : (process_col + 1) * local_cols,
            ] = buffers[rank]
    return expected


def _expected_block_cyclic_layout(host, grid, tile_shape):
    local_rows = host.shape[0] // grid.process_rows
    local_cols = host.shape[1] // grid.process_cols
    expected = np.empty_like(host)

    for global_row in range(host.shape[0]):
        tile_row = global_row // tile_shape.rows
        row_in_tile = global_row % tile_shape.rows
        process_row = tile_row % grid.process_rows
        local_row = (tile_row // grid.process_rows) * tile_shape.rows + row_in_tile

        for global_col in range(host.shape[1]):
            tile_col = global_col // tile_shape.cols
            col_in_tile = global_col % tile_shape.cols
            process_col = tile_col % grid.process_cols
            local_col = (
                (tile_col // grid.process_cols) * tile_shape.cols + col_in_tile
            )

            expected[
                process_row * local_rows + local_row,
                process_col * local_cols + local_col,
            ] = host[global_row, global_col]

    return expected


def _run_executor_case(grid, host, scratch_specs, *, layout="row_major"):
    if len(jax.devices("gpu")) < grid.num_processes:
        pytest.skip(f"{grid.num_processes} GPUs are required. Skipping")

    devices = np.asarray(jax.devices("gpu")[: grid.num_processes], dtype=object).reshape(
        grid.process_rows, grid.process_cols
    )
    mesh = Mesh(devices, ("pr", "pc"))
    plan = build_two_phase_2d_plan(
        logical_rows=host.shape[0],
        logical_cols=host.shape[1],
        tile_shape=TileShape(rows=2, cols=2),
        grid=grid,
    )
    batches = batch_executable_fragment_transfers(
        build_executable_fragment_transfer_schedule(plan)
    )
    expected = _expected_from_batches(host, grid, batches)
    scratch_per_rank = required_rect_transfer_scratch_size(batches)

    matrix = jax.device_put(
        jnp.asarray(host),
        NamedSharding(mesh, P("pr", "pc")),
    )
    scratch = jax.device_put(
        jnp.full((grid.num_processes * scratch_per_rank,), -1, dtype=jnp.float32),
        NamedSharding(mesh, scratch_specs),
    )

    out, scratch_out = execute_fragment_transfer_batches_shardmap(
        matrix,
        scratch,
        mesh,
        P("pr", "pc"),
        scratch_specs,
        batches,
        grid=grid,
        layout=layout,
    )
    out.block_until_ready()
    scratch_out.block_until_ready()

    np.testing.assert_array_equal(np.asarray(out), expected)


def _run_native_executor_case(grid, host, scratch_specs, *, layout="row_major"):
    if len(jax.devices("gpu")) < grid.num_processes:
        pytest.skip(f"{grid.num_processes} GPUs are required. Skipping")

    devices = np.asarray(jax.devices("gpu")[: grid.num_processes], dtype=object).reshape(
        grid.process_rows, grid.process_cols
    )
    mesh = Mesh(devices, ("pr", "pc"))
    tile_shape = TileShape(rows=2, cols=2)
    expected = _expected_block_cyclic_layout(host, grid, tile_shape)
    scratch_per_rank = required_native_2d_plan_scratch_size(
        local_rows=host.shape[0] // grid.process_rows,
        local_cols=host.shape[1] // grid.process_cols,
        tile_rows=tile_shape.rows,
        tile_cols=tile_shape.cols,
    )

    matrix = jax.device_put(
        jnp.asarray(host),
        NamedSharding(mesh, P("pr", "pc")),
    )
    scratch = jax.device_put(
        jnp.full((grid.num_processes * scratch_per_rank,), -1, dtype=jnp.float32),
        NamedSharding(mesh, scratch_specs),
    )

    out, scratch_out = execute_tile_aligned_native_2d_plan_shardmap(
        matrix,
        scratch,
        mesh,
        P("pr", "pc"),
        scratch_specs,
        grid=grid,
        tile_rows=tile_shape.rows,
        tile_cols=tile_shape.cols,
        layout=layout,
    )
    out.block_until_ready()
    scratch_out.block_until_ready()

    np.testing.assert_array_equal(np.asarray(out), expected)


def test_column_owner_executor_reorders_two_process_columns():
    host = np.arange(32, dtype=np.float32).reshape(4, 8)

    _run_executor_case(
        ProcessGrid(process_rows=1, process_cols=2),
        host,
        P("pc"),
    )


def test_row_owner_executor_reorders_two_process_rows():
    host = np.arange(32, dtype=np.float32).reshape(8, 4)

    _run_executor_case(
        ProcessGrid(process_rows=2, process_cols=1),
        host,
        P("pr"),
    )


def test_two_by_two_executor_runs_column_then_row_phases():
    host = np.arange(64, dtype=np.float32).reshape(8, 8)

    _run_executor_case(
        ProcessGrid(process_rows=2, process_cols=2),
        host,
        P(("pr", "pc")),
    )


def test_two_by_two_executor_runs_column_major_layout():
    host = np.arange(64, dtype=np.float32).reshape(8, 8)

    _run_executor_case(
        ProcessGrid(process_rows=2, process_cols=2),
        host,
        P(("pr", "pc")),
        layout="column_major",
    )


def test_native_two_by_two_executor_runs_row_major_layout():
    host = np.arange(64, dtype=np.float32).reshape(8, 8)

    _run_native_executor_case(
        ProcessGrid(process_rows=2, process_cols=2),
        host,
        P(("pr", "pc")),
    )


def test_native_two_by_two_executor_runs_column_major_layout():
    host = np.arange(64, dtype=np.float32).reshape(8, 8)

    _run_native_executor_case(
        ProcessGrid(process_rows=2, process_cols=2),
        host,
        P(("pr", "pc")),
        layout="column_major",
    )


def test_native_two_by_two_executor_runs_larger_slab_cycles():
    host = np.arange(256, dtype=np.float32).reshape(16, 16)

    _run_native_executor_case(
        ProcessGrid(process_rows=2, process_cols=2),
        host,
        P(("pr", "pc")),
    )


def test_native_two_by_two_executor_runs_larger_column_major_slab_cycles():
    host = np.arange(256, dtype=np.float32).reshape(16, 16)

    _run_native_executor_case(
        ProcessGrid(process_rows=2, process_cols=2),
        host,
        P(("pr", "pc")),
        layout="column_major",
    )
