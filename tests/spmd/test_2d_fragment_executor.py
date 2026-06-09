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
    execute_fragment_transfer_batches_shardmap,
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


def _run_two_rank_executor_case(grid, host, expected, scratch_specs):
    devices = np.asarray(jax.devices("gpu")[:2], dtype=object).reshape(
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
    )
    out.block_until_ready()
    scratch_out.block_until_ready()

    np.testing.assert_array_equal(np.asarray(out), expected)


def test_column_owner_executor_reorders_two_process_columns():
    host = np.arange(32, dtype=np.float32).reshape(4, 8)
    expected = host[:, [0, 1, 4, 5, 2, 3, 6, 7]]

    _run_two_rank_executor_case(
        ProcessGrid(process_rows=1, process_cols=2),
        host,
        expected,
        P("pc"),
    )


def test_row_owner_executor_reorders_two_process_rows():
    host = np.arange(32, dtype=np.float32).reshape(8, 4)
    expected = host[[0, 1, 4, 5, 2, 3, 6, 7], :]

    _run_two_rank_executor_case(
        ProcessGrid(process_rows=2, process_cols=1),
        host,
        expected,
        P("pr"),
    )
