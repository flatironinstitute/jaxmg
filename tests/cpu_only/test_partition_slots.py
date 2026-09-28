import json
import os
import subprocess
import sys

import pytest

from jax.sharding import AbstractMesh, PartitionSpec as P

from jaxmg._cusolvermp_layout import (
    partition_slots_from_mesh,
    validate_2d_matrix_specs,
)


def _slots(mesh, specs):
    row_axis, col_axis, grid = validate_2d_matrix_specs(mesh, specs)
    return partition_slots_from_mesh(
        mesh, row_axis=row_axis, col_axis=col_axis, grid=grid, caller="test"
    )


_AXIS_INDEX_SCRIPT = r"""
import json, sys
import numpy as np
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jaxmg._cusolvermp_layout import partition_slots_from_mesh, validate_2d_matrix_specs

order, shape, names, specs = json.loads(sys.argv[1])
specs = P(*specs)
mesh = Mesh(np.asarray(jax.devices())[order].reshape(shape), tuple(names))
row_axis, col_axis, grid = validate_2d_matrix_specs(mesh, specs)
slots = partition_slots_from_mesh(
    mesh, row_axis=row_axis, col_axis=col_axis, grid=grid, caller="test"
)

def body(x):
    row = 0 if row_axis is None else jax.lax.axis_index(row_axis)
    col = 0 if col_axis is None else jax.lax.axis_index(col_axis)
    return x * 0 + row * grid.process_cols + col

x = jax.device_put(np.zeros((12, 12)), NamedSharding(mesh, specs))
out = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=specs, out_specs=specs))(x)
flat = list(mesh.devices.flat)
for shard in out.addressable_shards:
    p = flat.index(shard.device)
    assert int(shard.data[0, 0]) == slots[p], (p, shard.device, slots)
print("ok")
"""


@pytest.mark.parametrize(
    "order, shape, names, specs",
    [
        ([3, 1, 0, 5, 2, 4], (2, 3), ("r", "c"), ("r", "c")),
        ([3, 1, 0, 5, 2, 4], (2, 3), ("r", "c"), ("c", "r")),
        ([5, 4, 3, 2, 1, 0], (6,), ("x",), ("x", None)),
        ([0, 2, 4, 1, 3, 5], (6,), ("x",), (None, "x")),
    ],
)
def test_partition_slots_match_axis_index_on_devices(order, shape, names, specs):
    """The slot of partition p is where jax.lax.axis_index puts the shard XLA
    runs as partition p, on `mesh.devices.flat[p]`, for any device order."""
    env = dict(
        os.environ,
        XLA_FLAGS="--xla_force_host_platform_device_count=6",
        JAX_PLATFORMS="cpu",
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _AXIS_INDEX_SCRIPT,
            json.dumps([order, shape, names, specs]),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")


def test_partition_slots_row_and_column_major():
    mesh = AbstractMesh((2, 3), ("r", "c"))
    # Partitions are laid out row-major over the mesh, so P("r", "c") gives the
    # identity, and the transposed matrix sharding a column-major order.
    assert _slots(mesh, P("r", "c")) == (0, 1, 2, 3, 4, 5)
    assert _slots(mesh, P("c", "r")) == (0, 2, 4, 1, 3, 5)


def test_partition_slots_rejects_extra_mesh_axes():
    mesh = AbstractMesh((2, 3), ("r", "c"))
    with pytest.raises(ValueError, match="exactly the axes"):
        _slots(mesh, P("r", None))
