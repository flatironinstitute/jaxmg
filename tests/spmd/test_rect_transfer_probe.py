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

from jaxmg._xla_comm_probe import xla_rect_transfer_probe_shardmap

platforms = {d.platform for d in jax.devices()}
if "gpu" not in platforms:
    pytest.skip("No GPUs found. Skipping", allow_module_level=True)
if len(jax.devices("gpu")) < 2:
    pytest.skip("At least two GPUs are required. Skipping", allow_module_level=True)


def _run_two_rank_rectangle_transfer(layout, packed_rank0, packed_rank1):
    devices = np.asarray(jax.devices("gpu")[:2], dtype=object)
    mesh = Mesh(devices, ("x",))

    host = np.stack(
        [
            np.arange(20, dtype=np.float32).reshape(4, 5),
            100 + np.arange(20, dtype=np.float32).reshape(4, 5),
        ]
    )
    matrix = jax.device_put(jnp.asarray(host), NamedSharding(mesh, P("x", None, None)))
    scratch = jax.device_put(
        jnp.full((2, 8), -1, dtype=jnp.float32),
        NamedSharding(mesh, P("x", None)),
    )

    out, scratch_out = xla_rect_transfer_probe_shardmap(
        matrix,
        scratch,
        mesh,
        P("x", None, None),
        P("x", None),
        layout=layout,
        targets=[1, 0],
        src_row_starts=[1, 2],
        src_col_starts=[2, 3],
        dst_row_starts=[0, 0],
        dst_col_starts=[0, 1],
        row_count=2,
        col_count=2,
    )
    out.block_until_ready()
    scratch_out.block_until_ready()

    expected = host.copy()
    expected[1, 0:2, 0:2] = host[0, 1:3, 2:4]
    expected[0, 0:2, 1:3] = host[1, 2:4, 3:5]

    expected_scratch = np.full((2, 8), -1, dtype=np.float32)
    expected_scratch[0, 0:4] = packed_rank0
    expected_scratch[0, 4:8] = packed_rank1
    expected_scratch[1, 0:4] = packed_rank1
    expected_scratch[1, 4:8] = packed_rank0

    np.testing.assert_array_equal(np.asarray(out), expected)
    np.testing.assert_array_equal(np.asarray(scratch_out), expected_scratch)


def test_rect_transfer_probe_row_major_float32():
    _run_two_rank_rectangle_transfer(
        "row_major",
        packed_rank0=np.array([7, 8, 12, 13], dtype=np.float32),
        packed_rank1=np.array([113, 114, 118, 119], dtype=np.float32),
    )


def test_rect_transfer_probe_column_major_float32():
    _run_two_rank_rectangle_transfer(
        "column_major",
        packed_rank0=np.array([7, 12, 8, 13], dtype=np.float32),
        packed_rank1=np.array([113, 118, 114, 119], dtype=np.float32),
    )
