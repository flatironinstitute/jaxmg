import os
import sys
from functools import partial

import numpy as np

this_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(this_dir, "..")
sys.path.append(src_path)

import jax
import jax.numpy as jnp
import pytest

from jaxmg._xla_comm_probe import xla_rect_pack_unpack_probe

platforms = {d.platform for d in jax.devices()}
if "gpu" not in platforms:
    pytest.skip("No GPUs found. Skipping", allow_module_level=True)


def test_rect_pack_unpack_probe_float32():
    matrix = jnp.arange(20, dtype=jnp.float32).reshape(4, 5)
    scratch = jnp.full((4,), -1, dtype=jnp.float32)

    fn = jax.jit(
        partial(
            xla_rect_pack_unpack_probe,
            row_start=2,
            col_start=3,
            row_count=2,
            col_count=2,
            target_row=0,
            target_col=1,
        ),
        donate_argnums=(0, 1),
    )
    out, packed = fn(matrix, scratch)
    out.block_until_ready()
    packed.block_until_ready()

    expected_packed = np.array([13, 14, 18, 19], dtype=np.float32)
    expected_matrix = np.arange(20, dtype=np.float32).reshape(4, 5)
    expected_matrix[0:2, 1:3] = expected_packed.reshape(2, 2)

    np.testing.assert_array_equal(np.asarray(packed), expected_packed)
    np.testing.assert_array_equal(np.asarray(out), expected_matrix)
