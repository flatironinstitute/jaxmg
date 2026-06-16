import sys
import os

this_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(this_dir, "..")
sys.path.append(src_path)
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, NamedSharding
import pytest
from jaxmg import potri
from jaxmg.utils import random_psd
from functools import partial

devices = jax.devices()
import os
import sys

this_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(this_dir, "..")
sys.path.append(src_path)

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import pytest
from jaxmg import potri


# Basic environment sanity: require at least one device (CPU is OK)
devices = jax.devices()
ndev = len(devices)
pytestmark = pytest.mark.skipif(ndev == 0, reason="Requires at least one JAX device")

# mesh used for calls (shape uses ndev so axis name 'x' is valid)
mesh = jax.make_mesh((ndev,), ("x",))


def test_potri_assert_a_2d():
    """Passing a 1D array should raise the assertion that `a` must be 2D.

    Use a valid `in_specs` so validation reaches the `a.ndim == 2` assertion
    and does not fail earlier.
    """
    A = jnp.ones((4,))
    T_A = 1
    # Valid PartitionSpec for potri is P(<axis_name>, None)
    with pytest.raises(AssertionError, match="a must be a 2D array"):
        potri(A, T_A=T_A, mesh=mesh, in_specs=P("x", None))


def test_potri_in_specs_length_valueerror():
    """Passing a tuple/list with length != 1 should raise ValueError."""
    A = jnp.eye(4)
    T_A = 1
    with pytest.raises(ValueError, match="in_specs must be a single PartitionSpec"):
        potri(A, T_A=T_A, mesh=mesh, in_specs=(P("x", None), P("x", None)))


def test_potri_in_specs_typeerror():
    """Non-PartitionSpec `in_specs` should raise a TypeError."""
    A = jnp.eye(4)
    T_A = 1
    with pytest.raises(TypeError, match="in_specs must be a PartitionSpec"):
        potri(A, T_A=T_A, mesh=mesh, in_specs="not_a_partitionspec")


def test_potri_in_specs_valueerror_wrong_partition():
    """A PartitionSpec with incorrect structure (first None or second not-None)
    should raise a ValueError explaining the expected PartitionSpec format.
    """
    A = jnp.eye(4)
    T_A = 1
    # second entry not None -> invalid
    with pytest.raises(ValueError, match="sharded along the rows with PartitionSpec P\(str, None\)"):
        potri(A, T_A=T_A, mesh=mesh, in_specs=P(None, "x"))
