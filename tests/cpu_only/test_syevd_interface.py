import sys
import os

this_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(this_dir, "..")

import pytest
import jax

# Setup JAX
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jaxmg import syevd
from jaxmg.utils import JaxMgWarning

# These tests exercise the argument-validation paths in `syevd` and avoid
# invoking the native FFI. They match the current implementation in
# `src/jaxmg/_syevd.py` (normalization of `in_specs`, type/length checks,
# PartitionSpec shape checks, and T_A bounds).


def mesh_for_tests():
    return jax.make_mesh((jax.local_device_count(),), ("x",))


def test_in_specs_wrong_length_raises_valueerror():
    a = jnp.eye(4)
    T_A = 32
    mesh = mesh_for_tests()
    # Passing a tuple of length != 1 should raise ValueError per implementation
    with pytest.raises(ValueError, match="in_specs must be a single PartitionSpec or a 1-element list/tuple"):
        syevd(a, T_A, mesh, (P(None, "x"), P(None, "x")))


def test_in_specs_non_partitionspec_raises_typeerror():
    a = jnp.eye(4)
    T_A = 32
    mesh = mesh_for_tests()
    with pytest.raises(TypeError, match="in_specs must be a PartitionSpec or a 1-element list/tuple containing one"):
        syevd(a, T_A, mesh, 12345)


def test_spec_a_invalid_partition_raises_valueerror():
    a = jnp.eye(4)
    T_A = 32
    mesh = mesh_for_tests()
    # invalid when second partition is not None or first partition is None
    bad_specs = [P(None, "x"), P(None, None), P("x", "y")]
    for spec in bad_specs:
        with pytest.raises(ValueError, match="A must be sharded along the rows with PartitionSpec P\(str, None\)"):
            syevd(a, T_A, mesh, (spec,))


def test_a_not_2d_raises_assertion():
    # Use a valid PartitionSpec so we reach the a.ndim check
    a = jnp.ones((3,))
    T_A = 32
    mesh = mesh_for_tests()
    with pytest.raises(AssertionError, match="a must be a 2D array"):
        syevd(a, T_A, mesh, (P("x", None),))


def test_T_A_too_large_raises_valueerror():
    a = jnp.eye(4)
    T_A = 2048
    mesh = mesh_for_tests()
    with pytest.warns(JaxMgWarning, match="T_A has a maximum value of 1024"):
        syevd(a, T_A, mesh, (P("x", None),))