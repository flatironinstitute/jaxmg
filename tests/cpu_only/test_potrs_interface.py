import os
import pytest
import jax

# Setup JAX
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jaxmg import potrs


def test_spec_a_valueerror_when_first_none_or_second_not_none():
    A = jnp.eye(4)
    b = jnp.ones((4, 1))
    T_A = 1
    mesh = jax.make_mesh((jax.local_device_count(),), ("x",))
    # First entry None -> invalid
    with pytest.raises(ValueError, match="A must be sharded along the columns"):
        potrs(A, b, T_A, mesh=mesh, in_specs=(P(None, "x"),))
    # Second entry not None -> invalid
    with pytest.raises(ValueError, match="A must be sharded along the columns"):
        potrs(A, b, T_A, mesh=mesh, in_specs=(P("x", "y"), ))


def test_shape_mismatch_raises_assertion():
    A = jnp.eye(4)
    b = jnp.ones((5, 1))
    T_A = 1
    mesh = jax.make_mesh((jax.local_device_count(),), ("x",))
    with pytest.raises(AssertionError, match="A and b must have the same number of columns"):
        potrs(A, b, T_A, mesh=mesh, in_specs=(P("x", None), ))


def test_a_1d_raises_indexerror():
    # Due to the current implementation ordering, passing a 1D `a` raises an
    # IndexError when the code attempts to access a.shape[1]. This documents
    # the current behavior (should be tightened in future).
    A = jnp.ones((4,))
    b = jnp.ones((4, 1))
    T_A = 1
    mesh = jax.make_mesh((jax.local_device_count(),), ("x",))
    with pytest.raises(IndexError):
        potrs(A, b, T_A, mesh=mesh, in_specs=(P("x", None), ))


def test_b_1d_raises_assertion():
    A = jnp.eye(4)
    b = jnp.ones((4,3,1))
    T_A = 1
    mesh = jax.make_mesh((jax.local_device_count(),), ("x",))
    with pytest.raises(AssertionError, match="b must be a 1D or 2D array."):
        potrs(A, b, T_A, mesh=mesh, in_specs=(P("x", None), ))
