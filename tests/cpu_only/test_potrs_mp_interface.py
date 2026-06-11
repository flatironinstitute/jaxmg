import numpy as np

import jax
import jax.numpy as jnp
import pytest
from jax.sharding import Mesh, PartitionSpec as P

from jaxmg import potrs_mp


def test_potrs_mp_rejects_non_2d_process_grid_specs():
    mesh = Mesh(np.asarray(jax.devices()[:1], dtype=object), ("x",))
    a = jnp.eye(4, dtype=jnp.float32)
    b = jnp.ones((4, 4), dtype=jnp.float32)

    with pytest.raises(ValueError, match="requires both matrix axes"):
        potrs_mp(a, b, 2, mesh=mesh, matrix_specs=P("x", None))


def test_potrs_mp_rejects_non_square_a():
    mesh = Mesh(np.asarray(jax.devices()[:1], dtype=object).reshape(1, 1), ("pr", "pc"))
    a = jnp.ones((4, 2), dtype=jnp.float32)
    b = jnp.ones((4, 4), dtype=jnp.float32)

    with pytest.raises(ValueError, match="A to be square"):
        potrs_mp(a, b, 2, mesh=mesh, matrix_specs=P("pr", "pc"))


def test_potrs_mp_rejects_rhs_leading_dimension_mismatch():
    mesh = Mesh(np.asarray(jax.devices()[:1], dtype=object).reshape(1, 1), ("pr", "pc"))
    a = jnp.eye(4, dtype=jnp.float32)
    b = jnp.ones((2, 4), dtype=jnp.float32)

    with pytest.raises(ValueError, match="matching leading dimensions"):
        potrs_mp(a, b, 2, mesh=mesh, matrix_specs=P("pr", "pc"))
