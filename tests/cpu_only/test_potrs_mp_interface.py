import numpy as np

import jax
import jax.numpy as jnp
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from jaxmg import potrs_mp


def test_potrs_mp_rejects_non_2d_process_grid_specs():
    mesh = Mesh(np.asarray(jax.devices()[:1], dtype=object), ("x",))
    a = jnp.eye(4, dtype=jnp.float32)
    b = jnp.ones((4, 4), dtype=jnp.float32)

    with pytest.raises(ValueError, match="requires both matrix axes"):
        potrs_mp(a, b, 2, mesh=mesh, matrix_specs=P("x", None))


def test_potrs_mp_infers_mesh_and_specs_from_a_sharding():
    mesh = Mesh(np.asarray(jax.devices()[:1], dtype=object), ("x",))
    sharding = NamedSharding(mesh, P("x", None))
    a = jax.device_put(jnp.eye(4, dtype=jnp.float32), sharding)
    b = jax.device_put(jnp.ones((4, 4), dtype=jnp.float32), sharding)

    with pytest.raises(ValueError, match="requires both matrix axes"):
        potrs_mp(a, b, 2)


def test_potrs_mp_requires_named_sharding_when_mesh_is_omitted():
    a = jnp.eye(4, dtype=jnp.float32)
    b = jnp.ones((4, 4), dtype=jnp.float32)

    with pytest.raises(ValueError, match="could not infer mesh"):
        potrs_mp(a, b, 2)


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


def test_potrs_mp_rejects_mismatched_dtypes():
    mesh = Mesh(np.asarray(jax.devices()[:1], dtype=object).reshape(1, 1), ("pr", "pc"))
    a = jnp.eye(4, dtype=jnp.float32)
    b = jnp.ones((4, 4), dtype=jnp.complex64)

    with pytest.raises(TypeError, match="matching A/B dtypes"):
        potrs_mp(a, b, 2, mesh=mesh, matrix_specs=P("pr", "pc"))


def test_potrs_mp_rejects_non_positive_tile_size():
    mesh = Mesh(np.asarray(jax.devices()[:1], dtype=object).reshape(1, 1), ("pr", "pc"))
    a = jnp.eye(4, dtype=jnp.float32)
    b = jnp.ones((4, 4), dtype=jnp.float32)

    with pytest.raises(ValueError, match="T_A must be positive"):
        potrs_mp(a, b, 0, mesh=mesh, matrix_specs=P("pr", "pc"))


def test_potrs_mp_rejects_padding_when_pad_false():
    mesh = Mesh(np.asarray(jax.devices()[:1], dtype=object).reshape(1, 1), ("pr", "pc"))
    a = jnp.eye(5, dtype=jnp.float32)
    b = jnp.ones((5, 1), dtype=jnp.float32)

    with pytest.raises(ValueError, match="tile-aligned local shards"):
        potrs_mp(a, b, 4, mesh=mesh, matrix_specs=P("pr", "pc"), pad=False)
