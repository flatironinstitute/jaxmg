import numpy as np

import jax
import jax.numpy as jnp
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from jaxmg import syevd_mp


def _single_device_mesh() -> Mesh:
    return Mesh(np.asarray(jax.devices()[:1], dtype=object).reshape(1, 1), ("pr", "pc"))


def test_syevd_mp_rejects_non_2d_process_grid_specs():
    mesh = Mesh(np.asarray(jax.devices()[:1], dtype=object), ("x",))
    a = jnp.eye(4, dtype=jnp.float32)

    with pytest.raises(ValueError, match="requires both matrix axes"):
        syevd_mp(a, 2, mesh=mesh, matrix_specs=P("x", None))


def test_syevd_mp_infers_mesh_and_specs_from_a_sharding():
    mesh = Mesh(np.asarray(jax.devices()[:1], dtype=object), ("x",))
    sharding = NamedSharding(mesh, P("x", None))
    a = jax.device_put(jnp.eye(4, dtype=jnp.float32), sharding)

    with pytest.raises(ValueError, match="requires both matrix axes"):
        syevd_mp(a, 2)


def test_syevd_mp_requires_named_sharding_when_mesh_is_omitted():
    a = jnp.eye(4, dtype=jnp.float32)

    with pytest.raises(ValueError, match="could not infer mesh"):
        syevd_mp(a, 2)


def test_syevd_mp_rejects_non_square_a():
    mesh = _single_device_mesh()
    a = jnp.ones((4, 2), dtype=jnp.float32)

    with pytest.raises(ValueError, match="A to be square"):
        syevd_mp(a, 2, mesh=mesh, matrix_specs=P("pr", "pc"))


def test_syevd_mp_rejects_unsupported_dtype():
    mesh = _single_device_mesh()
    a = jnp.eye(4, dtype=jnp.int32)

    with pytest.raises(TypeError, match="supports float32"):
        syevd_mp(a, 2, mesh=mesh, matrix_specs=P("pr", "pc"))


def test_syevd_mp_rejects_non_positive_tile_size():
    mesh = _single_device_mesh()
    a = jnp.eye(4, dtype=jnp.float32)

    with pytest.raises(ValueError, match="T_A must be positive"):
        syevd_mp(a, 0, mesh=mesh, matrix_specs=P("pr", "pc"))


def test_syevd_mp_rejects_eigenvalue_only_mode():
    a = jnp.eye(4, dtype=jnp.float32)

    with pytest.raises(NotImplementedError, match="eigvecs=True"):
        syevd_mp(a, 2, eigvecs=False)


def test_syevd_mp_rejects_padding_when_pad_false():
    mesh = _single_device_mesh()
    a = jnp.eye(5, dtype=jnp.float32)

    with pytest.raises(ValueError, match="tile-aligned local shards"):
        syevd_mp(a, 4, mesh=mesh, matrix_specs=P("pr", "pc"), pad=False)
