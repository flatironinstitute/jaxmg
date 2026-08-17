import numpy as np
import pytest

import jax
from jax.sharding import Mesh, PartitionSpec as P

from jaxmg._cusolvermp_layout import use_abstract_mesh_decorator


def _mesh(axis_name: str) -> Mesh:
    devices = np.asarray(jax.devices()[:1], dtype=object)
    return Mesh(devices, (axis_name,))


def test_use_abstract_mesh_decorator_sets_and_restores_context():
    mesh = _mesh("pr")
    before = jax.sharding.get_abstract_mesh()

    @use_abstract_mesh_decorator(mesh)
    def body():
        return jax.sharding.get_abstract_mesh()

    assert body() == mesh.abstract_mesh
    assert jax.sharding.get_abstract_mesh() == before


def test_use_abstract_mesh_decorator_applies_inside_jit():
    """Solvers are traced inside jax.jit, where set_mesh cannot be used."""
    mesh = _mesh("pr")
    seen = {}

    @jax.jit
    def f(x):
        @use_abstract_mesh_decorator(mesh)
        def body(x):
            seen["mesh"] = jax.sharding.get_abstract_mesh()
            return x + 1

        return body(x)

    f(jax.numpy.zeros((2,)))
    assert seen["mesh"] == mesh.abstract_mesh


def test_shard_map_runs_under_a_foreign_context_mesh():
    """A caller with its own context mesh must not have to switch it themselves."""
    mesh = _mesh("pr")
    caller_mesh = Mesh(np.asarray(jax.devices()[:1], dtype=object), ("caller",))

    @use_abstract_mesh_decorator(mesh)
    def shard_mapped(x):
        return jax.shard_map(
            lambda block: block * 2,
            mesh=mesh,
            in_specs=P("pr"),
            out_specs=P("pr"),
            check_vma=False,
        )(x)

    x = jax.numpy.ones((4,))
    with jax.sharding.use_abstract_mesh(caller_mesh.abstract_mesh):
        np.testing.assert_allclose(np.asarray(shard_mapped(x)), 2.0)
        # without the decorator, jax rejects the mismatched context mesh
        with pytest.raises(ValueError, match="context mesh"):
            jax.shard_map(
                lambda block: block * 2,
                mesh=mesh,
                in_specs=P("pr"),
                out_specs=P("pr"),
                check_vma=False,
            )(x)
