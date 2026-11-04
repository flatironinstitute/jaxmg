import numpy as np
import pytest

import jax
from jax.sharding import Mesh, PartitionSpec as P

from jaxmg._cusolvermp_layout import infer_rhs_specs


def _mesh(axis_name: str) -> Mesh:
    devices = np.asarray(jax.devices()[:1], dtype=object)
    return Mesh(devices, (axis_name,))


@pytest.mark.parametrize("compiled", [False, True])
def test_solver_restores_foreign_context_mesh(monkeypatch, compiled):
    """Global context calls temporarily select their mesh, including under jit."""
    from jaxmg import _potrf as module

    monkeypatch.setattr(module, "ensure_init_jaxmg_backend", lambda: None)

    def fake_ffi(target, result_types, **options):
        return lambda a, **attrs: (a, jax.numpy.zeros(result_types[1].shape, dtype="int32"))

    monkeypatch.setattr(jax.ffi, "ffi_call", fake_ffi)
    mesh = _mesh("pr")
    caller_mesh = _mesh("caller")

    def solve(a):
        return module.potrf_jit_ctx(a, T_A=2, mesh=mesh, matrix_specs=P("pr", None))

    if compiled:
        solve = jax.jit(solve)
    x = jax.numpy.eye(4)
    with jax.sharding.use_abstract_mesh(caller_mesh.abstract_mesh):
        result, _ = solve(x)
        np.testing.assert_array_equal(result, x)
        assert jax.sharding.get_abstract_mesh() == caller_mesh.abstract_mesh


@pytest.mark.parametrize(
    "axis_type, expected",
    [
        # Under jit, Auto shardings are not part of the type of an array, so
        # its replicated-looking type must not be restored onto the solution.
        (jax.sharding.AxisType.Auto, P("pr", None)),
        (jax.sharding.AxisType.Explicit, P(None, "pr")),
    ],
)
def test_infer_rhs_specs_under_jit(axis_type, expected):
    mesh = Mesh(
        np.asarray(jax.devices()[:1], dtype=object), ("pr",), axis_types=(axis_type,)
    )
    b = jax.device_put(
        jax.numpy.ones((4, 2)), jax.sharding.NamedSharding(mesh, P(None, "pr"))
    )
    seen = {}

    @jax.jit
    def f(b):
        seen["specs"] = infer_rhs_specs(b, matrix_specs=P("pr", None))
        return b

    with jax.set_mesh(mesh):
        f(b)
    assert seen["specs"] == expected
