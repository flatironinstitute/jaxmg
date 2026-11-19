from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

import jaxmg
import jaxmg._inverse as inverse_module
from jaxmg._cusolvermp_status import _CUSOLVERMP_INVERSE_STATUS_SIZE


@pytest.mark.parametrize("routine", ["potri", "getri"])
@pytest.mark.parametrize("interface", ["public", "jit_ctx", "shardmap_ctx"])
def test_inverse_layers_preserve_shape_and_native_contract(monkeypatch, routine, interface):
    """All three layers use one aliased FFI call with the original logical N."""
    captured = {}
    inverse_module._inverse_compiled.cache_clear()
    monkeypatch.setattr(inverse_module, "ensure_init_jaxmg_backend", lambda: None)

    def fake_ffi(target, results, **options):
        captured.update(target=target, results=results, options=options)

        def run(a, **attrs):
            captured.update(attrs)
            return a, jnp.zeros((_CUSOLVERMP_INVERSE_STATUS_SIZE,), jnp.int32)

        return run

    monkeypatch.setattr(jax.ffi, "ffi_call", fake_ffi)
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("pr", "pc"))
    specs = P("pr", "pc")
    a = jax.device_put(jnp.eye(3), NamedSharding(mesh, specs))
    if interface == "public":
        inverse, status = getattr(jaxmg, routine)(
            a, 2, mesh, specs, donate=False, return_status=True,
        )
        np.testing.assert_array_equal(a, np.eye(3))
    elif interface == "jit_ctx":
        inverse, status = jax.jit(partial(
            getattr(jaxmg, routine + "_jit_ctx"), T_A=2,
            mesh=mesh, matrix_specs=specs,
        ), donate_argnums=(0,))(a)
    else:
        inverse, status = jax.jit(jax.shard_map(
            partial(getattr(jaxmg, routine + "_shardmap_ctx"),
                    T_A=2, matrix_specs=specs),
            mesh=mesh, in_specs=specs, out_specs=(specs, P(("pr", "pc"))),
        ))(a)

    assert inverse.shape == (3, 3)
    assert status.shape == (_CUSOLVERMP_INVERSE_STATUS_SIZE,)
    assert captured["target"] == "cusolvermp_inverse"
    assert captured["options"]["input_output_aliases"] == {0: 0}
    assert captured["results"][0].shape == (4, 4)
    assert captured["cholesky"] is (routine == "potri")
    assert captured["n"] == 3
    assert captured["tile_size"] == 2
    np.testing.assert_array_equal(inverse, np.eye(3))


@pytest.mark.parametrize("routine", ["potri", "getri"])
def test_inverse_rejects_invalid_inputs(routine):
    """Reject incompatible mathematical and layout contracts before native code."""
    invert = getattr(jaxmg, routine)
    with pytest.raises(ValueError, match="rank-2"):
        invert(jnp.ones(4), 2)
    with pytest.raises(ValueError, match="square"):
        invert(jnp.ones((4, 3)), 2)
    with pytest.raises(TypeError, match="supports float32"):
        invert(jnp.eye(4, dtype=jnp.int32), 2)
    with pytest.raises(ValueError, match="positive"):
        invert(jnp.eye(4), 0)
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("pr", "pc"))
    with pytest.raises(ValueError, match="tile-aligned"):
        invert(jnp.eye(3), 2, mesh, P("pr", "pc"), pad=False)
    with pytest.raises(ValueError, match="Specify only one"):
        invert(jnp.eye(4), 2, mesh, P("pr", "pc"), in_specs=P("pr", "pc"))
