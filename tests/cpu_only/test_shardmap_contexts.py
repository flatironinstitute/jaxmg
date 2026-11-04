"""Exercise local context interfaces without loading the CUDA backend."""

import importlib
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, PartitionSpec as P


@pytest.mark.parametrize("routine", ["lu_solve"])
@pytest.mark.parametrize("check_vma", [True, False])
def test_local_context_padding_and_outputs(monkeypatch, routine, check_vma):
    """Run padding and output restoration with a shape-correct FFI stand-in."""
    module = importlib.import_module(f"jaxmg._{routine}")
    call = getattr(module, f"{routine}_shardmap_ctx")
    captured = {}

    def fake_ffi(target, result_types, **options):
        def run(*inputs, **attrs):
            captured.update(attrs)
            aliases = {out: arg for arg, out in options["input_output_aliases"].items()}
            return tuple(inputs[aliases[i]] if i in aliases else jnp.zeros(t.shape, t.dtype)
                         for i, t in enumerate(result_types))
        return run

    monkeypatch.setattr(module, "ensure_init_jaxmg_backend", lambda: None)
    monkeypatch.setattr(jax.ffi, "ffi_call", fake_ffi)
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("pr", "pc"))
    specs = P("pr", "pc")
    inputs = (jnp.eye(6), jnp.ones((6, 1)))
    solve = jax.jit(jax.shard_map(
        partial(call, T_A=4), mesh=mesh, in_specs=(specs, specs),
        out_specs=(specs, specs, P(("pr", "pc"))), check_vma=check_vma,
    ))
    work, x, status = solve(*inputs)
    assert work.shape == (8, 8)
    np.testing.assert_array_equal(x, inputs[1])
    assert status.ndim == 1
    assert captured["n"] == 6 and captured["nrhs"] == 1

    with pytest.raises(ValueError, match="inside jax.shard_map"):
        call(*inputs, T_A=4)
