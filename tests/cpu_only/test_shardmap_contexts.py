"""Exercise local context interfaces without loading the CUDA backend."""

import importlib
import os
import subprocess
import sys
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, PartitionSpec as P


def test_multi_device_local_contexts():
    """Run collective and real-FFI tracing checks on four host devices in CI."""
    env = dict(os.environ, XLA_FLAGS="--xla_force_host_platform_device_count=4",
               JAX_PLATFORMS="cpu")
    env.pop("JAX_NUM_CPU_DEVICES", None)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", __file__,
         "-k", "solution_rows or local_ffi"],
        env=env, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "skipped" not in result.stdout, result.stdout


@pytest.mark.parametrize("routine", ["potrs", "lu_solve"])
@pytest.mark.parametrize("shape", [(1, 2), (2, 2)])
def test_local_ffi_with_replicated_rhs_columns(monkeypatch, routine, shape):
    """Trace real FFI declarations rather than bypassing their typing checks."""
    count = int(np.prod(shape))
    if jax.device_count() < count:
        pytest.skip("requires multiple CPU devices or GPUs")
    module = importlib.import_module(f"jaxmg._{routine}")
    monkeypatch.setattr(module, "ensure_init_jaxmg_backend", lambda: None)
    mesh = Mesh(np.asarray(jax.devices()[:count]).reshape(shape), ("pr", "pc"))
    specs, rhs_specs = P("pr", "pc"), P("pr", None)
    solve = jax.shard_map(
        partial(getattr(module, f"{routine}_shardmap_ctx"), T_A=2), mesh=mesh,
        in_specs=(specs, rhs_specs),
        out_specs=(specs, rhs_specs, P(("pr", "pc"))),
    )
    result = jax.eval_shape(solve, jax.ShapeDtypeStruct((8, 8), jnp.float32),
                            jax.ShapeDtypeStruct((8, 1), jnp.float32))
    assert result[1].shape == (8, 1)


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
