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
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P


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


@pytest.mark.parametrize("routine,options", [
    ("lu_solve", {}), ("potrf", {}), ("getrf", {}),
    ("syevd", {"return_eigenvectors": True}),
    ("syevd", {"return_eigenvectors": False}),
    ("qr", {}),
    ("polar", {"compute_h": True}),
    ("polar", {"compute_h": False}),
    ("gesvd", {"compute_u": True, "compute_vh": True, "full_matrices": False}),
    ("gesvd", {"compute_u": True, "compute_vh": True, "full_matrices": True}),
    ("gesvd", {"compute_u": True, "compute_vh": False, "full_matrices": False}),
    ("gesvd", {"compute_u": False, "compute_vh": True, "full_matrices": False}),
    ("gesvd", {"compute_u": False, "compute_vh": False, "full_matrices": False}),
    ("least_squares", {}),
])
@pytest.mark.parametrize("check_vma", [True, False])
def test_local_context_padding_and_outputs(monkeypatch, routine, options, check_vma):
    """Run padding and output restoration with a shape-correct FFI stand-in."""
    module = importlib.import_module(f"jaxmg._{routine}")
    call = getattr(module, f"{routine}_shardmap_ctx")
    captured = {}

    def fake_ffi(target, result_types, **options):
        def run(*inputs, **attrs):
            captured.update(attrs)
            aliases = {out: arg for arg, out in options["input_output_aliases"].items()}
            result = [inputs[aliases[i]] if i in aliases else jnp.zeros(t.shape, t.dtype)
                      for i, t in enumerate(result_types)]
            if target == "cusolvermp_gels":
                assert options["input_output_aliases"] == {0: 0, 1: 1}
                result[2] = jnp.full(result_types[2].shape, 17, inputs[1].dtype)
            return tuple(result)
        return run

    monkeypatch.setattr(module, "ensure_init_jaxmg_backend", lambda: None)
    monkeypatch.setattr(jax.ffi, "ffi_call", fake_ffi)
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("pr", "pc"))
    specs = P("pr", "pc")
    inputs = (jnp.eye(6), jnp.ones((6, 1))) if routine == "lu_solve" else (jnp.eye(6),)
    if routine in ("qr", "polar", "gesvd"):
        inputs = (jnp.eye(10, 6),)
    if routine == "least_squares":
        inputs = (jnp.eye(10, 6, dtype=jnp.float32), jnp.arange(10., dtype=jnp.float32)[:, None])
    outputs = (specs, specs) if routine == "lu_solve" else (specs,)
    if routine == "getrf":
        outputs += (P(),)
    if routine == "syevd":
        outputs = (specs, P()) + ((specs,) if options["return_eigenvectors"] else ())
    if routine == "qr":
        outputs = (specs, specs)
    if routine == "polar":
        outputs = (specs,) * (2 if options["compute_h"] else 1)
    if routine == "gesvd":
        outputs = ((specs,) + (specs,) * options["compute_u"] + (P(),)
                   + (specs,) * options["compute_vh"])
    if routine == "least_squares":
        outputs = (specs, specs, specs)
    solve = jax.jit(jax.shard_map(
        partial(call, T_A=4, **options), mesh=mesh, in_specs=(specs,) * len(inputs),
        out_specs=outputs + (P(("pr", "pc")),), check_vma=check_vma,
    ))
    result = solve(*inputs)
    assert result[-1].ndim == 1
    assert captured["n"] == 6
    if routine == "lu_solve":
        assert result[0].shape == (8, 8)
        np.testing.assert_array_equal(result[1], inputs[1])
        assert captured["nrhs"] == 1
    elif routine == "syevd":
        assert result[0].shape == (8, 8)
        assert result[1].shape == (6,)
        assert result[1].sharding.is_fully_replicated
        if options["return_eigenvectors"]:
            assert result[2].shape == (6, 6)
    elif routine == "least_squares":
        assert result[0].shape == (12, 8)
        np.testing.assert_array_equal(result[1], inputs[1])
        np.testing.assert_array_equal(result[2], jnp.full((6, 1), 17))
        assert captured["nrhs"] == 1
    elif routine == "gesvd":
        assert result[0].shape == (12, 8)
        index = 1
        if options["compute_u"]:
            assert result[index].shape == (10, 10 if options["full_matrices"] else 6)
            index += 1
        assert result[index].shape == (6,)
        assert result[index].sharding.is_fully_replicated
        if options["compute_vh"]:
            assert result[index + 1].shape == (6, 6)
    else:
        np.testing.assert_array_equal(result[0], inputs[0])
        if routine == "getrf":
            assert result[1].shape == (6,)
            assert result[1].sharding.is_fully_replicated
        if routine == "qr":
            assert result[1].shape == (6, 6)
            assert captured["m"] == 10
        if routine == "polar" and options["compute_h"]:
            assert result[1].shape == (6, 6)

    with pytest.raises(ValueError, match="inside jax.shard_map"):
        call(*inputs, T_A=4, **options)


@pytest.mark.parametrize("shape,n,rhs_specs", [
    ((2, 1), 6, P("pr", None)),
    ((2, 1), 5, P()),
    ((3, 1), 5, P()),
    ((2, 2), 6, P("pc", None)),
    ((2, 2), 8, P(("pr", "pc"), None)),
])
@pytest.mark.parametrize("axis_type", [AxisType.Auto, AxisType.Explicit])
def test_least_squares_solution_rows(monkeypatch, shape, n, rhs_specs, axis_type):
    """Restore native X separately from B, including output-only routing rows."""
    from jaxmg import _least_squares as module

    count = int(np.prod(shape))
    if jax.device_count() < count:
        pytest.skip("requires multiple CPU devices or GPUs")
    mesh = Mesh(np.asarray(jax.devices()[:count]).reshape(shape), ("pr", "pc"),
                axis_types=(axis_type, axis_type))
    captured = {}

    def fake_ffi(target, result_types, **options):
        def run(a, b, **attrs):
            captured.update(attrs)
            local_rows = attrs["x_distribution_rows"] // shape[0]
            local_cols = attrs["b_distribution_cols"] // shape[1]
            rows = jax.lax.axis_index("pr") * local_rows + jnp.arange(local_rows)
            cols = jax.lax.axis_index("pc") * local_cols + jnp.arange(local_cols)
            x = (10 * rows[:, None] + cols[None, :] + 1).astype(b.dtype)
            x = jnp.pad(x, ((0, result_types[2].shape[0] - local_rows),
                            (0, result_types[2].shape[1] - local_cols)))
            return a, b, x, jnp.zeros(result_types[3].shape, jnp.int32)
        return run

    monkeypatch.setattr(module, "ensure_init_jaxmg_backend", lambda: None)
    monkeypatch.setattr(jax.ffi, "ffi_call", fake_ffi)
    a = jax.device_put(jnp.eye(12, n), NamedSharding(mesh, P("pr", "pc")))
    b = jax.device_put(jnp.arange(60.).reshape(12, 5), NamedSharding(mesh, rhs_specs))
    solve = partial(module.least_squares_jit_ctx, T_A=2)
    # Auto-axis shardings are available on concrete arrays, not tracer types.
    if axis_type == AxisType.Explicit:
        solve = jax.jit(solve)
    with jax.set_mesh(mesh):
        _, b_work, x, _ = solve(a, b)
    np.testing.assert_array_equal(b_work, b)
    np.testing.assert_array_equal(x, 10 * np.arange(n)[:, None] + np.arange(5) + 1)
    assert x.sharding.is_equivalent_to(b.sharding, 2)
    assert b_work.sharding.is_equivalent_to(b.sharding, 2)
    assert captured["n"] == n
    assert captured["x_distribution_rows"] >= n
    assert captured["x_distribution_rows"] % shape[0] == 0


@pytest.mark.parametrize("shape", [(1, 2), (2, 2)])
def test_least_squares_local_ffi_with_replicated_rhs_columns(monkeypatch, shape):
    """Trace the real FFI declaration, including its varying-axis requirements."""
    from jaxmg import _least_squares as module

    count = int(np.prod(shape))
    if jax.device_count() < count:
        pytest.skip("requires multiple CPU devices or GPUs")
    monkeypatch.setattr(module, "ensure_init_jaxmg_backend", lambda: None)
    mesh = Mesh(np.asarray(jax.devices()[:count]).reshape(shape), ("pr", "pc"))
    specs, rhs_specs = P("pr", "pc"), P("pr", None)
    solve = jax.shard_map(
        partial(module.least_squares_shardmap_ctx, T_A=2), mesh=mesh,
        in_specs=(specs, rhs_specs),
        out_specs=(specs, rhs_specs, rhs_specs, P(("pr", "pc"))),
    )
    result = jax.eval_shape(solve, jax.ShapeDtypeStruct((12, 8), jnp.float32),
                            jax.ShapeDtypeStruct((12, 2), jnp.float32))
    assert result[2].shape == (8, 2)
