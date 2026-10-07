import os
import subprocess
import sys
from functools import partial

import numpy as np
import pytest

import jax
if not jax.config.jax_enable_x64:
    jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

import jaxmg._potrs as potrs_module
from jaxmg import potrs, potrs_jit_ctx, potrs_shardmap_ctx
from jaxmg._cusolvermp_layout import mark_varying
from jaxmg._cusolvermp_status import _CUSOLVERMP_POTRS_STATUS_SIZE


def _one_rank_mesh() -> Mesh:
    """Return a regular 1x1 matrix mesh usable on CPU-only test hosts."""
    devices = np.asarray(jax.devices()[:1], dtype=object).reshape(1, 1)
    return Mesh(devices, ("pr", "pc"))


def _single_axis_mesh() -> Mesh:
    """Return a mesh with one axis, as used by callers that shard only rows."""
    devices = np.asarray(jax.devices()[:1], dtype=object)
    return Mesh(devices, ("pr",))


def _install_fake_potrs_backend(monkeypatch):
    """Replace the native FFI call with a small Python stand-in.

    Everything around the FFI call still runs: placement, ``shard_map``, tile
    padding, and restoration of the RHS layout.
    """
    captured = {}

    def fake_native_call(_a, _b, **kwargs):
        captured["native_kwargs"] = kwargs
        status = jnp.zeros((_CUSOLVERMP_POTRS_STATUS_SIZE,), dtype=jnp.int32)
        if kwargs["return_logdet"]:
            logdet_dtype = (
                jnp.float32 if _a.dtype in (jnp.float32, jnp.complex64) else jnp.float64
            )
            logdet = jnp.asarray([3.25], dtype=logdet_dtype)
            return _a, _b, logdet, status
        # The native call returns the A work buffer, the solved RHS, and status.
        return _a, _b, status

    pipeline = potrs_module._potrs_pipeline

    def spy_pipeline(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return pipeline(*args, **kwargs)

    # Cached pipelines would replay traces of an earlier test's fake.
    pipeline.cache_clear()
    monkeypatch.setattr(potrs_module, "ensure_init_jaxmg_backend", lambda: None)
    monkeypatch.setattr(potrs_module, "_potrs_native_call", fake_native_call)
    monkeypatch.setattr(potrs_module, "_potrs_pipeline", spy_pipeline)
    return captured


def test_potrs_accepts_current_2d_mesh_contract(monkeypatch):
    captured = _install_fake_potrs_backend(monkeypatch)
    a = jnp.eye(4, dtype=jnp.float32)
    b = jnp.ones((4, 2), dtype=jnp.float32)

    out = potrs(a, b, 2, mesh=_one_rank_mesh(), matrix_specs=P("pr", "pc"))

    assert out.shape == b.shape
    assert captured["native_kwargs"]["n"] == 4
    assert captured["native_kwargs"]["nrhs"] == 2
    assert captured["native_kwargs"]["tile_size"] == 2


def test_potrs_preserves_vector_rhs_rank(monkeypatch):
    captured = _install_fake_potrs_backend(monkeypatch)
    a = jnp.eye(4, dtype=jnp.float64)
    b = jnp.ones((4,), dtype=jnp.float64)

    out, status = potrs(
        a,
        b,
        2,
        mesh=_one_rank_mesh(),
        matrix_specs=P("pr", "pc"),
        return_status=True,
    )

    assert out.shape == b.shape
    assert status.shape == (_CUSOLVERMP_POTRS_STATUS_SIZE,)
    assert captured["native_kwargs"]["nrhs"] == 1


def test_potrs_preserves_single_column_rhs_rank(monkeypatch):
    _install_fake_potrs_backend(monkeypatch)
    a = jnp.eye(4, dtype=jnp.float64)
    b = jnp.ones((4, 1), dtype=jnp.float64)

    out = potrs(
        a,
        b,
        2,
        mesh=_one_rank_mesh(),
        matrix_specs=P("pr", "pc"),
    )

    assert out.shape == b.shape


@pytest.mark.parametrize("return_status", [False, True])
@pytest.mark.parametrize(
    "matrix_dtype,expected_logdet_dtype",
    [
        (jnp.float32, jnp.float32),
        (jnp.complex64, jnp.float32),
        (jnp.float64, jnp.float64),
        (jnp.complex128, jnp.float64),
    ],
)
def test_potrs_returns_optional_real_dtype_logdet(
    monkeypatch, return_status, matrix_dtype, expected_logdet_dtype
):
    captured = _install_fake_potrs_backend(monkeypatch)
    a = jnp.eye(4, dtype=matrix_dtype)
    b = jnp.ones((4, 1), dtype=matrix_dtype)

    result = potrs(
        a,
        b,
        2,
        mesh=_one_rank_mesh(),
        matrix_specs=P("pr", "pc"),
        return_logdet=True,
        return_status=return_status,
    )

    if return_status:
        out, logdet, status = result
        assert status.shape == (_CUSOLVERMP_POTRS_STATUS_SIZE,)
    else:
        out, logdet = result
    assert out.shape == b.shape
    assert logdet.shape == ()
    assert logdet.dtype == expected_logdet_dtype
    assert float(logdet) == pytest.approx(3.25)
    assert captured["kwargs"]["return_logdet"] is True


def test_potrs_public_api_can_be_wrapped_in_external_jit(monkeypatch):
    _install_fake_potrs_backend(monkeypatch)
    mesh = _one_rank_mesh()
    solve = jax.jit(
        partial(potrs, T_A=2, mesh=mesh, matrix_specs=P("pr", "pc")),
        donate_argnums=(0, 1),
    )

    out = solve(
        jnp.eye(4, dtype=jnp.float32),
        jnp.ones((4, 1), dtype=jnp.float32),
    )

    assert out.shape == (4, 1)


def test_potrs_jit_ctx_returns_work_solution_and_status(monkeypatch):
    captured = _install_fake_potrs_backend(monkeypatch)
    a = jnp.eye(4, dtype=jnp.float32)
    b = jnp.ones((4, 2), dtype=jnp.float32)

    a_work, out, status = potrs_jit_ctx(
        a,
        b,
        2,
        mesh=_one_rank_mesh(),
        matrix_specs=P("pr", "pc"),
    )

    assert a_work.shape == a.shape
    assert out.shape == b.shape
    assert status.shape == (_CUSOLVERMP_POTRS_STATUS_SIZE,)
    assert captured["native_kwargs"]["n"] == 4
    assert captured["native_kwargs"]["nrhs"] == 2
    assert captured["native_kwargs"]["tile_size"] == 2


def test_potrs_jit_ctx_preserves_vector_rhs_rank(monkeypatch):
    captured = _install_fake_potrs_backend(monkeypatch)
    a = jnp.eye(4, dtype=jnp.float64)
    b = jnp.ones((4,), dtype=jnp.float64)

    a_work, out, status = potrs_jit_ctx(
        a,
        b,
        2,
        mesh=_one_rank_mesh(),
        matrix_specs=P("pr", "pc"),
    )

    assert a_work.shape == a.shape
    assert out.shape == b.shape
    assert status.shape == (_CUSOLVERMP_POTRS_STATUS_SIZE,)
    assert captured["native_kwargs"]["nrhs"] == 1


def test_potrs_jit_ctx_returns_optional_logdet(monkeypatch):
    captured = _install_fake_potrs_backend(monkeypatch)
    a = jnp.eye(4, dtype=jnp.float32)
    b = jnp.ones((4, 2), dtype=jnp.float32)

    a_work, out, logdet, status = potrs_jit_ctx(
        a,
        b,
        2,
        mesh=_one_rank_mesh(),
        matrix_specs=P("pr", "pc"),
        return_logdet=True,
    )

    assert a_work.shape == a.shape
    assert out.shape == b.shape
    assert logdet.shape == ()
    assert logdet.dtype == jnp.float32
    assert status.shape == (_CUSOLVERMP_POTRS_STATUS_SIZE,)
    assert captured["native_kwargs"]["return_logdet"] is True


def test_potrs_jit_ctx_can_be_wrapped_in_external_jit(monkeypatch):
    _install_fake_potrs_backend(monkeypatch)
    mesh = _one_rank_mesh()
    solve = jax.jit(
        partial(potrs_jit_ctx, T_A=2, mesh=mesh, matrix_specs=P("pr", "pc")),
        donate_argnums=(0, 1),
    )

    a_work, out, status = solve(
        jnp.eye(4, dtype=jnp.float32),
        jnp.ones((4, 1), dtype=jnp.float32),
    )

    assert a_work.shape == (4, 4)
    assert out.shape == (4, 1)
    assert status.shape == (_CUSOLVERMP_POTRS_STATUS_SIZE,)


def test_potrs_external_jit_can_lower_from_shape_specs(monkeypatch):
    _install_fake_potrs_backend(monkeypatch)
    mesh = _one_rank_mesh()
    solve = jax.jit(
        partial(potrs, T_A=2, mesh=mesh, matrix_specs=P("pr", "pc")),
        donate_argnums=(0, 1),
    )

    compiled = solve.lower(
        jax.ShapeDtypeStruct((4, 4), jnp.float32),
        jax.ShapeDtypeStruct((4, 1), jnp.float32),
    ).compile()

    assert compiled.memory_analysis() is not None


def test_potrs_rejects_non_matrix_a():
    with pytest.raises(ValueError, match="rank-2 matrix A"):
        potrs(jnp.ones((4,)), jnp.ones((4, 1)), 2)


def test_potrs_rejects_non_matrix_rhs():
    with pytest.raises(ValueError, match="rank-1 or rank-2 RHS"):
        potrs(jnp.eye(4), jnp.ones((4, 1, 1)), 2)


def test_potrs_rejects_mismatched_dtypes():
    with pytest.raises(TypeError, match="matching A/B dtypes"):
        potrs(jnp.eye(4, dtype=jnp.float32), jnp.ones((4, 1), dtype=jnp.float64), 2)


def test_potrs_rejects_unsupported_dtype():
    with pytest.raises(TypeError, match="supports float32"):
        potrs(jnp.eye(4, dtype=jnp.int32), jnp.ones((4, 1), dtype=jnp.int32), 2)


def test_potrs_rejects_non_square_a():
    with pytest.raises(ValueError, match="A to be square"):
        potrs(jnp.ones((4, 3)), jnp.ones((4, 1)), 2)


def test_potrs_rejects_leading_dimension_mismatch():
    with pytest.raises(ValueError, match="matching leading dimensions"):
        potrs(jnp.eye(4), jnp.ones((5, 1)), 2)


def test_potrs_rejects_nonpositive_tile_size():
    with pytest.raises(ValueError, match="T_A must be positive"):
        potrs(jnp.eye(4), jnp.ones((4, 1)), 0)


def test_potrs_rejects_ambiguous_spec_arguments():
    with pytest.raises(ValueError, match="Specify only one"):
        potrs(
            jnp.eye(4),
            jnp.ones((4, 1)),
            2,
            mesh=_one_rank_mesh(),
            matrix_specs=P("pr", "pc"),
            in_specs=P("pr", "pc"),
        )


def test_potrs_accepts_degenerate_column_grid(monkeypatch):
    """A P_r x 1 grid leaves the matrix columns undistributed."""
    captured = _install_fake_potrs_backend(monkeypatch)
    a = jnp.eye(4, dtype=jnp.float32)
    b = jnp.ones((4, 1), dtype=jnp.float32)

    out = potrs(a, b, 2, mesh=_single_axis_mesh(), matrix_specs=P("pr", None))

    assert out.shape == b.shape
    grid = captured["native_kwargs"]["grid"]
    assert (grid.process_rows, grid.process_cols) == (1, 1)


def test_potrs_accepts_rank_1_matrix_specs(monkeypatch):
    """P('pr') is the same layout as P('pr', None)."""
    captured = _install_fake_potrs_backend(monkeypatch)
    a = jnp.eye(4, dtype=jnp.float32)
    b = jnp.ones((4, 1), dtype=jnp.float32)

    out = potrs(a, b, 2, mesh=_single_axis_mesh(), matrix_specs=P("pr"))

    assert out.shape == b.shape
    assert captured["args"][1] == P("pr", None)


def test_potrs_rejects_fully_replicated_matrix_specs():
    with pytest.raises(ValueError, match="at least one matrix axis"):
        potrs(
            jnp.eye(4),
            jnp.ones((4, 1)),
            2,
            mesh=_one_rank_mesh(),
            matrix_specs=P(None, None),
        )


def test_potrs_rejects_required_a_padding_when_disabled():
    with pytest.raises(ValueError, match="potrs\\(A\\) requires tile-aligned"):
        potrs(
            jnp.eye(3),
            jnp.ones((3, 1)),
            2,
            mesh=_one_rank_mesh(),
            matrix_specs=P("pr", "pc"),
            pad=False,
        )


def test_potrs_rejects_required_rhs_padding_when_disabled():
    with pytest.raises(ValueError, match="potrs\\(B\\) requires tile-aligned"):
        potrs(
            jnp.eye(4),
            jnp.ones((4, 1)),
            2,
            mesh=_one_rank_mesh(),
            matrix_specs=P("pr", "pc"),
            pad=False,
        )

def test_potrs_donates_by_default(monkeypatch):
    _install_fake_potrs_backend(monkeypatch)
    a = jnp.eye(4, dtype=jnp.float32)
    b = jnp.ones((4, 1), dtype=jnp.float32)

    potrs(a, b, 2, mesh=_one_rank_mesh(), matrix_specs=P("pr", "pc"))

    assert a.is_deleted() and b.is_deleted()


def test_potrs_donation_can_be_disabled(monkeypatch):
    _install_fake_potrs_backend(monkeypatch)
    a = jnp.eye(4, dtype=jnp.float32)
    b = jnp.ones((4, 1), dtype=jnp.float32)

    potrs(
        a, b, 2,
        mesh=_one_rank_mesh(),
        matrix_specs=P("pr", "pc"),
        donate=False,
    )

    assert not a.is_deleted() and not b.is_deleted()


def test_potrs_reshards_a_replicated_over_explicit_mesh_axes(monkeypatch):
    # The type of A carries no sharding, so the matrix specs default to the mesh
    # axes, P("pr", None), and A must be resharded to them.
    monkeypatch.setattr(potrs_module, "ensure_init_jaxmg_backend", lambda: None)
    devices = np.asarray(jax.devices()[:1], dtype=object)
    mesh = Mesh(devices, ("pr",), axis_types=(AxisType.Explicit,))
    with jax.set_mesh(mesh):
        a = jax.device_put(jnp.eye(4), NamedSharding(mesh, P()))
        b = jax.device_put(jnp.ones((4,)), NamedSharding(mesh, P()))
        a_work, x, _ = jax.eval_shape(partial(potrs_jit_ctx, T_A=2), a, b)

    assert a_work.sharding.spec == P("pr", None)
    assert x.sharding.spec == P(None)


def _local_potrs(mesh, b_specs, out_b_specs=None, *, check_vma=True, **kwargs):
    """Wrap potrs_shardmap_ctx in a caller-owned shard_map on the 1x1 mesh."""
    specs = P("pr", "pc")
    out_specs = (specs, specs if out_b_specs is None else out_b_specs)
    if kwargs.get("return_logdet", False):
        out_specs += (P(),)
    return jax.shard_map(
        partial(potrs_shardmap_ctx, **kwargs),
        mesh=mesh,
        in_specs=(specs, b_specs),
        out_specs=out_specs + (P(("pr", "pc")),),
        check_vma=check_vma,
    )


@pytest.mark.parametrize("check_vma", [True, False])
def test_potrs_shardmap_ctx_runs_inside_caller_shard_map(monkeypatch, check_vma):
    captured = _install_fake_potrs_backend(monkeypatch)
    solve = jax.jit(
        _local_potrs(_one_rank_mesh(), P("pr", "pc"), check_vma=check_vma, T_A=2),
        donate_argnums=(0, 1),
    )

    a_work, out, status = solve(
        jnp.eye(4, dtype=jnp.float32),
        jnp.ones((4, 2), dtype=jnp.float32),
    )

    assert a_work.shape == (4, 4)
    assert out.shape == (4, 2)
    assert status.shape == (_CUSOLVERMP_POTRS_STATUS_SIZE,)
    assert captured["native_kwargs"]["n"] == 4
    assert captured["native_kwargs"]["nrhs"] == 2
    assert captured["native_kwargs"]["tile_size"] == 2


@pytest.mark.parametrize("check_vma", [True, False])
def test_potrs_shardmap_ctx_pads_vector_rhs_and_returns_logdet(monkeypatch, check_vma):
    captured = _install_fake_potrs_backend(monkeypatch)
    solve = _local_potrs(
        _one_rank_mesh(),
        P("pr"),
        P("pr"),
        check_vma=check_vma,
        T_A=4,
        return_logdet=True,
    )
    b = jnp.arange(6, dtype=jnp.float64)

    a_work, out, logdet, status = jax.jit(solve)(jnp.eye(6), b)

    # A local 6x6 block is padded to the 8x8 tile capacity.
    assert a_work.shape == (8, 8)
    np.testing.assert_array_equal(out, b)
    assert logdet.shape == ()
    assert float(logdet) == pytest.approx(3.25)
    assert status.shape == (_CUSOLVERMP_POTRS_STATUS_SIZE,)
    assert captured["native_kwargs"]["nrhs"] == 1


def test_potrs_shardmap_ctx_rejects_global_arrays():
    with pytest.raises(ValueError, match="inside jax.shard_map"):
        potrs_shardmap_ctx(jnp.eye(4), jnp.ones((4, 1)), 2)


def test_potrs_shardmap_ctx_rejects_global_arrays_under_context_mesh():
    with jax.set_mesh(_one_rank_mesh()):
        with pytest.raises(ValueError, match="inside jax.shard_map"):
            potrs_shardmap_ctx(jnp.eye(4), jnp.ones((4, 1)), 2)


def test_potrs_shardmap_ctx_rejects_required_padding_when_disabled(monkeypatch):
    _install_fake_potrs_backend(monkeypatch)
    solve = _local_potrs(_one_rank_mesh(), P("pr", "pc"), T_A=2, pad=False)

    with pytest.raises(ValueError, match="potrs_shardmap_ctx\\(A\\) requires tile-aligned"):
        solve(jnp.eye(3), jnp.ones((3, 2)))


def test_mark_varying_casts_only_missing_axes():
    def body(a):
        status = mark_varying(jnp.zeros((2,)), ("pr", "pc"))
        assert jax.typeof(status).manual_axis_type.varying == {"pr", "pc"}
        # A value that already varies over both axes is returned unchanged.
        assert mark_varying(a, ("pr", "pc")) is a
        return status

    jax.shard_map(
        body,
        mesh=_one_rank_mesh(),
        in_specs=P("pr", "pc"),
        out_specs=P(("pr", "pc")),
    )(jnp.zeros((4, 4)))


_TWO_BY_TWO_SCRIPT = r"""
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

import jaxmg._potrs as potrs_module
from jaxmg import potrs_jit_ctx, potrs_shardmap_ctx

seen = []


def fake_native_call(a, b, **kwargs):
    # An identity solve: the solution is the (padded) right-hand side.
    seen.append(kwargs)
    return a, b, jnp.zeros((4,), dtype=jnp.int32)


potrs_module._potrs_native_call = fake_native_call
potrs_module.ensure_init_jaxmg_backend = lambda: None

mesh = Mesh(np.asarray(jax.devices()).reshape(2, 2), ("pr", "pc"))
specs = P("pr", "pc")
a = jax.device_put(jnp.eye(8), NamedSharding(mesh, specs))

# Global arrays: a skinny RHS gains one routing column to shard like A.
b = jax.device_put(jnp.arange(8.0)[:, None], NamedSharding(mesh, P("pr", None)))
_, x, _ = jax.jit(partial(potrs_jit_ctx, T_A=2))(a, b)
np.testing.assert_array_equal(x, b)
assert x.sharding.is_equivalent_to(b.sharding, 2), x.sharding
assert seen[-1]["nrhs"] == 2 and seen[-1]["n"] == 8, seen[-1]

# Local blocks with B sharded like A.
b = jax.device_put(jnp.arange(16.0).reshape(8, 2), NamedSharding(mesh, specs))
_, x, _ = jax.jit(jax.shard_map(
    partial(potrs_shardmap_ctx, T_A=2), mesh=mesh,
    in_specs=(specs, specs), out_specs=(specs, specs, P(("pr", "pc"))),
))(a, b)
np.testing.assert_array_equal(x, b)
assert seen[-1]["nrhs"] == 2, seen[-1]

# Local blocks with B replicated over the column axis.
b = jax.device_put(jnp.arange(8.0), NamedSharding(mesh, P("pr")))
_, x, _ = jax.jit(jax.shard_map(
    partial(potrs_shardmap_ctx, T_A=2), mesh=mesh,
    in_specs=(specs, P("pr")), out_specs=(specs, P("pr"), P(("pr", "pc"))),
))(a, b)
np.testing.assert_array_equal(x, b)
print("ok")
"""


def test_potrs_layers_move_data_on_a_2x2_cpu_mesh():
    """Run every layer on four host devices with an identity native solve."""
    env = dict(
        os.environ,
        XLA_FLAGS="--xla_force_host_platform_device_count=4",
        JAX_PLATFORMS="cpu",
    )
    result = subprocess.run(
        [sys.executable, "-c", _TWO_BY_TWO_SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")
