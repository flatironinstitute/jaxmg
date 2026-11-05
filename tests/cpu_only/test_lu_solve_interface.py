from functools import partial

import numpy as np
import pytest

import jax

if not jax.config.jax_enable_x64:
    jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P

import jaxmg._lu_solve as lu_solve_module
from jaxmg import lu_solve, lu_solve_jit_ctx
from jaxmg._cusolvermp_status import _CUSOLVERMP_LU_SOLVE_STATUS_SIZE


def _one_rank_mesh() -> Mesh:
    """Return a regular 1x1 matrix mesh usable on CPU-only test hosts."""
    devices = np.asarray(jax.devices()[:1], dtype=object).reshape(1, 1)
    return Mesh(devices, ("pr", "pc"))


def _single_axis_mesh() -> Mesh:
    """Return a mesh with one axis, as used by callers that shard only rows."""
    devices = np.asarray(jax.devices()[:1], dtype=object)
    return Mesh(devices, ("pr",))


def _install_fake_lu_solve_backend(monkeypatch):
    """Run the public wrappers with only the native FFI replaced."""
    captured = {}

    def fake_ffi(target, result_types, **options):
        def impl(*inputs, **attrs):
            captured["native_kwargs"] = attrs
            captured["target"] = target
            aliases = {out: arg for arg, out in options["input_output_aliases"].items()}
            return tuple(inputs[aliases[i]] if i in aliases else jnp.zeros(t.shape, t.dtype)
                         for i, t in enumerate(result_types))
        return impl

    compiled = lu_solve_module._lu_solve_compiled
    compiled.cache_clear()

    def record_compiled(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return compiled(*args, **kwargs)

    monkeypatch.setattr(lu_solve_module, "ensure_init_jaxmg_backend", lambda: None)
    monkeypatch.setattr(jax.ffi, "ffi_call", fake_ffi)
    monkeypatch.setattr(lu_solve_module, "_lu_solve_compiled", record_compiled)
    return captured


def test_lu_solve_accepts_current_2d_mesh_contract(monkeypatch):
    captured = _install_fake_lu_solve_backend(monkeypatch)
    a = jnp.eye(4, dtype=jnp.float32)
    b = jnp.ones((4, 2), dtype=jnp.float32)

    out = lu_solve(a, b, 2, mesh=_one_rank_mesh(), matrix_specs=P("pr", "pc"))

    assert out.shape == b.shape
    assert captured["native_kwargs"]["n"] == 4
    assert captured["native_kwargs"]["nrhs"] == 2
    assert captured["native_kwargs"]["tile_size"] == 2


def test_lu_solve_preserves_vector_rhs_rank(monkeypatch):
    captured = _install_fake_lu_solve_backend(monkeypatch)
    a = jnp.eye(4, dtype=jnp.float64)
    b = jnp.ones((4,), dtype=jnp.float64)

    out, status = lu_solve(
        a,
        b,
        2,
        mesh=_one_rank_mesh(),
        matrix_specs=P("pr", "pc"),
        return_status=True,
    )

    assert out.shape == b.shape
    assert status.shape == (_CUSOLVERMP_LU_SOLVE_STATUS_SIZE,)
    assert captured["native_kwargs"]["nrhs"] == 1


def test_lu_solve_preserves_single_column_rhs_rank(monkeypatch):
    _install_fake_lu_solve_backend(monkeypatch)
    a = jnp.eye(4, dtype=jnp.float64)
    b = jnp.ones((4, 1), dtype=jnp.float64)

    out = lu_solve(
        a,
        b,
        2,
        mesh=_one_rank_mesh(),
        matrix_specs=P("pr", "pc"),
    )

    assert out.shape == b.shape


def test_lu_solve_public_api_can_be_wrapped_in_external_jit(monkeypatch):
    _install_fake_lu_solve_backend(monkeypatch)
    mesh = _one_rank_mesh()
    solve = jax.jit(
        partial(lu_solve, T_A=2, mesh=mesh, matrix_specs=P("pr", "pc")),
        donate_argnums=(0, 1),
    )

    out = solve(
        jnp.eye(4, dtype=jnp.float32),
        jnp.ones((4, 1), dtype=jnp.float32),
    )

    assert out.shape == (4, 1)


def test_lu_solve_jit_ctx_returns_work_solution_and_status(monkeypatch):
    captured = _install_fake_lu_solve_backend(monkeypatch)
    a = jnp.eye(4, dtype=jnp.float32)
    b = jnp.ones((4, 2), dtype=jnp.float32)

    a_work, out, status = lu_solve_jit_ctx(
        a,
        b,
        2,
        mesh=_one_rank_mesh(),
        matrix_specs=P("pr", "pc"),
    )

    assert a_work.shape == a.shape
    assert out.shape == b.shape
    assert status.shape == (_CUSOLVERMP_LU_SOLVE_STATUS_SIZE,)
    assert captured["native_kwargs"]["n"] == 4
    assert captured["native_kwargs"]["nrhs"] == 2
    assert captured["native_kwargs"]["tile_size"] == 2


def test_lu_solve_jit_ctx_preserves_vector_rhs_rank(monkeypatch):
    captured = _install_fake_lu_solve_backend(monkeypatch)
    a = jnp.eye(4, dtype=jnp.float64)
    b = jnp.ones((4,), dtype=jnp.float64)

    a_work, out, status = lu_solve_jit_ctx(
        a,
        b,
        2,
        mesh=_one_rank_mesh(),
        matrix_specs=P("pr", "pc"),
    )

    assert a_work.shape == a.shape
    assert out.shape == b.shape
    assert status.shape == (_CUSOLVERMP_LU_SOLVE_STATUS_SIZE,)
    assert captured["native_kwargs"]["nrhs"] == 1


def test_lu_solve_jit_ctx_can_be_wrapped_in_external_jit(monkeypatch):
    _install_fake_lu_solve_backend(monkeypatch)
    mesh = _one_rank_mesh()
    solve = jax.jit(
        partial(lu_solve_jit_ctx, T_A=2, mesh=mesh, matrix_specs=P("pr", "pc")),
        donate_argnums=(0, 1),
    )

    a_work, out, status = solve(
        jnp.eye(4, dtype=jnp.float32),
        jnp.ones((4, 1), dtype=jnp.float32),
    )

    assert a_work.shape == (4, 4)
    assert out.shape == (4, 1)
    assert status.shape == (_CUSOLVERMP_LU_SOLVE_STATUS_SIZE,)


def test_lu_solve_external_jit_can_lower_from_shape_specs(monkeypatch):
    _install_fake_lu_solve_backend(monkeypatch)
    mesh = _one_rank_mesh()
    solve = jax.jit(
        partial(lu_solve, T_A=2, mesh=mesh, matrix_specs=P("pr", "pc")),
        donate_argnums=(0, 1),
    )

    compiled = solve.lower(
        jax.ShapeDtypeStruct((4, 4), jnp.float32),
        jax.ShapeDtypeStruct((4, 1), jnp.float32),
    ).compile()

    assert compiled.memory_analysis() is not None


def test_lu_solve_rejects_non_matrix_a():
    with pytest.raises(ValueError, match="rank-2 matrix A"):
        lu_solve(jnp.ones((4,)), jnp.ones((4, 1)), 2)


def test_lu_solve_rejects_non_matrix_rhs():
    with pytest.raises(ValueError, match="rank-1 or rank-2 RHS"):
        lu_solve(jnp.eye(4), jnp.ones((4, 1, 1)), 2)


def test_lu_solve_rejects_mismatched_dtypes():
    with pytest.raises(TypeError, match="matching A/B dtypes"):
        lu_solve(
            jnp.eye(4, dtype=jnp.float32),
            jnp.ones((4, 1), dtype=jnp.float64),
            2,
        )


def test_lu_solve_rejects_unsupported_dtype():
    with pytest.raises(TypeError, match="supports float32"):
        lu_solve(jnp.eye(4, dtype=jnp.int32), jnp.ones((4, 1), dtype=jnp.int32), 2)


def test_lu_solve_rejects_non_square_a():
    with pytest.raises(ValueError, match="A to be square"):
        lu_solve(jnp.ones((4, 3)), jnp.ones((4, 1)), 2)


def test_lu_solve_rejects_leading_dimension_mismatch():
    with pytest.raises(ValueError, match="matching leading dimensions"):
        lu_solve(jnp.eye(4), jnp.ones((5, 1)), 2)


def test_lu_solve_rejects_nonpositive_tile_size():
    with pytest.raises(ValueError, match="T_A must be positive"):
        lu_solve(jnp.eye(4), jnp.ones((4, 1)), 0)


def test_lu_solve_rejects_ambiguous_spec_arguments():
    with pytest.raises(ValueError, match="Specify only one"):
        lu_solve(
            jnp.eye(4),
            jnp.ones((4, 1)),
            2,
            mesh=_one_rank_mesh(),
            matrix_specs=P("pr", "pc"),
            in_specs=P("pr", "pc"),
        )


def test_lu_solve_accepts_degenerate_column_grid(monkeypatch):
    """A P_r x 1 grid leaves the matrix columns undistributed."""
    captured = _install_fake_lu_solve_backend(monkeypatch)
    b = jnp.ones((4, 1), dtype=jnp.float32)

    out = lu_solve(
        jnp.eye(4, dtype=jnp.float32),
        b,
        2,
        mesh=_single_axis_mesh(),
        matrix_specs=P("pr", None),
    )

    assert out.shape == b.shape
    grid = captured["args"][0][0].grid
    assert (grid.process_rows, grid.process_cols) == (1, 1)


def test_lu_solve_accepts_rank_1_matrix_specs(monkeypatch):
    """P('pr') is normalized to the equivalent P('pr', None)."""
    captured = _install_fake_lu_solve_backend(monkeypatch)
    b = jnp.ones((4, 1), dtype=jnp.float32)

    out = lu_solve(
        jnp.eye(4, dtype=jnp.float32),
        b,
        2,
        mesh=_single_axis_mesh(),
        matrix_specs=P("pr"),
    )

    assert out.shape == b.shape
    assert captured["args"][0][0].matrix_specs == P("pr", None)


def test_lu_solve_rejects_fully_replicated_matrix_specs():
    with pytest.raises(ValueError, match="at least one matrix axis"):
        lu_solve(
            jnp.eye(4),
            jnp.ones((4, 1)),
            2,
            mesh=_one_rank_mesh(),
            matrix_specs=P(None, None),
        )


def test_lu_solve_rejects_required_a_padding_when_disabled():
    with pytest.raises(ValueError, match="lu_solve\\(A\\) requires tile-aligned"):
        lu_solve(
            jnp.eye(3),
            jnp.ones((3, 1)),
            2,
            mesh=_one_rank_mesh(),
            matrix_specs=P("pr", "pc"),
            pad=False,
        )


def test_lu_solve_rejects_required_rhs_padding_when_disabled():
    with pytest.raises(ValueError, match="lu_solve\\(B\\) requires tile-aligned"):
        lu_solve(
            jnp.eye(4),
            jnp.ones((4, 1)),
            2,
            mesh=_one_rank_mesh(),
            matrix_specs=P("pr", "pc"),
            pad=False,
        )

def test_lu_solve_donates_by_default(monkeypatch):
    captured = _install_fake_lu_solve_backend(monkeypatch)

    lu_solve(jnp.eye(4, dtype=jnp.float32), jnp.ones((4, 1), dtype=jnp.float32), 2, mesh=_one_rank_mesh(), matrix_specs=P("pr", "pc"))

    assert captured["kwargs"]["donate"] is True


def test_lu_solve_donation_can_be_disabled(monkeypatch):
    captured = _install_fake_lu_solve_backend(monkeypatch)

    lu_solve(
        jnp.eye(4, dtype=jnp.float32), jnp.ones((4, 1), dtype=jnp.float32), 2,
        mesh=_one_rank_mesh(),
        matrix_specs=P("pr", "pc"),
        donate=False,
    )

    assert captured["kwargs"]["donate"] is False
