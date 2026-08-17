from functools import partial

import numpy as np
import pytest

import jax
if not jax.config.jax_enable_x64:
    jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P

import jaxmg._potrs as potrs_module
from jaxmg import potrs, potrs_shardmap_ctx
from jaxmg._cusolvermp_status import _CUSOLVERMP_POTRS_STATUS_SIZE


def _one_rank_mesh() -> Mesh:
    """Return a regular 1x1 matrix mesh usable on CPU-only test hosts."""
    devices = np.asarray(jax.devices()[:1], dtype=object).reshape(1, 1)
    return Mesh(devices, ("pr", "pc"))


def _install_fake_potrs_backend(monkeypatch):
    """Replace native backend entry points with a small Python stand-in."""
    captured = {}

    def fake_pipeline(*args, **kwargs):
        captured["pipeline_args"] = args
        captured["pipeline_kwargs"] = kwargs

        def impl(_a, _b):
            status = jnp.zeros((_CUSOLVERMP_POTRS_STATUS_SIZE,), dtype=jnp.int32)
            if kwargs.get("return_logdet", False):
                logdet_dtype = (
                    jnp.float32
                    if _a.dtype in (jnp.float32, jnp.complex64)
                    else jnp.float64
                )
                logdet = jnp.asarray([3.25], dtype=logdet_dtype)
                return _a, _b, logdet, status
            # impl returns the donated A work buffer, the solved RHS, and status.
            return _a, _b, status

        return impl

    def fake_compiled(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return fake_pipeline(*args, **kwargs)

    monkeypatch.setattr(potrs_module, "ensure_init_jaxmg_backend", lambda: None)
    monkeypatch.setattr(potrs_module, "_potrs_pipeline", fake_pipeline)
    monkeypatch.setattr(potrs_module, "_potrs_compiled", fake_compiled)
    return captured


def test_potrs_accepts_current_2d_mesh_contract(monkeypatch):
    captured = _install_fake_potrs_backend(monkeypatch)
    a = jnp.eye(4, dtype=jnp.float32)
    b = jnp.ones((4, 2), dtype=jnp.float32)

    out = potrs(a, b, 2, mesh=_one_rank_mesh(), matrix_specs=P("pr", "pc"))

    assert out.shape == b.shape
    assert captured["kwargs"]["n"] == 4
    assert captured["kwargs"]["nrhs"] == 2
    assert captured["kwargs"]["tile_size"] == 2


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
    assert captured["kwargs"]["nrhs"] == 1


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


def test_potrs_shardmap_ctx_returns_work_solution_and_status(monkeypatch):
    captured = _install_fake_potrs_backend(monkeypatch)
    a = jnp.eye(4, dtype=jnp.float32)
    b = jnp.ones((4, 2), dtype=jnp.float32)

    a_work, out, status = potrs_shardmap_ctx(
        a,
        b,
        2,
        mesh=_one_rank_mesh(),
        matrix_specs=P("pr", "pc"),
    )

    assert a_work.shape == a.shape
    assert out.shape == b.shape
    assert status.shape == (_CUSOLVERMP_POTRS_STATUS_SIZE,)
    assert captured["pipeline_kwargs"]["n"] == 4
    assert captured["pipeline_kwargs"]["nrhs"] == 2
    assert captured["pipeline_kwargs"]["tile_size"] == 2


def test_potrs_shardmap_ctx_preserves_vector_rhs_rank(monkeypatch):
    captured = _install_fake_potrs_backend(monkeypatch)
    a = jnp.eye(4, dtype=jnp.float64)
    b = jnp.ones((4,), dtype=jnp.float64)

    a_work, out, status = potrs_shardmap_ctx(
        a,
        b,
        2,
        mesh=_one_rank_mesh(),
        matrix_specs=P("pr", "pc"),
    )

    assert a_work.shape == a.shape
    assert out.shape == b.shape
    assert status.shape == (_CUSOLVERMP_POTRS_STATUS_SIZE,)
    assert captured["pipeline_kwargs"]["nrhs"] == 1


def test_potrs_shardmap_ctx_returns_optional_logdet(monkeypatch):
    captured = _install_fake_potrs_backend(monkeypatch)
    a = jnp.eye(4, dtype=jnp.float32)
    b = jnp.ones((4, 2), dtype=jnp.float32)

    a_work, out, logdet, status = potrs_shardmap_ctx(
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
    assert captured["pipeline_kwargs"]["return_logdet"] is True


def test_potrs_shardmap_ctx_can_be_wrapped_in_external_jit(monkeypatch):
    _install_fake_potrs_backend(monkeypatch)
    mesh = _one_rank_mesh()
    solve = jax.jit(
        partial(potrs_shardmap_ctx, T_A=2, mesh=mesh, matrix_specs=P("pr", "pc")),
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


def test_potrs_rejects_1d_matrix_specs():
    with pytest.raises(ValueError, match="both matrix axes"):
        potrs(
            jnp.eye(4),
            jnp.ones((4, 1)),
            2,
            mesh=_one_rank_mesh(),
            matrix_specs=P("pr", None),
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
    captured = _install_fake_potrs_backend(monkeypatch)

    potrs(jnp.eye(4, dtype=jnp.float32), jnp.ones((4, 1), dtype=jnp.float32), 2, mesh=_one_rank_mesh(), matrix_specs=P("pr", "pc"))

    assert captured["kwargs"]["donate"] is True


def test_potrs_donation_can_be_disabled(monkeypatch):
    captured = _install_fake_potrs_backend(monkeypatch)

    potrs(
        jnp.eye(4, dtype=jnp.float32), jnp.ones((4, 1), dtype=jnp.float32), 2,
        mesh=_one_rank_mesh(),
        matrix_specs=P("pr", "pc"),
        donate=False,
    )

    assert captured["kwargs"]["donate"] is False
