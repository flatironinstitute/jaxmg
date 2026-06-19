import numpy as np
import pytest

import jax
if not jax.config.jax_enable_x64:
    jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P

import jaxmg._potrs as potrs_module
from jaxmg import potrs
from jaxmg._cusolvermp_status import _CUSOLVERMP_POTRS_STATUS_SIZE


def _one_rank_mesh() -> Mesh:
    """Return a regular 1x1 matrix mesh usable on CPU-only test hosts."""
    devices = np.asarray(jax.devices()[:1], dtype=object).reshape(1, 1)
    return Mesh(devices, ("pr", "pc"))


def _install_fake_potrs_backend(monkeypatch):
    """Replace native backend entry points with a small Python stand-in."""
    captured = {}

    def fake_compiled(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

        def impl(_a, _b):
            status = jnp.zeros((_CUSOLVERMP_POTRS_STATUS_SIZE,), dtype=jnp.int32)
            return _b, status

        return impl

    monkeypatch.setattr(potrs_module, "ensure_init_jaxmg_backend", lambda: None)
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


def test_potrs_treats_vector_rhs_as_single_column(monkeypatch):
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

    assert out.shape == (4, 1)
    assert status.shape == (_CUSOLVERMP_POTRS_STATUS_SIZE,)
    assert captured["kwargs"]["nrhs"] == 1


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
