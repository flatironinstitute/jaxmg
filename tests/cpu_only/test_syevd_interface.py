import numpy as np
import pytest

import jax
if not jax.config.jax_enable_x64:
    jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P

import jaxmg._syevd as syevd_module
from jaxmg import syevd
from jaxmg._cusolvermp_status import _CUSOLVERMP_SYEVD_STATUS_SIZE


def _one_rank_mesh() -> Mesh:
    """Return a regular 1x1 matrix mesh usable on CPU-only test hosts."""
    devices = np.asarray(jax.devices()[:1], dtype=object).reshape(1, 1)
    return Mesh(devices, ("pr", "pc"))


def _install_fake_syevd_backend(monkeypatch):
    """Replace native backend entry points with a small Python stand-in."""
    captured = {}

    def fake_compiled(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

        def impl(_a):
            n = int(kwargs["n"])
            eigenvalues = jnp.arange(n, dtype=jnp.float32)
            status = jnp.zeros((_CUSOLVERMP_SYEVD_STATUS_SIZE,), dtype=jnp.int32)
            # impl returns eigenvalues, the donated work buffer, eigenvectors,
            # and status.
            return eigenvalues, _a, _a, status

        return impl

    monkeypatch.setattr(syevd_module, "ensure_init_jaxmg_backend", lambda: None)
    monkeypatch.setattr(syevd_module, "_syevd_compiled", fake_compiled)
    return captured


def test_syevd_accepts_current_2d_mesh_contract(monkeypatch):
    captured = _install_fake_syevd_backend(monkeypatch)
    a = jnp.eye(4, dtype=jnp.float32)

    eigenvalues, vectors = syevd(
        a,
        2,
        mesh=_one_rank_mesh(),
        matrix_specs=P("pr", "pc"),
    )

    assert eigenvalues.shape == (4,)
    assert vectors.shape == a.shape
    assert captured["kwargs"]["n"] == 4
    assert captured["kwargs"]["tile_size"] == 2


def test_syevd_can_return_native_status(monkeypatch):
    _install_fake_syevd_backend(monkeypatch)
    a = jnp.eye(4, dtype=jnp.float64)

    _, _, status = syevd(
        a,
        2,
        mesh=_one_rank_mesh(),
        matrix_specs=P("pr", "pc"),
        return_status=True,
    )

    assert status.shape == (_CUSOLVERMP_SYEVD_STATUS_SIZE,)


def test_syevd_rejects_non_matrix_a():
    with pytest.raises(ValueError, match="rank-2 matrix A"):
        syevd(jnp.ones((4,)), 2)


def test_syevd_rejects_unsupported_dtype():
    with pytest.raises(TypeError, match="supports float32"):
        syevd(jnp.eye(4, dtype=jnp.int32), 2)


def test_syevd_rejects_non_square_a():
    with pytest.raises(ValueError, match="A to be square"):
        syevd(jnp.ones((4, 3)), 2)


def test_syevd_rejects_nonpositive_tile_size():
    with pytest.raises(ValueError, match="T_A must be positive"):
        syevd(jnp.eye(4), 0)


def test_syevd_rejects_ambiguous_spec_arguments():
    with pytest.raises(ValueError, match="Specify only one"):
        syevd(
            jnp.eye(4),
            2,
            mesh=_one_rank_mesh(),
            matrix_specs=P("pr", "pc"),
            in_specs=P("pr", "pc"),
        )


def test_syevd_rejects_1d_matrix_specs():
    with pytest.raises(ValueError, match="both matrix axes"):
        syevd(
            jnp.eye(4),
            2,
            mesh=_one_rank_mesh(),
            matrix_specs=P("pr", None),
        )


def test_syevd_rejects_required_padding_when_disabled():
    with pytest.raises(ValueError, match="syevd requires tile-aligned"):
        syevd(
            jnp.eye(3),
            2,
            mesh=_one_rank_mesh(),
            matrix_specs=P("pr", "pc"),
            pad=False,
        )
