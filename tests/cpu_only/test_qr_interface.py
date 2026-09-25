from functools import partial

import numpy as np
import pytest

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P

import jaxmg._qr as qr_module
from jaxmg import qr, qr_shardmap_ctx
from jaxmg._cusolvermp_status import _CUSOLVERMP_QR_STATUS_SIZE


def _one_rank_mesh() -> Mesh:
    devices = np.asarray(jax.devices()[:1], dtype=object).reshape(1, 1)
    return Mesh(devices, ("pr", "pc"))


def _install_fake_qr_backend(monkeypatch):
    captured = {}

    def fake_pipeline(*args, **kwargs):
        captured["pipeline_args"] = args
        captured["pipeline_kwargs"] = kwargs

        def impl(_a):
            q = _a
            r = jnp.eye(kwargs["n"], dtype=_a.dtype)
            status = jnp.zeros((_CUSOLVERMP_QR_STATUS_SIZE,), dtype=jnp.int32)
            return q, r, status

        return impl

    def fake_compiled(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return fake_pipeline(*args, **kwargs)

    monkeypatch.setattr(qr_module, "ensure_init_jaxmg_backend", lambda: None)
    monkeypatch.setattr(qr_module, "_qr_pipeline", fake_pipeline)
    monkeypatch.setattr(qr_module, "_qr_compiled", fake_compiled)
    return captured


def test_qr_returns_reduced_factors_and_status(monkeypatch):
    _install_fake_qr_backend(monkeypatch)
    q, r, status = qr(
        jnp.ones((6, 4), dtype=jnp.float32),
        2,
        mesh=_one_rank_mesh(),
        matrix_specs=P("pr", "pc"),
        return_status=True,
    )
    assert q.shape == (6, 4)
    assert r.shape == (4, 4)
    assert status.shape == (_CUSOLVERMP_QR_STATUS_SIZE,)


def test_qr_shardmap_ctx_supports_outer_donation(monkeypatch):
    _install_fake_qr_backend(monkeypatch)
    mesh = _one_rank_mesh()
    decomposition = jax.jit(
        partial(
            qr_shardmap_ctx,
            T_A=2,
            mesh=mesh,
            matrix_specs=P("pr", "pc"),
        ),
        donate_argnums=(0,),
    )
    q, r, status = decomposition(jnp.ones((6, 4), dtype=jnp.float32))
    assert q.shape == (6, 4)
    assert r.shape == (4, 4)
    assert status.shape == (_CUSOLVERMP_QR_STATUS_SIZE,)


def test_qr_rejects_wide_matrix_invalid_rank_dtype_and_padding():
    with pytest.raises(ValueError, match="m >= n"):
        qr(jnp.ones((4, 6)), 2)
    with pytest.raises(ValueError, match="rank-2 matrix A"):
        qr(jnp.ones((4,)), 2)
    with pytest.raises(TypeError, match="supports float32"):
        qr(jnp.ones((4, 4), dtype=jnp.int32), 2)
    with pytest.raises(ValueError, match=r"qr\(A\) requires tile-aligned"):
        qr(
            jnp.ones((6, 4), dtype=jnp.float32),
            4,
            mesh=_one_rank_mesh(),
            matrix_specs=P("pr", "pc"),
            pad=False,
        )


def test_qr_donation_is_configurable(monkeypatch):
    captured = _install_fake_qr_backend(monkeypatch)
    qr(
        jnp.ones((6, 4), dtype=jnp.float32),
        2,
        mesh=_one_rank_mesh(),
        matrix_specs=P("pr", "pc"),
        donate=False,
    )
    assert captured["kwargs"]["donate"] is False
