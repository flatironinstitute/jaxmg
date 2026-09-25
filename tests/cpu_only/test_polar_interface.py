from functools import partial

import numpy as np
import pytest

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P

import jaxmg._polar as polar_module
from jaxmg import polar, polar_shardmap_ctx
from jaxmg._cusolvermp_status import _CUSOLVERMP_POLAR_STATUS_SIZE


def _one_rank_mesh() -> Mesh:
    devices = np.asarray(jax.devices()[:1], dtype=object).reshape(1, 1)
    return Mesh(devices, ("pr", "pc"))


def _install_fake_polar_backend(monkeypatch):
    captured = {}

    def fake_pipeline(*args, **kwargs):
        captured["pipeline_args"] = args
        captured["pipeline_kwargs"] = kwargs

        def impl(_a):
            status = jnp.zeros((_CUSOLVERMP_POLAR_STATUS_SIZE,), dtype=jnp.int32)
            if kwargs["compute_h"]:
                return _a, jnp.eye(kwargs["n"], dtype=_a.dtype), status
            return _a, status

        return impl

    def fake_compiled(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return fake_pipeline(*args, **kwargs)

    monkeypatch.setattr(polar_module, "ensure_init_jaxmg_backend", lambda: None)
    monkeypatch.setattr(polar_module, "_polar_pipeline", fake_pipeline)
    monkeypatch.setattr(polar_module, "_polar_compiled", fake_compiled)
    return captured


def test_polar_returns_both_factors_by_default(monkeypatch):
    _install_fake_polar_backend(monkeypatch)
    up, h = polar(
        jnp.ones((6, 4), dtype=jnp.float32),
        2,
        mesh=_one_rank_mesh(),
        matrix_specs=P("pr", "pc"),
    )
    assert up.shape == (6, 4)
    assert h.shape == (4, 4)


def test_polar_can_omit_h_and_append_status(monkeypatch):
    _install_fake_polar_backend(monkeypatch)
    up, status = polar(
        jnp.ones((6, 4), dtype=jnp.complex64),
        2,
        mesh=_one_rank_mesh(),
        matrix_specs=P("pr", "pc"),
        compute_h=False,
        return_status=True,
    )
    assert up.shape == (6, 4)
    assert status.shape == (_CUSOLVERMP_POLAR_STATUS_SIZE,)


def test_polar_shardmap_ctx_supports_outer_donation(monkeypatch):
    _install_fake_polar_backend(monkeypatch)
    mesh = _one_rank_mesh()
    decomposition = jax.jit(
        partial(
            polar_shardmap_ctx,
            T_A=2,
            mesh=mesh,
            matrix_specs=P("pr", "pc"),
        ),
        donate_argnums=(0,),
    )
    up, h, status = decomposition(jnp.ones((6, 4), dtype=jnp.float32))
    assert up.shape == (6, 4)
    assert h.shape == (4, 4)
    assert status.shape == (_CUSOLVERMP_POLAR_STATUS_SIZE,)


def test_polar_rejects_wide_matrix_and_non_boolean_mode():
    with pytest.raises(ValueError, match="m >= n"):
        polar(jnp.ones((4, 6)), 2)
    with pytest.raises(TypeError, match="compute_h must be a Python bool"):
        polar(jnp.ones((4, 4)), 2, compute_h=1)


def test_polar_rejects_invalid_rank_dtype_and_padding():
    with pytest.raises(ValueError, match="rank-2 matrix A"):
        polar(jnp.ones((4,)), 2)
    with pytest.raises(TypeError, match="supports float32"):
        polar(jnp.ones((4, 4), dtype=jnp.int32), 2)
    with pytest.raises(ValueError, match=r"polar\(A\) requires tile-aligned"):
        polar(
            jnp.ones((6, 4), dtype=jnp.float32),
            4,
            mesh=_one_rank_mesh(),
            matrix_specs=P("pr", "pc"),
            pad=False,
        )


def test_polar_donation_is_configurable(monkeypatch):
    captured = _install_fake_polar_backend(monkeypatch)
    polar(
        jnp.eye(4, dtype=jnp.float32),
        2,
        mesh=_one_rank_mesh(),
        matrix_specs=P("pr", "pc"),
        donate=False,
    )
    assert captured["kwargs"]["donate"] is False
