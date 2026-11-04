from functools import partial

import numpy as np
import pytest

import jax
import jax.numpy as jnp
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

import jaxmg._getrf as getrf_module
from jaxmg import getrf, getrf_shardmap_ctx
from jaxmg._cusolvermp_status import _CUSOLVERMP_GETRF_STATUS_SIZE


def _one_rank_mesh() -> Mesh:
    devices = np.asarray(jax.devices()[:1], dtype=object).reshape(1, 1)
    return Mesh(devices, ("pr", "pc"))


def _install_fake_getrf_backend(monkeypatch):
    captured = {}

    def fake_pipeline(*args, **kwargs):
        captured["pipeline_args"] = args
        captured["pipeline_kwargs"] = kwargs

        def impl(_a):
            piv = jnp.arange(kwargs["n"], dtype=jnp.int32)
            status = jnp.zeros((_CUSOLVERMP_GETRF_STATUS_SIZE,), dtype=jnp.int32)
            return _a, piv, status

        return impl

    def fake_compiled(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return fake_pipeline(*args, **kwargs)

    monkeypatch.setattr(getrf_module, "ensure_init_jaxmg_backend", lambda: None)
    monkeypatch.setattr(getrf_module, "_getrf_pipeline", fake_pipeline)
    monkeypatch.setattr(getrf_module, "_getrf_compiled", fake_compiled)
    return captured


def test_getrf_returns_factor_pivots_and_optional_status(monkeypatch):
    captured = _install_fake_getrf_backend(monkeypatch)

    lu, piv = getrf(
        jnp.eye(4, dtype=jnp.float32),
        2,
        mesh=_one_rank_mesh(),
        matrix_specs=P("pr", "pc"),
    )
    assert lu.shape == (4, 4)
    assert piv.shape == (4,)
    assert piv.dtype == jnp.int32
    assert captured["kwargs"]["n"] == 4
    assert captured["kwargs"]["tile_size"] == 2

    lu, piv, status = getrf(
        jnp.eye(4, dtype=jnp.float32),
        2,
        mesh=_one_rank_mesh(),
        matrix_specs=P("pr", "pc"),
        return_status=True,
    )
    assert lu.shape == (4, 4)
    assert piv.shape == (4,)
    assert status.shape == (_CUSOLVERMP_GETRF_STATUS_SIZE,)


def test_getrf_shardmap_ctx_supports_outer_donation(monkeypatch):
    _install_fake_getrf_backend(monkeypatch)
    mesh = _one_rank_mesh()
    factorize = jax.jit(
        partial(getrf_shardmap_ctx, T_A=2, mesh=mesh, matrix_specs=P("pr", "pc")),
        donate_argnums=(0,),
    )
    lu, piv, status = factorize(jnp.eye(4, dtype=jnp.float32))
    assert lu.shape == (4, 4)
    assert piv.shape == (4,)
    assert status.shape == (_CUSOLVERMP_GETRF_STATUS_SIZE,)


def test_getrf_rejects_invalid_rank_shape_dtype_and_tile_size():
    with pytest.raises(ValueError, match="rank-2 matrix A"):
        getrf(jnp.ones((4,)), 2)
    with pytest.raises(ValueError, match="A to be square"):
        getrf(jnp.ones((4, 3)), 2)
    with pytest.raises(TypeError, match="supports float32"):
        getrf(jnp.eye(4, dtype=jnp.int32), 2)
    with pytest.raises(ValueError, match="T_A must be positive"):
        getrf(jnp.eye(4), 0)


def test_getrf_rejects_required_padding_when_disabled():
    with pytest.raises(ValueError, match=r"getrf\(A\) requires tile-aligned"):
        getrf(
            jnp.eye(3, dtype=jnp.float32),
            2,
            mesh=_one_rank_mesh(),
            matrix_specs=P("pr", "pc"),
            pad=False,
        )


def test_getrf_rejects_ambiguous_spec_arguments():
    with pytest.raises(ValueError, match="Specify only one"):
        getrf(
            jnp.eye(4),
            2,
            mesh=_one_rank_mesh(),
            matrix_specs=P("pr", "pc"),
            in_specs=P("pr", "pc"),
        )


def test_getrf_donation_is_configurable(monkeypatch):
    captured = _install_fake_getrf_backend(monkeypatch)
    getrf(
        jnp.eye(4, dtype=jnp.float32),
        2,
        mesh=_one_rank_mesh(),
        matrix_specs=P("pr", "pc"),
    )
    assert captured["kwargs"]["donate"] is True
    getrf(
        jnp.eye(4, dtype=jnp.float32),
        2,
        mesh=_one_rank_mesh(),
        matrix_specs=P("pr", "pc"),
        donate=False,
    )
    assert captured["kwargs"]["donate"] is False


def test_getrf_pipeline_replicates_pivots(monkeypatch):
    monkeypatch.setattr(getrf_module, "ensure_init_jaxmg_backend", lambda: None)
    devices = np.asarray(jax.devices()[:1], dtype=object)
    mesh = Mesh(devices, ("pr",), axis_types=(AxisType.Explicit,))
    with jax.set_mesh(mesh):
        a = jax.device_put(jnp.eye(4), NamedSharding(mesh, P("pr", None)))
        lu, piv, status = jax.eval_shape(partial(getrf_shardmap_ctx, T_A=2), a)

    assert lu.shape == (4, 4)
    assert lu.sharding.spec == P("pr", None)
    assert piv.shape == (4,)
    assert piv.dtype == jnp.int32
    assert piv.sharding.spec == P(None)
    assert status.shape == (_CUSOLVERMP_GETRF_STATUS_SIZE,)
