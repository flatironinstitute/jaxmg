from functools import partial

import numpy as np
import pytest

import jax
import jax.numpy as jnp
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

import jaxmg._potrf as potrf_module
from jaxmg import potrf, potrf_shardmap_ctx
from jaxmg._cusolvermp_status import _CUSOLVERMP_POTRF_STATUS_SIZE


def _one_rank_mesh() -> Mesh:
    devices = np.asarray(jax.devices()[:1], dtype=object).reshape(1, 1)
    return Mesh(devices, ("pr", "pc"))


def _install_fake_potrf_backend(monkeypatch):
    captured = {}

    def fake_pipeline(*args, **kwargs):
        captured["pipeline_args"] = args
        captured["pipeline_kwargs"] = kwargs

        def impl(_a):
            status = jnp.zeros((_CUSOLVERMP_POTRF_STATUS_SIZE,), dtype=jnp.int32)
            return jnp.tril(_a), status

        return impl

    def fake_compiled(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return fake_pipeline(*args, **kwargs)

    monkeypatch.setattr(potrf_module, "ensure_init_jaxmg_backend", lambda: None)
    monkeypatch.setattr(potrf_module, "_potrf_pipeline", fake_pipeline)
    monkeypatch.setattr(potrf_module, "_potrf_compiled", fake_compiled)
    return captured


def test_potrf_returns_factor_and_optional_status(monkeypatch):
    captured = _install_fake_potrf_backend(monkeypatch)
    a = jnp.eye(4, dtype=jnp.float32)

    factor = potrf(a, 2, mesh=_one_rank_mesh(), matrix_specs=P("pr", "pc"))
    assert factor.shape == (4, 4)
    assert captured["kwargs"]["n"] == 4
    assert captured["kwargs"]["tile_size"] == 2

    factor, status = potrf(
        jnp.eye(4, dtype=jnp.float32),
        2,
        mesh=_one_rank_mesh(),
        matrix_specs=P("pr", "pc"),
        return_status=True,
    )
    assert factor.shape == (4, 4)
    assert status.shape == (_CUSOLVERMP_POTRF_STATUS_SIZE,)


def test_potrf_shardmap_ctx_supports_outer_donation(monkeypatch):
    _install_fake_potrf_backend(monkeypatch)
    mesh = _one_rank_mesh()
    factorize = jax.jit(
        partial(potrf_shardmap_ctx, T_A=2, mesh=mesh, matrix_specs=P("pr", "pc")),
        donate_argnums=(0,),
    )
    factor, status = factorize(jnp.eye(4, dtype=jnp.float32))
    assert factor.shape == (4, 4)
    assert status.shape == (_CUSOLVERMP_POTRF_STATUS_SIZE,)


def test_potrf_rejects_invalid_rank_shape_dtype_and_tile_size():
    with pytest.raises(ValueError, match="rank-2 matrix A"):
        potrf(jnp.ones((4,)), 2)
    with pytest.raises(ValueError, match="A to be square"):
        potrf(jnp.ones((4, 3)), 2)
    with pytest.raises(TypeError, match="supports float32"):
        potrf(jnp.eye(4, dtype=jnp.int32), 2)
    with pytest.raises(ValueError, match="T_A must be positive"):
        potrf(jnp.eye(4), 0)


def test_potrf_rejects_required_padding_when_disabled():
    with pytest.raises(ValueError, match=r"potrf\(A\) requires tile-aligned"):
        potrf(
            jnp.eye(3, dtype=jnp.float32),
            2,
            mesh=_one_rank_mesh(),
            matrix_specs=P("pr", "pc"),
            pad=False,
        )


def test_potrf_rejects_ambiguous_spec_arguments():
    with pytest.raises(ValueError, match="Specify only one"):
        potrf(
            jnp.eye(4),
            2,
            mesh=_one_rank_mesh(),
            matrix_specs=P("pr", "pc"),
            in_specs=P("pr", "pc"),
        )


def test_potrf_donation_is_configurable(monkeypatch):
    captured = _install_fake_potrf_backend(monkeypatch)
    potrf(
        jnp.eye(4, dtype=jnp.float32),
        2,
        mesh=_one_rank_mesh(),
        matrix_specs=P("pr", "pc"),
    )
    assert captured["kwargs"]["donate"] is True
    potrf(
        jnp.eye(4, dtype=jnp.float32),
        2,
        mesh=_one_rank_mesh(),
        matrix_specs=P("pr", "pc"),
        donate=False,
    )
    assert captured["kwargs"]["donate"] is False


def test_potrf_pipeline_keeps_factor_sharding(monkeypatch):
    monkeypatch.setattr(potrf_module, "ensure_init_jaxmg_backend", lambda: None)
    devices = np.asarray(jax.devices()[:1], dtype=object)
    mesh = Mesh(devices, ("pr",), axis_types=(AxisType.Explicit,))
    with jax.set_mesh(mesh):
        a = jax.device_put(jnp.eye(4), NamedSharding(mesh, P("pr", None)))
        factor, status = jax.eval_shape(partial(potrf_shardmap_ctx, T_A=2), a)

    assert factor.shape == (4, 4)
    assert factor.dtype == a.dtype
    assert factor.sharding.spec == P("pr", None)
    assert status.shape == (_CUSOLVERMP_POTRF_STATUS_SIZE,)
