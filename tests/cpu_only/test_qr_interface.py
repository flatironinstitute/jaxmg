from functools import partial

import numpy as np
import pytest

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P

import jaxmg._qr as qr_module
from jaxmg import qr, qr_jit_ctx
from jaxmg._cusolvermp_status import _CUSOLVERMP_QR_STATUS_SIZE


def _one_rank_mesh() -> Mesh:
    devices = np.asarray(jax.devices()[:1], dtype=object).reshape(1, 1)
    return Mesh(devices, ("pr", "pc"))


def _install_fake_qr_backend(monkeypatch):
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

    compiled = qr_module._qr_compiled
    compiled.cache_clear()

    def record_compiled(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return compiled(*args, **kwargs)

    monkeypatch.setattr(qr_module, "ensure_init_jaxmg_backend", lambda: None)
    monkeypatch.setattr(jax.ffi, "ffi_call", fake_ffi)
    monkeypatch.setattr(qr_module, "_qr_compiled", record_compiled)
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


def test_qr_jit_ctx_supports_outer_donation(monkeypatch):
    _install_fake_qr_backend(monkeypatch)
    mesh = _one_rank_mesh()
    decomposition = jax.jit(
        partial(
            qr_jit_ctx,
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
