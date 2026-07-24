import pytest

import jaxmg._setup as jaxmg_setup


def test_rank_per_gpu_runtime_accepts_one_local_device(monkeypatch):
    """The cuSOLVERMp backend expects one Python rank per local GPU."""
    monkeypatch.setattr(jaxmg_setup.jax, "local_device_count", lambda: 1)

    jaxmg_setup._validate_rank_per_gpu_runtime()


def test_rank_per_gpu_runtime_rejects_multi_gpu_process(monkeypatch):
    """The old single-process multi-GPU regime is rejected explicitly."""
    monkeypatch.setattr(jaxmg_setup.jax, "local_device_count", lambda: 2)

    with pytest.raises(RuntimeError, match="rank-per-GPU"):
        jaxmg_setup._validate_rank_per_gpu_runtime()


def test_backend_initialization_can_retry_after_failure(monkeypatch):
    """A failed native load must not leave initialization marked complete."""
    calls = 0

    def initialize():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("backend unavailable")

    monkeypatch.setattr(jaxmg_setup, "_initialized", False)
    monkeypatch.setattr(jaxmg_setup, "_initialize", initialize)

    with pytest.raises(OSError, match="backend unavailable"):
        jaxmg_setup.ensure_init_jaxmg_backend()

    assert jaxmg_setup._initialized is False

    jaxmg_setup.ensure_init_jaxmg_backend()

    assert calls == 2
    assert jaxmg_setup._initialized is True
