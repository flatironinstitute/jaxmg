import pytest

import jaxmg._setup as setup


def test_detect_runtime_mode_spmd(monkeypatch):
    monkeypatch.delenv("JAXMG_NUMBER_OF_DEVICES", raising=False)
    monkeypatch.setattr(setup.jax.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(setup.jax, "local_device_count", lambda: 4)

    assert setup._detect_runtime_mode() == ("SPMD", 4)


def test_detect_runtime_mode_mpmd_uses_explicit_devices(monkeypatch):
    monkeypatch.setenv("JAXMG_NUMBER_OF_DEVICES", "8")
    monkeypatch.setattr(setup.jax.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(setup.jax, "device_count", lambda: 16)

    assert setup._detect_runtime_mode() == ("MPMD", 8)


def test_detect_runtime_mode_mpmd_rejects_bad_device_count(monkeypatch):
    monkeypatch.setenv("JAXMG_NUMBER_OF_DEVICES", "0")
    monkeypatch.setattr(setup.jax.distributed, "is_initialized", lambda: True)

    with pytest.raises(ValueError, match="JAXMG_NUMBER_OF_DEVICES"):
        setup._detect_runtime_mode()


def test_detect_runtime_mode_mpmd_falls_back_to_global_count(monkeypatch):
    monkeypatch.delenv("JAXMG_NUMBER_OF_DEVICES", raising=False)
    monkeypatch.setattr(setup.jax.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(setup.jax, "device_count", lambda: 2)

    assert setup._detect_runtime_mode() == ("MPMD", 2)
