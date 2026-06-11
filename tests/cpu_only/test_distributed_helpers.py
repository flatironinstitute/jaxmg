import os

import numpy as np
import pytest

import jaxmg._distributed as distributed


def test_infer_local_device_count_prefers_explicit_jaxmg_env():
    environ = {
        "JAXMG_LOCAL_DEVICE_COUNT": "8",
        "CUDA_VISIBLE_DEVICES": "0,1",
    }

    assert distributed._infer_local_device_count(environ) == 8


def test_infer_local_device_count_from_cuda_visible_devices():
    assert (
        distributed._infer_local_device_count({"CUDA_VISIBLE_DEVICES": "3,4,7,8"})
        == 4
    )


def test_infer_local_device_count_from_slurm_gpu_count():
    assert (
        distributed._infer_local_device_count({"SLURM_GPUS_ON_NODE": "gpu:a100:4"})
        == 4
    )
    assert distributed._infer_local_device_count({"SLURM_GPUS_ON_NODE": "4(S:0-1)"}) == 4
    assert distributed._infer_local_device_count({"SLURM_STEP_GPUS": "0,1,2,3"}) == 4


def test_normalize_local_device_ids_rejects_duplicates():
    with pytest.raises(ValueError, match="duplicates"):
        distributed._normalize_local_device_ids([0, 0])


def test_initialize_node_process_passes_all_local_device_ids(monkeypatch):
    calls = []

    def fake_initialize(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(distributed.jax.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(distributed.jax.distributed, "initialize", fake_initialize)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    monkeypatch.delenv("JAXMG_EXECUTION_MODE", raising=False)
    monkeypatch.delenv("JAXMG_NUMBER_OF_DEVICES", raising=False)

    ids = distributed.initialize_node_process()

    assert ids == (0, 1, 2, 3)
    assert calls[0]["local_device_ids"] == [0, 1, 2, 3]
    assert os.environ["JAXMG_EXECUTION_MODE"] == "SPMD"
    assert os.environ["JAXMG_NUMBER_OF_DEVICES"] == "4"


def test_initialize_node_process_noops_when_already_initialized(monkeypatch):
    def fail_initialize(**_kwargs):
        raise AssertionError("initialize should not be called twice")

    monkeypatch.setattr(distributed.jax.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(distributed.jax.distributed, "initialize", fail_initialize)
    monkeypatch.delenv("JAXMG_EXECUTION_MODE", raising=False)
    monkeypatch.delenv("JAXMG_NUMBER_OF_DEVICES", raising=False)

    ids = distributed.initialize_node_process(local_device_ids=[0, 1])

    assert ids == (0, 1)
    assert os.environ["JAXMG_EXECUTION_MODE"] == "SPMD"
    assert os.environ["JAXMG_NUMBER_OF_DEVICES"] == "2"


def test_initialize_node_process_requires_inferable_device_count(monkeypatch):
    monkeypatch.setattr(distributed.jax.distributed, "is_initialized", lambda: False)
    for name in (
        "JAXMG_LOCAL_DEVICE_COUNT",
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "SLURM_GPUS_ON_NODE",
        "SLURM_GPUS_PER_NODE",
        "SLURM_STEP_GPUS",
    ):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(RuntimeError, match="Could not infer"):
        distributed.initialize_node_process()


def test_row_major_device_grid():
    devices = ["d0", "d1", "d2", "d3"]

    grid = distributed._row_major_device_grid(devices, 2, 2)

    np.testing.assert_array_equal(
        grid,
        np.asarray([["d0", "d1"], ["d2", "d3"]], dtype=object),
    )


def test_row_major_device_grid_rejects_mismatched_shape():
    with pytest.raises(ValueError, match="exactly 6 devices"):
        distributed._row_major_device_grid(["d0", "d1", "d2", "d3"], 2, 3)
