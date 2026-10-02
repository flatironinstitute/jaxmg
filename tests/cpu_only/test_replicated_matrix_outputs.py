from functools import partial

import numpy as np
import pytest

import jax
import jax.numpy as jnp
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

import jaxmg._gesvd as gesvd_module
import jaxmg._polar as polar_module
import jaxmg._qr as qr_module
import jaxmg._syevd as syevd_module


@pytest.mark.parametrize(
    "module,fn,result_indices,work_index",
    [
        (syevd_module, syevd_module.syevd_shardmap_ctx, (2,), 0),
        (gesvd_module, gesvd_module.gesvd_shardmap_ctx, (1, 3), 0),
        (qr_module, qr_module.qr_shardmap_ctx, (0, 1), None),
        (polar_module, polar_module.polar_shardmap_ctx, (0, 1), None),
    ],
)
def test_decompositions_restore_replicated_matrix_results(
    monkeypatch, module, fn, result_indices, work_index
):
    monkeypatch.setattr(module, "ensure_init_jaxmg_backend", lambda: None)
    devices = np.asarray(jax.devices()[:1], dtype=object)
    mesh = Mesh(devices, ("pr",), axis_types=(AxisType.Explicit,))
    with jax.set_mesh(mesh):
        a = jax.device_put(jnp.eye(4), NamedSharding(mesh, P()))
        outputs = jax.eval_shape(partial(fn, T_A=2), a)

    for index in result_indices:
        assert outputs[index].sharding.spec == P(None, None)
    if work_index is not None:
        assert outputs[work_index].sharding.spec == P("pr", None)
