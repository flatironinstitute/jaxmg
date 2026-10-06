"""A native failure must give NaN results, as JAX's own linear algebra does."""

import numpy as np
import pytest

import jax

if not jax.config.jax_enable_x64:
    jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.sharding import Mesh

import jaxmg
from jaxmg._cusolvermp_layout import nan_on_native_failure
from jaxmg import _gesvd, _least_squares, _lu_solve, _polar, _potrs, _qr, _syevd

_MODULES = (_gesvd, _least_squares, _lu_solve, _polar, _potrs, _qr, _syevd)
_FAILED = 26  # any non-zero status code


@pytest.fixture
def fake_native_backend(monkeypatch):
    """Run the real pipelines with a native call returning finite output.

    The fake fills every numerical output with ones, and writes ``status_code``
    in the first word of each rank's status vector.
    """

    def install(status_code):
        def ffi_call(target_name, out_types, **kwargs):
            def call(*args, **attrs):
                outputs = []
                for out_type in out_types:
                    if out_type.dtype == jnp.int32:
                        status = jnp.zeros(out_type.shape, jnp.int32)
                        outputs.append(status.at[0].set(status_code))
                    else:
                        outputs.append(jnp.ones(out_type.shape, out_type.dtype))
                return tuple(outputs)

            return call

        monkeypatch.setattr(jax.ffi, "ffi_call", ffi_call)
        for module in _MODULES:
            monkeypatch.setattr(module, "ensure_init_jaxmg_backend", lambda: None)
            for name in dir(module):
                cached = getattr(module, name)
                if hasattr(cached, "cache_clear"):
                    cached.cache_clear()
        jax.clear_caches()

    return install


def _calls():
    a = jnp.eye(8)
    b = jnp.ones((8,))
    return {
        "potrs": lambda: jaxmg.potrs(a, b, 4, donate=False, return_status=True),
        "potrs_logdet": lambda: jaxmg.potrs(
            a, b, 4, donate=False, return_logdet=True, return_status=True
        ),
        "potrs_shardmap_ctx": lambda: jaxmg.potrs_shardmap_ctx(a, b, 4)[1:],
        "lu_solve": lambda: jaxmg.lu_solve(a, b, 4, donate=False, return_status=True),
        "lu_solve_shardmap_ctx": lambda: jaxmg.lu_solve_shardmap_ctx(a, b, 4)[1:],
        "least_squares": lambda: jaxmg.least_squares(
            a, b, 4, donate=False, return_status=True
        ),
        "least_squares_shardmap_ctx": lambda: jaxmg.least_squares_shardmap_ctx(
            a, b, 4
        )[2:],
        "syevd": lambda: jaxmg.syevd(a, 4, donate=False, return_status=True),
        "syevd_values": lambda: jaxmg.syevd(
            a, 4, return_eigenvectors=False, donate=False, return_status=True
        ),
        "syevd_shardmap_ctx": lambda: jaxmg.syevd_shardmap_ctx(a, 4)[1:],
        "gesvd": lambda: jaxmg.gesvd(a, 4, donate=False, return_status=True),
        "gesvd_values": lambda: jaxmg.gesvd(
            a, 4, compute_u=False, compute_vh=False, donate=False, return_status=True
        ),
        "gesvd_shardmap_ctx": lambda: jaxmg.gesvd_shardmap_ctx(a, 4)[1:],
        "polar": lambda: jaxmg.polar(a, 4, donate=False, return_status=True),
        "polar_up": lambda: jaxmg.polar(
            a, 4, compute_h=False, donate=False, return_status=True
        ),
        "polar_shardmap_ctx": lambda: jaxmg.polar_shardmap_ctx(a, 4),
        "qr": lambda: jaxmg.qr(a, 4, donate=False, return_status=True),
        "qr_shardmap_ctx": lambda: jaxmg.qr_shardmap_ctx(a, 4),
    }


@pytest.mark.parametrize("status_code", [0, _FAILED], ids=["ok", "failed"])
@pytest.mark.parametrize("name", list(_calls()))
def test_native_failure_fills_results_with_nans(
    fake_native_backend, name, status_code
):
    fake_native_backend(status_code)
    mesh = Mesh(np.asarray(jax.devices()[:1], dtype=object), ("x",))
    with jax.set_mesh(mesh):
        *results, status = _calls()[name]()

    assert int(status[0]) == status_code
    # The work buffers, which are not results, are not part of `results`.
    for result in results:
        assert np.all(np.isnan(result)) == (status_code != 0), result


@pytest.mark.parametrize("failed_rank", [None, 0, 2])
def test_a_failure_on_any_rank_fills_every_result(failed_rank):
    num_processes, status_size = 4, 5
    status = np.full((num_processes, status_size), 7, np.int32)
    status[:, 0] = 0
    if failed_rank is not None:
        status[failed_rank, 0] = _FAILED
    x = jnp.ones((3,))
    z = jnp.ones((2, 2), jnp.complex64)

    x, z = nan_on_native_failure(jnp.asarray(status.reshape(-1)), num_processes, x, z)

    assert np.all(np.isnan(x)) == (failed_rank is not None)
    assert np.all(np.isnan(z)) == (failed_rank is not None)
    assert z.dtype == jnp.complex64
