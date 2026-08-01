from functools import partial

import numpy as np
import pytest

import jax

if not jax.config.jax_enable_x64:
    jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P

import jaxmg._gesvd as gesvd_module
from jaxmg import gesvd, gesvd_shardmap_ctx
from jaxmg._cusolvermp_status import _CUSOLVERMP_GESVD_STATUS_SIZE
from jaxmg._layout_types import ProcessGrid, TileShape


def _one_rank_mesh() -> Mesh:
    """Return a regular 1x1 matrix mesh usable on CPU-only test hosts."""
    devices = np.asarray(jax.devices()[:1], dtype=object).reshape(1, 1)
    return Mesh(devices, ("pr", "pc"))


def _install_fake_gesvd_backend(monkeypatch):
    """Replace native GESVD factories with shape-correct Python stand-ins."""
    captured = {}

    def fake_pipeline(*args, **kwargs):
        captured["pipeline_args"] = args
        captured["pipeline_kwargs"] = kwargs

        def impl(_a):
            m = int(kwargs["m"])
            n = int(kwargs["n"])
            k = min(m, n)
            vector_dtype = _a.dtype
            singular_dtype = (
                jnp.float32
                if vector_dtype in (jnp.float32, jnp.complex64)
                else jnp.float64
            )
            singular_values = jnp.arange(k, dtype=singular_dtype)
            status = jnp.zeros((_CUSOLVERMP_GESVD_STATUS_SIZE,), dtype=jnp.int32)
            outputs = [singular_values, _a]
            if kwargs["compute_u"]:
                u_cols = m if kwargs["full_matrices"] else k
                outputs.append(jnp.zeros((m, u_cols), dtype=vector_dtype))
            if kwargs["compute_vh"]:
                vh_rows = n if kwargs["full_matrices"] else k
                outputs.append(jnp.zeros((vh_rows, n), dtype=vector_dtype))
            outputs.append(status)
            return tuple(outputs)

        return impl

    def fake_compiled(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return fake_pipeline(*args, **kwargs)

    monkeypatch.setattr(gesvd_module, "ensure_init_jaxmg_backend", lambda: None)
    monkeypatch.setattr(gesvd_module, "_gesvd_pipeline", fake_pipeline)
    monkeypatch.setattr(gesvd_module, "_gesvd_compiled", fake_compiled)
    return captured


@pytest.mark.parametrize(
    "compute_u,compute_vh,expected_shapes",
    [
        (True, True, ((6, 4), (4,), (4, 4))),
        (True, False, ((6, 4), (4,))),
        (False, True, ((4,), (4, 4))),
        (False, False, ((4,),)),
    ],
)
def test_gesvd_supports_independent_vector_outputs(
    monkeypatch, compute_u, compute_vh, expected_shapes
):
    """Return only the requested reduced SVD outputs in NumPy/JAX order."""
    _install_fake_gesvd_backend(monkeypatch)
    result = gesvd(
        jnp.ones((6, 4), dtype=jnp.float32),
        2,
        mesh=_one_rank_mesh(),
        matrix_specs=P("pr", "pc"),
        compute_u=compute_u,
        compute_vh=compute_vh,
    )
    outputs = result if isinstance(result, tuple) else (result,)
    assert tuple(output.shape for output in outputs) == expected_shapes


@pytest.mark.parametrize(
    "matrix_shape,expected_u_shape,expected_vh_shape",
    [
        ((6, 4), (6, 6), (4, 4)),
        ((4, 6), (4, 4), (6, 6)),
    ],
)
def test_gesvd_full_matrices_uses_full_vector_shapes(
    monkeypatch, matrix_shape, expected_u_shape, expected_vh_shape
):
    """Return square U and Vh matrices when full decomposition is requested."""
    captured = _install_fake_gesvd_backend(monkeypatch)
    u, singular_values, vh = gesvd(
        jnp.ones(matrix_shape, dtype=jnp.float64),
        2,
        mesh=_one_rank_mesh(),
        matrix_specs=P("pr", "pc"),
        full_matrices=True,
    )
    assert u.shape == expected_u_shape
    assert singular_values.shape == (4,)
    assert vh.shape == expected_vh_shape
    assert captured["kwargs"]["full_matrices"] is True


def test_gesvd_can_append_native_status(monkeypatch):
    """Append the fixed per-rank status vector after selected SVD outputs."""
    _install_fake_gesvd_backend(monkeypatch)
    singular_values, status = gesvd(
        jnp.ones((6, 4), dtype=jnp.complex64),
        2,
        mesh=_one_rank_mesh(),
        matrix_specs=P("pr", "pc"),
        compute_u=False,
        compute_vh=False,
        return_status=True,
    )
    assert singular_values.dtype == jnp.float32
    assert status.shape == (_CUSOLVERMP_GESVD_STATUS_SIZE,)


def test_gesvd_shardmap_ctx_exposes_work_and_selected_outputs(monkeypatch):
    """Return A work storage so an enclosing JIT can preserve input donation."""
    _install_fake_gesvd_backend(monkeypatch)
    mesh = _one_rank_mesh()
    decomposition = jax.jit(
        partial(
            gesvd_shardmap_ctx,
            T_A=2,
            mesh=mesh,
            matrix_specs=P("pr", "pc"),
            compute_vh=False,
        ),
        donate_argnums=(0,),
    )
    a_work, u, singular_values, status = decomposition(
        jnp.ones((6, 4), dtype=jnp.float32)
    )
    assert a_work.shape == (6, 4)
    assert u.shape == (6, 4)
    assert singular_values.shape == (4,)
    assert status.shape == (_CUSOLVERMP_GESVD_STATUS_SIZE,)


@pytest.mark.parametrize("option", ["compute_u", "compute_vh", "full_matrices"])
def test_gesvd_requires_static_python_boolean_options(option):
    """Reject dynamic or non-boolean mode selectors before tracing."""
    kwargs = {option: 1}
    with pytest.raises(TypeError, match=f"{option} must be a Python bool"):
        gesvd(jnp.ones((4, 4)), 2, **kwargs)


def test_gesvd_rejects_non_matrix_and_unsupported_dtype():
    """Reject invalid rank and dtype metadata at the public boundary."""
    with pytest.raises(ValueError, match="rank-2 matrix A"):
        gesvd(jnp.ones((4,)), 2)
    with pytest.raises(TypeError, match="supports float32"):
        gesvd(jnp.ones((4, 4), dtype=jnp.int32), 2)


def test_gesvd_rejects_required_padding_when_disabled():
    """Apply pad=False consistently to rectangular A and vector outputs."""
    with pytest.raises(ValueError, match=r"gesvd\(A\) requires tile-aligned"):
        gesvd(
            jnp.ones((6, 4), dtype=jnp.float32),
            4,
            mesh=_one_rank_mesh(),
            matrix_specs=P("pr", "pc"),
            pad=False,
        )


def test_gesvd_rejects_output_shape_not_divisible_by_process_grid():
    """Reject a requested thin output that the shared JAX sharding cannot hold."""
    with pytest.raises(ValueError, match="must be divisible by process grid"):
        gesvd_module._prepare_gesvd_matrix_layout(
            8,
            3,
            ProcessGrid(2, 2),
            TileShape(2, 2),
            pad=True,
            caller="gesvd(U)",
        )
