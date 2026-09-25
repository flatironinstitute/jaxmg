from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P

if not jax.config.jax_enable_x64:
    jax.config.update("jax_enable_x64", True)

import jaxmg._least_squares as least_squares_module
from jaxmg import least_squares, least_squares_shardmap_ctx
from jaxmg._cusolvermp_status import _CUSOLVERMP_LEAST_SQUARES_STATUS_SIZE
from jaxmg._layout_types import ProcessGrid, TileShape


def _one_rank_mesh() -> Mesh:
    devices = np.asarray(jax.devices()[:1], dtype=object).reshape(1, 1)
    return Mesh(devices, ("pr", "pc"))


def _install_fake_backend(monkeypatch):
    """Replace the native pipeline with a shape-correct stand-in."""
    captured = {}

    def fake_pipeline(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

        def impl(_a, _b):
            status = jnp.zeros(
                (_CUSOLVERMP_LEAST_SQUARES_STATUS_SIZE,), dtype=jnp.int32
            )
            return _a, _b, _b[: kwargs["n"]], status

        return impl

    monkeypatch.setattr(
        least_squares_module, "ensure_init_jaxmg_backend", lambda: None
    )
    monkeypatch.setattr(
        least_squares_module, "_least_squares_pipeline", fake_pipeline
    )
    monkeypatch.setattr(
        least_squares_module, "_least_squares_compiled", fake_pipeline
    )
    return captured


def test_least_squares_returns_solution_shape_and_status(monkeypatch):
    captured = _install_fake_backend(monkeypatch)
    x, status = least_squares(
        jnp.ones((6, 4), dtype=jnp.float32),
        jnp.ones((6, 2), dtype=jnp.float32),
        2,
        mesh=_one_rank_mesh(),
        matrix_specs=P("pr", "pc"),
        return_status=True,
    )
    assert x.shape == (4, 2)
    assert status.shape == (_CUSOLVERMP_LEAST_SQUARES_STATUS_SIZE,)
    assert captured["kwargs"]["m"] == 6
    assert captured["kwargs"]["n"] == 4


def test_least_squares_preserves_vector_rank(monkeypatch):
    _install_fake_backend(monkeypatch)
    x = least_squares(
        jnp.ones((6, 4)),
        jnp.ones((6,)),
        2,
        mesh=_one_rank_mesh(),
        matrix_specs=P("pr", "pc"),
    )
    assert x.shape == (4,)


def test_least_squares_context_supports_external_jit(monkeypatch):
    _install_fake_backend(monkeypatch)
    mesh = _one_rank_mesh()
    solve = jax.jit(
        partial(
            least_squares_shardmap_ctx,
            T_A=2,
            mesh=mesh,
            matrix_specs=P("pr", "pc"),
        ),
        donate_argnums=(0, 1),
    )
    a_work, b_work, x, status = solve(jnp.ones((6, 4)), jnp.ones((6, 1)))
    assert a_work.shape == (6, 4)
    assert b_work.shape == (6, 1)
    assert x.shape == (4, 1)
    assert status.shape == (_CUSOLVERMP_LEAST_SQUARES_STATUS_SIZE,)


def test_least_squares_rejects_incompatible_solution_sharding(monkeypatch):
    class FakeMesh:
        shape = {"pr": 2}

    monkeypatch.setattr(
        least_squares_module,
        "prepare_input_matrix_layout",
        lambda *args, **kwargs: SimpleNamespace(
            mesh=FakeMesh(),
            matrix_specs=P("pr", None),
            grid=ProcessGrid(2, 1),
            tile_shape=TileShape(1, 1),
        ),
    )
    monkeypatch.setattr(
        least_squares_module,
        "infer_rhs_specs",
        lambda *args, **kwargs: P("pr", None),
    )

    with pytest.raises(ValueError, match="row-axis extent divides N"):
        least_squares_module._prepare_least_squares_call(
            jnp.ones((6, 3)),
            jnp.ones((6, 1)),
            1,
            None,
            None,
            in_specs=None,
            pad=True,
            caller="least_squares",
        )


def test_least_squares_rejects_empty_rhs_process_column(monkeypatch):
    class FakeMesh:
        shape = {"pc": 2}

    monkeypatch.setattr(
        least_squares_module,
        "prepare_input_matrix_layout",
        lambda *args, **kwargs: SimpleNamespace(
            mesh=FakeMesh(),
            matrix_specs=P(None, "pc"),
            grid=ProcessGrid(1, 2),
            tile_shape=TileShape(64, 64),
        ),
    )
    monkeypatch.setattr(
        least_squares_module,
        "infer_rhs_specs",
        lambda *args, **kwargs: P(None, None),
    )
    with pytest.raises(ValueError, match=r"least_squares\(B\).*own at least one"):
        least_squares_module._prepare_least_squares_call(
            jnp.ones((192, 96)),
            jnp.ones((192, 3)),
            64,
            None,
            None,
            in_specs=None,
            pad=True,
            caller="least_squares",
        )


@pytest.mark.parametrize(
    "a,b,message",
    [
        (jnp.ones((6,)), jnp.ones((6,)), "rank-2 matrix A"),
        (jnp.ones((4, 6)), jnp.ones((4,)), "requires M >= N"),
        (jnp.ones((6, 4)), jnp.ones((5,)), "matching leading dimensions"),
    ],
)
def test_least_squares_rejects_invalid_shapes(a, b, message):
    with pytest.raises(ValueError, match=message):
        least_squares(a, b, 2)


def test_least_squares_rejects_mismatched_or_unsupported_dtypes():
    with pytest.raises(TypeError, match="matching A/B dtypes"):
        least_squares(
            jnp.ones((6, 4), dtype=jnp.float32),
            jnp.ones((6,), dtype=jnp.float64),
            2,
        )
    with pytest.raises(TypeError, match="supports float32"):
        least_squares(
            jnp.ones((6, 4), dtype=jnp.int32),
            jnp.ones((6,), dtype=jnp.int32),
            2,
        )
