from functools import partial

import numpy as np
import pytest

import jax
if not jax.config.jax_enable_x64:
    jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P

import jaxmg._syevd as syevd_module
from jaxmg import syevd, syevd_shardmap_ctx
from jaxmg._cusolvermp_status import _CUSOLVERMP_SYEVD_STATUS_SIZE


def _one_rank_mesh() -> Mesh:
    """Return a regular 1x1 matrix mesh usable on CPU-only test hosts."""
    devices = np.asarray(jax.devices()[:1], dtype=object).reshape(1, 1)
    return Mesh(devices, ("pr", "pc"))


def _single_axis_mesh() -> Mesh:
    """Return a mesh with one axis, as used by callers that shard only rows."""
    devices = np.asarray(jax.devices()[:1], dtype=object)
    return Mesh(devices, ("pr",))


def _install_fake_syevd_backend(monkeypatch):
    """Replace native backend entry points with a small Python stand-in."""
    captured = {}

    def fake_pipeline(*args, **kwargs):
        captured["pipeline_args"] = args
        captured["pipeline_kwargs"] = kwargs

        def impl(_a):
            n = int(kwargs["n"])
            eigenvalue_dtype = (
                jnp.float32
                if _a.dtype in (jnp.float32, jnp.complex64)
                else jnp.float64
            )
            eigenvalues = jnp.arange(n, dtype=eigenvalue_dtype)
            status = jnp.zeros((_CUSOLVERMP_SYEVD_STATUS_SIZE,), dtype=jnp.int32)
            if not kwargs["return_eigenvectors"]:
                return eigenvalues, _a, status
            return eigenvalues, _a, _a, status

        return impl

    def fake_compiled(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return fake_pipeline(*args, **kwargs)

    monkeypatch.setattr(syevd_module, "ensure_init_jaxmg_backend", lambda: None)
    monkeypatch.setattr(syevd_module, "_syevd_pipeline", fake_pipeline)
    monkeypatch.setattr(syevd_module, "_syevd_compiled", fake_compiled)
    return captured


def test_syevd_accepts_current_2d_mesh_contract(monkeypatch):
    captured = _install_fake_syevd_backend(monkeypatch)
    a = jnp.eye(4, dtype=jnp.float32)

    eigenvalues, vectors = syevd(
        a,
        2,
        mesh=_one_rank_mesh(),
        matrix_specs=P("pr", "pc"),
    )

    assert eigenvalues.shape == (4,)
    assert vectors.shape == a.shape
    assert captured["kwargs"]["n"] == 4
    assert captured["kwargs"]["tile_size"] == 2
    assert captured["kwargs"]["return_eigenvectors"] is True


def test_syevd_can_return_eigenvalues_only(monkeypatch):
    captured = _install_fake_syevd_backend(monkeypatch)
    a = jnp.eye(4, dtype=jnp.float32)

    eigenvalues = syevd(
        a,
        2,
        mesh=_one_rank_mesh(),
        matrix_specs=P("pr", "pc"),
        return_eigenvectors=False,
    )

    assert eigenvalues.shape == (4,)
    assert captured["kwargs"]["return_eigenvectors"] is False


def test_syevd_values_only_can_return_native_status(monkeypatch):
    _install_fake_syevd_backend(monkeypatch)
    a = jnp.eye(4, dtype=jnp.float64)

    eigenvalues, status = syevd(
        a,
        2,
        mesh=_one_rank_mesh(),
        matrix_specs=P("pr", "pc"),
        return_eigenvectors=False,
        return_status=True,
    )

    assert eigenvalues.shape == (4,)
    assert status.shape == (_CUSOLVERMP_SYEVD_STATUS_SIZE,)


def test_syevd_can_return_native_status(monkeypatch):
    _install_fake_syevd_backend(monkeypatch)
    a = jnp.eye(4, dtype=jnp.float64)

    _, _, status = syevd(
        a,
        2,
        mesh=_one_rank_mesh(),
        matrix_specs=P("pr", "pc"),
        return_status=True,
    )

    assert status.shape == (_CUSOLVERMP_SYEVD_STATUS_SIZE,)


def test_syevd_shardmap_ctx_returns_work_eigensystem_and_status(monkeypatch):
    captured = _install_fake_syevd_backend(monkeypatch)
    a = jnp.eye(4, dtype=jnp.float32)

    a_work, eigenvalues, eigenvectors, status = syevd_shardmap_ctx(
        a,
        2,
        mesh=_one_rank_mesh(),
        matrix_specs=P("pr", "pc"),
    )

    assert a_work.shape == a.shape
    assert eigenvalues.shape == (4,)
    assert eigenvectors.shape == a.shape
    assert status.shape == (_CUSOLVERMP_SYEVD_STATUS_SIZE,)
    assert captured["pipeline_kwargs"]["n"] == 4
    assert captured["pipeline_kwargs"]["tile_size"] == 2
    assert captured["pipeline_kwargs"]["return_eigenvectors"] is True


def test_syevd_shardmap_ctx_returns_values_only_work_and_status(monkeypatch):
    captured = _install_fake_syevd_backend(monkeypatch)
    a = jnp.eye(4, dtype=jnp.float32)

    a_work, eigenvalues, status = syevd_shardmap_ctx(
        a,
        2,
        mesh=_one_rank_mesh(),
        matrix_specs=P("pr", "pc"),
        return_eigenvectors=False,
    )

    assert a_work.shape == a.shape
    assert eigenvalues.shape == (4,)
    assert status.shape == (_CUSOLVERMP_SYEVD_STATUS_SIZE,)
    assert captured["pipeline_kwargs"]["return_eigenvectors"] is False


def test_syevd_shardmap_ctx_can_be_wrapped_in_external_jit(monkeypatch):
    _install_fake_syevd_backend(monkeypatch)
    mesh = _one_rank_mesh()
    eigensolve = jax.jit(
        partial(
            syevd_shardmap_ctx,
            T_A=2,
            mesh=mesh,
            matrix_specs=P("pr", "pc"),
        ),
        donate_argnums=(0,),
    )

    a_work, eigenvalues, eigenvectors, status = eigensolve(
        jnp.eye(4, dtype=jnp.float32)
    )

    assert a_work.shape == (4, 4)
    assert eigenvalues.shape == (4,)
    assert eigenvectors.shape == (4, 4)
    assert status.shape == (_CUSOLVERMP_SYEVD_STATUS_SIZE,)


def test_syevd_external_jit_can_lower_from_shape_specs(monkeypatch):
    _install_fake_syevd_backend(monkeypatch)
    mesh = _one_rank_mesh()
    eigensolve = jax.jit(
        partial(
            syevd_shardmap_ctx,
            T_A=2,
            mesh=mesh,
            matrix_specs=P("pr", "pc"),
        ),
        donate_argnums=(0,),
    )

    compiled = eigensolve.lower(
        jax.ShapeDtypeStruct((4, 4), jnp.float32)
    ).compile()

    assert compiled.memory_analysis() is not None


def test_syevd_values_only_external_jit_can_lower(monkeypatch):
    _install_fake_syevd_backend(monkeypatch)
    mesh = _one_rank_mesh()
    eigensolve = jax.jit(
        partial(
            syevd_shardmap_ctx,
            T_A=2,
            mesh=mesh,
            matrix_specs=P("pr", "pc"),
            return_eigenvectors=False,
        ),
        donate_argnums=(0,),
    )

    compiled = eigensolve.lower(
        jax.ShapeDtypeStruct((4, 4), jnp.float32)
    ).compile()

    assert compiled.memory_analysis() is not None


@pytest.mark.parametrize("value", [0, 1, "N", None])
def test_syevd_rejects_non_boolean_return_eigenvectors(value):
    with pytest.raises(TypeError, match="return_eigenvectors must be a Python bool"):
        syevd(jnp.eye(4), 2, return_eigenvectors=value)


def test_syevd_rejects_non_matrix_a():
    with pytest.raises(ValueError, match="rank-2 matrix A"):
        syevd(jnp.ones((4,)), 2)


def test_syevd_rejects_unsupported_dtype():
    with pytest.raises(TypeError, match="supports float32"):
        syevd(jnp.eye(4, dtype=jnp.int32), 2)


def test_syevd_rejects_non_square_a():
    with pytest.raises(ValueError, match="A to be square"):
        syevd(jnp.ones((4, 3)), 2)


def test_syevd_rejects_nonpositive_tile_size():
    with pytest.raises(ValueError, match="T_A must be positive"):
        syevd(jnp.eye(4), 0)


def test_syevd_rejects_ambiguous_spec_arguments():
    with pytest.raises(ValueError, match="Specify only one"):
        syevd(
            jnp.eye(4),
            2,
            mesh=_one_rank_mesh(),
            matrix_specs=P("pr", "pc"),
            in_specs=P("pr", "pc"),
        )


def test_syevd_accepts_degenerate_column_grid(monkeypatch):
    """A P_r x 1 grid leaves the matrix columns undistributed."""
    captured = _install_fake_syevd_backend(monkeypatch)

    eigenvalues, vectors = syevd(
        jnp.eye(4, dtype=jnp.float32),
        2,
        mesh=_single_axis_mesh(),
        matrix_specs=P("pr", None),
    )

    assert eigenvalues.shape == (4,)
    assert vectors.shape == (4, 4)
    grid = captured["args"][3]
    assert (grid.process_rows, grid.process_cols) == (1, 1)


def test_syevd_accepts_rank_1_matrix_specs(monkeypatch):
    """P('pr') is the same layout as P('pr', None)."""
    captured = _install_fake_syevd_backend(monkeypatch)

    syevd(
        jnp.eye(4, dtype=jnp.float32),
        2,
        mesh=_single_axis_mesh(),
        matrix_specs=P("pr"),
    )

    assert captured["args"][1] == P("pr", None)


def test_syevd_rejects_fully_replicated_matrix_specs():
    with pytest.raises(ValueError, match="at least one matrix axis"):
        syevd(
            jnp.eye(4),
            2,
            mesh=_one_rank_mesh(),
            matrix_specs=P(None, None),
        )


def test_syevd_rejects_required_padding_when_disabled():
    with pytest.raises(ValueError, match="syevd requires tile-aligned"):
        syevd(
            jnp.eye(3),
            2,
            mesh=_one_rank_mesh(),
            matrix_specs=P("pr", "pc"),
            pad=False,
        )


def test_syevd_shardmap_ctx_rejects_required_padding_when_disabled():
    with pytest.raises(
        ValueError, match="syevd_shardmap_ctx requires tile-aligned"
    ):
        syevd_shardmap_ctx(
            jnp.eye(3),
            2,
            mesh=_one_rank_mesh(),
            matrix_specs=P("pr", "pc"),
            pad=False,
        )

def test_syevd_donates_by_default(monkeypatch):
    captured = _install_fake_syevd_backend(monkeypatch)

    syevd(jnp.eye(4, dtype=jnp.float32), 2, mesh=_one_rank_mesh(), matrix_specs=P("pr", "pc"))

    assert captured["kwargs"]["donate"] is True


def test_syevd_donation_can_be_disabled(monkeypatch):
    captured = _install_fake_syevd_backend(monkeypatch)

    syevd(
        jnp.eye(4, dtype=jnp.float32), 2,
        mesh=_one_rank_mesh(),
        matrix_specs=P("pr", "pc"),
        donate=False,
    )

    assert captured["kwargs"]["donate"] is False
