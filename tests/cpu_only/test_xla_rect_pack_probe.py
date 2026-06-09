import jax.numpy as jnp
import pytest

from jaxmg._xla_comm_probe import _validate_rect_pack_args


def test_rect_pack_validation_accepts_valid_fragment():
    matrix = jnp.zeros((4, 5), dtype=jnp.float32)
    scratch = jnp.zeros((6,), dtype=jnp.float32)

    args = _validate_rect_pack_args(
        matrix,
        scratch,
        row_start=1,
        col_start=2,
        row_count=2,
        col_count=3,
        target_row=0,
        target_col=1,
    )

    assert args == (0, 1, 2, 2, 3, 0, 1)


def test_rect_pack_validation_accepts_column_major_layout():
    matrix = jnp.zeros((4, 5), dtype=jnp.float32)
    scratch = jnp.zeros((6,), dtype=jnp.float32)

    args = _validate_rect_pack_args(
        matrix,
        scratch,
        layout="column_major",
        row_start=1,
        col_start=2,
        row_count=2,
        col_count=3,
        target_row=0,
        target_col=1,
    )

    assert args == (1, 1, 2, 2, 3, 0, 1)


def test_rect_pack_validation_rejects_unknown_layout():
    matrix = jnp.zeros((4, 5), dtype=jnp.float32)
    scratch = jnp.zeros((4,), dtype=jnp.float32)

    with pytest.raises(ValueError, match="layout"):
        _validate_rect_pack_args(
            matrix,
            scratch,
            layout="blocked",
            row_start=0,
            col_start=0,
            row_count=1,
            col_count=1,
            target_row=0,
            target_col=0,
        )


def test_rect_pack_validation_rejects_bad_rank():
    matrix = jnp.zeros((4,), dtype=jnp.float32)
    scratch = jnp.zeros((4,), dtype=jnp.float32)

    with pytest.raises(ValueError, match="rank-2 matrix"):
        _validate_rect_pack_args(
            matrix,
            scratch,
            row_start=0,
            col_start=0,
            row_count=1,
            col_count=1,
            target_row=0,
            target_col=0,
        )


def test_rect_pack_validation_rejects_dtype_mismatch():
    matrix = jnp.zeros((4, 5), dtype=jnp.float32)
    scratch = jnp.zeros((4,), dtype=jnp.complex64)

    with pytest.raises(TypeError, match="dtypes must match"):
        _validate_rect_pack_args(
            matrix,
            scratch,
            row_start=0,
            col_start=0,
            row_count=1,
            col_count=1,
            target_row=0,
            target_col=0,
        )


def test_rect_pack_validation_rejects_out_of_bounds_fragment():
    matrix = jnp.zeros((4, 5), dtype=jnp.float32)
    scratch = jnp.zeros((6,), dtype=jnp.float32)

    with pytest.raises(ValueError, match="fit inside matrix"):
        _validate_rect_pack_args(
            matrix,
            scratch,
            row_start=2,
            col_start=2,
            row_count=3,
            col_count=2,
            target_row=0,
            target_col=0,
        )


def test_rect_pack_validation_rejects_small_scratch():
    matrix = jnp.zeros((4, 5), dtype=jnp.float32)
    scratch = jnp.zeros((5,), dtype=jnp.float32)

    with pytest.raises(ValueError, match="row_count \\* col_count"):
        _validate_rect_pack_args(
            matrix,
            scratch,
            row_start=0,
            col_start=0,
            row_count=2,
            col_count=3,
            target_row=0,
            target_col=0,
        )
