import jax.numpy as jnp
import pytest

from jaxmg._xla_comm_probe import (
    _validate_rect_pack_args,
    _validate_rect_transfer_args,
)


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


def test_rect_transfer_validation_accepts_valid_schedule():
    matrix = jnp.zeros((4, 5), dtype=jnp.float32)
    scratch = jnp.zeros((8,), dtype=jnp.float32)

    args = _validate_rect_transfer_args(
        matrix,
        scratch,
        layout="column_major",
        targets=[1, 0],
        src_row_starts=[1, 2],
        src_col_starts=[2, 3],
        dst_row_starts=[0, 1],
        dst_col_starts=[1, 0],
        row_count=2,
        col_count=2,
    )

    assert args[0] == 1
    assert args[1].tolist() == [1, 0]
    assert args[2].tolist() == [1, 2]
    assert args[3].tolist() == [2, 3]
    assert args[4].tolist() == [0, 1]
    assert args[5].tolist() == [1, 0]
    assert args[6:] == (2, 2)


def test_rect_transfer_validation_rejects_mismatched_schedule_lengths():
    matrix = jnp.zeros((4, 5), dtype=jnp.float32)
    scratch = jnp.zeros((8,), dtype=jnp.float32)

    with pytest.raises(ValueError, match="match in shape"):
        _validate_rect_transfer_args(
            matrix,
            scratch,
            targets=[1, 0],
            src_row_starts=[1],
            src_col_starts=[2, 3],
            dst_row_starts=[0, 1],
            dst_col_starts=[1, 0],
            row_count=2,
            col_count=2,
        )


def test_rect_transfer_validation_rejects_single_slot_scratch():
    matrix = jnp.zeros((4, 5), dtype=jnp.float32)
    scratch = jnp.zeros((4,), dtype=jnp.float32)

    with pytest.raises(ValueError, match="2 \\* row_count \\* col_count"):
        _validate_rect_transfer_args(
            matrix,
            scratch,
            targets=[1, 0],
            src_row_starts=[1, 2],
            src_col_starts=[2, 3],
            dst_row_starts=[0, 1],
            dst_col_starts=[1, 0],
            row_count=2,
            col_count=2,
        )
