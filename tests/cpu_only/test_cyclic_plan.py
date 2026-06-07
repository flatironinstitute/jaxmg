import pytest

from jaxmg._cyclic_1d import (
    calculate_padding,
    get_cols_cyclic,
    validate_padded_matrix_rows,
    validate_padded_shard_rows,
)
from jaxmg._cyclic_plan import (
    build_low_memory_cyclic_transfer_steps,
    build_1d_cyclic_column_plan,
    build_cyclic_column_cycles,
    build_slot_mapping,
    calculate_cyclic_padding,
    iter_rank_local_moves,
    slot_to_rank_local,
)


@pytest.mark.parametrize("num_devices", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("tile_size", [1, 2, 3, 5, 8])
@pytest.mark.parametrize("shard_factor", [2, 3, 5, 11])
def test_1d_cyclic_column_plan_matches_legacy_mapping(
    num_devices, tile_size, shard_factor
):
    n = num_devices * shard_factor
    shard_size = n // num_devices
    n_batch = shard_size + calculate_padding(shard_size, tile_size)

    plan = build_1d_cyclic_column_plan(n, n_batch, tile_size, num_devices)
    legacy = get_cols_cyclic(n, n_batch, tile_size, num_devices)

    assert [(m.column, m.source_slot, m.target_slot) for m in plan.moves] == legacy
    assert plan.padding_per_device == calculate_padding(shard_size, tile_size)
    assert plan.padding_per_device == calculate_cyclic_padding(shard_size, tile_size)

    for move in plan.moves:
        assert move.source_slot == move.source_rank * n_batch + move.source_local_col
        assert move.target_slot == move.target_rank * n_batch + move.target_local_col
        assert 0 <= move.source_rank < num_devices
        assert 0 <= move.target_rank < num_devices
        assert 0 <= move.source_local_col < n_batch
        assert 0 <= move.target_local_col < n_batch


@pytest.mark.parametrize("num_devices", [2, 3, 4, 8])
@pytest.mark.parametrize("tile_size", [1, 2, 3, 5])
@pytest.mark.parametrize("shard_factor", [2, 4, 7])
def test_reverse_plan_inverts_forward_mapping(num_devices, tile_size, shard_factor):
    n = num_devices * shard_factor
    shard_size = n // num_devices
    n_batch = shard_size + calculate_padding(shard_size, tile_size)

    forward = build_1d_cyclic_column_plan(n, n_batch, tile_size, num_devices)
    reverse = build_1d_cyclic_column_plan(
        n, n_batch, tile_size, num_devices, reverse=True
    )

    assert len(forward.moves) == len(reverse.moves)
    for fwd, rev in zip(forward.moves, reverse.moves):
        assert rev.column == fwd.column
        assert rev.source_slot == fwd.target_slot
        assert rev.target_slot == fwd.source_slot


@pytest.mark.parametrize("num_devices", [2, 3, 4, 8])
@pytest.mark.parametrize("tile_size", [1, 2, 3, 5, 8])
@pytest.mark.parametrize("shard_factor", [2, 3, 5, 11])
def test_cycle_paths_cover_nontrivial_slot_mapping(
    num_devices, tile_size, shard_factor
):
    n = num_devices * shard_factor
    shard_size = n // num_devices
    n_batch = shard_size + calculate_padding(shard_size, tile_size)
    plan = build_1d_cyclic_column_plan(n, n_batch, tile_size, num_devices)

    mapping = build_slot_mapping(plan)
    cycles = build_cyclic_column_cycles(plan)

    expected_edges = {
        (source, target)
        for source, target in mapping.items()
        if source != target
    }
    cycle_edges = set()
    for cycle in cycles:
        assert len(cycle.slots) > 1
        for source, target in zip(cycle.slots, cycle.slots[1:]):
            if source in mapping:
                cycle_edges.add((source, target))

    assert cycle_edges == expected_edges


def test_iter_rank_local_moves_uses_rank_local_coordinates():
    plan = build_1d_cyclic_column_plan(n=12, n_batch=4, tile_size=2, num_devices=3)

    rank_local_moves = list(iter_rank_local_moves(plan.active_moves))

    assert rank_local_moves
    for move, rank_local in zip(plan.active_moves, rank_local_moves):
        assert rank_local == (
            move.source_rank,
            move.source_local_col,
            move.target_rank,
            move.target_local_col,
        )


def test_validate_padded_rows_uses_logical_matrix_size():
    assert validate_padded_matrix_rows(
        physical_rows=16, logical_size=10, num_devices=2, T_A=4
    ) == 3
    assert validate_padded_shard_rows(
        physical_shard_rows=8, logical_size=10, num_devices=2, T_A=4
    ) == 3

    with pytest.raises(ValueError, match="Expected 16 rows"):
        validate_padded_matrix_rows(
            physical_rows=10, logical_size=10, num_devices=2, T_A=4
        )


@pytest.mark.parametrize("num_devices", [2, 3, 4, 8])
@pytest.mark.parametrize("tile_size", [1, 2, 3, 5, 8])
@pytest.mark.parametrize("shard_factor", [2, 3, 5, 11])
def test_low_memory_transfer_steps_apply_cyclic_mapping(
    num_devices, tile_size, shard_factor
):
    n = num_devices * shard_factor
    shard_size = n // num_devices
    n_batch = shard_size + calculate_padding(shard_size, tile_size)
    plan = build_1d_cyclic_column_plan(n, n_batch, tile_size, num_devices)
    mapping = build_slot_mapping(plan)
    steps = build_low_memory_cyclic_transfer_steps(plan)

    initial = list(range(num_devices * n_batch))
    actual = initial.copy()
    scratch = None
    for step in steps:
        if step.kind == "save_scratch":
            assert step.source_slot is not None
            assert step.target_slot is None
            assert scratch is None
            scratch = actual[step.source_slot]
        elif step.kind == "move":
            assert step.source_slot is not None
            assert step.target_slot is not None
            actual[step.target_slot] = actual[step.source_slot]
        elif step.kind == "restore_scratch":
            assert step.source_slot is None
            assert step.target_slot is not None
            assert scratch is not None
            actual[step.target_slot] = scratch
            scratch = None
        else:
            raise AssertionError(f"unexpected step kind {step.kind!r}")

    assert scratch is None
    for source_slot, target_slot in mapping.items():
        assert actual[target_slot] == initial[source_slot]


@pytest.mark.parametrize("num_devices", [2, 3, 4, 8])
@pytest.mark.parametrize("tile_size", [1, 2, 3, 5])
@pytest.mark.parametrize("shard_factor", [2, 4, 7])
def test_low_memory_transfer_steps_use_one_scratch_per_closed_cycle(
    num_devices, tile_size, shard_factor
):
    n = num_devices * shard_factor
    shard_size = n // num_devices
    n_batch = shard_size + calculate_padding(shard_size, tile_size)
    plan = build_1d_cyclic_column_plan(n, n_batch, tile_size, num_devices)
    cycles = build_cyclic_column_cycles(plan)
    steps = build_low_memory_cyclic_transfer_steps(plan)

    closed_cycles = sum(cycle.is_closed for cycle in cycles)
    assert sum(step.kind == "save_scratch" for step in steps) == closed_cycles
    assert sum(step.kind == "restore_scratch" for step in steps) == closed_cycles
    for step in steps:
        if step.source_slot is not None:
            rank, local_col = slot_to_rank_local(step.source_slot, n_batch)
            assert 0 <= rank < num_devices
            assert 0 <= local_col < n_batch
        if step.target_slot is not None:
            rank, local_col = slot_to_rank_local(step.target_slot, n_batch)
            assert 0 <= rank < num_devices
            assert 0 <= local_col < n_batch


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n": -1, "n_batch": 1, "tile_size": 1, "num_devices": 1},
        {"n": 4, "n_batch": 0, "tile_size": 1, "num_devices": 1},
        {"n": 4, "n_batch": 4, "tile_size": 0, "num_devices": 1},
        {"n": 4, "n_batch": 4, "tile_size": 1, "num_devices": 0},
        {"n": 5, "n_batch": 3, "tile_size": 1, "num_devices": 2},
        {"n": 8, "n_batch": 3, "tile_size": 1, "num_devices": 2},
    ],
)
def test_invalid_plan_parameters_raise(kwargs):
    with pytest.raises(ValueError):
        build_1d_cyclic_column_plan(**kwargs)
