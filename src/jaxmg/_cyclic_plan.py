from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class CyclicColumnMove:
    """One logical column movement in the padded distributed address space."""

    column: int
    source_slot: int
    target_slot: int
    source_rank: int
    target_rank: int
    source_local_col: int
    target_local_col: int


@dataclass(frozen=True)
class CyclicColumnPlan:
    """1D block-to-cyclic layout plan for the current cuSolverMg layout."""

    n: int
    n_batch: int
    tile_size: int
    num_devices: int
    reverse: bool
    moves: tuple[CyclicColumnMove, ...]

    @property
    def shard_size(self) -> int:
        return self.n // self.num_devices

    @property
    def padding_per_device(self) -> int:
        return self.n_batch - self.shard_size

    @property
    def active_moves(self) -> tuple[CyclicColumnMove, ...]:
        return tuple(move for move in self.moves if move.source_slot != move.target_slot)


@dataclass(frozen=True)
class CyclicColumnCycle:
    """Permutation/open-chain path over padded global column slots."""

    slots: tuple[int, ...]

    @property
    def is_closed(self) -> bool:
        return len(self.slots) > 1 and self.slots[0] == self.slots[-1]


@dataclass(frozen=True)
class CyclicColumnTransferStep:
    """One low-memory movement step for a cyclic column path.

    ``kind`` is one of:

    - ``"move"``: copy ``source_slot`` into ``target_slot``.
    - ``"save_scratch"``: copy ``source_slot`` into a one-column scratch buffer.
    - ``"restore_scratch"``: copy the scratch buffer into ``target_slot``.

    The steps are deliberately slot-based so the same schedule can drive local
    CUDA copies, NCCL/XLA communicator sends, or CPU-only tests.
    """

    kind: str
    cycle_index: int
    source_slot: int | None = None
    target_slot: int | None = None


def calculate_cyclic_padding(shard_size: int, tile_size: int) -> int:
    """Return per-device padding needed to make local storage tile-aligned."""
    if shard_size < 0:
        raise ValueError("shard_size must be non-negative.")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive.")
    return (-shard_size) % tile_size


def build_1d_cyclic_column_plan(
    n: int,
    n_batch: int,
    tile_size: int,
    num_devices: int,
    *,
    reverse: bool = False,
) -> CyclicColumnPlan:
    """Build the same padded-column mapping currently used by JAXMg.

    Slots are indexed globally as ``rank * n_batch + local_column``. For the
    forward direction, each move maps an original block-sharded source slot to
    the cuSolverMg 1D block-cyclic target slot. ``reverse=True`` inverts that
    mapping for moving results back to the original block-sharded layout.
    """
    if n < 0:
        raise ValueError("n must be non-negative.")
    if n_batch <= 0:
        raise ValueError("n_batch must be positive.")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive.")
    if num_devices <= 0:
        raise ValueError("num_devices must be positive.")
    if n % num_devices != 0:
        raise ValueError("n must be divisible by num_devices.")

    shard_size = n // num_devices
    if n_batch < shard_size:
        raise ValueError("n_batch must be at least n // num_devices.")

    dst_cols = [0] * num_devices
    dst_rank = -1
    offset = n_batch - shard_size
    moves: list[CyclicColumnMove] = []

    for column in range(n):
        if column % tile_size == 0:
            dst_rank = (dst_rank + 1) % num_devices

        source_slot = column + offset * (column // shard_size)
        target_slot = dst_cols[dst_rank] + dst_rank * n_batch

        if reverse:
            source_slot, target_slot = target_slot, source_slot

        moves.append(
            CyclicColumnMove(
                column=column,
                source_slot=source_slot,
                target_slot=target_slot,
                source_rank=source_slot // n_batch,
                target_rank=target_slot // n_batch,
                source_local_col=source_slot % n_batch,
                target_local_col=target_slot % n_batch,
            )
        )
        dst_cols[dst_rank] += 1

    return CyclicColumnPlan(
        n=n,
        n_batch=n_batch,
        tile_size=tile_size,
        num_devices=num_devices,
        reverse=reverse,
        moves=tuple(moves),
    )


def build_slot_mapping(plan: CyclicColumnPlan) -> dict[int, int]:
    """Return ``source_slot -> target_slot`` for all real columns in ``plan``."""
    return {move.source_slot: move.target_slot for move in plan.moves}


def build_cyclic_column_cycles(plan: CyclicColumnPlan) -> tuple[CyclicColumnCycle, ...]:
    """Build disjoint slot chains/cycles matching the existing C++ algorithm.

    Padding slots have no entry in the source mapping, so paths that reach
    padding are open chains. Closed cycles end by repeating their starting slot.
    """
    col_map: dict[int, tuple[int, bool]] = {
        move.source_slot: (move.target_slot, False) for move in plan.moves
    }
    cycles: dict[int, list[int]] = {}

    for key in list(col_map):
        target, visited = col_map[key]
        if visited:
            continue
        if target == key:
            col_map[key] = (target, True)
            continue

        cycle = [key]
        col_map[key] = (target, True)

        while True:
            dst_entry = col_map.get(target)
            if dst_entry is None:
                cycle.append(target)
                break

            next_target, dst_visited = dst_entry
            if next_target == key:
                cycle.append(target)
                col_map[target] = (next_target, True)
                cycle.append(next_target)
                break

            if dst_visited:
                prior_cycle = cycles.pop(target, None)
                if prior_cycle is not None:
                    cycle.extend(prior_cycle)
                else:
                    cycle.append(target)
                break

            cycle.append(target)
            col_map[target] = (next_target, True)
            target = next_target

        if len(cycle) > 1:
            cycles[key] = cycle

    return tuple(CyclicColumnCycle(tuple(cycle)) for cycle in cycles.values())


def slot_to_rank_local(slot: int, n_batch: int) -> tuple[int, int]:
    """Convert a padded global slot into ``rank, local_column`` coordinates."""
    if slot < 0:
        raise ValueError("slot must be non-negative.")
    if n_batch <= 0:
        raise ValueError("n_batch must be positive.")
    return slot // n_batch, slot % n_batch


def build_low_memory_cyclic_transfer_steps(
    plan: CyclicColumnPlan,
) -> tuple[CyclicColumnTransferStep, ...]:
    """Schedule a one-scratch-column implementation of the cyclic permutation.

    Open chains end in source-side padding, so they can be executed safely from
    the tail back toward the head without staging. Closed cycles need one saved
    column: save the final real slot, shift the remaining slots backward, then
    restore the saved value into the first slot.
    """
    steps: list[CyclicColumnTransferStep] = []

    for cycle_index, cycle in enumerate(build_cyclic_column_cycles(plan)):
        slots = cycle.slots
        if cycle.is_closed:
            steps.append(
                CyclicColumnTransferStep(
                    kind="save_scratch",
                    cycle_index=cycle_index,
                    source_slot=slots[-2],
                )
            )
            for index in range(len(slots) - 3, -1, -1):
                steps.append(
                    CyclicColumnTransferStep(
                        kind="move",
                        cycle_index=cycle_index,
                        source_slot=slots[index],
                        target_slot=slots[index + 1],
                    )
                )
            steps.append(
                CyclicColumnTransferStep(
                    kind="restore_scratch",
                    cycle_index=cycle_index,
                    target_slot=slots[0],
                )
            )
        else:
            for index in range(len(slots) - 2, -1, -1):
                steps.append(
                    CyclicColumnTransferStep(
                        kind="move",
                        cycle_index=cycle_index,
                        source_slot=slots[index],
                        target_slot=slots[index + 1],
                    )
                )

    return tuple(steps)


def iter_rank_local_moves(
    moves: Iterable[CyclicColumnMove],
) -> Iterable[tuple[int, int, int, int]]:
    """Yield ``src_rank, src_local_col, dst_rank, dst_local_col`` tuples."""
    for move in moves:
        yield (
            move.source_rank,
            move.source_local_col,
            move.target_rank,
            move.target_local_col,
        )
