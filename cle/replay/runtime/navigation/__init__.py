"""Replay navigation: undo, goto_fast, goto_sequential, goto_divergence."""

from __future__ import annotations

from cle.replay.contracts import (
    BroadcastFn,
    ReplayOutcome,
    ReplayRuntimeState,
    ReplayStepFn,
)

from .goto import (
    replay_goto_divergence_transaction,
    replay_goto_sequential_transaction,
)
from .undo import replay_undo_transaction

__all__ = [
    "replay_goto_divergence_logic",
    "replay_goto_fast_logic",
    "replay_goto_sequential_logic",
    "replay_undo_logic",
]


def replay_undo_logic(
    state: ReplayRuntimeState,
    broadcast_fn: BroadcastFn,
) -> ReplayOutcome:
    """Undo the last replay step. Returns dict to jsonify."""
    mutation_lock = getattr(state, "replay_mutation_lock", None)
    if mutation_lock is None:
        return replay_undo_transaction(state, broadcast_fn)
    with mutation_lock:
        return replay_undo_transaction(state, broadcast_fn)


def replay_goto_fast_logic(
    state: ReplayRuntimeState,
    target_step: int,
    replay_step_fn: ReplayStepFn,
    broadcast_fn: BroadcastFn,
) -> ReplayOutcome:
    """Compatibility alias for authoritative sequential reconstruction."""
    return replay_goto_sequential_logic(
        state,
        target_step,
        replay_step_fn,
        broadcast_fn,
    )


def replay_goto_sequential_logic(
    state: ReplayRuntimeState,
    target_step: int,
    replay_step_fn: ReplayStepFn,
    broadcast_fn: BroadcastFn,
) -> ReplayOutcome:
    """Jump to a replay step under the shared replay mutation lock."""
    mutation_lock = getattr(state, "replay_mutation_lock", None)
    if mutation_lock is None:
        return replay_goto_sequential_transaction(
            state, target_step, replay_step_fn, broadcast_fn
        )
    with mutation_lock:
        return replay_goto_sequential_transaction(
            state, target_step, replay_step_fn, broadcast_fn
        )


def replay_goto_divergence_logic(
    state: ReplayRuntimeState,
    max_steps: int,
    replay_step_fn: ReplayStepFn,
    broadcast_fn: BroadcastFn,
) -> ReplayOutcome:
    """Step until divergence under the shared replay mutation lock."""
    mutation_lock = getattr(state, "replay_mutation_lock", None)
    if mutation_lock is None:
        return replay_goto_divergence_transaction(
            state, max_steps, replay_step_fn, broadcast_fn
        )
    with mutation_lock:
        return replay_goto_divergence_transaction(
            state, max_steps, replay_step_fn, broadcast_fn
        )
