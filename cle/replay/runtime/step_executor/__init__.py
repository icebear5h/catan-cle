"""Core replay_step logic, decomposed from the monolithic function."""

from __future__ import annotations

from cle.replay.contracts import (
    BroadcastFn,
    ReplayOutcome,
    ReplayRuntimeState,
    parsed_actions_field,
)
from cle.replay.runtime.access import get_game_engine
from cle.replay.runtime.checkpoint import (
    ReplayStepCheckpoint,
    ensure_replay_checkpoint_state,
)
from cle.replay.runtime.revision import bump_replay_revision

from .context import TURN_OWNER_ACTIONS
from .forcing import apply_trade_closures as _apply_trade_closures
from .offers import ensure_root_offer as _ensure_root_offer
from .publishing import publish_replay_action as _publish_replay_action
from .publishing import publish_trade_overlay as _publish_trade_overlay
from .step import replay_step_logic as _replay_step_logic

__all__ = [
    "TURN_OWNER_ACTIONS",
    "_apply_trade_closures",
    "_ensure_root_offer",
    "_publish_replay_action",
    "_publish_trade_overlay",
    "_replay_step_logic",
    "replay_step_logic",
]


def _replay_step_transaction(
    state: ReplayRuntimeState,
    broadcast_fn: BroadcastFn,
    allow_lookahead: bool,
) -> ReplayOutcome:
    game = get_game_engine(state)
    replay_data = state.replay_data
    parsed_actions = parsed_actions_field(replay_data) if replay_data else []

    if (
        not state.replay_mode
        or not replay_data
        or not game
        or state.replay_index >= len(parsed_actions)
    ):
        return _replay_step_logic(
            state, broadcast_fn, allow_lookahead=allow_lookahead
        )

    ensure_replay_checkpoint_state(state)
    checkpoint = ReplayStepCheckpoint.capture(state)

    try:
        result = _replay_step_logic(
            state, broadcast_fn, allow_lookahead=allow_lookahead
        )
    except Exception:
        checkpoint.restore(state)
        raise

    if state.replay_index == checkpoint.replay_index + 1:
        state.replay_actions_per_step[-1] = (
            len(game.state.actions) - len(checkpoint.game_state.actions)
        )
        state.replay_step_checkpoints.append(checkpoint)
        bump_replay_revision(state)
    else:
        checkpoint.restore(state)

    return result


def replay_step_logic(
    state: ReplayRuntimeState,
    broadcast_fn: BroadcastFn,
    allow_lookahead: bool = True,
) -> ReplayOutcome:
    """Execute one parsed replay action as an atomic, undoable transaction.

    Set ``allow_lookahead=False`` for causal consumers that must not inspect
    future parsed rows while revealing the current event.
    """
    mutation_lock = getattr(state, "replay_mutation_lock", None)
    if mutation_lock is None:
        return _replay_step_transaction(state, broadcast_fn, allow_lookahead)
    with mutation_lock:
        return _replay_step_transaction(state, broadcast_fn, allow_lookahead)
