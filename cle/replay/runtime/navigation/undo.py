"""Undo the last authoritative replay step."""

from __future__ import annotations

from cle.replay.contracts import BroadcastFn, ReplayOutcome, ReplayRuntimeState
from cle.replay.runtime.access import get_game_engine
from cle.replay.runtime.checkpoint import ensure_replay_checkpoint_state
from cle.replay.runtime.revision import bump_replay_revision

__all__ = ["replay_undo_transaction"]


def replay_undo_transaction(
    state: ReplayRuntimeState,
    broadcast_fn: BroadcastFn,
) -> ReplayOutcome:
    game = get_game_engine(state)

    if not state.replay_mode or not game:
        return {"error": "No replay loaded"}, 400

    ensure_replay_checkpoint_state(state)
    if state.replay_step_checkpoints:
        checkpoint = state.replay_step_checkpoints[-1]
        if checkpoint.replay_index != state.replay_index - 1:
            return {"error": "Replay undo checkpoint is out of sync"}, 409

        actions_to_undo = (
            state.replay_actions_per_step[-1]
            if state.replay_actions_per_step
            else 0
        )
        action_start = len(checkpoint.game_state.actions)
        undone_actions = [
            str(action) for action in game.state.actions[action_start:]
        ]

        state.replay_step_checkpoints.pop()
        checkpoint.restore(state)
        bump_replay_revision(state)
        broadcast_fn()

        return {
            "status": "ok",
            "event_index": state.replay_index,
            "actions_undone": actions_to_undo,
            "undone_actions": undone_actions,
        }

    if not game.can_undo():
        return {"error": "Nothing to undo"}, 400

    if not state.replay_actions_per_step:
        return {"error": "No steps to undo"}, 400

    actions_to_undo = state.replay_actions_per_step.pop()
    undone_actions = []

    for _ in range(actions_to_undo):
        if game.can_undo():
            undone_action = game.undo()
            undone_actions.append(str(undone_action))

    state.replay_index = max(0, state.replay_index - 1)
    state.game_running = True
    bump_replay_revision(state)

    for _ in range(len(undone_actions)):
        if state.game_log:
            state.game_log.pop()

    broadcast_fn()

    return {
        "status": "ok",
        "event_index": state.replay_index,
        "actions_undone": len(undone_actions),
        "undone_actions": undone_actions,
    }
