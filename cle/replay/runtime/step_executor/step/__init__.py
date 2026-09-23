"""Execute one parsed Colonist row against the engine."""

from __future__ import annotations

from collections.abc import Sequence

from cle.game_engine.models.enums import Action
from cle.replay.contracts import (
    BroadcastFn,
    ReplayOutcome,
    ReplayRuntimeState,
    parsed_actions_field,
)
from cle.replay.runtime.access import get_game_engine
from cle.replay.runtime.action_matcher import find_matching_action
from cle.replay.runtime.audit import ensure_replay_audit_state
from cle.replay.runtime.trade_ledger import apply_replay_trade_event

from ..direct import handle_direct_execute
from ..forcing import sync_turn_owner_from_hint
from ..trade_steps import handle_trade_response
from .execute import execute_matched_action
from .prepare import (
    DIRECT_EXECUTE_TYPES,
    SKIP_ACTIONS,
    auto_resolve,
    resolve_engine_requirement,
)
from .unmatched import handle_unmatched

__all__ = ["replay_step_logic"]


def replay_step_logic(
    state: ReplayRuntimeState,
    broadcast_fn: BroadcastFn,
    allow_lookahead: bool = True,
) -> ReplayOutcome:
    """Execute one replay step. Returns a dict to be jsonified.

    Args:
        state: ServerState instance
        broadcast_fn: callable to broadcast game state to clients
    """
    game = get_game_engine(state)
    replay_data = state.replay_data
    ensure_replay_audit_state(state)

    if not state.replay_mode or not replay_data or not game:
        return {"error": "No replay loaded"}, 400

    parsed_actions = parsed_actions_field(replay_data)
    if state.replay_index >= len(parsed_actions):
        state.game_running = False
        return {
            "status": "finished",
            "message": "Replay complete",
            "event_index": state.replay_index,
            "total_events": len(parsed_actions),
            "finished": True,
        }

    action_hint = parsed_actions[state.replay_index]
    action_type = action_hint.get("type")

    apply_replay_trade_event(state, action_hint)
    sync_turn_owner_from_hint(action_hint, state)

    playable: Sequence[Action] = game.state.playable_actions
    # A recorded road is authoritative even when the engine rejects its topology.
    # Do not cancel offers, advance turns, or execute future rows to make it legal.
    if action_type == "BUILD_ROAD" and find_matching_action(playable, action_hint, state=state) is None:
        road_result, _ = handle_direct_execute(action_type, action_hint, state)
        broadcast_fn()
        if road_result is None:
            # BUILD_ROAD always resolves to a direct-execution branch.
            raise ValueError("Replay BUILD_ROAD has no direct execution branch")
        return road_result
    if not playable:
        state.game_running = False
        return {"error": "No playable actions", "event_index": state.replay_index}, 400

    auto_actions_taken = 0
    playable, engine_requires, lookahead_executed_type = resolve_engine_requirement(
        state, game, action_hint, action_type, parsed_actions, playable, allow_lookahead
    )

    # Handle async trade responses
    if action_type in ["ACCEPT_TRADE", "REJECT_TRADE"]:
        response, should_continue = handle_trade_response(action_hint, action_type, state)
        if not should_continue and response is not None:
            broadcast_fn()
            return response

    # Try to find matching action
    action = find_matching_action(playable, action_hint, state=state)

    if action is None and action_type in SKIP_ACTIONS:
        pass
    elif action is None and action_type in DIRECT_EXECUTE_TYPES:
        pass
    elif action is None:
        # Auto-execute trade resolution or END_TURN
        action, playable, auto_actions_taken = auto_resolve(
            state, game, action_hint, playable
        )

    if action is None:
        unmatched = handle_unmatched(
            state, game, action_hint, action_type, parsed_actions, playable,
            engine_requires, lookahead_executed_type,
        )
        broadcast_fn()
        return unmatched

    outcome = execute_matched_action(
        state, game, replay_data, action_hint, action, parsed_actions,
        auto_actions_taken,
    )
    if isinstance(outcome, tuple):
        return outcome
    broadcast_fn()
    return outcome
