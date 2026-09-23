"""Reconcile the engine's required prompt with the replay cursor."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from cle.game_engine.game import GameEngine
from cle.game_engine.models.enums import Action
from cle.replay.colonist.types import ActionHint
from cle.replay.contracts import ParsedActions, ReplayRuntimeState
from cle.replay.runtime.action_matcher import find_matching_action

from ..forcing import force_clear_stale_steal_prompt

__all__ = [
    "DIRECT_EXECUTE_TYPES",
    "REQUIRED_ACTION_TYPES",
    "SKIP_ACTIONS",
    "auto_resolve",
    "resolve_engine_requirement",
]

SKIP_ACTIONS: Final[list[str]] = [
    "OFFER_TRADE",
    "ACCEPT_TRADE",
    "REJECT_TRADE",
    "CLEAR_TRADE_RESPONSE",
    "COUNTER_OFFER",
]
REQUIRED_ACTION_TYPES: Final[set[str]] = {"MOVE_ROBBER", "STEAL", "DISCARD"}
DIRECT_EXECUTE_TYPES: Final[set[str]] = {
    "CLOSE_TRADE", "MARITIME_TRADE", "PLAY_MONOPOLY", "PLAY_YEAR_OF_PLENTY",
    "PLAY_KNIGHT_CARD", "PLAY_ROAD_BUILDING",
    "MONOPOLY_RESOURCE", "YEAR_OF_PLENTY_RESOURCES", "CONFIRM_TRADE",
    "DISCARD", "MOVE_ROBBER", "STEAL",
}


def resolve_engine_requirement(
    state: ReplayRuntimeState,
    game: GameEngine,
    action_hint: ActionHint,
    action_type: str | None,
    parsed_actions: ParsedActions,
    playable: Sequence[Action],
    allow_lookahead: bool,
) -> tuple[Sequence[Action], str | None, str | None]:
    """Return the refreshed playable set plus what the engine demanded."""
    engine_requires = None
    lookahead_executed_type = None
    for a in playable:
        if hasattr(a, 'action_type'):
            atype = str(a.action_type).replace("ActionType.", "")
            if atype in REQUIRED_ACTION_TYPES:
                engine_requires = atype
                break

    if engine_requires and action_type != engine_requires:
        if action_type == "DISCARD" and engine_requires == "MOVE_ROBBER":
            print("[Replay] Engine is ready to move robber, but Colonist has another discard - executing discard first")
        elif action_type == "MOVE_ROBBER" and engine_requires == "STEAL":
            print("[Replay] MOVE_ROBBER already satisfied; waiting for Colonist STEAL action")
        elif action_type in SKIP_ACTIONS:
            print(
                f"[Replay] Engine requires {engine_requires}, but replay has "
                f"{action_type} trade response - waiting for replay cursor"
            )
        elif engine_requires == "STEAL":
            print(
                f"[Replay] Engine requires STEAL, but replay advanced to {action_type}; "
                "clearing stale steal prompt"
            )
            force_clear_stale_steal_prompt(
                state,
                action_hint,
                f"Replay advanced to {action_type} without a STEAL row",
            )
            playable = game.state.playable_actions
        elif allow_lookahead:
            print(f"[Replay] Engine requires {engine_requires}, but replay has {action_type} - looking ahead")
            for lookahead_idx in range(state.replay_index, min(state.replay_index + 50, len(parsed_actions))):
                lookahead_hint = parsed_actions[lookahead_idx]
                if lookahead_hint.get("type") == engine_requires:
                    action = find_matching_action(playable, lookahead_hint, state=state)
                    if action:
                        print(f"[Replay] Found {engine_requires} at replay index {lookahead_idx}, executing")
                        game.step(action, force=True)
                        playable = game.state.playable_actions
                        lookahead_executed_type = engine_requires
                        break

    return playable, engine_requires, lookahead_executed_type


def auto_resolve(
    state: ReplayRuntimeState,
    game: GameEngine,
    action_hint: ActionHint,
    playable: Sequence[Action],
) -> tuple[Action | None, Sequence[Action], int]:
    """Auto-execute trade resolution or END_TURN until the row can match."""
    auto_actions_taken = 0
    action: Action | None = None
    max_auto_actions = 10

    while auto_actions_taken < max_auto_actions:
        cancel_trade_action = None
        for a in playable:
            if hasattr(a, 'action_type') and "CANCEL_TRADE" in str(a.action_type):
                cancel_trade_action = a
                break

        if cancel_trade_action:
            print("[Replay] Auto-executing CANCEL_TRADE to exit trade state")
            game.step(cancel_trade_action, force=True)
            playable = game.state.playable_actions
            auto_actions_taken += 1
            action = find_matching_action(playable, action_hint, state=state)
            if action is not None:
                break
            continue

        end_turn_action = None
        for a in playable:
            if hasattr(a, 'action_type') and "END_TURN" in str(a.action_type):
                end_turn_action = a
                break

        if end_turn_action:
            print("[Replay] Auto-executing END_TURN to sync with Colonist replay")
            game.step(end_turn_action, force=True)
            playable = game.state.playable_actions
            auto_actions_taken += 1
            action = find_matching_action(playable, action_hint, state=state)
            if action is not None:
                break
            continue

        break

    return action, playable, auto_actions_taken
