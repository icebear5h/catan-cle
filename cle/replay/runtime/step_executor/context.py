"""Shared replay-step bookkeeping: seat mapping, cursor advance, overlays."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from cle.game_engine.game import GameEngine
from cle.game_engine.models.actions import generate_playable_actions
from cle.game_engine.models.player import Color
from cle.game_engine.state import GameState
from cle.replay.colonist.types import ActionHint
from cle.replay.contracts import ParsedActions, ReplayRuntimeState, mapping_field
from cle.replay.runtime.access import get_game_engine
from cle.replay.runtime.audit import record_replay_issue, sync_final_replay_state

__all__ = [
    "TURN_OWNER_ACTIONS",
    "engine_color_for_colonist",
    "engine_of",
    "mark_finished_if_needed",
    "record_forced_overlay",
    "regenerate_playable_actions",
    "seating_map",
    "seat_index",
]


TURN_OWNER_ACTIONS: Final[set[str]] = {
    "ROLL",
    "BUILD_ROAD",
    "BUILD_SETTLEMENT",
    "BUILD_CITY",
    "BUY_DEVELOPMENT_CARD",
    "PLAY_KNIGHT_CARD",
    "PLAY_ROAD_BUILDING",
    "PLAY_MONOPOLY",
    "PLAY_YEAR_OF_PLENTY",
    "MONOPOLY_RESOURCE",
    "YEAR_OF_PLENTY_RESOURCES",
    "MOVE_ROBBER",
    "STEAL",
    "DISCARD",
    "OFFER_TRADE",
    "CONFIRM_TRADE",
    "MARITIME_TRADE",
    "END_TURN",
}


def engine_of(state: ReplayRuntimeState) -> GameEngine:
    """The live engine; every replay path here already requires one."""
    game = get_game_engine(state)
    if game is None:
        raise AttributeError("'NoneType' object has no attribute 'state'")
    return game


def seating_map(state: ReplayRuntimeState) -> Mapping[str, object]:
    """Colonist player id -> engine seat index, as the archive records it."""
    replay_data = state.replay_data or {}
    return mapping_field(replay_data, "colonist_color_to_engine_idx")


def seat_index(seating: Mapping[str, object], colonist_player: object) -> int | None:
    """Read one seat index; non-integer entries were never usable."""
    index = seating.get(str(colonist_player))
    return index if isinstance(index, int) else None


def mark_finished_if_needed(
    state: ReplayRuntimeState,
    parsed_actions: ParsedActions,
) -> bool:
    finished = state.replay_index >= len(parsed_actions)
    state.game_running = not finished
    if finished:
        sync_final_replay_state(state)
    return finished


def engine_color_for_colonist(
    state: ReplayRuntimeState,
    colonist_player: object,
) -> tuple[Color | None, int | None]:
    if colonist_player is None:
        return None, None
    game = engine_of(state)
    player_idx = seat_index(seating_map(state), colonist_player)
    if player_idx is None or player_idx >= len(game.state.colors):
        return None, player_idx
    return game.state.colors[player_idx], player_idx


def regenerate_playable_actions(game_state: GameState) -> None:
    game_state.playable_actions = generate_playable_actions(game_state)


def record_forced_overlay(
    state: ReplayRuntimeState,
    action_hint: ActionHint,
    message: str,
    details: Mapping[str, object] | None = None,
) -> None:
    record_replay_issue(
        state,
        kind="forced_replay_overlay",
        action_hint=action_hint,
        message=message,
        severity="warning",
        details=details,
    )
