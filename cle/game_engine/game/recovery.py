"""Repairs applied when restoring engine snapshots written by older engines."""

from __future__ import annotations

import copy
from collections.abc import Iterable

from cle.game_engine.events import HistoryEntry
from cle.game_engine.models.actions import generate_playable_actions
from cle.game_engine.models.enums import ActionType
from cle.game_engine.state import GameState
from cle.game_engine.state_functions import maintain_longest_road, player_key


def copy_history(entries: Iterable[HistoryEntry]) -> list[HistoryEntry]:
    """Deep-copy undo history entries, keeping the event-count integer as is."""
    return [
        (
            copy.deepcopy(state),
            copy.deepcopy(action),
            event_count,
            tuple(copy.deepcopy(commitments)),
        )
        for state, action, event_count, commitments in entries
    ]


def repair_restored_road_state(state: GameState) -> None:
    """Rebuild derived road facts when a restored board disagrees with its pieces.

    Pre-fix saves can retain enemy-crossing candidates and road awards.
    Probe only derived board fields, keeping the saved holder as the tie input.
    """
    board = state.board
    rebuilt = copy.copy(board)
    road_result = rebuilt.recompute_road_state()
    road_changed = (
        rebuilt.road_color != board.road_color
        or rebuilt.road_length != board.road_length
    )
    for color in state.colors:
        key = player_key(state, color)
        length = rebuilt.road_lengths.get(color, 0)
        if (
            length != board.road_lengths.get(color, 0)
            or length != state.player_state[f"{key}_LONGEST_ROAD_LENGTH"]
            or (color == rebuilt.road_color) != state.player_state[f"{key}_HAS_ROAD"]
            or {frozenset(nodes) for nodes in rebuilt.connected_components.get(color, [])}
            != {frozenset(nodes) for nodes in board.connected_components.get(color, [])}
        ):
            road_changed = True
            break
    if not (
        road_changed
        or any(
            set(edges) != set(rebuilt.buildable_edges(color))
            for color, edges in board.buildable_edges_cache.items()
        )
        or any(
            action.action_type == ActionType.BUILD_ROAD
            and action.value not in rebuilt.buildable_edges(action.color)
            for action in state.playable_actions
        )
    ):
        return
    for color, edges in board.buildable_edges_cache.items():
        if set(edges) == set(rebuilt.buildable_edges(color)):
            rebuilt.buildable_edges_cache[color] = edges
    state.board = rebuilt
    maintain_longest_road(state, *road_result)
    playable_actions = generate_playable_actions(state)
    # Keep saved menu indices only for an exact multiset match; values can be mutable.
    unmatched = state.playable_actions.copy()
    for action in playable_actions:
        try:
            unmatched.remove(action)
        except ValueError:
            break
    else:
        if not unmatched:
            playable_actions = state.playable_actions
    state.playable_actions = playable_actions
