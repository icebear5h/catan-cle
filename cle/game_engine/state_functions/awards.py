"""Synchronize the public awards and their visible and hidden victory points."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

from cle.game_engine import state_functions
from cle.game_engine.models.player import Color

if TYPE_CHECKING:
    from cle.game_engine.state import GameState


def maintain_longest_road(
    state: GameState,
    previous_road_color: Color | None,
    road_color: Color | None,
    road_lengths: Mapping[Color, int],
) -> None:
    """Synchronize lengths and award VP, including revocation without a successor."""
    for color in state.colors:
        key = state_functions.player_key(state, color)
        state.player_state[f"{key}_LONGEST_ROAD_LENGTH"] = road_lengths.get(color, 0)
        has_road = color == road_color
        delta = 2 * (int(has_road) - int(state.player_state[f"{key}_HAS_ROAD"]))
        state.player_state[f"{key}_HAS_ROAD"] = has_road
        state.player_state[f"{key}_VICTORY_POINTS"] += delta
        state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"] += delta


def maintain_largest_army(
    state: GameState,
    color: Color,
    previous_army_color: Color | None,
    previous_army_size: int | None,
) -> None:
    candidate_size = state_functions.get_played_dev_cards(state, color, "KNIGHT")

    # Skip if army is too small to be considered.
    if candidate_size < 3:
        return

    if previous_army_color is None:
        winner_key = state_functions.player_key(state, color)
        state.player_state[f"{winner_key}_HAS_ARMY"] = True
        state.player_state[f"{winner_key}_VICTORY_POINTS"] += 2
        state.player_state[f"{winner_key}_ACTUAL_VICTORY_POINTS"] += 2
    # get_largest_army supplies an integer size whenever there is a holder.
    elif cast(int, previous_army_size) < candidate_size and previous_army_color != color:
        # switch, remove previous points and award to new king
        winner_key = state_functions.player_key(state, color)
        state.player_state[f"{winner_key}_HAS_ARMY"] = True
        state.player_state[f"{winner_key}_VICTORY_POINTS"] += 2
        state.player_state[f"{winner_key}_ACTUAL_VICTORY_POINTS"] += 2

        loser_key = state_functions.player_key(state, previous_army_color)
        state.player_state[f"{loser_key}_HAS_ARMY"] = False
        state.player_state[f"{loser_key}_VICTORY_POINTS"] -= 2
        state.player_state[f"{loser_key}_ACTUAL_VICTORY_POINTS"] -= 2
    # else: someone else has army and we dont compete
