"""Mutate piece caches, supply, hands, and points after board placement."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cle.game_engine import state_functions
from cle.game_engine.models.enums import CITY, ROAD, SETTLEMENT
from cle.game_engine.models.player import Color

if TYPE_CHECKING:
    from cle.game_engine.state import GameState


def build_settlement(state: GameState, color: Color, node_id: int, is_free: bool) -> None:
    state.buildings_by_color[color][SETTLEMENT].append(node_id)

    key = state_functions.player_key(state, color)
    state.player_state[f"{key}_SETTLEMENTS_AVAILABLE"] -= 1

    state.player_state[f"{key}_VICTORY_POINTS"] += 1
    state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"] += 1

    if not is_free:
        state.player_state[f"{key}_WOOD_IN_HAND"] -= 1
        state.player_state[f"{key}_BRICK_IN_HAND"] -= 1
        state.player_state[f"{key}_SHEEP_IN_HAND"] -= 1
        state.player_state[f"{key}_WHEAT_IN_HAND"] -= 1


def build_road(state: GameState, color: Color, edge: tuple[int, int], is_free: bool) -> None:
    state.buildings_by_color[color][ROAD].append(edge)

    key = state_functions.player_key(state, color)
    state.player_state[f"{key}_ROADS_AVAILABLE"] -= 1
    if not is_free:
        state.player_state[f"{key}_WOOD_IN_HAND"] -= 1
        state.player_state[f"{key}_BRICK_IN_HAND"] -= 1
        state.resource_freqdeck = state_functions.freqdeck_add(
            state.resource_freqdeck, state_functions.ROAD_COST_FREQDECK
        )  # replenish bank


def build_city(state: GameState, color: Color, node_id: int) -> None:
    state.buildings_by_color[color][SETTLEMENT].remove(node_id)
    state.buildings_by_color[color][CITY].append(node_id)

    key = state_functions.player_key(state, color)
    state.player_state[f"{key}_SETTLEMENTS_AVAILABLE"] += 1
    state.player_state[f"{key}_CITIES_AVAILABLE"] -= 1

    state.player_state[f"{key}_VICTORY_POINTS"] += 1
    state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"] += 1

    state.player_state[f"{key}_WHEAT_IN_HAND"] -= 2
    state.player_state[f"{key}_ORE_IN_HAND"] -= 3
