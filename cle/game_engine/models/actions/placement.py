"""Board-placement moves: roads, settlements, cities, the robber, steals, discards."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cle.game_engine.models.decks import (
    CITY_COST_FREQDECK,
    ROAD_COST_FREQDECK,
    SETTLEMENT_COST_FREQDECK,
)
from cle.game_engine.models.enums import SETTLEMENT, Action, ActionType
from cle.game_engine.models.player import Color
from cle.game_engine.state_functions import (
    get_player_buildings,
    player_key,
    player_num_resource_cards,
    player_resource_freqdeck_contains,
)

if TYPE_CHECKING:
    from cle.game_engine.state import GameState


def road_building_possibilities(
    state: GameState, color: Color, check_money: bool = True
) -> list[Action]:
    key = player_key(state, color)

    # Check if can't build any more roads.
    has_roads_available = state.player_state[f"{key}_ROADS_AVAILABLE"] > 0
    if not has_roads_available:
        return []

    # Check if need to pay for roads but can't afford them.
    has_money = player_resource_freqdeck_contains(state, color, ROAD_COST_FREQDECK)
    if check_money and not has_money:
        return []

    buildable_edges = state.board.buildable_edges(color)
    return [Action(color, ActionType.BUILD_ROAD, edge) for edge in buildable_edges]


def settlement_possibilities(
    state: GameState, color: Color, initial_build_phase: bool = False
) -> list[Action]:
    key = player_key(state, color)
    has_settlements_available = (
        state.player_state[f"{key}_SETTLEMENTS_AVAILABLE"] > 0
    )
    if not has_settlements_available:
        return []

    if initial_build_phase:
        buildable_node_ids = state.board.buildable_node_ids(
            color, initial_build_phase=True
        )
        return [
            Action(color, ActionType.BUILD_SETTLEMENT, node_id)
            for node_id in buildable_node_ids
        ]

    has_money = player_resource_freqdeck_contains(
        state, color, SETTLEMENT_COST_FREQDECK
    )
    if not has_money:
        return []

    buildable_node_ids = state.board.buildable_node_ids(color)
    return [
        Action(color, ActionType.BUILD_SETTLEMENT, node_id)
        for node_id in buildable_node_ids
    ]


def city_possibilities(state: GameState, color: Color) -> list[Action]:
    key = player_key(state, color)

    can_buy_city = player_resource_freqdeck_contains(state, color, CITY_COST_FREQDECK)
    if not can_buy_city:
        return []

    has_cities_available = state.player_state[f"{key}_CITIES_AVAILABLE"] > 0
    if not has_cities_available:
        return []

    return [
        Action(color, ActionType.BUILD_CITY, node_id)
        for node_id in get_player_buildings(state, color, SETTLEMENT)
    ]


def robber_possibilities(state: GameState, color: Color) -> list[Action]:
    """Generate possible MOVE_ROBBER actions (tile coordinates only).

    STEAL is now a separate action that happens after moving the robber.
    """
    actions: list[Action] = []
    for coordinate, tile in state.board.map.land_tiles.items():
        if coordinate == state.board.robber_coordinate:
            continue  # ignore. must move robber.

        actions.append(Action(color, ActionType.MOVE_ROBBER, coordinate))

    return actions


def steal_possibilities(state: GameState, color: Color) -> list[Action]:
    """Generate possible STEAL actions from players at robber's tile.

    Called after MOVE_ROBBER to determine who to steal from.
    Returns empty list if no valid targets (no players with cards).
    """
    actions: list[Action] = []
    robber_tile = state.board.map.land_tiles.get(state.board.robber_coordinate)
    if robber_tile is None:
        return actions

    to_steal_from: set[Color] = set()
    for node_id in robber_tile.nodes.values():
        building = state.board.buildings.get(node_id, None)
        if building is not None:
            candidate_color = building[0]
            if (
                player_num_resource_cards(state, candidate_color) >= 1
                and color != candidate_color  # can't steal from yourself
            ):
                to_steal_from.add(candidate_color)

    for enemy_color in to_steal_from:
        actions.append(Action(color, ActionType.STEAL, (enemy_color, None)))

    return actions


def initial_road_possibilities(state: GameState, color: Color) -> list[Action]:
    key = player_key(state, color)
    if state.player_state[f"{key}_ROADS_AVAILABLE"] <= 0:
        return []

    # Must be connected to last settlement
    last_settlement_node_id = state.buildings_by_color[color][SETTLEMENT][-1]

    buildable_edges = filter(
        lambda edge: last_settlement_node_id in edge,
        state.board.buildable_edges(color),
    )
    return [Action(color, ActionType.BUILD_ROAD, edge) for edge in buildable_edges]


def discard_possibilities(color: Color) -> list[Action]:
    """One parameterized choice; None retains explicit engine auto-discard."""
    return [Action(color, ActionType.DISCARD, None)]
