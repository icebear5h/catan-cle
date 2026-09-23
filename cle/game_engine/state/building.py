"""Piece placement and development card purchases, with supply and cost guards."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cle.game_engine.models.actions import (
    generate_playable_actions,
    road_building_possibilities,
)
from cle.game_engine.models.decks import (
    CITY_COST_FREQDECK,
    DEVELOPMENT_CARD_COST_FREQDECK,
    ROAD_COST_FREQDECK,
    SETTLEMENT_COST_FREQDECK,
    draw_from_listdeck,
    freqdeck_add,
    freqdeck_draw,
)
from cle.game_engine.models.enums import SETTLEMENT, Action, ActionPrompt
from cle.game_engine.models.player import Color
from cle.game_engine.state.turns import advance_turn
from cle.game_engine.state_functions import (
    build_city,
    build_road,
    build_settlement,
    buy_dev_card,
    maintain_longest_road,
    player_can_afford_dev_card,
    player_key,
    player_resource_freqdeck_contains,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cle.game_engine.state.core import GameState


def require_available_piece(
    state: GameState,
    color: Color,
    field: str,
    label: str,
) -> None:
    """Reject construction before the board mutates if no piece remains."""
    key = player_key(state, color)
    if state.player_state[f"{key}_{field}_AVAILABLE"] <= 0:
        raise ValueError(f"No {label} pieces available for {color.value}")


def require_build_resources(
    state: GameState,
    color: Color,
    cost: Sequence[int],
    label: str,
) -> None:
    """Reject paid construction before it can create cards in the bank."""
    if not player_resource_freqdeck_contains(state, color, cost):
        raise ValueError(f"{color.value} cannot afford to build a {label}")


def apply_build_settlement(state: GameState, action: Action, force: bool = False) -> None:
    require_available_piece(
        state,
        action.color,
        "SETTLEMENTS",
        "settlement",
    )
    if not state.is_initial_build_phase:
        require_build_resources(
            state,
            action.color,
            SETTLEMENT_COST_FREQDECK,
            "settlement",
        )
    node_id = action.value
    if state.is_initial_build_phase:
        result = state.board.build_settlement(action.color, node_id, True)
        build_settlement(state, action.color, node_id, True)
        maintain_longest_road(state, *result)
        buildings = state.buildings_by_color[action.color][SETTLEMENT]

        # yield resources if second settlement
        is_second_house = len(buildings) == 2
        if is_second_house:
            key = player_key(state, action.color)
            for tile in state.board.map.adjacent_tiles[node_id]:
                if tile.resource is not None:
                    freqdeck_draw(state.resource_freqdeck, 1, tile.resource)
                    state.player_state[f"{key}_{tile.resource}_IN_HAND"] += 1

        # state.current_player_index stays the same
        state.current_prompt = ActionPrompt.BUILD_INITIAL_ROAD
        state.playable_actions = generate_playable_actions(state)
    else:
        (
            previous_road_color,
            road_color,
            road_lengths,
        ) = state.board.build_settlement(action.color, node_id, False)
        build_settlement(state, action.color, node_id, False)
        state.resource_freqdeck = freqdeck_add(
            state.resource_freqdeck, SETTLEMENT_COST_FREQDECK
        )  # replenish bank
        maintain_longest_road(state, previous_road_color, road_color, road_lengths)

        # state.current_player_index stays the same
        # state.current_prompt stays as PLAY
        state.playable_actions = generate_playable_actions(state)


def apply_build_road(state: GameState, action: Action, force: bool = False) -> None:
    require_available_piece(state, action.color, "ROADS", "road")
    is_free = state.is_initial_build_phase or (
        state.is_road_building and state.free_roads_available > 0
    )
    if not is_free:
        require_build_resources(state, action.color, ROAD_COST_FREQDECK, "road")
    edge = action.value
    if state.is_initial_build_phase:
        result = state.board.build_road(action.color, edge)
        build_road(state, action.color, edge, True)
        maintain_longest_road(state, *result)

        # state.current_player_index depend on what index are we
        # state.current_prompt too
        buildings = [
            len(state.buildings_by_color[color][SETTLEMENT])
            for color in state.color_to_index.keys()
        ]
        num_buildings = sum(buildings)
        num_players = len(buildings)
        going_forward = num_buildings < num_players
        at_the_end = num_buildings == num_players
        if going_forward:
            advance_turn(state)
            state.current_prompt = ActionPrompt.BUILD_INITIAL_SETTLEMENT
        elif at_the_end:
            state.current_prompt = ActionPrompt.BUILD_INITIAL_SETTLEMENT
        elif num_buildings == 2 * num_players:
            state.is_initial_build_phase = False
            state.current_prompt = ActionPrompt.PLAY_TURN
        else:
            advance_turn(state, -1)
            state.current_prompt = ActionPrompt.BUILD_INITIAL_SETTLEMENT
        state.playable_actions = generate_playable_actions(state)
    elif state.is_road_building and state.free_roads_available > 0:
        result = state.board.build_road(action.color, edge)
        previous_road_color, road_color, road_lengths = result
        build_road(state, action.color, edge, True)
        maintain_longest_road(state, previous_road_color, road_color, road_lengths)

        state.free_roads_available -= 1
        if (
            state.free_roads_available == 0
            or len(road_building_possibilities(state, action.color, False)) == 0
        ):
            state.is_road_building = False
            state.free_roads_available = 0
            # state.current_player_index stays the same
            # state.current_prompt stays as PLAY
        state.playable_actions = generate_playable_actions(state)
    else:
        result = state.board.build_road(action.color, edge)
        previous_road_color, road_color, road_lengths = result
        build_road(state, action.color, edge, False)
        maintain_longest_road(state, previous_road_color, road_color, road_lengths)

        # state.current_player_index stays the same
        # state.current_prompt stays as PLAY
        state.playable_actions = generate_playable_actions(state)


def apply_build_city(state: GameState, action: Action, force: bool = False) -> None:
    require_available_piece(state, action.color, "CITIES", "city")
    require_build_resources(state, action.color, CITY_COST_FREQDECK, "city")
    node_id = action.value
    state.board.build_city(action.color, node_id)
    build_city(state, action.color, node_id)
    state.resource_freqdeck = freqdeck_add(
        state.resource_freqdeck, CITY_COST_FREQDECK
    )  # replenish bank

    # state.current_player_index stays the same
    # state.current_prompt stays as PLAY
    state.playable_actions = generate_playable_actions(state)


def apply_buy_development_card(state: GameState, action: Action, force: bool = False) -> Action:
    if len(state.development_listdeck) == 0:
        raise ValueError("No more development cards")
    if not player_can_afford_dev_card(state, action.color):
        raise ValueError("No money to buy development card")

    if action.value is None:
        if force:
            raise ValueError("Forced BUY_DEVELOPMENT_CARD requires explicit card type")
        card = state.development_listdeck.pop()  # already shuffled
    else:
        card = action.value
        draw_from_listdeck(state.development_listdeck, 1, card)

    buy_dev_card(state, action.color, card)
    state.resource_freqdeck = freqdeck_add(
        state.resource_freqdeck, DEVELOPMENT_CARD_COST_FREQDECK
    )

    action = Action(action.color, action.action_type, card)
    # state.current_player_index stays the same
    # state.current_prompt stays as PLAY
    state.playable_actions = generate_playable_actions(state)
    return action
