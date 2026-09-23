"""Playing development cards."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cle.game_engine.models.actions import generate_playable_actions
from cle.game_engine.models.decks import (
    freqdeck_contains,
    freqdeck_from_listdeck,
    freqdeck_replenish,
    freqdeck_subtract,
)
from cle.game_engine.models.enums import MONOPOLY, YEAR_OF_PLENTY, Action, ActionPrompt
from cle.game_engine.state_functions import (
    play_dev_card,
    player_can_play_dev,
    player_deck_draw,
    player_freqdeck_add,
    player_key,
)

if TYPE_CHECKING:
    from cle.game_engine.state.core import GameState


def apply_play_knight_card(state: GameState, action: Action, force: bool = False) -> None:
    if not player_can_play_dev(state, action.color, "KNIGHT"):
        raise ValueError("Player cant play knight card now")

    play_dev_card(state, action.color, "KNIGHT")

    # state.current_player_index stays the same
    state.current_prompt = ActionPrompt.MOVE_ROBBER
    state.playable_actions = generate_playable_actions(state)


def apply_play_year_of_plenty(state: GameState, action: Action, force: bool = False) -> None:
    cards_selected = freqdeck_from_listdeck(action.value)
    if not player_can_play_dev(state, action.color, YEAR_OF_PLENTY):
        raise ValueError("Player cant play year of plenty now")
    if not freqdeck_contains(state.resource_freqdeck, cards_selected):
        raise ValueError("Not enough resources of this type (these types?) in bank")
    player_freqdeck_add(state, action.color, cards_selected)
    state.resource_freqdeck = freqdeck_subtract(state.resource_freqdeck, cards_selected)
    play_dev_card(state, action.color, YEAR_OF_PLENTY)

    # state.current_player_index stays the same
    state.current_prompt = ActionPrompt.PLAY_TURN
    state.playable_actions = generate_playable_actions(state)


def apply_play_monopoly(state: GameState, action: Action, force: bool = False) -> None:
    mono_resource = action.value
    cards_stolen = [0, 0, 0, 0, 0]
    if not player_can_play_dev(state, action.color, MONOPOLY):
        raise ValueError("Player cant play monopoly now")
    for color in state.colors:
        if not color == action.color:
            key = player_key(state, color)
            number_of_cards_to_steal = state.player_state[
                f"{key}_{mono_resource}_IN_HAND"
            ]
            freqdeck_replenish(cards_stolen, number_of_cards_to_steal, mono_resource)
            player_deck_draw(state, color, mono_resource, number_of_cards_to_steal)
    player_freqdeck_add(state, action.color, cards_stolen)
    play_dev_card(state, action.color, MONOPOLY)

    # state.current_player_index stays the same
    state.current_prompt = ActionPrompt.PLAY_TURN
    state.playable_actions = generate_playable_actions(state)


def apply_play_road_building(state: GameState, action: Action, force: bool = False) -> None:
    if not player_can_play_dev(state, action.color, "ROAD_BUILDING"):
        raise ValueError("Player cant play road building now")

    play_dev_card(state, action.color, "ROAD_BUILDING")
    state.is_road_building = True
    state.free_roads_available = 2

    # state.current_player_index stays the same
    state.current_prompt = ActionPrompt.PLAY_TURN
    state.playable_actions = generate_playable_actions(state)
