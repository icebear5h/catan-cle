"""Player hands and development-card play, preserving draw and mutation order."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

from cle.game_engine import state_functions
from cle.game_engine.models.enums import (
    BRICK,
    ORE,
    SHEEP,
    VICTORY_POINT,
    WHEAT,
    WOOD,
    FastDevCard,
    FastResource,
)
from cle.game_engine.models.player import Color

if TYPE_CHECKING:
    from cle.game_engine.state import GameState


def player_can_afford_dev_card(state: GameState, color: Color) -> bool:
    key = state_functions.player_key(state, color)
    return (
        state.player_state[f"{key}_SHEEP_IN_HAND"] >= 1
        and state.player_state[f"{key}_WHEAT_IN_HAND"] >= 1
        and state.player_state[f"{key}_ORE_IN_HAND"] >= 1
    )


def player_resource_freqdeck_contains(
    state: GameState, color: Color, freqdeck: Sequence[int],
) -> bool:
    key = state_functions.player_key(state, color)
    return (
        state.player_state[f"{key}_WOOD_IN_HAND"] >= freqdeck[0]
        and state.player_state[f"{key}_BRICK_IN_HAND"] >= freqdeck[1]
        and state.player_state[f"{key}_SHEEP_IN_HAND"] >= freqdeck[2]
        and state.player_state[f"{key}_WHEAT_IN_HAND"] >= freqdeck[3]
        and state.player_state[f"{key}_ORE_IN_HAND"] >= freqdeck[4]
    )


def player_can_play_dev(state: GameState, color: Color, dev_card: FastDevCard) -> int | bool:
    key = state_functions.player_key(state, color)
    return (
        not state.player_state[f"{key}_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"]
        and state.player_state[f"{key}_{dev_card}_IN_HAND"] >= 1
        and state.player_state[f"{key}_{dev_card}_OWNED_AT_START"]
    )


def player_freqdeck_add(state: GameState, color: Color, freqdeck: Sequence[int]) -> None:
    key = state_functions.player_key(state, color)
    state.player_state[f"{key}_WOOD_IN_HAND"] += freqdeck[0]
    state.player_state[f"{key}_BRICK_IN_HAND"] += freqdeck[1]
    state.player_state[f"{key}_SHEEP_IN_HAND"] += freqdeck[2]
    state.player_state[f"{key}_WHEAT_IN_HAND"] += freqdeck[3]
    state.player_state[f"{key}_ORE_IN_HAND"] += freqdeck[4]


def player_freqdeck_subtract(state: GameState, color: Color, freqdeck: Sequence[int]) -> None:
    key = state_functions.player_key(state, color)
    state.player_state[f"{key}_WOOD_IN_HAND"] -= freqdeck[0]
    state.player_state[f"{key}_BRICK_IN_HAND"] -= freqdeck[1]
    state.player_state[f"{key}_SHEEP_IN_HAND"] -= freqdeck[2]
    state.player_state[f"{key}_WHEAT_IN_HAND"] -= freqdeck[3]
    state.player_state[f"{key}_ORE_IN_HAND"] -= freqdeck[4]


def buy_dev_card(state: GameState, color: Color, dev_card: FastDevCard) -> None:
    key = state_functions.player_key(state, color)

    assert state.player_state[f"{key}_SHEEP_IN_HAND"] >= 1
    assert state.player_state[f"{key}_WHEAT_IN_HAND"] >= 1
    assert state.player_state[f"{key}_ORE_IN_HAND"] >= 1

    state.player_state[f"{key}_{dev_card}_IN_HAND"] += 1
    if dev_card == VICTORY_POINT:
        state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"] += 1

    state.player_state[f"{key}_SHEEP_IN_HAND"] -= 1
    state.player_state[f"{key}_WHEAT_IN_HAND"] -= 1
    state.player_state[f"{key}_ORE_IN_HAND"] -= 1


def player_num_resource_cards(
    state: GameState, color: Color, card: FastResource | None = None,
) -> int:
    key = state_functions.player_key(state, color)
    if card is None:
        return (
            state.player_state[f"{key}_WOOD_IN_HAND"]
            + state.player_state[f"{key}_BRICK_IN_HAND"]
            + state.player_state[f"{key}_SHEEP_IN_HAND"]
            + state.player_state[f"{key}_WHEAT_IN_HAND"]
            + state.player_state[f"{key}_ORE_IN_HAND"]
        )
    else:
        return state.player_state[f"{key}_{card}_IN_HAND"]


def player_num_dev_cards(state: GameState, color: Color) -> int:
    key = state_functions.player_key(state, color)
    return (
        state.player_state[f"{key}_YEAR_OF_PLENTY_IN_HAND"]
        + state.player_state[f"{key}_MONOPOLY_IN_HAND"]
        + state.player_state[f"{key}_VICTORY_POINT_IN_HAND"]
        + state.player_state[f"{key}_KNIGHT_IN_HAND"]
        + state.player_state[f"{key}_ROAD_BUILDING_IN_HAND"]
    )


def player_deck_to_array(state: GameState, color: Color) -> list[FastResource]:
    key = state_functions.player_key(state, color)
    # List arithmetic widens the resource literals to str in mypy.
    return cast(list[FastResource], (
        state.player_state[f"{key}_WOOD_IN_HAND"] * [WOOD]
        + state.player_state[f"{key}_BRICK_IN_HAND"] * [BRICK]
        + state.player_state[f"{key}_SHEEP_IN_HAND"] * [SHEEP]
        + state.player_state[f"{key}_WHEAT_IN_HAND"] * [WHEAT]
        + state.player_state[f"{key}_ORE_IN_HAND"] * [ORE]
    ))


def player_deck_draw(
    state: GameState, color: Color, card: FastResource | FastDevCard, amount: int = 1,
) -> None:
    key = state_functions.player_key(state, color)
    assert state.player_state[f"{key}_{card}_IN_HAND"] >= amount
    state.player_state[f"{key}_{card}_IN_HAND"] -= amount


def player_deck_replenish(
    state: GameState, color: Color, resource: FastResource, amount: int = 1,
) -> None:
    key = state_functions.player_key(state, color)
    state.player_state[f"{key}_{resource}_IN_HAND"] += amount


def player_deck_random_draw(state: GameState, color: Color) -> FastResource:
    deck_array = state_functions.player_deck_to_array(state, color)
    resource = state.rng.choice(deck_array)
    state_functions.player_deck_draw(state, color, resource)
    return resource


def play_dev_card(state: GameState, color: Color, dev_card: FastDevCard) -> None:
    if dev_card == "KNIGHT":
        previous_army_color, previous_army_size = state_functions.get_largest_army(state)
    key = state_functions.player_key(state, color)
    state_functions.player_deck_draw(state, color, dev_card)
    state.player_state[f"{key}_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"] = True
    state.player_state[f"{key}_PLAYED_{dev_card}"] += 1
    if dev_card == "KNIGHT":
        state_functions.maintain_largest_army(state, color, previous_army_color, previous_army_size)


def player_clean_turn(state: GameState, color: Color) -> None:
    key = state_functions.player_key(state, color)
    state.player_state[f"{key}_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"] = False
    state.player_state[f"{key}_HAS_ROLLED"] = False
    # Dev cards owned this turn will be playable next turn
    state.player_state[f"{key}_KNIGHT_OWNED_AT_START"] = (
        state.player_state[f"{key}_KNIGHT_IN_HAND"] > 0
    )
    state.player_state[f"{key}_MONOPOLY_OWNED_AT_START"] = (
        state.player_state[f"{key}_MONOPOLY_IN_HAND"] > 0
    )
    state.player_state[f"{key}_YEAR_OF_PLENTY_OWNED_AT_START"] = (
        state.player_state[f"{key}_YEAR_OF_PLENTY_IN_HAND"] > 0
    )
    state.player_state[f"{key}_ROAD_BUILDING_OWNED_AT_START"] = (
        state.player_state[f"{key}_ROAD_BUILDING_IN_HAND"] > 0
    )
