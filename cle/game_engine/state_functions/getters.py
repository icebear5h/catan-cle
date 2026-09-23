"""Read state without changing scalar values or mutable cache references."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal, TypedDict, overload

from cle.game_engine import state_functions
from cle.game_engine.models.enums import FastBuildingType, FastDevCard
from cle.game_engine.models.player import Color

if TYPE_CHECKING:
    from cle.game_engine.state import GameState


class PlayerBuildings(TypedDict, total=False):
    """Sparse keys in the per-color defaultdict; roads contain endpoint pairs."""

    SETTLEMENT: list[int]
    CITY: list[int]
    ROAD: list[tuple[int, int]]


def player_key(state: GameState, color: Color) -> str:
    return f"P{state.color_to_index[color]}"


def get_enemy_colors(colors: Iterable[Color], player_color: Color) -> filter[Color]:
    return filter(lambda c: c != player_color, colors)


def get_actual_victory_points(state: GameState, color: Color) -> int:
    key = state_functions.player_key(state, color)
    return state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"]


def get_visible_victory_points(state: GameState, color: Color) -> int:
    key = state_functions.player_key(state, color)
    return state.player_state[f"{key}_VICTORY_POINTS"]


def get_longest_road_color(state: GameState) -> Color | None:
    for index in range(len(state.colors)):
        if state.player_state[f"P{index}_HAS_ROAD"]:
            return state.colors[index]
    return None


def get_largest_army(state: GameState) -> tuple[Color, int] | tuple[None, None]:
    for index in range(len(state.colors)):
        if state.player_state[f"P{index}_HAS_ARMY"]:
            return (
                state.colors[index],
                state.player_state[f"P{index}_PLAYED_KNIGHT"],
            )
    return None, None


def player_has_rolled(state: GameState, color: Color) -> int | bool:
    key = state_functions.player_key(state, color)
    return state.player_state[f"{key}_HAS_ROLLED"]


def get_longest_road_length(state: GameState, color: Color) -> int:
    key = state_functions.player_key(state, color)
    return state.player_state[key + "_LONGEST_ROAD_LENGTH"]


def get_played_dev_cards(
    state: GameState, color: Color, dev_card: FastDevCard | None = None,
) -> int:
    key = state_functions.player_key(state, color)
    if dev_card is None:
        return (
            state.player_state[f"{key}_PLAYED_KNIGHT"]
            + state.player_state[f"{key}_PLAYED_MONOPOLY"]
            + state.player_state[f"{key}_PLAYED_ROAD_BUILDING"]
            + state.player_state[f"{key}_PLAYED_YEAR_OF_PLENTY"]
        )
    else:
        return state.player_state[f"{key}_PLAYED_{dev_card}"]


def get_dev_cards_in_hand(
    state: GameState, color: Color, dev_card: FastDevCard | None = None,
) -> int:
    key = state_functions.player_key(state, color)
    if dev_card is None:
        return (
            state.player_state[f"{key}_KNIGHT_IN_HAND"]
            + state.player_state[f"{key}_MONOPOLY_IN_HAND"]
            + state.player_state[f"{key}_ROAD_BUILDING_IN_HAND"]
            + state.player_state[f"{key}_YEAR_OF_PLENTY_IN_HAND"]
            + state.player_state[f"{key}_VICTORY_POINT_IN_HAND"]
        )
    else:
        return state.player_state[f"{key}_{dev_card}_IN_HAND"]


@overload
def get_player_buildings(
    state: GameState, color_param: Color, building_type_param: Literal["SETTLEMENT", "CITY"],
) -> list[int]: ...


@overload
def get_player_buildings(
    state: GameState, color_param: Color, building_type_param: Literal["ROAD"],
) -> list[tuple[int, int]]: ...


def get_player_buildings(
    state: GameState, color_param: Color, building_type_param: FastBuildingType,
) -> list[int] | list[tuple[int, int]]:
    return state.buildings_by_color.get(color_param, {}).get(building_type_param, [])


def get_player_freqdeck(state: GameState, color: Color) -> list[int]:
    """Returns a 'freqdeck' of a player's resource hand."""
    key = state_functions.player_key(state, color)
    return [
        state.player_state[f"{key}_WOOD_IN_HAND"],
        state.player_state[f"{key}_BRICK_IN_HAND"],
        state.player_state[f"{key}_SHEEP_IN_HAND"],
        state.player_state[f"{key}_WHEAT_IN_HAND"],
        state.player_state[f"{key}_ORE_IN_HAND"],
    ]


def get_state_index(state: GameState) -> int:
    return len(state.actions)
