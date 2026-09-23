"""Per-seat player, hand, and bank features, plus small board predicates."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from cle.game_engine.models.decks import freqdeck_count
from cle.game_engine.models.enums import (
    DEVELOPMENT_CARDS,
    RESOURCES,
    VICTORY_POINT,
    ActionType,
    FastBuildingType,
)
from cle.game_engine.models.player import Color
from cle.game_engine.state_functions import (
    player_key,
    player_num_dev_cards,
    player_num_resource_cards,
)

if TYPE_CHECKING:
    from cle.game_engine.game import GameEngine


# ===== Helpers
def is_building(
    game: GameEngine, node_id: int, color: Color, building_type: FastBuildingType
) -> bool:
    building = game.state.board.buildings.get(node_id, None)
    if building is None:
        return False
    else:
        return building[0] == color and building[1] == building_type


def is_road(game: GameEngine, edge: tuple[int, int], color: Color) -> bool:
    return game.state.board.get_edge_color(edge) == color


@functools.lru_cache(1024)
def iter_players(colors: tuple[Color, ...], p0_color: Color) -> list[tuple[int, Color]]:
    """Iterator: for i, player in iter_players(game, p0.color)"""
    start_index = colors.index(p0_color)
    result: list[tuple[int, Color]] = []
    for i in range(len(colors)):
        actual_index = (start_index + i) % len(colors)
        result.append((i, colors[actual_index]))
    return result


# ===== Extractors
def player_features(game: GameEngine, p0_color: Color) -> dict[str, int | bool]:
    # P0_ACTUAL_VPS
    # P{i}_PUBLIC_VPS, P1_PUBLIC_VPS, ...
    # P{i}_HAS_ARMY, P{i}_HAS_ROAD, P1_HAS_ARMY, ...
    # P{i}_ROADS_LEFT, P{i}_SETTLEMENTS_LEFT, P{i}_CITIES_LEFT, P1_...
    # P{i}_HAS_ROLLED, P{i}_LONGEST_ROAD_LENGTH
    features: dict[str, int | bool] = dict()
    for i, color in iter_players(game.state.colors, p0_color):
        key = player_key(game.state, color)
        if color == p0_color:
            features["P0_ACTUAL_VPS"] = game.state.player_state[
                key + "_ACTUAL_VICTORY_POINTS"
            ]

        features[f"P{i}_PUBLIC_VPS"] = game.state.player_state[key + "_VICTORY_POINTS"]
        features[f"P{i}_HAS_ARMY"] = game.state.player_state[key + "_HAS_ARMY"]
        features[f"P{i}_HAS_ROAD"] = game.state.player_state[key + "_HAS_ROAD"]
        features[f"P{i}_ROADS_LEFT"] = game.state.player_state[key + "_ROADS_AVAILABLE"]
        features[f"P{i}_SETTLEMENTS_LEFT"] = game.state.player_state[
            key + "_SETTLEMENTS_AVAILABLE"
        ]
        features[f"P{i}_CITIES_LEFT"] = game.state.player_state[
            key + "_CITIES_AVAILABLE"
        ]
        features[f"P{i}_HAS_ROLLED"] = game.state.player_state[key + "_HAS_ROLLED"]
        features[f"P{i}_LONGEST_ROAD_LENGTH"] = game.state.player_state[
            key + "_LONGEST_ROAD_LENGTH"
        ]

    return features


def resource_hand_features(game: GameEngine, p0_color: Color) -> dict[str, int | bool]:
    # P0_WHEATS_IN_HAND, P0_WOODS_IN_HAND, ...
    # P0_ROAD_BUILDINGS_IN_HAND, P0_KNIGHT_IN_HAND, ..., P0_VPS_IN_HAND
    # P0_ROAD_BUILDINGS_PLAYABLE, P0_KNIGHT_PLAYABLE, ...
    # P0_ROAD_BUILDINGS_PLAYED, P0_KNIGHT_PLAYED, ...

    # P1_ROAD_BUILDINGS_PLAYED, P1_KNIGHT_PLAYED, ...
    # TODO: P1_WHEATS_INFERENCE, P1_WOODS_INFERENCE, ...
    # TODO: P1_ROAD_BUILDINGS_INFERENCE, P1_KNIGHT_INFERENCE, ...

    state = game.state
    player_state = state.player_state

    features: dict[str, int | bool] = {}
    for i, color in iter_players(game.state.colors, p0_color):
        key = player_key(game.state, color)

        if color == p0_color:
            for resource in RESOURCES:
                features[f"P0_{resource}_IN_HAND"] = player_state[
                    key + f"_{resource}_IN_HAND"
                ]
            for card in DEVELOPMENT_CARDS:
                features[f"P0_{card}_IN_HAND"] = player_state[key + f"_{card}_IN_HAND"]
            features["P0_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"] = player_state[
                key + "_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"
            ]

        for card in DEVELOPMENT_CARDS:
            if card == VICTORY_POINT:
                continue  # cant play VPs
            features[f"P{i}_{card}_PLAYED"] = player_state[key + f"_PLAYED_{card}"]

        features[f"P{i}_NUM_RESOURCES_IN_HAND"] = player_num_resource_cards(
            state, color
        )
        features[f"P{i}_NUM_DEVS_IN_HAND"] = player_num_dev_cards(state, color)

    return features


def game_features(game: GameEngine, p0_color: Color) -> dict[str, int | bool]:
    # BANK_WOODS, BANK_WHEATS, ..., BANK_DEV_CARDS
    possibilities = set([a.action_type for a in game.state.playable_actions])
    features: dict[str, int | bool] = {
        "BANK_DEV_CARDS": len(game.state.development_listdeck),
        "IS_MOVING_ROBBER": ActionType.MOVE_ROBBER in possibilities,
        "IS_DISCARDING": ActionType.DISCARD in possibilities,
    }
    for resource in RESOURCES:
        features[f"BANK_{resource}"] = freqdeck_count(
            game.state.resource_freqdeck, resource
        )
    return features
