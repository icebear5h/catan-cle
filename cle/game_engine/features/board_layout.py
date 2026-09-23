"""Static-map tile and port features, plus one-hot building and road occupancy."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from cle.game_engine.models.board import get_edges
from cle.game_engine.models.enums import CITY, RESOURCES, ROAD, SETTLEMENT
from cle.game_engine.models.map import NUM_TILES, CatanMap, number_probability
from cle.game_engine.models.map_types import Coordinate
from cle.game_engine.models.player import Color

from .players import iter_players

if TYPE_CHECKING:
    from cle.game_engine.game import GameEngine


@functools.lru_cache(NUM_TILES * 2)  # one for each robber, and acount for Minimap
def map_tile_features(
    catan_map: CatanMap, robber_coordinate: Coordinate
) -> dict[str, bool | float]:
    # Returns list of functions that take a game and output a feature.
    # build features like tile0_is_wood, tile0_is_wheat, ..., tile0_proba, tile0_hasrobber
    features: dict[str, bool | float] = {}

    for tile_id, tile in catan_map.tiles_by_id.items():
        for resource in RESOURCES:
            features[f"TILE{tile_id}_IS_{resource}"] = tile.resource == resource
        features[f"TILE{tile_id}_IS_DESERT"] = tile.resource is None
        features[f"TILE{tile_id}_PROBA"] = (
            0 if tile.resource is None else number_probability(tile.number)
        )
        features[f"TILE{tile_id}_HAS_ROBBER"] = (
            catan_map.tiles[robber_coordinate] == tile
        )
    return features


def tile_features(game: GameEngine, p0_color: Color) -> dict[str, bool | float]:
    # Returns list of functions that take a game and output a feature.
    # build features like tile0_is_wood, tile0_is_wheat, ..., tile0_proba, tile0_hasrobber
    return map_tile_features(game.state.board.map, game.state.board.robber_coordinate)


@functools.lru_cache(1)
def map_port_features(catan_map: CatanMap) -> dict[str, bool]:
    features: dict[str, bool] = {}
    for port_id, port in catan_map.ports_by_id.items():
        for resource in RESOURCES:
            features[f"PORT{port_id}_IS_{resource}"] = port.resource == resource
        features[f"PORT{port_id}_IS_THREE_TO_ONE"] = port.resource is None
    return features


def port_features(game: GameEngine, p0_color: Color) -> dict[str, bool]:
    # PORT0_WOOD, PORT0_THREE_TO_ONE, ...
    return map_port_features(game.state.board.map)


@functools.lru_cache(4)
def initialize_graph_features_template(
    num_players: int, catan_map: CatanMap
) -> dict[str, bool]:
    features: dict[str, bool] = {}
    for i in range(num_players):
        for node_id in range(len(catan_map.land_nodes)):
            for building in [SETTLEMENT, CITY]:
                features[f"NODE{node_id}_P{i}_{building}"] = False
        for edge in get_edges(catan_map.land_nodes):
            features[f"EDGE{edge}_P{i}_ROAD"] = False
    return features


@functools.lru_cache(1024 * 2 * 2 * 2)
def get_node_hot_encoded(
    player_index: int,
    colors: tuple[Color, ...],
    settlements: tuple[int, ...],
    cities: tuple[int, ...],
    roads: tuple[tuple[int, int], ...],
) -> dict[str, bool]:
    features: dict[str, bool] = {}

    for node_id in settlements:
        features[f"NODE{node_id}_P{player_index}_SETTLEMENT"] = True
    for node_id in cities:
        features[f"NODE{node_id}_P{player_index}_CITY"] = True
    for edge in roads:
        features[f"EDGE{tuple(sorted(edge))}_P{player_index}_ROAD"] = True

    return features


def graph_features(game: GameEngine, p0_color: Color) -> dict[str, bool]:
    features = initialize_graph_features_template(
        len(game.state.colors), game.state.board.map
    ).copy()

    for i, color in iter_players(game.state.colors, p0_color):
        settlements = tuple(game.state.buildings_by_color[color][SETTLEMENT])
        cities = tuple(game.state.buildings_by_color[color][CITY])
        roads = tuple(game.state.buildings_by_color[color][ROAD])
        to_update = get_node_hot_encoded(
            i, game.state.colors, settlements, cities, roads
        )
        features.update(to_update)

    return features
