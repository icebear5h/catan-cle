"""Flat, name-keyed feature extractors over a GameEngine from one player's seat.

Every extractor and helper stays importable from this package path. create_sample
reads feature_extractors through this module's globals at call time, so replacing
either name here still steers the composed sample.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from typing import Literal

from cle.game_engine.game import GameEngine
from cle.game_engine.models.map import build_map
from cle.game_engine.models.player import Color

from .board_layout import (
    get_node_hot_encoded,
    graph_features,
    initialize_graph_features_template,
    map_port_features,
    map_tile_features,
    port_features,
    tile_features,
)
from .expansion import expansion_features, port_distance_features
from .players import (
    game_features,
    is_building,
    is_road,
    iter_players,
    player_features,
    resource_hand_features,
)
from .production import (
    REACHABLE_FEATURES_MAX,
    build_production_features,
    count_production,
    get_node_production,
    get_owned_or_buildable,
    get_player_expandable_nodes,
    get_zero_nodes,
    iter_level_nodes,
    reachability_features,
)
from .types import EdgePath, FeatureExtractor, FeatureValue, LevelNodes

__all__ = [
    "REACHABLE_FEATURES_MAX",
    "EdgePath",
    "FeatureExtractor",
    "FeatureValue",
    "LevelNodes",
    "build_production_features",
    "count_production",
    "create_sample",
    "create_sample_vector",
    "expansion_features",
    "feature_extractors",
    "game_features",
    "get_feature_ordering",
    "get_node_hot_encoded",
    "get_node_production",
    "get_owned_or_buildable",
    "get_player_expandable_nodes",
    "get_zero_nodes",
    "graph_features",
    "initialize_graph_features_template",
    "is_building",
    "is_road",
    "iter_level_nodes",
    "iter_players",
    "map_port_features",
    "map_tile_features",
    "player_features",
    "port_distance_features",
    "port_features",
    "reachability_features",
    "resource_hand_features",
    "tile_features",
]

feature_extractors: list[FeatureExtractor] = [
    # PLAYER FEATURES =====
    player_features,
    resource_hand_features,
    # TRANSFERABLE BOARD FEATURES =====
    # build_production_features(True),
    # build_production_features(False),
    # expansion_features,
    # reachability_features,
    # RAW BASE-MAP FEATURES =====
    tile_features,
    port_features,
    graph_features,
    # GAME FEATURES =====
    game_features,
]


# TODO: Use OrderedDict instead? To minimize mis-aligned features errors.
def create_sample(game: GameEngine, p0_color: Color) -> dict[str, FeatureValue]:
    record: dict[str, FeatureValue] = {}
    for extractor in feature_extractors:
        record.update(extractor(game, p0_color))
    return record


def create_sample_vector(
    game: GameEngine, p0_color: Color, features: Sequence[str] | None = None
) -> list[float]:
    features = features or get_feature_ordering(len(game.state.colors))
    sample_dict = create_sample(game, p0_color)
    return [float(sample_dict[i]) for i in features if i in sample_dict]


@functools.lru_cache(4 * 3)
def get_feature_ordering(
    num_players: int = 4, map_type: Literal["BASE", "MINI", "TOURNAMENT"] = "BASE"
) -> list[str]:
    colors = [
        Color.RED,
        Color.BLUE,
        Color.WHITE,
        Color.ORANGE,
    ]
    colors = colors[:num_players]
    game = GameEngine(colors, catan_map=build_map(map_type))
    sample = create_sample(game, colors[0])
    return sorted(sample.keys())
