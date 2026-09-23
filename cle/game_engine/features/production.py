"""Resource production owned now and reachable within a few roads."""

from __future__ import annotations

import functools
from collections import Counter
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

from cle.game_engine.models.board import STATIC_GRAPH
from cle.game_engine.models.enums import CITY, RESOURCES, SETTLEMENT, FastResource
from cle.game_engine.models.map import CatanMap, number_probability
from cle.game_engine.models.map_types import Production
from cle.game_engine.models.player import Color
from cle.game_engine.state_functions import get_player_buildings

from .players import iter_players
from .types import EdgePath, LevelNodes

if TYPE_CHECKING:
    from cle.game_engine.game import GameEngine


def build_production_features(
    consider_robber: bool,
) -> Callable[[GameEngine, Color], dict[str, float]]:
    prefix = "EFFECTIVE_" if consider_robber else "TOTAL_"

    def production_features(game: GameEngine, p0_color: Color) -> dict[str, float]:
        # P0_WHEAT_PRODUCTION, P0_ORE_PRODUCTION, ..., P1_WHEAT_PRODUCTION, ...
        features: dict[str, float] = {}
        board = game.state.board
        robbed_nodes = set(board.map.tiles[board.robber_coordinate].nodes.values())
        for resource in RESOURCES:
            for i, color in iter_players(game.state.colors, p0_color):
                production: float = 0
                for node_id in get_player_buildings(game.state, color, SETTLEMENT):
                    if consider_robber and node_id in robbed_nodes:
                        continue
                    production += get_node_production(
                        game.state.board.map, node_id, resource
                    )
                for node_id in get_player_buildings(game.state, color, CITY):
                    if consider_robber and node_id in robbed_nodes:
                        continue
                    production += 2 * get_node_production(
                        game.state.board.map, node_id, resource
                    )
                features[f"{prefix}P{i}_{resource}_PRODUCTION"] = production

        return features

    return production_features


@functools.lru_cache(maxsize=1000)
def get_node_production(catan_map: CatanMap, node_id: int, resource: FastResource) -> float:
    tiles = catan_map.adjacent_tiles[node_id]
    return sum([number_probability(t.number) for t in tiles if t.resource == resource])


def get_player_expandable_nodes(game: GameEngine, color: Color) -> list[int]:
    node_sets = game.state.board.find_connected_components(color)
    enemy_colors = [
        enemy_color for enemy_color in game.state.colors if enemy_color != color
    ]
    enemy_node_ids: set[int] = set()
    for enemy_color in enemy_colors:
        enemy_node_ids.update(get_player_buildings(game.state, enemy_color, SETTLEMENT))
        enemy_node_ids.update(get_player_buildings(game.state, enemy_color, CITY))

    expandable_node_ids = [
        node_id
        for node_set in node_sets
        for node_id in node_set
        if node_id not in enemy_node_ids  # not plowed
    ]  # not exactly "buildable_node_ids" b.c. we could expand from non-buildable nodes
    return expandable_node_ids


REACHABLE_FEATURES_MAX = 2  # inclusive


def get_zero_nodes(game: GameEngine, color: Color) -> set[int]:
    zero_nodes: set[int] = set()
    for component in game.state.board.connected_components[color]:
        for node_id in component:
            zero_nodes.add(node_id)
    return zero_nodes


@functools.lru_cache(maxsize=2000)
def iter_level_nodes(
    enemy_nodes: frozenset[int],
    enemy_roads: frozenset[tuple[int, int]],
    num_roads: int,
    zero_nodes: frozenset[int],
) -> list[LevelNodes]:
    """Iterates over possible expansion paths.

    Args:
        enemy_nodes (frozenset[NodeId]): node_ids owned by enemy colors
        enemy_roads (frozenset[EdgeId]): edge_ids owned by enemy colors
        num_roads (int): Max-depth of BFS (inclusive). e.g. 2 will yield
            possible expansions with up to 2 roads.
        zero_nodes (frozenset[NodeId]): Nodes reachable per board.connected_components

    Yields:
        Tuple[int, Set[NodeId], Dict[NodeId, List[EdgeId]]:
            First element is level (roads needed to get there).
            Second element is set of node_ids reachable at this level.
            Third is mapping of NodeId to the list of edges
            that leads to shortest path to that NodeId.
    """
    last_layer_nodes: set[int] | frozenset[int] = zero_nodes
    paths: dict[int, EdgePath] = {i: [] for i in zero_nodes}
    results: list[LevelNodes] = []
    for level in range(1, num_roads + 1):
        level_nodes = set(last_layer_nodes)
        for node_id in last_layer_nodes:
            if node_id in enemy_nodes:
                continue  # not expandable.

            # here we can assume node is empty or owned
            expandable: list[int] = []
            for neighbor_id in STATIC_GRAPH.neighbors(node_id):
                edge = (node_id, neighbor_id)
                can_follow_edge = edge not in enemy_roads
                if can_follow_edge:
                    expandable.append(neighbor_id)
                    if neighbor_id not in paths:
                        paths[neighbor_id] = paths[node_id] + [(node_id, neighbor_id)]

            level_nodes.update(expandable)

        results.append((level, level_nodes, paths))

        last_layer_nodes = level_nodes

    return results


def get_owned_or_buildable(
    game: GameEngine, color: Color, board_buildable: list[int]
) -> frozenset[int]:
    return frozenset(
        get_player_buildings(game.state, color, SETTLEMENT)
        + get_player_buildings(game.state, color, CITY)
        + board_buildable
    )


def reachability_features(
    game: GameEngine, p0_color: Color, levels: int = REACHABLE_FEATURES_MAX
) -> dict[str, float]:
    features: dict[str, float] = {}

    board_buildable = game.state.board.buildable_node_ids(p0_color, True)
    for i, color in iter_players(game.state.colors, p0_color):
        owned_or_buildable = get_owned_or_buildable(game, color, board_buildable)

        # do layer 0
        zero_nodes = get_zero_nodes(game, color)
        production = count_production(
            frozenset(owned_or_buildable.intersection(zero_nodes)),
            game.state.board.map,
        )
        for resource in RESOURCES:
            features[f"P{i}_0_ROAD_REACHABLE_{resource}"] = production[resource]

        # do rest of layers
        enemy_nodes = frozenset(
            k
            for k, v in game.state.board.buildings.items()
            if v is not None and v[0] != color
        )
        enemy_roads = frozenset(
            k for k, v in game.state.board.roads.items() if v is not None and v != color
        )
        for level, level_nodes, paths in iter_level_nodes(
            enemy_nodes, enemy_roads, levels, frozenset(zero_nodes)
        ):
            production = count_production(
                frozenset(owned_or_buildable.intersection(level_nodes)),
                game.state.board.map,
            )
            for resource in RESOURCES:
                features[f"P{i}_{level}_ROAD_REACHABLE_{resource}"] = production[
                    resource
                ]

    return features


@functools.lru_cache(maxsize=1000)
def count_production(nodes: frozenset[int], catan_map: CatanMap) -> Production:
    # Counter supports fractional counts at runtime (see map.init_node_production);
    # its stubs only admit ints, so the accumulator is exposed as Production.
    production: Counter[FastResource] = Counter()
    for node_id in nodes:
        production += cast("Counter[FastResource]", catan_map.node_production[node_id])
    return cast(Production, production)
