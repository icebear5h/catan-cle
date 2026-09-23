"""Best production reachable by expanding along empty edges, and port distances."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING

import networkx as nx

from cle.game_engine.models.board import STATIC_GRAPH, get_edges, get_node_distances
from cle.game_engine.models.enums import RESOURCES, ROAD, FastResource
from cle.game_engine.models.player import Color
from cle.game_engine.state_functions import get_player_buildings

from .players import iter_players
from .production import get_node_production, get_player_expandable_nodes

if TYPE_CHECKING:
    from cle.game_engine.game import GameEngine


def expansion_features(game: GameEngine, p0_color: Color) -> dict[str, float]:
    MAX_EXPANSION_DISTANCE = 3  # exclusive

    features: dict[str, float] = {}

    # For each connected component node, bfs_edges (skipping enemy edges and nodes nodes)
    empty_edges = set(get_edges(game.state.board.map.land_nodes))
    for i, color in iter_players(game.state.colors, p0_color):
        empty_edges.difference_update(get_player_buildings(game.state, color, ROAD))
    searchable_subgraph = STATIC_GRAPH.edge_subgraph(empty_edges)

    board_buildable_node_ids = game.state.board.buildable_node_ids(
        p0_color, True
    )  # this should be the same for all players. TODO: Can maintain internally (instead of re-compute).

    for i, color in iter_players(game.state.colors, p0_color):
        expandable_node_ids = get_player_expandable_nodes(game, color)

        def skip_blocked_by_enemy(neighbor_ids: Iterable[int]) -> Iterator[int]:
            for node_id in neighbor_ids:
                node_color = game.state.board.get_node_color(node_id)
                if node_color is None or node_color == color:
                    yield node_id  # not owned by enemy, can explore

        # owned_edges = get_player_buildings(state, color, ROAD)
        dis_res_prod: dict[int, dict[FastResource, float]] = {
            distance: {k: 0.0 for k in RESOURCES}
            for distance in range(MAX_EXPANSION_DISTANCE)
        }
        for node_id in expandable_node_ids:
            if node_id in board_buildable_node_ids:  # node itself is buildable
                for resource in RESOURCES:
                    production = get_node_production(
                        game.state.board.map, node_id, resource
                    )
                    dis_res_prod[0][resource] = max(
                        production, dis_res_prod[0][resource]
                    )

            if node_id not in searchable_subgraph.nodes():
                continue  # must be internal node, no need to explore

            bfs_iteration = nx.bfs_edges(
                searchable_subgraph,
                node_id,
                depth_limit=MAX_EXPANSION_DISTANCE - 1,
                sort_neighbors=skip_blocked_by_enemy,
            )

            paths: dict[int, list[int]] = {node_id: []}
            for edge in bfs_iteration:
                a, b = edge
                path_until_now = paths[a]
                distance = len(path_until_now) + 1
                paths[b] = paths[a] + [b]

                if b not in board_buildable_node_ids:
                    continue

                # means we can get to node b, at distance=d, starting from path[0]
                for resource in RESOURCES:
                    production = get_node_production(game.state.board.map, b, resource)
                    dis_res_prod[distance][resource] = max(
                        production, dis_res_prod[distance][resource]
                    )

        for distance, res_prod in dis_res_prod.items():
            for resource, prod in res_prod.items():
                features[f"P{i}_{resource}_AT_DISTANCE_{int(distance)}"] = prod

    return features


def port_distance_features(game: GameEngine, p0_color: Color) -> dict[str, bool | float]:
    # P0_HAS_WHEAT_PORT, P0_WHEAT_PORT_DISTANCE, ..., P1_HAS_WHEAT_PORT,
    features: dict[str, bool | float] = {}
    ports = game.state.board.map.port_nodes
    distances = get_node_distances()
    resources_and_none: list[FastResource | None] = [*RESOURCES]
    resources_and_none += [None]
    for resource_or_none in resources_and_none:
        port_name = resource_or_none or "3:1"
        for i, color in iter_players(game.state.colors, p0_color):
            expandable_node_ids = get_player_expandable_nodes(game, color)
            if len(expandable_node_ids) == 0:
                features[f"P{i}_HAS_{port_name}_PORT"] = False
                features[f"P{i}_{port_name}_PORT_DISTANCE"] = float("inf")
            else:
                min_distance = min(
                    [
                        distances[port_node_id][my_node]
                        for my_node in expandable_node_ids
                        for port_node_id in ports[resource_or_none]
                    ]
                )
                features[f"P{i}_HAS_{port_name}_PORT"] = min_distance == 0
                features[f"P{i}_{port_name}_PORT_DISTANCE"] = min_distance
    return features
