"""Road-network analysis: per-color road components and longest trails."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, TypeAlias

from cle.game_engine.models.board.graph import STATIC_GRAPH, sorted_edge
from cle.game_engine.models.map import EdgeId, NodeId
from cle.game_engine.models.player import Color

if TYPE_CHECKING:
    from cle.game_engine.models.board.core import Board

# (previous longest-road owner, new longest-road owner, road length per color)
RoadState: TypeAlias = tuple[Color | None, Color | None, dict[Color, int]]
RoadComponents: TypeAlias = defaultdict[Color, list[set[NodeId]]]


def road_components_and_lengths(board: Board) -> tuple[RoadComponents, defaultdict[Color, int]]:
    """Rebuild each color's road components and its edge-simple longest length.

    Enemy vertices can be endpoints shared by multiple components, but
    never join roads through that vertex.
    """
    colors = (
        set(board.connected_components)
        | set(board.road_lengths)
        | set(board.roads.values())
        | {owner for owner, _ in board.buildings.values()}
    )
    components: RoadComponents = defaultdict(list)
    lengths: defaultdict[Color, int] = defaultdict(int)
    for color in sorted(colors, key=lambda item: item.value):
        remaining = {sorted_edge(edge) for edge, owner in board.roads.items() if owner == color}
        road_nodes = {node for edge in remaining for node in edge}
        while remaining:
            agenda = [min(remaining)]
            nodes: set[NodeId] = set()
            while agenda:
                edge = agenda.pop()
                if edge not in remaining:
                    continue
                remaining.remove(edge)
                nodes.update(edge)
                for node in edge:
                    if board.is_enemy_node(node, color):
                        continue
                    for neighbor in STATIC_GRAPH.neighbors(node):
                        candidate = sorted_edge((node, neighbor))
                        if candidate in remaining:
                            agenda.append(candidate)
            components[color].append(nodes)
        components[color].extend(
            {node}
            for node, (owner, _) in board.buildings.items()
            if owner == color and node not in road_nodes
        )
        lengths[color] = max(
            (len(longest_acyclic_path(board, nodes, color)) for nodes in components[color]),
            default=0,
        )
    return components, lengths


def longest_acyclic_path(board: Board, node_set: set[NodeId], color: Color) -> list[EdgeId]:
    """Return a longest trail: vertices may repeat, undirected edges may not."""
    longest: list[EdgeId] = []
    for start_node in sorted(node_set):
        agenda: list[tuple[NodeId, list[EdgeId]]] = [(start_node, [])]
        while len(agenda) > 0:
            node, path_thus_far = agenda.pop()
            if len(path_thus_far) > len(longest):
                longest = path_thus_far
            if path_thus_far and board.is_enemy_node(node, color):
                continue
            for neighbor_node in STATIC_GRAPH.neighbors(node):
                if neighbor_node not in node_set:
                    continue
                edge = sorted_edge((node, neighbor_node))
                if board.is_friendly_road(edge, color) and edge not in path_thus_far:
                    agenda.append((neighbor_node, path_thus_far + [edge]))
    return longest
