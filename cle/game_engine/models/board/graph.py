"""Static node/edge graph shared by every map, plus cached graph queries."""

from __future__ import annotations

import functools
from collections import defaultdict
from collections.abc import Iterable
from typing import TypeAlias

import networkx as nx

from cle.game_engine.models.map import (
    BASE_MAP_TEMPLATE,
    MINI_MAP_TEMPLATE,
    NUM_NODES,
    CatanMap,
    EdgeId,
    NodeId,
)

# Node/edge attribute dicts are never populated; ``object`` keeps them Any-free.
# Quoted because networkx.Graph is not subscriptable at runtime.
NodeGraph: TypeAlias = "nx.Graph[NodeId, dict[str, object], dict[str, object]]"

# Used to find relationships between nodes and edges
base_map = CatanMap.from_template(BASE_MAP_TEMPLATE)
mini_map = CatanMap.from_template(MINI_MAP_TEMPLATE)
STATIC_GRAPH: NodeGraph = nx.Graph()
for tile in base_map.tiles.values():
    STATIC_GRAPH.add_nodes_from(tile.nodes.values())
    STATIC_GRAPH.add_edges_from(tile.edges.values())


@functools.lru_cache(1)
def get_node_distances() -> dict[NodeId, defaultdict[NodeId, float]]:
    return nx.floyd_warshall(STATIC_GRAPH)


@functools.lru_cache(3)  # None, range(54), range(24)
def get_edges(land_nodes: Iterable[NodeId] | None = None) -> list[EdgeId]:
    return list(STATIC_GRAPH.subgraph(land_nodes or range(NUM_NODES)).edges())


def sorted_edge(edge: tuple[NodeId, NodeId]) -> EdgeId:
    """Return ``edge`` with its endpoints in ascending order (``tuple(sorted(edge))``)."""
    first, second = sorted(edge)
    return (first, second)
