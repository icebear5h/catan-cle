"""Engine topology lookups shared by contracts and question selection."""

from __future__ import annotations

from typing import Dict, List, Tuple

from cle.game_engine.models.board import get_edges
from cle.game_engine.models.map import (
    NUM_EDGES,
    NUM_NODES,
    PORT_DIRECTION_TO_NODEREFS,
    CatanMap,
    LandTile,
    Port,
)
from cle.game_engine.models.player import Color
from evals.catan_board_bench.builder.constants import EdgeId
from evals.catan_board_bench.tokens import base_edges, canonical_edge


def _tile_coordinates(catan_map: CatanMap) -> Dict[int, Tuple[int, int, int]]:
    return {
        tile.id: coordinate
        for coordinate, tile in catan_map.land_tiles.items()
        if isinstance(tile, LandTile)
    }


def _port_coordinates(catan_map: CatanMap) -> Dict[int, Tuple[int, int, int]]:
    return {
        port.id: coordinate
        for coordinate, port in catan_map.tiles.items()
        if isinstance(port, Port)
    }


def _port_nodes_by_id(catan_map: CatanMap) -> Dict[int, List[int]]:
    port_nodes = {}
    for port_id, port in sorted(catan_map.ports_by_id.items()):
        node_refs = PORT_DIRECTION_TO_NODEREFS[port.direction]
        port_nodes[port_id] = [port.nodes[node_ref] for node_ref in node_refs]
    return port_nodes


def _node_ports(catan_map: CatanMap) -> Dict[int, List[int]]:
    node_ports: Dict[int, List[int]] = {node_id: [] for node_id in range(NUM_NODES)}
    for port_id, nodes in _port_nodes_by_id(catan_map).items():
        for node_id in nodes:
            node_ports[node_id].append(port_id)
    return node_ports


def _playable_edges(catan_map: CatanMap) -> List[EdgeId]:
    edges = sorted(canonical_edge(edge) for edge in get_edges(catan_map.land_nodes))
    if len(edges) != NUM_EDGES:
        fallback_edges = base_edges()
        if len(fallback_edges) != NUM_EDGES:
            raise RuntimeError(f"expected {NUM_EDGES} playable edges, found {len(edges)}")
        return fallback_edges
    return edges


def _owned_road_count(roads: Dict[EdgeId, Color], color: Color) -> int:
    return len({canonical_edge(edge) for edge, owner in roads.items() if owner == color})

