"""Closed-world static-topology fact tables over the land graph."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import networkx as nx

from cle.game_engine.models.board import base_map
from cle.game_engine.models.map import PORT_DIRECTION_TO_NODEREFS
from sft.board.board_atlas._tokens import (
    LAND_GRAPH,
    LAND_NODES,
    _positions,
    edge_token,
    node_token,
    port_token,
    tile_token,
)
from sft.board.symbolic_board_tasks import OFFSETS

# --- fact tables -----------------------------------------------------------
#
# Every table maps a string key (already tokenized) to a list of answer atoms.
# An empty list renders as NONE. Distances render as a single numeric atom.


def _node_neighbors() -> Dict[str, List[str]]:
    return {
        node_token(n): [node_token(m) for m in sorted(LAND_GRAPH.neighbors(n))]
        for n in LAND_NODES
    }


def _node_step() -> Dict[str, List[str]]:
    """Oriented adjacency: one compass step from a node, or nothing.

    Verified total and unambiguous on this board -- every node-to-node edge lies
    along one of the six `OFFSETS` directions, 24 edges per direction, and no
    node has two neighbours in the same direction. The 180 empty slots are the
    coastline, so they carry real information rather than padding the table.

    This subsumes `node_neighbors` (union over the six directions) and is the
    only table here that grounds orientation, which the board-fluency corpus
    drops entirely.
    """
    positions = _positions()
    out: Dict[str, List[str]] = {}
    for node in LAND_NODES:
        token = node_token(node)
        x, y = positions[token]
        neighbours = {
            (positions[node_token(m)][0] - x, positions[node_token(m)][1] - y): node_token(m)
            for m in LAND_GRAPH.neighbors(node)
        }
        for name, offset in OFFSETS.items():
            match = neighbours.get(offset)
            out[f"{token} {name}"] = [match] if match else []
    return out


def _node_edges() -> Dict[str, List[str]]:
    return {
        node_token(n): [edge_token(n, m) for m in sorted(LAND_GRAPH.neighbors(n))]
        for n in LAND_NODES
    }


def _edge_endpoints() -> Dict[str, List[str]]:
    return {
        edge_token(a, b): [node_token(x) for x in sorted((a, b))]
        for a, b in LAND_GRAPH.edges()
    }


def _node_tiles() -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for n in LAND_NODES:
        tiles = base_map.adjacent_tiles.get(n, [])
        out[node_token(n)] = [tile_token(t.id) for t in sorted(tiles, key=lambda t: t.id)]
    return out


def _tile_nodes() -> Dict[str, List[str]]:
    return {
        tile_token(tile.id): [node_token(n) for n in sorted(tile.nodes.values())]
        for tile in base_map.tiles_by_id.values()
    }


def _tile_neighbors() -> Dict[str, List[str]]:
    """Land tiles sharing an edge, i.e. two or more corner nodes."""
    corners = {tile.id: set(tile.nodes.values()) for tile in base_map.tiles_by_id.values()}
    out: Dict[str, List[str]] = {}
    for tid, own in corners.items():
        adjacent = sorted(
            other for other, nodes in corners.items() if other != tid and len(own & nodes) >= 2
        )
        out[tile_token(tid)] = [tile_token(t) for t in adjacent]
    return out


def _port_nodes() -> Dict[str, List[str]]:
    """The two nodes a port actually grants access through.

    Not every land node of the port hex counts: the port sits on one edge of that
    hex, and `PORT_DIRECTION_TO_NODEREFS` names the two corners of that edge.
    Intersecting the hex's six node refs with the land set instead yields three
    nodes for several ports and silently overstates access.
    """
    out: Dict[str, List[str]] = {}
    for port in base_map.ports_by_id.values():
        refs = PORT_DIRECTION_TO_NODEREFS[port.direction]
        nodes = sorted(port.nodes[ref] for ref in refs)
        out[port_token(port.id)] = [node_token(n) for n in nodes]
    return out


def _node_port() -> Dict[str, List[str]]:
    owner: Dict[int, List[str]] = {n: [] for n in LAND_NODES}
    for token, nodes in _port_nodes().items():
        for node in nodes:
            owner[int(node[2:4])].append(token)
    return {node_token(n): sorted(set(v)) for n, v in owner.items()}


def _pair_key(a: int, b: int) -> str:
    lo, hi = (a, b) if a < b else (b, a)
    return f"{node_token(lo)} {node_token(hi)}"


def _node_distance() -> Dict[str, List[str]]:
    lengths = dict(nx.all_pairs_shortest_path_length(LAND_GRAPH))
    return {
        _pair_key(a, b): [str(lengths[a][b])]
        for a, b in itertools.combinations(LAND_NODES, 2)
    }


def _node_path() -> Dict[str, List[str]]:
    """One canonical shortest path per pair, endpoints included.

    Ties are broken by networkx's traversal order, which is deterministic for a
    fixed graph; the path is a *witness*, so any shortest path is a valid target
    as long as the same one is used consistently between training and scoring.
    """
    out: Dict[str, List[str]] = {}
    for a, b in itertools.combinations(LAND_NODES, 2):
        path = nx.shortest_path(LAND_GRAPH, a, b)
        out[_pair_key(a, b)] = [node_token(n) for n in path]
    return out


@dataclass(frozen=True)
class FactTable:
    name: str
    question: str
    answer_kind: str  # "set" | "scalar" | "sequence"
    facts: Dict[str, List[str]]

    def __len__(self) -> int:
        return len(self.facts)


def build_fact_tables() -> Dict[str, FactTable]:
    """Every static-topology fact, keyed by table name. Pure, no board state."""
    specs: Sequence[Tuple[str, str, str, Callable[[], Dict[str, List[str]]]]] = (
        (
            "node_neighbors",
            "For each listed node, give the nodes joined to it by one edge.",
            "set",
            _node_neighbors,
        ),
        (
            "node_step",
            "For each listed node and compass direction, give the node one edge step "
            "that way, or NONE if the board ends there.",
            "set",
            _node_step,
        ),
        (
            "node_edges",
            "For each listed node, give the edges incident to it.",
            "set",
            _node_edges,
        ),
        (
            "edge_endpoints",
            "For each listed edge, give its two endpoint nodes.",
            "set",
            _edge_endpoints,
        ),
        (
            "node_tiles",
            "For each listed node, give the land tiles touching it.",
            "set",
            _node_tiles,
        ),
        (
            "tile_nodes",
            "For each listed tile, give its six corner nodes.",
            "set",
            _tile_nodes,
        ),
        (
            "tile_neighbors",
            "For each listed tile, give the land tiles sharing an edge with it.",
            "set",
            _tile_neighbors,
        ),
        (
            "port_nodes",
            "For each listed port, give the nodes attached to it.",
            "set",
            _port_nodes,
        ),
        (
            "node_port",
            "For each listed node, give the port attached to it.",
            "set",
            _node_port,
        ),
        (
            "node_distance",
            "For each listed node pair, give the minimum number of edges between them.",
            "scalar",
            _node_distance,
        ),
        (
            "node_path",
            "For each listed node pair, give a shortest node path between them, endpoints included.",
            "sequence",
            _node_path,
        ),
    )
    return {name: FactTable(name, q, kind, fn()) for name, q, kind, fn in specs}
