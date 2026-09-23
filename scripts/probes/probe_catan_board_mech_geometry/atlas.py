"""Atlas token index maps and part adjacency graphs."""

from __future__ import annotations

from collections import defaultdict
from typing import TypedDict

from evals.catan_board_bench.tokens import (
    atlas_metadata,
    canonical_edge,
    edge_token,
    node_token,
    port_token,
    tile_token,
)

__all__ = ["TokenIndexMaps", "build_atlas_graphs", "find_subsequence_indices", "token_index_maps"]


class TokenIndexMaps(TypedDict):
    """Token-to-index maps per part, plus the canonical edge token order."""

    tile: dict[str, int]
    node: dict[str, int]
    edge: dict[str, int]
    port: dict[str, int]
    edge_tokens: list[str]


def find_subsequence_indices(sequence: list[int], needle: list[int]) -> list[list[int]]:
    if not sequence or not needle:
        return []
    out: list[list[int]] = []
    for start in range(0, len(sequence) - len(needle) + 1):
        if sequence[start : start + len(needle)] == needle:
            out.append(list(range(start, start + len(needle))))
    return out


def token_index_maps() -> TokenIndexMaps:
    meta = atlas_metadata()
    tile_map = {tile_token(tile["id"]): tile["id"] for tile in meta["tiles"]}
    node_map = {node_token(i): i for i in range(54)}
    edge_tokens = [
        edge_token(canonical_edge((edge["id"][0], edge["id"][1]))) for edge in meta["edges"]
    ]
    edge_map = {tok: idx for idx, tok in enumerate(edge_tokens)}
    edge_token_list = edge_tokens
    port_map = {port_token(port["id"]): port["id"] for port in meta["ports"]}
    return TokenIndexMaps(
        tile=tile_map,
        node=node_map,
        edge=edge_map,
        port=port_map,
        edge_tokens=edge_token_list,
    )


def build_atlas_graphs() -> dict[str, dict[int, set[int]]]:
    meta = atlas_metadata()
    token_maps = token_index_maps()

    tile_neighbors: dict[int, set[int]] = {tile_id: set() for tile_id in range(19)}
    node_neighbors: dict[int, set[int]] = {node_id: set() for node_id in range(54)}
    port_neighbors: dict[int, set[int]] = {port_id: set() for port_id in range(9)}

    # Edge token -> edge index
    edge_token_to_index = {tok: idx for idx, tok in enumerate(token_maps["edge_tokens"])}
    edge_graph: dict[int, set[int]] = {idx: set() for idx in range(len(edge_token_to_index))}

    # Build tile neighbors from shared edges.
    edge_to_tiles: dict[tuple[int, int], set[int]] = defaultdict(set)
    for tile in meta["tiles"]:
        tile_id = int(tile["id"])
        for raw_edge in tile["edges"].values():
            edge_tuple = canonical_edge((raw_edge[0], raw_edge[1]))
            edge_to_tiles[edge_tuple].add(tile_id)

    for edge_tiles in edge_to_tiles.values():
        if len(edge_tiles) < 2:
            continue
        tiles = sorted(edge_tiles)
        t0 = tiles[0]
        t1 = tiles[1]
        tile_neighbors[t0].add(t1)
        tile_neighbors[t1].add(t0)

    # Node neighbors and edge neighbors from canonical edge list.
    edge_indices_by_node: dict[int, set[int]] = defaultdict(set)
    for atlas_edge in meta["edges"]:
        edge_tuple = canonical_edge((atlas_edge["id"][0], atlas_edge["id"][1]))
        edge_index = edge_token_to_index.get(edge_token(edge_tuple))
        if edge_index is None:
            continue
        n0, n1 = edge_tuple
        node_neighbors[n0].add(n1)
        node_neighbors[n1].add(n0)
        edge_indices_by_node[n0].add(edge_index)
        edge_indices_by_node[n1].add(edge_index)

    for edges_at_node in edge_indices_by_node.values():
        edges_sorted = sorted(edges_at_node)
        for i, e0 in enumerate(edges_sorted):
            for e1 in edges_sorted[i + 1 :]:
                edge_graph[e0].add(e1)
                edge_graph[e1].add(e0)

    # Ports are adjacent if they share any node.
    node_to_ports: dict[int, set[int]] = defaultdict(set)
    for port in meta["ports"]:
        port_id = int(port["id"])
        for node_id in port["attached_nodes"]:
            node_to_ports[node_id].add(port_id)
    for ports in node_to_ports.values():
        for p in ports:
            port_neighbors[p].update(ports - {p})

    # Reindex edge neighbors into token index space.
    edge_neighbors_by_index: dict[int, set[int]] = {}
    for _token, idx in edge_token_to_index.items():
        edgeset = edge_graph.get(idx, set())
        if edgeset:
            edge_neighbors_by_index[idx] = edgeset
        else:
            edge_neighbors_by_index[idx] = set()

    return {
        "tile": {tile_id: neighbors for tile_id, neighbors in tile_neighbors.items()},
        "node": {node_id: neighbors for node_id, neighbors in node_neighbors.items()},
        "edge": edge_neighbors_by_index,
        "port": port_neighbors,
    }
