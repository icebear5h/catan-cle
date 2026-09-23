"""Canonical atlas topology cache and token-level solvers."""

from __future__ import annotations

from collections import deque
from functools import lru_cache

from cle.game_engine.models.enums import RESOURCES
from cle.game_engine.models.player import Color
from evals.catan_board_bench.tokens import AtlasTile, atlas_metadata

TASK_TYPES = ("node_tiles", "shortest_node_path", "local_node_tiles", "dice_production")
RESOURCE_KEYS = tuple(resource.lower() for resource in RESOURCES)
COLORS = tuple(color.value for color in Color)


@lru_cache(maxsize=1)
def _topology() -> tuple[dict[str, set[str]], dict[str, list[str]], dict[int, AtlasTile]]:
    atlas = atlas_metadata()
    nodes = {node["id"]: node["token"] for node in atlas["nodes"]}
    graph: dict[str, set[str]] = {token: set() for token in nodes.values()}
    for edge in atlas["edges"]:
        a, b = (nodes[node_id] for node_id in edge["id"])
        graph[a].add(b)
        graph[b].add(a)
    touching = {
        token: sorted(tile["token"] for tile in atlas["tiles"] if node_id in tile["nodes"].values())
        for node_id, token in nodes.items()
    }
    return graph, touching, {tile["id"]: tile for tile in atlas["tiles"]}


def atlas_node_graph() -> dict[str, set[str]]:
    """Return a detached copy of the canonical 54-node, 72-edge land graph."""

    return {node: set(neighbors) for node, neighbors in _topology()[0].items()}


def node_tile_tokens(node: str) -> list[str]:
    """Return the sorted canonical land-tile tokens touching a valid node."""

    touching = _topology()[1]
    if not isinstance(node, str) or node not in touching:
        raise ValueError(f"invalid canonical node: {node!r}")
    return list(touching[node])


def shortest_node_path(start: str, end: str) -> list[str]:
    """Return sorted-neighbor BFS's route, including both endpoints."""

    node_tile_tokens(start)
    node_tile_tokens(end)
    graph = _topology()[0]
    parents: dict[str, str | None] = {start: None}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        if node == end:
            path = [node]
            parent = parents[node]
            while parent is not None:
                path.append(parent)
                parent = parents[parent]
            return path[::-1]
        for neighbor in sorted(graph[node]):
            if neighbor not in parents:
                parents[neighbor] = node
                queue.append(neighbor)
    raise ValueError(f"no canonical path from {start!r} to {end!r}")
