"""Atlas geometry, graph distance, and compass direction helpers."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence

from data_pipeline.board_recognition.robber_impl._config import SpatialRobberError
from evals.catan_board_bench.tokens import atlas_metadata


def _pixel(coord: Sequence[int]) -> tuple[float, float]:
    q, _cube_y, r = coord
    return math.sqrt(3) * (q + r / 2), 1.5 * r


def _atlas_geometry() -> tuple[
    dict[str, tuple[float, float]],
    dict[str, set[str]],
    dict[str, tuple[float, float]],
    dict[str, set[str]],
]:
    atlas = atlas_metadata()
    tile_positions = {row["token"]: _pixel(row["coord"]) for row in atlas["tiles"]}
    node_positions: dict[str, tuple[float, float]] = {}
    offsets = {
        "NORTH": (0.0, -1.0),
        "NORTHEAST": (math.sqrt(3) / 2, -0.5),
        "SOUTHEAST": (math.sqrt(3) / 2, 0.5),
        "SOUTH": (0.0, 1.0),
        "SOUTHWEST": (-math.sqrt(3) / 2, 0.5),
        "NORTHWEST": (-math.sqrt(3) / 2, -0.5),
    }
    for tile in atlas["tiles"]:
        center = tile_positions[tile["token"]]
        for direction, node_id in tile["nodes"].items():
            dx, dy = offsets[direction]
            node_positions.setdefault(f"<N{node_id:02d}>", (center[0] + dx, center[1] + dy))

    node_graph: dict[str, set[str]] = {token: set() for token in node_positions}
    for edge in atlas["edges"]:
        left, right = (f"<N{node_id:02d}>" for node_id in edge["id"])
        node_graph[left].add(right)
        node_graph[right].add(left)

    tile_graph: dict[str, set[str]] = {token: set() for token in tile_positions}
    tile_nodes = {row["token"]: set(row["nodes"].values()) for row in atlas["tiles"]}
    tokens = sorted(tile_positions)
    for index, left in enumerate(tokens):
        for right in tokens[index + 1 :]:
            if len(tile_nodes[left] & tile_nodes[right]) == 2:
                tile_graph[left].add(right)
                tile_graph[right].add(left)
    return node_positions, node_graph, tile_positions, tile_graph


def _graph_distance(graph: dict[str, set[str]], source: str, target: str) -> int:
    queue = deque([(source, 0)])
    seen = {source}
    while queue:
        current, distance = queue.popleft()
        if current == target:
            return distance
        for neighbor in sorted(graph[current]):
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append((neighbor, distance + 1))
    raise SpatialRobberError(f"disconnected atlas graph: {source}, {target}")


def _direction(
    left: str,
    right: str,
    positions: dict[str, tuple[float, float]],
) -> str:
    dx = positions[left][0] - positions[right][0]
    dy = positions[left][1] - positions[right][1]
    if abs(dx) > abs(dy):
        return "right of" if dx > 0 else "left of"
    return "below" if dy > 0 else "above"


def _opposite(relation: str) -> str:
    return {
        "above": "below",
        "below": "above",
        "left of": "right of",
        "right of": "left of",
    }[relation]

