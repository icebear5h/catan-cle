"""Pointy-top hex geometry: cube coordinates to pixels, and node averaging."""

import math
from collections import defaultdict
from collections.abc import Callable

from cle.game_engine.models.coordinate_system import cube_to_axial
from cle.game_engine.models.enums import NodeRef
from cle.game_engine.models.map import CatanMap

__all__ = [
    "NODE_OFFSETS",
    "compute_node_positions",
    "cube_to_pixel",
    "hex_to_pixel",
    "hexagon_vertices",
]


def hex_to_pixel(q: int, r: int, size: float) -> tuple[float, float]:
    """Convert axial hex coordinates to pixel position (pointy-top, y-down)."""
    x = size * (math.sqrt(3) * q + math.sqrt(3) / 2 * r)
    y = size * (3.0 / 2 * r)
    return (x, y)


def cube_to_pixel(cube_coord: tuple[int, int, int], size: float) -> tuple[float, float]:
    """Convert engine cube coordinate to pixel position."""
    q, r = cube_to_axial(cube_coord)
    return hex_to_pixel(q, r, size)


# Node offsets from hex center for pointy-top hex (y-down)
NODE_OFFSETS: dict[NodeRef, Callable[[float], tuple[float, float]]] = {
    NodeRef.NORTH: lambda s: (0, -s),
    NodeRef.NORTHEAST: lambda s: (s * math.sqrt(3) / 2, -s / 2),
    NodeRef.SOUTHEAST: lambda s: (s * math.sqrt(3) / 2, s / 2),
    NodeRef.SOUTH: lambda s: (0, s),
    NodeRef.SOUTHWEST: lambda s: (-s * math.sqrt(3) / 2, s / 2),
    NodeRef.NORTHWEST: lambda s: (-s * math.sqrt(3) / 2, -s / 2),
}


def hexagon_vertices(cx: float, cy: float, size: float) -> list[tuple[float, float]]:
    """Get 6 vertices of a pointy-top hexagon centered at (cx, cy)."""
    vertices: list[tuple[float, float]] = []
    for i in range(6):
        angle = math.radians(60 * i - 30)  # pointy-top: start at -30 deg
        vx = cx + size * math.cos(angle)
        vy = cy + size * math.sin(angle)
        vertices.append((vx, vy))
    return vertices


def compute_node_positions(
    catan_map: CatanMap, hex_size: float
) -> dict[int, tuple[float, float]]:
    """Compute pixel positions for all nodes by averaging across shared tiles."""
    node_accum: dict[int, list[tuple[float, float]]] = defaultdict(list)

    for coord, tile in catan_map.tiles.items():
        if not hasattr(tile, "nodes"):
            continue
        tile_center = cube_to_pixel(coord, hex_size)
        for node_ref, node_id in tile.nodes.items():
            offset_fn = NODE_OFFSETS.get(node_ref)
            if offset_fn is None:
                continue
            dx, dy = offset_fn(hex_size)
            node_accum[node_id].append((tile_center[0] + dx, tile_center[1] + dy))
    result: dict[int, tuple[float, float]] = {}
    result = {}
    for node_id, positions in node_accum.items():
        avg_x = sum(p[0] for p in positions) / len(positions)
        avg_y = sum(p[1] for p in positions) / len(positions)
        result[node_id] = (avg_x, avg_y)

    return result
