"""ASCII topology placement and canvas primitives."""

from __future__ import annotations

import math
from collections.abc import Sequence

from evals.catan_board_bench.ascii_variations.facts import FullFacts


def _topology_diagram(facts: FullFacts) -> list[str]:
    tile_centers = {tile["id"]: _cube_point(tile["cube"]) for tile in facts["tiles"]}
    node_points: dict[str, tuple[float, float]] = {}
    corner_offsets = {
        "NORTH": (0.0, -1.0),
        "NORTHEAST": (math.sqrt(3) / 2, -0.5),
        "SOUTHEAST": (math.sqrt(3) / 2, 0.5),
        "SOUTH": (0.0, 1.0),
        "SOUTHWEST": (-math.sqrt(3) / 2, 0.5),
        "NORTHWEST": (-math.sqrt(3) / 2, -0.5),
    }
    for tile in facts["tiles"]:
        center = tile_centers[tile["id"]]
        for direction, node_id in tile["corners"].items():
            offset = corner_offsets[direction]
            point = (center[0] + offset[0], center[1] + offset[1])
            previous = node_points.get(node_id)
            if previous is not None and (
                abs(previous[0] - point[0]) > 1e-6 or abs(previous[1] - point[1]) > 1e-6
            ):
                raise ValueError(f"inconsistent diagram position for {node_id}")
            node_points[node_id] = point

    all_points = [*node_points.values(), *tile_centers.values()]
    min_x = min(point[0] for point in all_points)
    min_y = min(point[1] for point in all_points)

    def grid(point: tuple[float, float]) -> tuple[int, int]:
        return (
            round((point[0] - min_x) * 12) + 4,
            round((point[1] - min_y) * 5) + 2,
        )

    grid_nodes = {node_id: grid(point) for node_id, point in node_points.items()}
    grid_tiles = {tile_id: grid(point) for tile_id, point in tile_centers.items()}
    max_x = max(point[0] for point in [*grid_nodes.values(), *grid_tiles.values()]) + 5
    max_y = max(point[1] for point in [*grid_nodes.values(), *grid_tiles.values()]) + 3
    canvas = [[" " for _ in range(max_x + 1)] for _ in range(max_y + 1)]

    for edge in facts["edges"]:
        start = grid_nodes[edge["nodes"][0]]
        end = grid_nodes[edge["nodes"][1]]
        _draw_ascii_line(canvas, start, end)
        midpoint = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
        _overlay(canvas, midpoint, edge["id"])
    for tile in facts["tiles"]:
        label = tile["id"] + ("*" if tile["robber"] else "")
        _overlay(canvas, grid_tiles[tile["id"]], label)
    for node_id, point in grid_nodes.items():
        _overlay(canvas, point, node_id)

    return [line.rstrip() for line in ("".join(row) for row in canvas) if line.rstrip()]


def _draw_ascii_line(
    canvas: list[list[str]],
    start: tuple[int, int],
    end: tuple[int, int],
) -> None:
    x0, y0 = start
    x1, y1 = end
    steps = max(abs(x1 - x0), abs(y1 - y0))
    if steps == 0:
        return
    glyph = "-" if y0 == y1 else ("\\" if (x1 - x0) * (y1 - y0) > 0 else "/")
    for step in range(1, steps):
        x = round(x0 + (x1 - x0) * step / steps)
        y = round(y0 + (y1 - y0) * step / steps)
        if canvas[y][x] == " ":
            canvas[y][x] = glyph


def _overlay(
    canvas: list[list[str]],
    center: tuple[int, int],
    label: str,
) -> None:
    start_x = center[0] - len(label) // 2
    y = center[1]
    for offset, character in enumerate(label):
        x = start_x + offset
        if 0 <= y < len(canvas) and 0 <= x < len(canvas[y]):
            canvas[y][x] = character


def _cube_point(cube: Sequence[int]) -> tuple[float, float]:
    q = cube[0]
    r = cube[2]
    return (math.sqrt(3) * (q + r / 2), 1.5 * r)
