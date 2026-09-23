"""Occupation-integrated board placement with strict glyph collision checks."""

from __future__ import annotations

import math
from collections.abc import Sequence

from evals.catan_board_bench.ascii_variations.facts import FullFacts
from evals.catan_board_bench.full_graph_formats.schema import _BUILDING_TO_CODE, _COLOR_TO_CODE


def _integrated_diagram(facts: FullFacts) -> list[str]:
    tile_centers = {tile["id"]: _cube_point(tile["cube"]) for tile in facts["tiles"]}
    corner_offsets = {
        "NORTH": (0.0, -1.0),
        "NORTHEAST": (math.sqrt(3) / 2, -0.5),
        "SOUTHEAST": (math.sqrt(3) / 2, 0.5),
        "SOUTH": (0.0, 1.0),
        "SOUTHWEST": (-math.sqrt(3) / 2, 0.5),
        "NORTHWEST": (-math.sqrt(3) / 2, -0.5),
    }
    node_points: dict[str, tuple[float, float]] = {}
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
            round((point[0] - min_x) * 22) + 18,
            round((point[1] - min_y) * 9) + 3,
        )

    grid_nodes = {node_id: grid(point) for node_id, point in node_points.items()}
    grid_tiles = {tile_id: grid(point) for tile_id, point in tile_centers.items()}
    max_x = max(point[0] for point in [*grid_nodes.values(), *grid_tiles.values()]) + 18
    max_y = max(point[1] for point in [*grid_nodes.values(), *grid_tiles.values()]) + 4
    canvas = [[" " for _ in range(max_x + 1)] for _ in range(max_y + 1)]

    node_by_id = {node["id"]: node for node in facts["nodes"]}
    for edge in facts["edges"]:
        start = grid_nodes[edge["nodes"][0]]
        end = grid_nodes[edge["nodes"][1]]
        _draw_line(canvas, start, end)
    for edge in facts["edges"]:
        start = grid_nodes[edge["nodes"][0]]
        end = grid_nodes[edge["nodes"][1]]
        midpoint = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
        state = "-" if edge["road"] is None else _COLOR_TO_CODE[edge["road"]]
        _overlay_strict(canvas, midpoint, f"{edge['id']}[{state}]")
    for tile in facts["tiles"]:
        number = "-" if tile["number"] is None else tile["number"]
        robber = "*" if tile["robber"] else ""
        label = f"{tile['id']}[{tile['resource']}/{number}{robber}]"
        _overlay_strict(canvas, grid_tiles[tile["id"]], label)
    for node_id, point in grid_nodes.items():
        node = node_by_id[node_id]
        if node["building"] is None:
            state = "-"
        else:
            state = f"{_color_code(node['color'])}/{_BUILDING_TO_CODE[node['building']]}"
        _overlay_strict(canvas, point, f"{node_id}[{state}]")

    return [line.rstrip() for line in ("".join(row) for row in canvas) if line.rstrip()]


def _color_code(color: str | None) -> str:
    if color is None:
        raise KeyError(color)
    return _COLOR_TO_CODE[color]


def _cube_point(cube: Sequence[int]) -> tuple[float, float]:
    q = cube[0]
    r = cube[2]
    return (math.sqrt(3) * (q + r / 2), 1.5 * r)


def _draw_line(
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


def _overlay_strict(
    canvas: list[list[str]],
    center: tuple[int, int],
    label: str,
) -> None:
    start_x = center[0] - len(label) // 2
    y = center[1]
    for offset, character in enumerate(label):
        x = start_x + offset
        if not (0 <= y < len(canvas) and 0 <= x < len(canvas[y])):
            raise ValueError(f"diagram label out of bounds: {label}")
        existing = canvas[y][x]
        if existing not in {" ", "-", "/", "\\"}:
            raise ValueError(f"diagram label collision for {label!r} at {(x, y)} with {existing!r}")
        canvas[y][x] = character
