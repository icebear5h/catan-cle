"""Pixel transforms, bbox math, and the cached atlas reference tables."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import TYPE_CHECKING, TypedDict

from evals.catan_board_bench.annotations.constants import (
    HEX_SIZE,
    HEX_SPACING,
)
from evals.catan_board_bench.tokens import atlas_metadata, canonical_edge
from evals.json_types import JsonDict, JsonValue

if TYPE_CHECKING:
    from evals.catan_board_bench.annotations.render_state import RenderState


class AtlasRenderRef(TypedDict):
    tile_coordinate: list[int]
    direction: str


class AtlasRenderRefs(TypedDict):
    nodes: dict[int, AtlasRenderRef]
    edges: dict[tuple[int, ...], AtlasRenderRef]


@lru_cache(maxsize=1)
def _atlas_render_refs() -> AtlasRenderRefs:
    atlas = atlas_metadata()

    node_refs: dict[int, AtlasRenderRef] = {}
    edge_refs: dict[tuple[int, ...], AtlasRenderRef] = {}
    for tile in atlas["tiles"]:
        coord = tile["coord"]
        for direction, node_id in tile["nodes"].items():
            node_refs.setdefault(node_id, {"tile_coordinate": coord, "direction": direction})
        for direction, edge in tile["edges"].items():
            edge_refs.setdefault(tuple(edge), {"tile_coordinate": coord, "direction": direction})

    return {"nodes": node_refs, "edges": edge_refs}


def _hex_to_pixel(coord: list[int] | tuple[int, int, int]) -> dict[str, float]:
    cube_x, _cube_y, cube_z = coord
    size = HEX_SIZE + HEX_SPACING
    q = cube_x
    r = cube_z
    return {
        "x": size * math.sqrt(3) * (q + r / 2),
        "y": size * (3 / 2) * r,
    }


def _node_offset(direction: str) -> dict[str, float]:
    s = HEX_SIZE + HEX_SPACING
    w = s * math.sqrt(3) / 2
    offsets = {
        "NORTH": {"x": 0.0, "y": -s},
        "NORTHEAST": {"x": w, "y": -s / 2},
        "SOUTHEAST": {"x": w, "y": s / 2},
        "SOUTH": {"x": 0.0, "y": s},
        "SOUTHWEST": {"x": -w, "y": s / 2},
        "NORTHWEST": {"x": -w, "y": -s / 2},
    }
    return offsets.get(direction, {"x": 0.0, "y": 0.0})


def _node_positions(render_state: RenderState) -> dict[int, dict[str, float]]:
    positions: dict[int, dict[str, float]] = {}
    for node in render_state.get("nodes", {}).values():
        center = _hex_to_pixel(node["tile_coordinate"])
        offset = _node_offset(node["direction"])
        positions[node["id"]] = {
            "x": center["x"] + offset["x"],
            "y": center["y"] + offset["y"],
        }
    return positions


def _view_box(render_state: RenderState) -> dict[str, float]:
    positions = [_hex_to_pixel(placed["coordinate"]) for placed in render_state.get("tiles", [])]
    min_x = min(point["x"] for point in positions) - HEX_SIZE * 2.5
    max_x = max(point["x"] for point in positions) + HEX_SIZE * 2.5
    min_y = min(point["y"] for point in positions) - HEX_SIZE * 2.5
    max_y = max(point["y"] for point in positions) + HEX_SIZE * 2.5
    return {"min_x": min_x, "min_y": min_y, "width": max_x - min_x, "height": max_y - min_y}


def _view_box_payload(render_state: RenderState, image_size: int) -> dict[str, float]:
    view_box = _view_box(render_state)
    transform = _pixel_transform(view_box, image_size)
    return {**view_box, **transform}


def _pixel_transform(view_box: dict[str, float], image_size: int) -> dict[str, float]:
    scale = min(image_size / view_box["width"], image_size / view_box["height"])
    return {
        "scale": scale,
        "offset_x": (image_size - view_box["width"] * scale) / 2,
        "offset_y": (image_size - view_box["height"] * scale) / 2,
        "image_size": float(image_size),
        "min_x": view_box["min_x"],
        "min_y": view_box["min_y"],
    }


def _annotation(
    *,
    kind: str,
    token: str,
    bbox_svg: dict[str, float],
    point_svg: dict[str, float],
    transform: dict[str, float],
    semantic: JsonDict,
    label: str | None = None,
) -> JsonDict:
    bbox = _bbox_to_pixels(bbox_svg, transform)
    center = _point_to_pixels(point_svg, transform)
    bbox_json: list[JsonValue] = list(bbox)
    center_json: list[JsonValue] = list(center)
    annotation: JsonDict = {
        "id": f"{kind}:{label or token}",
        "kind": kind,
        "token": token,
        "bbox": bbox_json,
        "bbox_2d": bbox_json,
        "center": center_json,
        "point": center_json,
        "bbox_svg": dict(_round_bbox(bbox_svg)),
        "point_svg": dict(_round_point(point_svg)),
        "semantic": semantic,
    }
    if label:
        annotation["label"] = label
    return annotation


def _rect(x: float, y: float, width: float, height: float) -> dict[str, float]:
    return {"x1": x, "y1": y, "x2": x + width, "y2": y + height}


def _bbox_to_pixels(bbox: dict[str, float], transform: dict[str, float]) -> list[int]:
    p1 = _point_to_pixels({"x": bbox["x1"], "y": bbox["y1"]}, transform)
    p2 = _point_to_pixels({"x": bbox["x2"], "y": bbox["y2"]}, transform)
    image_size = int(transform["image_size"])
    return [
        _clamp_int(min(p1[0], p2[0]), 0, image_size),
        _clamp_int(min(p1[1], p2[1]), 0, image_size),
        _clamp_int(max(p1[0], p2[0]), 0, image_size),
        _clamp_int(max(p1[1], p2[1]), 0, image_size),
    ]


def _point_to_pixels(point: dict[str, float], transform: dict[str, float]) -> list[int]:
    x = (point["x"] - transform["min_x"]) * transform["scale"] + transform["offset_x"]
    y = (point["y"] - transform["min_y"]) * transform["scale"] + transform["offset_y"]
    image_size = int(transform["image_size"])
    return [_clamp_int(x, 0, image_size), _clamp_int(y, 0, image_size)]


def _clamp_int(value: float, lower: int, upper: int) -> int:
    return max(lower, min(upper, int(round(value))))


def _round_bbox(bbox: dict[str, float]) -> dict[str, float]:
    return {key: round(value, 3) for key, value in bbox.items()}


def _round_point(point: dict[str, float]) -> dict[str, float]:
    return {key: round(value, 3) for key, value in point.items()}


def _object_token(prefix: str, object_id: int) -> str:
    return f"<{prefix}{object_id:02d}>"


def _edge_token(edge: list[int] | tuple[int, int]) -> str:
    first, second = edge
    a, b = canonical_edge((first, second))
    return f"<E{a:02d}_{b:02d}>"

