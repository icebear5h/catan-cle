"""Marker glyphs, geometry, and spatial-target row construction."""

from __future__ import annotations

import math
from typing import Sequence, TypeAlias

from PIL import Image, ImageDraw, ImageFont

from data_pipeline.board_recognition import spatial_localization as api
from data_pipeline.board_recognition.replay_dataset import JsonDict
from data_pipeline.json_coerce import as_dict, as_list, as_str


def _font(radius: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", max(18, int(radius * 1.25)))
    except OSError:
        try:
            return ImageFont.load_default(size=max(18, int(radius * 1.25)))
        except TypeError:
            return ImageFont.load_default()


# Marker geometry is an in-process payload, not JSON: points arrive as tuples.
MarkerGeometry: TypeAlias = dict[str, object]


def _coordinate(part: object) -> float:
    if isinstance(part, bool) or not isinstance(part, (int, float)):
        raise TypeError(f"marker coordinate must be a number, got {type(part).__name__}")
    return float(part)


def _point(value: object) -> tuple[float, float]:
    """Read one (x, y) pixel pair out of a marker geometry payload."""

    if not isinstance(value, (list, tuple)):
        raise TypeError(f"marker point must be a sequence, got {type(value).__name__}")
    x, y = (_coordinate(part) for part in value)
    return x, y


def entity_marker_polygon(
    geometry: MarkerGeometry, radius: int
) -> list[tuple[float, float]] | None:
    """Polygon for an entity-shaped marker, or None to fall back to the style glyph.

    Edges get a bar along the edge at its true angle, so the marker stage asks
    the model to ground a thin slanted shape the way a road is drawn; tiles
    get a hexagon at tile scale. Nodes keep the style glyph.
    """

    kind = geometry.get("kind")
    if kind == "edge" and geometry.get("endpoints"):
        endpoints = geometry["endpoints"]
        if not isinstance(endpoints, (list, tuple)):
            raise TypeError("marker endpoints must be a sequence")
        (x1, y1), (x2, y2) = (_point(endpoint) for endpoint in endpoints)
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length == 0:
            return None
        ux, uy = dx / length, dy / length
        nx, ny = -uy, ux
        half = radius * 0.55
        inset = length * 0.12
        ax, ay = x1 + ux * inset, y1 + uy * inset
        bx, by = x2 - ux * inset, y2 - uy * inset
        return [(ax + nx * half, ay + ny * half), (bx + nx * half, by + ny * half), (bx - nx * half, by - ny * half), (ax - nx * half, ay - ny * half)]
    if kind == "tile":
        cx, cy = _point(geometry["center"])
        size = radius * 1.8
        return [(cx + size * math.cos(math.radians(60 * index + 30)), cy + size * math.sin(math.radians(60 * index + 30))) for index in range(6)]
    return None


def render_markers(
    base_image: Image.Image,
    assignments: Sequence[
        tuple[str, tuple[int, int]] | tuple[str, tuple[int, int], MarkerGeometry]
    ],
    *,
    style_name: str,
    entity_shaped: bool = False,
) -> Image.Image:
    """Overlay readable markers without changing the underlying board layout.

    With ``entity_shaped`` an assignment may carry a third element describing
    the entity (``{"kind": "edge", "endpoints": [...]}`` or ``{"kind": "tile",
    "center": ...}``); those markers take the entity's shape and the letter
    sits at the centre as before.
    """

    if style_name not in api.MARKER_STYLES:
        raise ValueError(f"unknown marker style: {style_name}")
    shape, fill, text_fill = api.MARKER_STYLES[style_name]
    image = base_image.convert("RGB").copy()
    draw = ImageDraw.Draw(image)
    radius = max(16, round(min(image.size) * 0.023))
    font = api._font(radius)
    outline = (255, 255, 255)
    for assignment in assignments:
        marker, (cx, cy) = assignment[0], assignment[1]
        geometry = assignment[2] if len(assignment) > 2 else {}
        if marker not in api.MARKERS:
            raise ValueError(f"unknown marker: {marker}")
        entity_polygon = api.entity_marker_polygon(geometry, radius) if entity_shaped else None
        if entity_polygon is not None:
            draw.polygon(entity_polygon, fill=fill, outline=outline)
        elif shape == "circle":
            bounds = [cx - radius, cy - radius, cx + radius, cy + radius]
            draw.ellipse(bounds, fill=fill, outline=outline, width=3)
        elif shape == "square":
            bounds = [cx - radius, cy - radius, cx + radius, cy + radius]
            draw.rounded_rectangle(bounds, radius=4, fill=fill, outline=outline, width=3)
        elif shape == "diamond":
            points = [(cx, cy - radius), (cx + radius, cy), (cx, cy + radius), (cx - radius, cy)]
            draw.polygon(points, fill=fill, outline=outline)
        else:
            points = [(cx, cy - radius), (cx + radius, cy + radius), (cx - radius, cy + radius)]
            draw.polygon(points, fill=fill, outline=outline)
        box = draw.textbbox((0, 0), marker, font=font)
        width, height = box[2] - box[0], box[3] - box[1]
        draw.text((cx - width / 2, cy - height / 2 - box[1]), marker, font=font, fill=text_fill)
    return image


def _marker_style(split: str, board_index: int) -> str:
    if split == "train":
        return "train_circle" if board_index % 2 == 0 else "train_square"
    if split == "validation":
        return "validation_diamond"
    return "test_triangle"


def _control_regions(regions: dict[str, JsonDict]) -> dict[str, JsonDict]:
    controls: dict[str, JsonDict] = {}
    for tokens in api._atlas_tokens_by_kind().values():
        for index, token in enumerate(tokens):
            controls[token] = regions[tokens[(index + 1) % len(tokens)]]
    return controls


def _spatial_target(region: JsonDict, control: JsonDict) -> JsonDict:
    return {
        "token": region["token"],
        "entity_type": region["entity_type"],
        "bbox": region["bbox"],
        "center": region["center"],
        "control_bbox": control["bbox"],
        "control_token": control["token"],
    }


def _training_row(
    *,
    row_id: str,
    image_name: str,
    prompt: str,
    answer: str,
    grounding_stage: str,
    task_type: str,
    metadata: JsonDict,
    spatial_target: JsonDict | None = None,
) -> JsonDict:
    row: JsonDict = {
        "schema": api.ROW_SCHEMA,
        "row_id": row_id,
        "curriculum_stage": "spatial_grounding",
        "grounding_stage": grounding_stage,
        "task_family": "spatial_localization",
        "task_type": task_type,
        "images": [image_name],
        "messages": [
            {"role": "user", "content": f"<image>\n{prompt}"},
            {"role": "assistant", "content": answer},
        ],
        **metadata,
    }
    if spatial_target is not None:
        row["spatial_targets"] = [spatial_target]
    return row


def _marker_geometry(
    token: str, regions: dict[str, JsonDict], contract: JsonDict
) -> MarkerGeometry:
    kind = regions[token]["entity_type"]
    if kind == "edge":
        edge = next(
            as_dict(entry)
            for entry in as_list(contract["edges"])
            if as_dict(entry)["token"] == token
        )
        return {
            "kind": "edge",
            "endpoints": [
                tuple(as_list(regions[as_str(node)]["center_pixels"]))
                for node in as_list(edge["node_tokens"])
            ],
        }
    return {"kind": kind, "center": tuple(as_list(regions[token]["center_pixels"]))}
