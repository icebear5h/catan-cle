"""Python renderer for Catan public-board contracts.

This mirrors the production path in ``playground/frontend/src/components/HexBoard.tsx``:

- same Catan render-state shape via ``contract_to_render_state``
- same geometry constants and coordinate transform as ``annotations.py``
- same frontend SVG/PNG assets from ``playground/frontend/public/assets``

The goal is bulk image generation without Playwright while keeping Playwright
available as the calibration oracle for spot checks.
"""

from __future__ import annotations

import io
import math
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from evals.catan_board_bench.annotations import (
    HEX_SIZE,
    PORT_SHIP_SIZE,
    PORT_SHIP_X_OFFSET,
    PORT_SHIP_Y_OFFSET,
    TILE_HEIGHT,
    TILE_WIDTH,
    NUMBER_TOKEN_SIZE,
    NUMBER_TOKEN_Y_OFFSET,
    contract_to_render_state,
    _hex_to_pixel,
    _node_positions,
    _pixel_transform,
    _view_box,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = PROJECT_ROOT / "playground/frontend/public/assets"
WATER_RGB = (9, 103, 165)
SETTLEMENT_SIZE = HEX_SIZE * 0.63
CITY_SIZE = HEX_SIZE * 0.80
ROBBER_SIZE = HEX_SIZE * 0.55
COAST_EDGE_ASSET = "/assets/tiles/coast_edge.png"
COAST_CORNER_ASSET = "/assets/tiles/coast_corner.png"
COAST_PIECE_SIZE = TILE_HEIGHT
DOCK_ASSET = "/assets/tiles/dock.svg"
RESOURCE_COLORS = {
    "WOOD": (34, 139, 34, 230),
    "BRICK": (184, 83, 26, 230),
    "SHEEP": (144, 238, 144, 230),
    "WHEAT": (244, 208, 63, 230),
    "ORE": (112, 128, 144, 230),
}
CUBE_DIRS = [
    (1, 0, -1),
    (1, -1, 0),
    (0, -1, 1),
    (-1, 0, 1),
    (-1, 1, 0),
    (0, 1, -1),
]


@dataclass(frozen=True)
class RenderStyle:
    """Tunable renderer dimensions in HexBoard SVG coordinate units."""

    road_width_factor: float = 0.81
    road_length_factor: float = 1.02
    dock_width: float = 8.5
    dock_extend_factor: float = 0.12
    dock_ship_clearance_factor: float = 0.12
    dock_port_gap: float = 12.5
    dock_sand_gap: float = 7.0
    coast_size_factor: float = 1.04
    view_padding_factor: float = 1.20
    robber_size_factor: float = 0.71
    settlement_size_factor: float = 0.63
    settlement_x_offset: float = 0.0
    settlement_y_factor: float = 0.70
    city_size_factor: float = 0.69
    city_width_factor: float = 1.12
    city_x_offset: float = 0.0
    city_y_factor: float = 0.65


DEFAULT_RENDER_STYLE = RenderStyle()
ROAD_WIDTH = HEX_SIZE * DEFAULT_RENDER_STYLE.road_width_factor
ROAD_LENGTH_FACTOR = DEFAULT_RENDER_STYLE.road_length_factor
DOCK_WIDTH = DEFAULT_RENDER_STYLE.dock_width
DOCK_EXTEND_FACTOR = DEFAULT_RENDER_STYLE.dock_extend_factor
DOCK_SHIP_CLEARANCE_FACTOR = DEFAULT_RENDER_STYLE.dock_ship_clearance_factor
DOCK_PORT_GAP = DEFAULT_RENDER_STYLE.dock_port_gap
DOCK_SAND_GAP = DEFAULT_RENDER_STYLE.dock_sand_gap
COAST_SIZE_FACTOR = DEFAULT_RENDER_STYLE.coast_size_factor
VIEW_PADDING_FACTOR = DEFAULT_RENDER_STYLE.view_padding_factor
ROBBER_SIZE_FACTOR = DEFAULT_RENDER_STYLE.robber_size_factor
SETTLEMENT_SIZE_FACTOR = DEFAULT_RENDER_STYLE.settlement_size_factor
SETTLEMENT_X_OFFSET = DEFAULT_RENDER_STYLE.settlement_x_offset
SETTLEMENT_Y_FACTOR = DEFAULT_RENDER_STYLE.settlement_y_factor
CITY_SIZE_FACTOR = DEFAULT_RENDER_STYLE.city_size_factor
CITY_WIDTH_FACTOR = DEFAULT_RENDER_STYLE.city_width_factor
CITY_X_OFFSET = DEFAULT_RENDER_STYLE.city_x_offset
CITY_Y_FACTOR = DEFAULT_RENDER_STYLE.city_y_factor


def render_contract_image(
    contract: dict[str, Any],
    *,
    image_size: int = 512,
    style: RenderStyle | None = None,
) -> Image.Image:
    """Render a public-board contract into a square RGB board image."""

    return render_state_image(
        contract_to_render_state(contract), image_size=image_size, style=style
    )


def render_state_image(
    render_state: dict[str, Any],
    *,
    image_size: int = 512,
    style: RenderStyle | None = None,
) -> Image.Image:
    """Render a HexBoard-compatible state into a square RGB board image."""

    style = style or DEFAULT_RENDER_STYLE
    view_box = _view_box_with_padding(render_state, style.view_padding_factor)
    transform = _pixel_transform(view_box, image_size)
    canvas = Image.new("RGBA", (image_size, image_size), (*WATER_RGB, 255))
    node_positions = _node_positions(render_state)

    _render_coast(canvas, render_state, transform, style)
    _render_land_tiles(canvas, render_state, transform, style)
    _render_ports(canvas, render_state, transform, node_positions, style)
    _render_roads(canvas, render_state, transform, node_positions, style)
    _render_buildings(canvas, render_state, transform, node_positions, style)
    _render_robber(canvas, render_state, transform, style)
    return canvas.convert("RGB")


def _view_box_with_padding(render_state: dict[str, Any], padding_factor: float) -> dict[str, float]:
    positions = [_hex_to_pixel(placed["coordinate"]) for placed in render_state.get("tiles", [])]
    if not positions:
        return _view_box(render_state)
    padding = HEX_SIZE * max(0.0, padding_factor)
    min_x = min(point["x"] for point in positions) - padding
    max_x = max(point["x"] for point in positions) + padding
    min_y = min(point["y"] for point in positions) - padding
    max_y = max(point["y"] for point in positions) + padding
    return {"min_x": min_x, "min_y": min_y, "width": max_x - min_x, "height": max_y - min_y}


def _render_coast(
    canvas: Image.Image,
    render_state: dict[str, Any],
    transform: dict[str, float],
    style: RenderStyle,
) -> None:
    land_coords = {
        tuple(placed["coordinate"])
        for placed in render_state.get("tiles", [])
        if placed["tile"]["type"] in {"RESOURCE_TILE", "DESERT"}
    }
    explicit_water_coords = {
        tuple(placed["coordinate"])
        for placed in render_state.get("tiles", [])
        if placed["tile"]["type"] not in {"RESOURCE_TILE", "DESERT"}
    }
    inferred_water_coords = {
        (cx + dx, cy + dy, cz + dz)
        for cx, cy, cz in land_coords
        for dx, dy, dz in CUBE_DIRS
        if (cx + dx, cy + dy, cz + dz) not in land_coords
    }

    for cx, cy, cz in sorted(explicit_water_coords | inferred_water_coords):
        center = _hex_to_pixel((cx, cy, cz))
        land_dirs = [
            index
            for index, (dx, dy, dz) in enumerate(CUBE_DIRS)
            if (cx + dx, cy + dy, cz + dz) in land_coords
        ]
        if not land_dirs:
            continue

        if len(land_dirs) == 2:
            d1, d2 = land_dirs
            diff = ((d2 - d1) + 6) % 6
            if diff == 1:
                coast_size = COAST_PIECE_SIZE * style.coast_size_factor
                _paste_rotated_asset(
                    canvas,
                    _asset_path(COAST_CORNER_ASSET),
                    center_svg=(center["x"], center["y"]),
                    width_svg=coast_size,
                    height_svg=coast_size,
                    angle_deg=d1 * 60,
                    transform=transform,
                )
                continue
            if diff == 5:
                coast_size = COAST_PIECE_SIZE * style.coast_size_factor
                _paste_rotated_asset(
                    canvas,
                    _asset_path(COAST_CORNER_ASSET),
                    center_svg=(center["x"], center["y"]),
                    width_svg=coast_size,
                    height_svg=coast_size,
                    angle_deg=d2 * 60,
                    transform=transform,
                )
                continue

        for direction in land_dirs:
            coast_size = COAST_PIECE_SIZE * style.coast_size_factor
            _paste_rotated_asset(
                canvas,
                _asset_path(COAST_EDGE_ASSET),
                center_svg=(center["x"], center["y"]),
                width_svg=coast_size,
                height_svg=coast_size,
                angle_deg=direction * 60,
                transform=transform,
            )


def _render_land_tiles(
    canvas: Image.Image,
    render_state: dict[str, Any],
    transform: dict[str, float],
    style: RenderStyle,
) -> None:
    for placed in render_state.get("tiles", []):
        tile = placed["tile"]
        if tile["type"] == "PORT":
            continue

        center = _hex_to_pixel(placed["coordinate"])
        is_desert = tile["type"] == "DESERT"
        is_resource = tile["type"] == "RESOURCE_TILE"
        if not is_desert and not is_resource:
            continue

        if is_desert:
            asset = _asset_path("/assets/tiles/desert.svg")
        else:
            asset = _asset_path(f"/assets/tiles/{tile['resource'].lower()}.svg")
        _paste_asset(
            canvas,
            asset,
            x_svg=center["x"] - TILE_WIDTH / 2,
            y_svg=center["y"] - TILE_HEIGHT / 2,
            width_svg=TILE_WIDTH,
            height_svg=TILE_HEIGHT,
            transform=transform,
        )

        number = tile.get("number") if is_resource else None
        if number:
            _paste_asset(
                canvas,
                _asset_path(f"/assets/numbers/{number}.svg"),
                x_svg=center["x"] - NUMBER_TOKEN_SIZE / 2,
                y_svg=center["y"] - NUMBER_TOKEN_SIZE / 2 + NUMBER_TOKEN_Y_OFFSET,
                width_svg=NUMBER_TOKEN_SIZE,
                height_svg=NUMBER_TOKEN_SIZE,
                transform=transform,
            )


def _render_ports(
    canvas: Image.Image,
    render_state: dict[str, Any],
    transform: dict[str, float],
    node_positions: dict[int, dict[str, float]],
    style: RenderStyle,
) -> None:
    for placed in render_state.get("tiles", []):
        tile = placed["tile"]
        if tile["type"] != "PORT":
            continue

        port_nodes = tile.get("port_nodes")
        if not port_nodes:
            continue
        pos1 = node_positions.get(port_nodes[0])
        pos2 = node_positions.get(port_nodes[1])
        if not pos1 or not pos2:
            continue

        center = _hex_to_pixel(placed["coordinate"])
        _render_dock(canvas, center, pos1, transform, style)
        _render_dock(canvas, center, pos2, transform, style)

        resource = tile.get("resource")
        asset_name = "generic" if resource is None else str(resource).lower()
        _paste_asset(
            canvas,
            _asset_path(f"/assets/tiles/port_{asset_name}.svg"),
            x_svg=center["x"] - PORT_SHIP_SIZE / 2 + PORT_SHIP_X_OFFSET,
            y_svg=center["y"] - PORT_SHIP_SIZE / 2 + PORT_SHIP_Y_OFFSET,
            width_svg=PORT_SHIP_SIZE,
            height_svg=PORT_SHIP_SIZE,
            transform=transform,
        )


def _render_dock(
    canvas: Image.Image,
    port_center: dict[str, float],
    node_pos: dict[str, float],
    transform: dict[str, float],
    style: RenderStyle,
) -> None:
    x1, y1 = port_center["x"], port_center["y"]
    x2, y2 = node_pos["x"], node_pos["y"]
    dx = x2 - x1
    dy = y2 - y1
    raw_len = math.sqrt(dx * dx + dy * dy)
    if raw_len <= 0:
        return
    nx = dx / raw_len
    ny = dy / raw_len
    extend = raw_len * style.dock_extend_factor
    ship_clearance = PORT_SHIP_SIZE * style.dock_ship_clearance_factor
    start_x = x1 + nx * (ship_clearance + style.dock_port_gap)
    start_y = y1 + ny * (ship_clearance + style.dock_port_gap)
    end_x = x2 + nx * extend - nx * style.dock_sand_gap
    end_y = y2 + ny * extend - ny * style.dock_sand_gap
    trim_dx = end_x - start_x
    trim_dy = end_y - start_y
    length = math.sqrt(trim_dx * trim_dx + trim_dy * trim_dy)
    if length <= 0:
        return
    angle = math.degrees(math.atan2(trim_dy, trim_dx)) - 90
    mx = (start_x + end_x) / 2
    my = (start_y + end_y) / 2
    _paste_rotated_asset(
        canvas,
        _asset_path(DOCK_ASSET),
        center_svg=(mx, my),
        width_svg=style.dock_width,
        height_svg=length,
        angle_deg=angle,
        transform=transform,
    )


def _render_roads(
    canvas: Image.Image,
    render_state: dict[str, Any],
    transform: dict[str, float],
    node_positions: dict[int, dict[str, float]],
    style: RenderStyle,
) -> None:
    for edge in render_state.get("edges", []):
        color = edge.get("color")
        if not color:
            continue
        node1_id, node2_id = edge["id"]
        pos1 = node_positions.get(node1_id)
        pos2 = node_positions.get(node2_id)
        if not pos1 or not pos2:
            continue

        dx = pos2["x"] - pos1["x"]
        dy = pos2["y"] - pos1["y"]
        length = math.sqrt(dx * dx + dy * dy)
        mx = (pos1["x"] + pos2["x"]) / 2
        my = (pos1["y"] + pos2["y"]) / 2
        angle = math.degrees(math.atan2(dy, dx)) + 90
        road_len = length * style.road_length_factor
        _paste_rotated_asset(
            canvas,
            _asset_path(f"/assets/pieces/road_{str(color).lower()}.svg"),
            center_svg=(mx, my),
            width_svg=HEX_SIZE * style.road_width_factor,
            height_svg=road_len,
            angle_deg=angle,
            transform=transform,
        )


def _render_buildings(
    canvas: Image.Image,
    render_state: dict[str, Any],
    transform: dict[str, float],
    node_positions: dict[int, dict[str, float]],
    style: RenderStyle,
) -> None:
    for node in render_state.get("nodes", {}).values():
        building = node.get("building")
        color = node.get("color")
        if not building or not color:
            continue
        pos = node_positions.get(node["id"])
        if not pos:
            continue

        if building == "SETTLEMENT":
            size = HEX_SIZE * style.settlement_size_factor
            _paste_asset(
                canvas,
                _asset_path(f"/assets/pieces/settlement_{str(color).lower()}.svg"),
                x_svg=pos["x"] - size / 2 + style.settlement_x_offset,
                y_svg=pos["y"] - size * style.settlement_y_factor,
                width_svg=size,
                height_svg=size,
                transform=transform,
            )
        elif building == "CITY":
            size = HEX_SIZE * style.city_size_factor
            width = size * style.city_width_factor
            _paste_asset(
                canvas,
                _asset_path(f"/assets/pieces/city_{str(color).lower()}.svg"),
                x_svg=pos["x"] - width / 2 + style.city_x_offset,
                y_svg=pos["y"] - size * style.city_y_factor,
                width_svg=width,
                height_svg=size,
                transform=transform,
            )


def _render_robber(
    canvas: Image.Image,
    render_state: dict[str, Any],
    transform: dict[str, float],
    style: RenderStyle,
) -> None:
    robber_coord = render_state.get("robber_coordinate")
    if not robber_coord:
        return

    for placed in render_state.get("tiles", []):
        tile = placed["tile"]
        if tile["type"] == "PORT":
            continue
        if list(robber_coord) != list(placed["coordinate"]):
            continue

        center = _hex_to_pixel(placed["coordinate"])
        robber_size = HEX_SIZE * style.robber_size_factor
        _paste_asset(
            canvas,
            _asset_path("/assets/pieces/robber.svg"),
            x_svg=center["x"] - robber_size / 2 - HEX_SIZE * 0.52,
            y_svg=center["y"] - robber_size / 2 - HEX_SIZE * 0.15,
            width_svg=robber_size,
            height_svg=robber_size,
            transform=transform,
        )
        return


def _asset_path(web_path: str) -> Path:
    if not web_path.startswith("/assets/"):
        raise ValueError(f"expected frontend asset path, got {web_path}")
    return ASSET_ROOT / web_path.removeprefix("/assets/")


def _paste_asset(
    canvas: Image.Image,
    path: Path,
    *,
    x_svg: float,
    y_svg: float,
    width_svg: float,
    height_svg: float,
    transform: dict[str, float],
) -> list[int]:
    x, y = _svg_point_to_px(x_svg, y_svg, transform)
    width = max(1, int(round(width_svg * transform["scale"])))
    height = max(1, int(round(height_svg * transform["scale"])))
    asset = _raster_asset(path, width, height)
    canvas.alpha_composite(asset, (x, y))
    return [x, y, x + width, y + height]


def _paste_rotated_asset(
    canvas: Image.Image,
    path: Path,
    *,
    center_svg: tuple[float, float],
    width_svg: float,
    height_svg: float,
    angle_deg: float,
    transform: dict[str, float],
) -> list[int]:
    center_x, center_y = _svg_point_to_px(center_svg[0], center_svg[1], transform)
    width = max(1, int(round(width_svg * transform["scale"])))
    height = max(1, int(round(height_svg * transform["scale"])))
    asset = _raster_asset(path, width, height)
    rotated = asset.rotate(-angle_deg, expand=True, resample=Image.Resampling.BICUBIC)
    x = int(round(center_x - rotated.width / 2))
    y = int(round(center_y - rotated.height / 2))
    canvas.alpha_composite(rotated, (x, y))
    return [x, y, x + rotated.width, y + rotated.height]


def _svg_point_to_px(x: float, y: float, transform: dict[str, float]) -> tuple[int, int]:
    px = (x - transform["min_x"]) * transform["scale"] + transform["offset_x"]
    py = (y - transform["min_y"]) * transform["scale"] + transform["offset_y"]
    return int(round(px)), int(round(py))


@lru_cache(maxsize=512)
def _raster_asset(path: Path, width: int, height: int) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".svg":
        return _raster_svg(path, width, height)
    return Image.open(path).convert("RGBA").resize((width, height), Image.Resampling.LANCZOS)


def _raster_svg(path: Path, width: int, height: int) -> Image.Image:
    binary = shutil.which("rsvg-convert")
    if binary is None:
        raise RuntimeError("rsvg-convert is required for Python Catan board rendering")
    result = subprocess.run(
        [binary, "-f", "png", "-w", str(width), "-h", str(height), str(path)],
        check=True,
        capture_output=True,
    )
    return Image.open(io.BytesIO(result.stdout)).convert("RGBA")


def _draw_hex_fallback(
    canvas: Image.Image,
    center: dict[str, float],
    resource: str | None,
    transform: dict[str, float],
) -> None:
    draw = ImageDraw.Draw(canvas, "RGBA")
    points = []
    for index in range(6):
        angle = math.pi / 3 * index - math.pi / 6
        x = center["x"] + HEX_SIZE * math.cos(angle)
        y = center["y"] + HEX_SIZE * math.sin(angle)
        points.append(_svg_point_to_px(x, y, transform))
    draw.polygon(points, fill=RESOURCE_COLORS.get(str(resource), (193, 154, 107, 230)))
