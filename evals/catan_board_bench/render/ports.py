"""Port ship placement and the dock planks that join ports to their nodes."""

from __future__ import annotations

import math

from PIL import Image

from evals.catan_board_bench.annotations import (
    PORT_SHIP_SIZE,
    PORT_SHIP_X_OFFSET,
    PORT_SHIP_Y_OFFSET,
    RenderState,
    _hex_to_pixel,
)
from evals.catan_board_bench.render.raster import (
    _asset_path,
    _paste_asset,
    _paste_rotated_asset,
)
from evals.catan_board_bench.render.style import DOCK_ASSET, RenderStyle


def _render_ports(
    canvas: Image.Image,
    render_state: RenderState,
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

