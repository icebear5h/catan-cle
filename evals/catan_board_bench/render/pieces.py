"""Road, settlement, city, and robber layers of the board raster."""

from __future__ import annotations

import math

from PIL import Image

from evals.catan_board_bench.annotations import HEX_SIZE, RenderState, _hex_to_pixel
from evals.catan_board_bench.render.raster import (
    _asset_path,
    _paste_asset,
    _paste_rotated_asset,
)
from evals.catan_board_bench.render.style import RenderStyle


def _render_roads(
    canvas: Image.Image,
    render_state: RenderState,
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
    render_state: RenderState,
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
    render_state: RenderState,
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

