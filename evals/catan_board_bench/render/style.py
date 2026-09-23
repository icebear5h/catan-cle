"""Geometry constants, tunable render style, and shared palettes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from evals.catan_board_bench.annotations import HEX_SIZE, TILE_HEIGHT

PROJECT_ROOT = Path(__file__).resolve().parents[3]
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
