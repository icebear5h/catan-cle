"""Python renderer for Catan public-board contracts.

This mirrors the production path in ``playground/frontend/src/components/HexBoard.tsx``:

- same Catan render-state shape via ``contract_to_render_state``
- same geometry constants and coordinate transform as ``annotations.py``
- same frontend SVG/PNG assets from ``playground/frontend/public/assets``

The goal is bulk image generation without Playwright while keeping Playwright
available as the calibration oracle for spot checks. The drawing layers live in
sibling modules; every pre-split name stays importable from this package.
"""

from __future__ import annotations

# Names the pre-split module also exposed, kept importable at this path.
import io as io
import math as math
import shutil as shutil
import subprocess as subprocess
from dataclasses import dataclass as dataclass
from functools import lru_cache as lru_cache
from pathlib import Path as Path
from typing import Any as Any

from PIL import Image
from PIL import ImageDraw as ImageDraw

from evals.catan_board_bench.annotations import HEX_SIZE as HEX_SIZE
from evals.catan_board_bench.annotations import NUMBER_TOKEN_SIZE as NUMBER_TOKEN_SIZE
from evals.catan_board_bench.annotations import NUMBER_TOKEN_Y_OFFSET as NUMBER_TOKEN_Y_OFFSET
from evals.catan_board_bench.annotations import PORT_SHIP_SIZE as PORT_SHIP_SIZE
from evals.catan_board_bench.annotations import PORT_SHIP_X_OFFSET as PORT_SHIP_X_OFFSET
from evals.catan_board_bench.annotations import PORT_SHIP_Y_OFFSET as PORT_SHIP_Y_OFFSET
from evals.catan_board_bench.annotations import TILE_HEIGHT as TILE_HEIGHT
from evals.catan_board_bench.annotations import TILE_WIDTH as TILE_WIDTH
from evals.catan_board_bench.annotations import (
    RenderState,
    _node_positions,
    _pixel_transform,
    contract_to_render_state,
)
from evals.catan_board_bench.annotations import _hex_to_pixel as _hex_to_pixel
from evals.catan_board_bench.annotations import _view_box as _view_box
from evals.catan_board_bench.render.pieces import _render_buildings as _render_buildings
from evals.catan_board_bench.render.pieces import _render_roads as _render_roads
from evals.catan_board_bench.render.pieces import _render_robber as _render_robber
from evals.catan_board_bench.render.ports import _render_dock as _render_dock
from evals.catan_board_bench.render.ports import _render_ports as _render_ports
from evals.catan_board_bench.render.raster import _asset_path as _asset_path
from evals.catan_board_bench.render.raster import _draw_hex_fallback as _draw_hex_fallback
from evals.catan_board_bench.render.raster import _paste_asset as _paste_asset
from evals.catan_board_bench.render.raster import _paste_rotated_asset as _paste_rotated_asset
from evals.catan_board_bench.render.raster import _raster_asset as _raster_asset
from evals.catan_board_bench.render.raster import _raster_svg as _raster_svg
from evals.catan_board_bench.render.raster import _svg_point_to_px as _svg_point_to_px
from evals.catan_board_bench.render.style import ASSET_ROOT as ASSET_ROOT
from evals.catan_board_bench.render.style import CITY_SIZE as CITY_SIZE
from evals.catan_board_bench.render.style import CITY_SIZE_FACTOR as CITY_SIZE_FACTOR
from evals.catan_board_bench.render.style import CITY_WIDTH_FACTOR as CITY_WIDTH_FACTOR
from evals.catan_board_bench.render.style import CITY_X_OFFSET as CITY_X_OFFSET
from evals.catan_board_bench.render.style import CITY_Y_FACTOR as CITY_Y_FACTOR
from evals.catan_board_bench.render.style import COAST_CORNER_ASSET as COAST_CORNER_ASSET
from evals.catan_board_bench.render.style import COAST_EDGE_ASSET as COAST_EDGE_ASSET
from evals.catan_board_bench.render.style import COAST_PIECE_SIZE as COAST_PIECE_SIZE
from evals.catan_board_bench.render.style import COAST_SIZE_FACTOR as COAST_SIZE_FACTOR
from evals.catan_board_bench.render.style import CUBE_DIRS as CUBE_DIRS
from evals.catan_board_bench.render.style import DEFAULT_RENDER_STYLE as DEFAULT_RENDER_STYLE
from evals.catan_board_bench.render.style import DOCK_ASSET as DOCK_ASSET
from evals.catan_board_bench.render.style import DOCK_EXTEND_FACTOR as DOCK_EXTEND_FACTOR
from evals.catan_board_bench.render.style import DOCK_PORT_GAP as DOCK_PORT_GAP
from evals.catan_board_bench.render.style import DOCK_SAND_GAP as DOCK_SAND_GAP
from evals.catan_board_bench.render.style import (
    DOCK_SHIP_CLEARANCE_FACTOR as DOCK_SHIP_CLEARANCE_FACTOR,
)
from evals.catan_board_bench.render.style import DOCK_WIDTH as DOCK_WIDTH
from evals.catan_board_bench.render.style import PROJECT_ROOT as PROJECT_ROOT
from evals.catan_board_bench.render.style import RESOURCE_COLORS as RESOURCE_COLORS
from evals.catan_board_bench.render.style import ROAD_LENGTH_FACTOR as ROAD_LENGTH_FACTOR
from evals.catan_board_bench.render.style import ROAD_WIDTH as ROAD_WIDTH
from evals.catan_board_bench.render.style import ROBBER_SIZE as ROBBER_SIZE
from evals.catan_board_bench.render.style import ROBBER_SIZE_FACTOR as ROBBER_SIZE_FACTOR
from evals.catan_board_bench.render.style import SETTLEMENT_SIZE as SETTLEMENT_SIZE
from evals.catan_board_bench.render.style import (
    SETTLEMENT_SIZE_FACTOR as SETTLEMENT_SIZE_FACTOR,
)
from evals.catan_board_bench.render.style import SETTLEMENT_X_OFFSET as SETTLEMENT_X_OFFSET
from evals.catan_board_bench.render.style import SETTLEMENT_Y_FACTOR as SETTLEMENT_Y_FACTOR
from evals.catan_board_bench.render.style import VIEW_PADDING_FACTOR as VIEW_PADDING_FACTOR
from evals.catan_board_bench.render.style import WATER_RGB as WATER_RGB
from evals.catan_board_bench.render.style import RenderStyle as RenderStyle
from evals.catan_board_bench.render.terrain import _render_coast as _render_coast
from evals.catan_board_bench.render.terrain import _render_land_tiles as _render_land_tiles
from evals.catan_board_bench.render.terrain import (
    _view_box_with_padding as _view_box_with_padding,
)
from evals.json_types import JsonDict


def render_contract_image(
    contract: JsonDict,
    *,
    image_size: int = 512,
    style: RenderStyle | None = None,
) -> Image.Image:
    """Render a public-board contract into a square RGB board image."""

    return render_state_image(
        contract_to_render_state(contract), image_size=image_size, style=style
    )


def render_state_image(
    render_state: RenderState,
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
