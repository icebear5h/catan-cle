"""Server-side board renderer: engine State -> PIL Image -> PNG.

Renders Catan board with colored hex tiles, number tokens with pip dots,
roads, settlements, cities, robber, ports, and player color legend.

Dual purpose: VLM benchmark now, training data pipeline later.
"""

from .geometry import (
    NODE_OFFSETS,
    compute_node_positions,
    cube_to_pixel,
    hex_to_pixel,
    hexagon_vertices,
)
from .palette import (
    OCEAN_COLOR,
    PIPS,
    PLAYER_COLORS,
    PLAYER_OUTLINE,
    RESOURCE_COLORS,
    RESOURCE_LABELS,
    ROBBER_COLOR,
    TOKEN_BG,
    TOKEN_BORDER,
)
from .renderer import CatanBoardRenderer
from .shapes import Font, draw_city, draw_legend, draw_settlement, load_fonts

__all__ = [
    "NODE_OFFSETS",
    "OCEAN_COLOR",
    "PIPS",
    "PLAYER_COLORS",
    "PLAYER_OUTLINE",
    "RESOURCE_COLORS",
    "RESOURCE_LABELS",
    "ROBBER_COLOR",
    "TOKEN_BG",
    "TOKEN_BORDER",
    "CatanBoardRenderer",
    "Font",
    "compute_node_positions",
    "cube_to_pixel",
    "draw_city",
    "draw_legend",
    "draw_settlement",
    "hex_to_pixel",
    "hexagon_vertices",
    "load_fonts",
]
