"""Hex geometry constants shared with the frontend HexBoard component."""

from __future__ import annotations

import math

HEX_SIZE = 50.0
HEX_SPACING = 0.0
TILE_BLEED = 1.5
TILE_WIDTH = math.sqrt(3) * HEX_SIZE + TILE_BLEED
TILE_HEIGHT = 2 * HEX_SIZE + TILE_BLEED
NUMBER_TOKEN_SIZE = HEX_SIZE * 0.7
NUMBER_TOKEN_Y_OFFSET = HEX_SIZE * 0.30
PORT_SHIP_SIZE = HEX_SIZE * 1.05
PORT_SHIP_X_OFFSET = -PORT_SHIP_SIZE * 0.1
PORT_SHIP_Y_OFFSET = -PORT_SHIP_SIZE * 0.25
NODE_BOX_SIZE = 16.0
EDGE_BOX_PAD = 8.0
