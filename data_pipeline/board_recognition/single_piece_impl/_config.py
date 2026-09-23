"""Schemas, colour tables, sprite settings, and row budgets."""

from __future__ import annotations

import re

from data_pipeline.json_types import JsonDict

EXPORT_SCHEMA = "catan_single_piece_localization/v1"
ROW_SCHEMA = "catan_single_piece_localization_row/v1"
DEFAULT_OUTPUT_NAME = "spatial_localization_v2"
COLORS = (
    "RED",
    "BLUE",
    "ORANGE",
    "WHITE",
    "BLACK",
    "GREEN",
    "BRONZE",
    "SILVER",
    "GOLD",
    "PINK",
    "MYSTIC_BLUE",
)
NOVEL_COLOR_FRACTION = 0.2
# Hue intervals (degrees) with at least 25 degrees of clearance from every
# saturated stop in the shipped piece sprites.
NOVEL_HUE_INTERVALS = ((70, 105), (145, 190), (250, 275), (295, 325))
NOVEL_SPRITE_BASE = "red"
SPRITE_PIECES = ("settlement", "city", "road")
HEX_COLOR_RE = re.compile(r"#([0-9a-fA-F]{6})")
NODE_PIECES = ("SETTLEMENT", "CITY")
EDGE_PIECE = "ROAD"
TRAIN_IMAGES_PER_BOARD_PER_ENTITY = 70
EVAL_IMAGES_PER_BOARD_PER_ENTITY = 30
FORWARD_QUERY = {"node": "building?", "edge": "road?"}
LOCATION_NOUN = {"node": "node", "edge": "edge"}
TILE_ROWS_PER_IMAGE = 3
NAMED_ROWS_PER_IMAGE = 3
NEGATIVE_KINDS = ("adjacent", "far")
DEFAULT_NEGATIVES = {"adjacent": 1, "far": 1}
# Coastal nodes have only five same-type locations within two hops; three
# hops gives every placement at least seven near candidates when a heavier
# mix is requested. The default single near negative always touches the piece.
NEAR_MAX_HOPS = 3

__all__ = ["JsonDict"]
