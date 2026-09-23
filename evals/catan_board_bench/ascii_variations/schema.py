"""Stable schemas, record order, and board-coordinate conventions."""

from cle.game_engine.models.coordinate_system import Direction

# The frozen scorer resolves its ``JsonDict`` annotation name through this module.
from evals.json_types import JsonDict as JsonDict

FACT_SCHEMA = "catan_full_public_graph/v1"
DATASET_SCHEMA = "catan_ascii_variation_probe/v1"
ASCII_VARIANTS = (
    "flat_sorted",
    "flat_shuffled",
    "sectioned",
    "tile_rows",
    "local_blocks",
    "topology_diagram",
)
SCREEN_DIRECTIONS = {
    Direction.WEST: "LEFT",
    Direction.EAST: "RIGHT",
    Direction.NORTHWEST: "UP-LEFT",
    Direction.NORTHEAST: "UP-RIGHT",
    Direction.SOUTHWEST: "DOWN-LEFT",
    Direction.SOUTHEAST: "DOWN-RIGHT",
}
DIRECTION_ORDER = (
    Direction.WEST,
    Direction.EAST,
    Direction.NORTHWEST,
    Direction.NORTHEAST,
    Direction.SOUTHWEST,
    Direction.SOUTHEAST,
)
CORNER_ORDER = (
    "NORTH",
    "NORTHEAST",
    "SOUTHEAST",
    "SOUTH",
    "SOUTHWEST",
    "NORTHWEST",
)
SIDE_ORDER = (
    "EAST",
    "SOUTHEAST",
    "SOUTHWEST",
    "WEST",
    "NORTHWEST",
    "NORTHEAST",
)
RECORD_KINDS = ("T", "N", "E", "P")
