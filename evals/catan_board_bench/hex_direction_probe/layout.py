"""Probe geometry: screen-direction names, palettes, and cube-to-pixel layout."""

from __future__ import annotations

import math

from cle.game_engine.models.coordinate_system import Direction
from evals.catan_board_bench.paths import DATASETS_DIR

DEFAULT_OUTPUT_DIR = DATASETS_DIR / "hex_direction_probe"
IMAGE_SIZE = 512
LAYOUT_COUNT = 6
ANCHOR_LABEL = "X"
CANDIDATE_LABELS = ("1", "2", "3", "4", "5", "6")

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

_PALETTES = (
    ((29, 78, 216), (234, 88, 12), (22, 163, 74), (147, 51, 234), (202, 138, 4), (219, 39, 119)),
    ((3, 105, 161), (190, 24, 93), (101, 163, 13), (194, 65, 12), (124, 58, 237), (13, 148, 136)),
)
_OFFSETS = ((0, 0), (8, -5), (-7, 6), (5, 7), (-6, -7), (3, -2))


def cube_to_pixel(
    coordinate: tuple[int, int, int],
    *,
    center: tuple[float, float],
    radius: float,
) -> tuple[float, float]:
    """Project engine cube coordinates with the frontend pointy-top convention."""
    q = coordinate[0]
    r = coordinate[2]
    return (
        center[0] + math.sqrt(3) * radius * (q + r / 2),
        center[1] + 1.5 * radius * r,
    )


def label_assignment(layout_index: int) -> dict[Direction, str]:
    """Use a Latin-square shift so every label occupies every direction once."""
    return {
        direction: CANDIDATE_LABELS[(direction_index + layout_index) % 6]
        for direction_index, direction in enumerate(DIRECTION_ORDER)
    }

