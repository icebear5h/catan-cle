"""All Colonist <-> Engine mapping constants."""

from typing import Final

from cle.game_engine.models.coordinate_system import Coordinate
from cle.game_engine.models.enums import FastResource
from cle.game_engine.models.map import BRICK, ORE, SHEEP, WHEAT, WOOD

# Colonist tile type to engine resource mapping
# VERIFIED: Tile type 1 produces resource 1, and resource 1 = WOOD (roads use 1+2)
# So: 1=WOOD, 2=BRICK, 3=SHEEP, 4=WHEAT, 5=ORE
COLONIST_RESOURCE: Final[dict[int, FastResource | None]] = {
    0: None,  # desert
    1: WOOD,   # Colonist tile type 1 = Wood (verified: produces resource 1 = wood)
    2: BRICK,
    3: SHEEP,
    4: WHEAT,  # Colonist tile type 4 = Wheat (verified: produces resource 4)
    5: ORE,
}

# Colonist port type to engine resource
# Verified from actual Colonist UI
COLONIST_PORT_RESOURCE: Final[dict[int, FastResource | None]] = {
    1: None,   # 3:1 generic port
    2: WOOD,
    3: BRICK,
    4: SHEEP,
    5: WHEAT,
    6: ORE,
}

# Engine port water hex positions
ENGINE_PORT_COORDS: Final[list[Coordinate]] = [
    (3, -3, 0),   # 0
    (1, -3, 2),   # 1
    (-1, -2, 3),  # 2
    (-3, 0, 3),   # 3
    (-3, 2, 1),   # 4
    (-2, 3, -1),  # 5
    (0, 3, -3),   # 6
    (2, 1, -3),   # 7
    (3, -1, -2),  # 8
]
ENGINE_PORT_MAP: Final[dict[Coordinate, int]] = {coord: idx for idx, coord in enumerate(ENGINE_PORT_COORDS)}

# Hex direction offsets
HEX_DIRECTIONS: Final[dict[str, Coordinate]] = {
    'EAST': (1, 0, -1),
    'WEST': (-1, 0, 1),
    'NORTHEAST': (1, -1, 0),
    'NORTHWEST': (0, -1, 1),
    'SOUTHEAST': (0, 1, -1),
    'SOUTHWEST': (-1, 1, 0),
}

# Resource emojis
RESOURCE_EMOJIS: Final[dict[str, str]] = {
    "WOOD": "\U0001FAB5",
    "BRICK": "\U0001F9F1",
    "SHEEP": "\U0001F411",
    "WHEAT": "\U0001F33E",
    "ORE": "\u26F0\uFE0F"
}

# Colonist resource ID to engine RESOURCES index
# VERIFIED: Road builds use resources [1, 2] which must be WOOD+BRICK
# So: 1=WOOD, 2=BRICK, 3=SHEEP, 4=WHEAT, 5=ORE (README was wrong about 1/4 swap)
# Engine RESOURCES order: WOOD=0, BRICK=1, SHEEP=2, WHEAT=3, ORE=4
COLONIST_RES_TO_ENGINE_IDX: Final[dict[int, int]] = {
    1: 0,  # WOOD -> index 0 (verified: roads use resource 1+2 = wood+brick)
    2: 1,  # brick -> index 1
    3: 2,  # sheep -> index 2
    4: 3,  # WHEAT -> index 3 (verified: tile type 4 produces resource 4)
    5: 4,  # ore -> index 4
    # 9 = "any" resource (flexible trade) - not mappable to specific engine resource
}

# Colonist dev card ID to engine type
# Verified from Colonist UI observation
COLONIST_DEV_CARD: Final[dict[int, str]] = {
    11: "KNIGHT",
    12: "VICTORY_POINT",
    13: "MONOPOLY",
    14: "ROAD_BUILDING",
    15: "YEAR_OF_PLENTY",
}

# Colonist resource ID to engine Resource string
# VERIFIED: 1=WOOD (used in road builds), 4=WHEAT (from tile type 4)
COLONIST_RES_TO_ENGINE: Final[dict[int, FastResource]] = {
    1: "WOOD",
    2: "BRICK",
    3: "SHEEP",
    4: "WHEAT",
    5: "ORE",
}

# Engine resource strings (for lookups)
ENGINE_RESOURCES: Final[list[FastResource]] = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]

# Colonist player color IDs to names (matching COLONIST_COLOR_NAMES)
COLONIST_PLAYER_COLORS: Final[dict[int, str]] = {
    1: "RED",
    2: "BLUE",
    3: "ORANGE",
    4: "GREEN",
    5: "BLACK",
    6: "BRONZE",
    7: "SILVER",
    8: "GOLD",
    9: "WHITE",
    10: "PINK",
    11: "MYSTIC_BLUE",
}
