"""Map inventories and insertion-ordered topology definitions."""

from cle.game_engine.models.coordinate_system import Direction
from cle.game_engine.models.enums import BRICK, ORE, SHEEP, WHEAT, WOOD
from cle.game_engine.models.map_types import LandTile, MapTemplate, Port, Water

# Small 7-tile map, no ports.
MINI_MAP_TEMPLATE = MapTemplate(
    [3, 4, 5, 6, 8, 9, 10],
    [],
    [WOOD, None, BRICK, SHEEP, WHEAT, WHEAT, ORE],
    {
        # center
        (0, 0, 0): LandTile,
        # first layer
        (1, -1, 0): LandTile,
        (0, -1, 1): LandTile,
        (-1, 0, 1): LandTile,
        (-1, 1, 0): LandTile,
        (0, 1, -1): LandTile,
        (1, 0, -1): LandTile,
        # second layer
        (2, -2, 0): Water,
        (1, -2, 1): Water,
        (0, -2, 2): Water,
        (-1, -1, 2): Water,
        (-2, 0, 2): Water,
        (-2, 1, 1): Water,
        (-2, 2, 0): Water,
        (-1, 2, -1): Water,
        (0, 2, -2): Water,
        (1, 1, -2): Water,
        (2, 0, -2): Water,
        (2, -1, -1): Water,
    },
)

# Standard 19-hex inventory: 3 hills (brick), 4 forests (wood),
# 4 pastures (sheep), 4 fields (wheat), 3 mountains (ore), 1 desert.
# Random layouts shuffle this inventory; they never sample terrain with replacement.
BASE_MAP_TEMPLATE = MapTemplate(
    [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12],
    [
        # These are 2:1 ports
        WOOD,
        BRICK,
        SHEEP,
        WHEAT,
        ORE,
        # These represet 3:1 ports
        None,
        None,
        None,
        None,
    ],
    [
        # Four wood tiles
        WOOD,
        WOOD,
        WOOD,
        WOOD,
        # Three brick tiles
        BRICK,
        BRICK,
        BRICK,
        # Four sheep tiles
        SHEEP,
        SHEEP,
        SHEEP,
        SHEEP,
        # Four wheat tiles
        WHEAT,
        WHEAT,
        WHEAT,
        WHEAT,
        # Three ore tiles
        ORE,
        ORE,
        ORE,
        # One desert
        None,
    ],
    # 3 layers, where last layer is water
    {
        # center
        (0, 0, 0): LandTile,
        # first layer
        (1, -1, 0): LandTile,
        (0, -1, 1): LandTile,
        (-1, 0, 1): LandTile,
        (-1, 1, 0): LandTile,
        (0, 1, -1): LandTile,
        (1, 0, -1): LandTile,
        # second layer
        (2, -2, 0): LandTile,
        (1, -2, 1): LandTile,
        (0, -2, 2): LandTile,
        (-1, -1, 2): LandTile,
        (-2, 0, 2): LandTile,
        (-2, 1, 1): LandTile,
        (-2, 2, 0): LandTile,
        (-1, 2, -1): LandTile,
        (0, 2, -2): LandTile,
        (1, 1, -2): LandTile,
        (2, 0, -2): LandTile,
        (2, -1, -1): LandTile,
        # third (water) layer
        (3, -3, 0): (Port, Direction.WEST),
        (2, -3, 1): Water,
        (1, -3, 2): (Port, Direction.NORTHWEST),
        (0, -3, 3): Water,
        (-1, -2, 3): (Port, Direction.NORTHWEST),
        (-2, -1, 3): Water,
        (-3, 0, 3): (Port, Direction.NORTHEAST),
        (-3, 1, 2): Water,
        (-3, 2, 1): (Port, Direction.EAST),
        (-3, 3, 0): Water,
        (-2, 3, -1): (Port, Direction.EAST),
        (-1, 3, -2): Water,
        (0, 3, -3): (Port, Direction.SOUTHEAST),
        (1, 2, -3): Water,
        (2, 1, -3): (Port, Direction.SOUTHWEST),
        (3, 0, -3): Water,
        (3, -1, -2): (Port, Direction.SOUTHWEST),
        (3, -2, -1): Water,
    },
)
