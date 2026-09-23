from enum import Enum
from typing import TypeAlias

Coordinate: TypeAlias = tuple[int, int, int]


# We'll be using Cube coordinates in https://math.stackexchange.com/questions/2254655/hexagon-grid-coordinate-system
class Direction(Enum):
    EAST = "EAST"
    SOUTHEAST = "SOUTHEAST"
    SOUTHWEST = "SOUTHWEST"
    WEST = "WEST"
    NORTHWEST = "NORTHWEST"
    NORTHEAST = "NORTHEAST"


UNIT_VECTORS: dict[Direction, Coordinate] = {
    # X-axis
    Direction.NORTHEAST: (1, 0, -1),
    Direction.SOUTHWEST: (-1, 0, 1),
    # Y-axis
    Direction.NORTHWEST: (0, 1, -1),
    Direction.SOUTHEAST: (0, -1, 1),
    # Z-axis
    Direction.EAST: (1, -1, 0),
    Direction.WEST: (-1, 1, 0),
}


def add(acoord: Coordinate, bcoord: Coordinate) -> Coordinate:
    (x, y, z) = acoord
    (u, v, w) = bcoord
    return (x + u, y + v, z + w)


def num_tiles_for(layer: int) -> int:
    """Including inner-layer tiles"""
    if layer == 0:
        return 1

    return 6 * layer + num_tiles_for(layer - 1)


def generate_coordinate_system(num_layers: int) -> set[Coordinate]:
    """
    Generates a set of coordinates by expanding outward from a center tile on
    (0,0,0) with the given number of layers (as in an onion :)). Follows BFS.
    """
    num_tiles = num_tiles_for(num_layers)

    agenda: list[Coordinate] = [(0, 0, 0)]
    visited: set[Coordinate] = set()
    while len(visited) < num_tiles:
        node = agenda.pop(0)
        visited.add(node)

        neighbors = [add(node, UNIT_VECTORS[d]) for d in Direction]
        new_neighbors = filter(
            lambda x: x not in visited and x not in agenda, neighbors
        )
        agenda.extend(new_neighbors)
    return visited


def cube_to_axial(cube: Coordinate) -> tuple[int, int]:
    """Convert cube coordinates to axial (pointy-top convention).

    For pointy-top hexes: q = x, r = z (standard conversion)
    """
    q = cube[0]
    r = cube[2]
    return (q, r)


def cube_to_offset(cube: Coordinate) -> tuple[int, int]:
    """Convert cube to offset coordinates (pointy-top, odd-r convention)."""
    col = cube[0] + (cube[2] - (cube[2] & 1)) // 2
    row = cube[2]
    return (col, row)


def offset_to_cube(offset: tuple[int, int]) -> Coordinate:
    """Convert offset coordinates to cube (pointy-top, odd-r convention)."""
    x = offset[0] - (offset[1] - (offset[1] & 1)) // 2
    z = offset[1]
    y = -x - z
    return (x, y, z)
