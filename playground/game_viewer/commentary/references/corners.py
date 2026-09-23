"""The 54-corner index one randomized board layout produces."""

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from cle.game_engine.models.map import CatanMap

__all__ = [
    "CornerFact",
    "TileFact",
    "build_corner_index",
    "load_colonist_corner_mapping",
]

_CORNER_MAP_PATH = Path(__file__).resolve().parents[2] / "corner_to_node_map.json"
_NODE_OFFSETS = {
    "NORTH": (0.0, -1.0),
    "NORTHEAST": (math.sqrt(3) / 2, -0.5),
    "SOUTHEAST": (math.sqrt(3) / 2, 0.5),
    "SOUTH": (0.0, 1.0),
    "SOUTHWEST": (-math.sqrt(3) / 2, 0.5),
    "NORTHWEST": (-math.sqrt(3) / 2, -0.5),
}


@dataclass(frozen=True)
class TileFact:
    tile_id: int
    number: int | None
    resource: str | None
    coordinate: tuple[int, int, int]


@dataclass(frozen=True)
class CornerFact:
    colonist_corner_id: int
    engine_node_id: int
    tiles: tuple[TileFact, ...]
    numbers: tuple[int, ...]
    ports: tuple[str, ...]
    coast: bool
    cube_coordinate: tuple[int, ...]
    screen_coordinate: tuple[float, float]


@lru_cache(maxsize=1)
def load_colonist_corner_mapping() -> dict[int, int]:
    """Return Colonist corner ID -> engine node ID."""
    raw = json.loads(_CORNER_MAP_PATH.read_text(encoding="utf-8"))
    return {int(key.removeprefix("_")): int(value) for key, value in raw.items()}


def _node_screen_coordinates(catan_map: CatanMap) -> dict[int, tuple[float, float]]:
    positions: dict[int, tuple[float, float]] = {}
    for coordinate, tile in catan_map.tiles.items():
        if not hasattr(tile, "nodes"):
            continue
        q = coordinate[0]
        r = coordinate[2]
        tile_x = math.sqrt(3) * (q + r / 2)
        tile_y = 1.5 * r
        for node_ref, node_id in tile.nodes.items():
            if node_id in positions:
                continue
            offset_x, offset_y = _NODE_OFFSETS[node_ref.value]
            positions[node_id] = (tile_x + offset_x, tile_y + offset_y)
    return positions


def _node_cube_coordinates(catan_map: CatanMap) -> dict[int, tuple[int, ...]]:
    coordinates: defaultdict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for coordinate, tile in catan_map.tiles.items():
        if hasattr(tile, "nodes"):
            for node_id in tile.nodes.values():
                coordinates[node_id].append(coordinate)
    return {
        node_id: tuple(sum(coord[axis] for coord in values) for axis in range(3))
        for node_id, values in coordinates.items()
    }


def _port_labels(catan_map: CatanMap, node_id: int) -> tuple[str, ...]:
    labels: list[str] = []
    for resource, nodes in catan_map.port_nodes.items():
        if node_id in nodes:
            labels.append("3:1" if resource is None else f"2:1 {resource}")
    return tuple(sorted(labels))


def build_corner_index(catan_map: CatanMap) -> tuple[CornerFact, ...]:
    """Build the canonical 54-corner index for one randomized board layout."""
    colonist_to_engine = load_colonist_corner_mapping()
    engine_to_colonist = {
        engine_node: colonist_corner
        for colonist_corner, engine_node in colonist_to_engine.items()
    }
    screen_coordinates = _node_screen_coordinates(catan_map)
    cube_coordinates = _node_cube_coordinates(catan_map)
    corners: list[CornerFact] = []

    for engine_node_id in sorted(catan_map.land_nodes):
        tile_entries: list[TileFact] = []
        for tile in catan_map.adjacent_tiles[engine_node_id]:
            coordinate = next(
                coord for coord, candidate in catan_map.land_tiles.items() if candidate is tile
            )
            tile_entries.append(
                TileFact(
                    tile_id=tile.id,
                    number=tile.number,
                    resource=tile.resource,
                    coordinate=coordinate,
                )
            )
        tiles = tuple(sorted(tile_entries, key=lambda tile: tile.tile_id))
        numbers = tuple(sorted(tile.number for tile in tiles if tile.number is not None))
        corners.append(
            CornerFact(
                colonist_corner_id=engine_to_colonist[engine_node_id],
                engine_node_id=engine_node_id,
                tiles=tiles,
                numbers=numbers,
                ports=_port_labels(catan_map, engine_node_id),
                coast=len(tiles) < 3,
                cube_coordinate=cube_coordinates[engine_node_id],
                screen_coordinate=screen_coordinates[engine_node_id],
            )
        )

    return tuple(sorted(corners, key=lambda corner: corner.colonist_corner_id))
