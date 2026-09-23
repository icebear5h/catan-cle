"""Shared map records and contracts, independent of templates and generation."""

from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Protocol, TypeAlias, TypeVar

from cle.game_engine.models.coordinate_system import Coordinate as Coordinate
from cle.game_engine.models.coordinate_system import Direction
from cle.game_engine.models.enums import EdgeRef, FastResource, NodeRef

NUM_NODES = 54
NUM_EDGES = 72
NUM_TILES = 19

EdgeId: TypeAlias = tuple[int, int]
NodeId: TypeAlias = int
# Counter's stubs require integer counts, but production contains probabilities.
Production: TypeAlias = MutableMapping[FastResource, float]
_T = TypeVar("_T")


class MapRNG(Protocol):
    """Sampling shared by random.Random instances and the global random module."""

    def sample(self, population: Sequence[_T], k: int) -> list[_T]: ...


@dataclass
class LandTile:
    # Keep the historical pickle path without importing the map facade.
    __module__ = "cle.game_engine.models.map"

    id: int
    resource: FastResource | None  # None means desert tile
    number: int | None  # None if desert
    nodes: dict[NodeRef, NodeId]  # node_ref => node_id
    edges: dict[EdgeRef, EdgeId]  # edge_ref => edge

    # The id is unique among the tiles, so we can use it as the hash.
    def __hash__(self) -> int:
        return self.id


@dataclass
class Port:
    __module__ = "cle.game_engine.models.map"

    id: int
    resource: FastResource | None  # None means desert tile
    direction: Direction
    nodes: dict[NodeRef, NodeId]  # node_ref => node_id
    edges: dict[EdgeRef, EdgeId]  # edge_ref => edge

    # The id is unique among the tiles, so we can use it as the hash.
    def __hash__(self) -> int:
        return self.id


@dataclass(frozen=True)
class Water:
    __module__ = "cle.game_engine.models.map"

    nodes: dict[NodeRef, int]
    edges: dict[EdgeRef, EdgeId]


Tile: TypeAlias = LandTile | Port | Water


@dataclass(frozen=True)
class MapTemplate:
    __module__ = "cle.game_engine.models.map"

    numbers: list[int]
    port_resources: list[FastResource | None]
    tile_resources: list[FastResource | None]
    topology: Mapping[Coordinate, type[LandTile] | type[Water] | tuple[type[Port], Direction]]


# TODO: Could consolidate Direction with EdgeRef.
PORT_DIRECTION_TO_NODEREFS: dict[Direction, tuple[NodeRef, NodeRef]] = {
    Direction.WEST: (NodeRef.NORTHWEST, NodeRef.SOUTHWEST),
    Direction.NORTHWEST: (NodeRef.NORTH, NodeRef.NORTHWEST),
    Direction.NORTHEAST: (NodeRef.NORTHEAST, NodeRef.NORTH),
    Direction.EAST: (NodeRef.SOUTHEAST, NodeRef.NORTHEAST),
    Direction.SOUTHEAST: (NodeRef.SOUTH, NodeRef.SOUTHEAST),
    Direction.SOUTHWEST: (NodeRef.SOUTHWEST, NodeRef.SOUTH),
}
