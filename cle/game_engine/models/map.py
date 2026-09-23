from collections import Counter, defaultdict
from typing import Literal, cast

from cle.game_engine.models.coordinate_system import UNIT_VECTORS as UNIT_VECTORS
from cle.game_engine.models.coordinate_system import Direction as Direction
from cle.game_engine.models.coordinate_system import add as add
from cle.game_engine.models.enums import BRICK as BRICK
from cle.game_engine.models.enums import ORE as ORE
from cle.game_engine.models.enums import SHEEP as SHEEP
from cle.game_engine.models.enums import WHEAT as WHEAT
from cle.game_engine.models.enums import WOOD as WOOD
from cle.game_engine.models.enums import EdgeRef as EdgeRef
from cle.game_engine.models.enums import FastResource as FastResource
from cle.game_engine.models.enums import NodeRef as NodeRef
from cle.game_engine.models.map_generation import get_edge_nodes as get_edge_nodes
from cle.game_engine.models.map_generation import get_nodes_and_edges as get_nodes_and_edges
from cle.game_engine.models.map_generation import initialize_tiles as initialize_tiles
from cle.game_engine.models.map_templates import BASE_MAP_TEMPLATE as BASE_MAP_TEMPLATE
from cle.game_engine.models.map_templates import MINI_MAP_TEMPLATE as MINI_MAP_TEMPLATE
from cle.game_engine.models.map_types import NUM_EDGES as NUM_EDGES
from cle.game_engine.models.map_types import NUM_NODES as NUM_NODES
from cle.game_engine.models.map_types import NUM_TILES as NUM_TILES
from cle.game_engine.models.map_types import (
    PORT_DIRECTION_TO_NODEREFS as PORT_DIRECTION_TO_NODEREFS,
)
from cle.game_engine.models.map_types import Coordinate as Coordinate
from cle.game_engine.models.map_types import EdgeId as EdgeId
from cle.game_engine.models.map_types import LandTile as LandTile
from cle.game_engine.models.map_types import MapRNG, Production
from cle.game_engine.models.map_types import MapTemplate as MapTemplate
from cle.game_engine.models.map_types import NodeId as NodeId
from cle.game_engine.models.map_types import Port as Port
from cle.game_engine.models.map_types import Tile as Tile
from cle.game_engine.models.map_types import Water as Water


class CatanMap:
    """Represents a randomly initialized map."""

    def __init__(
        self,
        tiles: dict[Coordinate, Tile] = dict(),
        land_tiles: dict[Coordinate, LandTile] = dict(),
        port_nodes: dict[FastResource | None, set[int]] = dict(),
        land_nodes: frozenset[NodeId] = frozenset(),
        adjacent_tiles: dict[int, list[LandTile]] = dict(),
        node_production: dict[NodeId, Production] = dict(),
        tiles_by_id: dict[int, LandTile] = dict(),
        ports_by_id: dict[int, Port] = dict(),
    ) -> None:
        self.tiles = tiles
        self.land_tiles = land_tiles
        self.port_nodes = port_nodes
        self.land_nodes = land_nodes
        self.adjacent_tiles = adjacent_tiles
        self.node_production = node_production
        self.tiles_by_id = tiles_by_id
        self.ports_by_id = ports_by_id

    @staticmethod
    def from_template(map_template: MapTemplate, rng: MapRNG | None = None) -> "CatanMap":
        tiles = initialize_tiles(map_template, rng=rng)

        return CatanMap.from_tiles(tiles)

    @staticmethod
    def from_tiles(tiles: dict[Coordinate, Tile]) -> "CatanMap":
        self = CatanMap()
        self.tiles = tiles

        self.land_tiles = {
            k: v for k, v in self.tiles.items() if isinstance(v, LandTile)
        }

        # initialize auxiliary data structures for fast-lookups
        self.port_nodes = init_port_nodes_cache(self.tiles)

        land_nodes_list = map(lambda t: set(t.nodes.values()), self.land_tiles.values())
        self.land_nodes = frozenset[NodeId]().union(*land_nodes_list)

        # TODO: Rename to self.node_to_tiles
        self.adjacent_tiles = init_adjacent_tiles(self.land_tiles)
        self.node_production = init_node_production(self.adjacent_tiles)
        self.tiles_by_id = {
            t.id: t for t in self.tiles.values() if isinstance(t, LandTile)
        }
        self.ports_by_id = {p.id: p for p in self.tiles.values() if isinstance(p, Port)}

        return self


def init_port_nodes_cache(
    tiles: dict[Coordinate, Tile],
) -> dict[FastResource | None, set[int]]:
    """Initializes board.port_nodes cache.

    Args:
        tiles (Dict[Coordinate, Tile]): initialized tiles datastructure

    Returns:
        Dict[Union[FastResource, None], Set[int]]: Mapping from FastResource to node_ids that
            enable port trading. None key represents 3:1 port.
    """
    port_nodes: defaultdict[FastResource | None, set[int]] = defaultdict(set)
    for tile in tiles.values():
        if not isinstance(tile, Port):
            continue

        (a_noderef, b_noderef) = PORT_DIRECTION_TO_NODEREFS[tile.direction]
        port_nodes[tile.resource].add(tile.nodes[a_noderef])
        port_nodes[tile.resource].add(tile.nodes[b_noderef])
    return port_nodes


def init_adjacent_tiles(
    land_tiles: dict[Coordinate, LandTile],
) -> dict[int, list[LandTile]]:
    adjacent_tiles: defaultdict[int, list[LandTile]] = defaultdict(list)  # node_id => tile[3]
    for tile in land_tiles.values():
        for node_id in tile.nodes.values():
            adjacent_tiles[node_id].append(tile)
    return adjacent_tiles


def init_node_production(
    adjacent_tiles: dict[int, list[LandTile]],
) -> dict[NodeId, Production]:
    """Returns node_id => Counter({WHEAT: 0.123, ...})"""
    node_production: dict[NodeId, Production] = dict()
    for node_id in adjacent_tiles.keys():
        node_production[node_id] = get_node_counter_production(adjacent_tiles, node_id)
    return node_production


def get_node_counter_production(
    adjacent_tiles: dict[int, list[LandTile]], node_id: NodeId
) -> Production:
    tiles = adjacent_tiles[node_id]
    production: defaultdict[FastResource, float] = defaultdict(float)
    for tile in tiles:
        if tile.resource is not None:
            production[tile.resource] += number_probability(tile.number)
    # Counter supports fractional counts at runtime; expose their actual value type.
    counter: Counter[FastResource] = Counter()
    probabilities = cast(Production, counter)
    probabilities.update(production)
    return probabilities


def build_dice_probas() -> defaultdict[int | None, float]:
    probas: defaultdict[int | None, float] = defaultdict(float)
    for i in range(1, 7):
        for j in range(1, 7):
            probas[i + j] += 1 / 36
    return probas


DICE_PROBAS = build_dice_probas()


def number_probability(number: int | None) -> float:
    return DICE_PROBAS[number]


TOURNAMENT_MAP_TILES = initialize_tiles(
    BASE_MAP_TEMPLATE,
    [10, 8, 3, 6, 2, 5, 10, 8, 4, 11, 12, 9, 5, 4, 9, 11, 3, 6],
    [
        None,
        SHEEP,
        None,
        ORE,
        WHEAT,
        None,
        WOOD,
        BRICK,
        None,
    ],
    [
        WOOD,
        SHEEP,
        SHEEP,
        WOOD,
        WHEAT,
        WOOD,
        WHEAT,
        BRICK,
        SHEEP,
        BRICK,
        SHEEP,
        WHEAT,
        WHEAT,
        ORE,
        BRICK,
        ORE,
        WOOD,
        ORE,
        None,
    ],
)
TOURNAMENT_MAP = CatanMap.from_tiles(TOURNAMENT_MAP_TILES)


def build_map(map_type: Literal["BASE", "TOURNAMENT", "MINI"]) -> CatanMap:
    if map_type == "TOURNAMENT":
        return TOURNAMENT_MAP  # this assumes map is read-only data struct
    elif map_type == "MINI":
        return CatanMap.from_template(MINI_MAP_TEMPLATE)
    else:
        return CatanMap.from_template(BASE_MAP_TEMPLATE)
