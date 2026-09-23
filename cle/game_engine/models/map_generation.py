"""Tile placement with stable RNG consumption, topology order, and entity IDs."""

import random
from collections import Counter
from collections.abc import Iterable
from typing import cast

from cle.game_engine.models.coordinate_system import UNIT_VECTORS, Direction, add
from cle.game_engine.models.enums import EdgeRef, FastResource, NodeRef
from cle.game_engine.models.map_types import (
    Coordinate,
    EdgeId,
    LandTile,
    MapRNG,
    MapTemplate,
    NodeId,
    Port,
    Tile,
    Water,
)


def initialize_tiles(
    map_template: MapTemplate,
    shuffled_numbers_param: list[int] | None = None,
    shuffled_port_resources_param: list[FastResource | None] | None = None,
    shuffled_tile_resources_param: Iterable[FastResource | None] | None = None,
    rng: MapRNG | None = None,
) -> dict[Coordinate, Tile]:
    """Initializes a new random board, based on the MapTemplate.

    It first shuffles tiles, ports, and numbers. Then goes satisfying the
    topology (i.e. placing tiles on coordinates); ensuring to "attach" these to
    neighbor tiles (so as to not repeat nodes or edges objects).
    Explicit terrain orders must be permutations of the template's inventory.

    Args:
        map_template (MapTemplate): Template to initialize.

    Raises:
        ValueError: Invalid terrain inventory or tile in topology.

    Returns:
        dict[Coordinate, Tile]: Coordinate to initialized Tile mapping.
    """
    source = rng if rng is not None else random.Random()
    shuffled_port_resources = shuffled_port_resources_param or source.sample(
        map_template.port_resources, len(map_template.port_resources)
    )
    shuffled_tile_resources = (
        source.sample(map_template.tile_resources, len(map_template.tile_resources))
        if shuffled_tile_resources_param is None
        else list(shuffled_tile_resources_param)
    )
    if Counter(shuffled_tile_resources) != Counter(map_template.tile_resources):
        raise ValueError(
            "Tile resources must be a permutation of the map template's terrain inventory"
        )
    land_tile_count = sum(tile_type is LandTile for tile_type in map_template.topology.values())
    if len(shuffled_tile_resources) != land_tile_count:
        raise ValueError("Terrain inventory must contain exactly one resource per land tile")
    shuffled_numbers = shuffled_numbers_param or source.sample(
        map_template.numbers, len(map_template.numbers)
    )

    # for each topology entry, place a tile. keep track of nodes and edges
    all_tiles: dict[Coordinate, Tile] = {}
    node_autoinc = 0
    tile_autoinc = 0
    port_autoinc = 0
    for coordinate, tile_type in map_template.topology.items():
        nodes, edges, node_autoinc = get_nodes_and_edges(all_tiles, coordinate, node_autoinc)

        # create and save tile
        if isinstance(tile_type, tuple):  # is port
            (_, direction) = tile_type
            port = Port(port_autoinc, shuffled_port_resources.pop(), direction, nodes, edges)
            all_tiles[coordinate] = port
            port_autoinc += 1
        elif tile_type == LandTile:
            resource = shuffled_tile_resources.pop()
            if resource is not None:
                number = shuffled_numbers.pop()
                tile = LandTile(tile_autoinc, resource, number, nodes, edges)
            else:
                tile = LandTile(tile_autoinc, None, None, nodes, edges)  # desert
            all_tiles[coordinate] = tile
            tile_autoinc += 1
        elif tile_type == Water:
            water_tile = Water(nodes, edges)
            all_tiles[coordinate] = water_tile
        else:
            raise ValueError("Invalid tile")

    return all_tiles


def get_nodes_and_edges(
    tiles: dict[Coordinate, Tile], coordinate: Coordinate, node_autoinc: int
) -> tuple[dict[NodeRef, NodeId], dict[EdgeRef, EdgeId], int]:
    """Get pre-existing nodes and edges in board for given tile coordinate"""
    nodes: dict[NodeRef, NodeId | None] = {
        NodeRef.NORTH: None,
        NodeRef.NORTHEAST: None,
        NodeRef.SOUTHEAST: None,
        NodeRef.SOUTH: None,
        NodeRef.SOUTHWEST: None,
        NodeRef.NORTHWEST: None,
    }
    edges: dict[EdgeRef, EdgeId | None] = {
        EdgeRef.EAST: None,
        EdgeRef.SOUTHEAST: None,
        EdgeRef.SOUTHWEST: None,
        EdgeRef.WEST: None,
        EdgeRef.NORTHWEST: None,
        EdgeRef.NORTHEAST: None,
    }

    # Find pre-existing ones
    neighbor_tiles = [(add(coordinate, UNIT_VECTORS[d]), d) for d in Direction]
    for coord, neighbor_direction in neighbor_tiles:
        if coord not in tiles:
            continue

        neighbor = tiles[coord]
        if neighbor_direction == Direction.EAST:
            nodes[NodeRef.NORTHEAST] = neighbor.nodes[NodeRef.NORTHWEST]
            nodes[NodeRef.SOUTHEAST] = neighbor.nodes[NodeRef.SOUTHWEST]
            edges[EdgeRef.EAST] = neighbor.edges[EdgeRef.WEST]
        elif neighbor_direction == Direction.SOUTHEAST:
            nodes[NodeRef.SOUTH] = neighbor.nodes[NodeRef.NORTHWEST]
            nodes[NodeRef.SOUTHEAST] = neighbor.nodes[NodeRef.NORTH]
            edges[EdgeRef.SOUTHEAST] = neighbor.edges[EdgeRef.NORTHWEST]
        elif neighbor_direction == Direction.SOUTHWEST:
            nodes[NodeRef.SOUTH] = neighbor.nodes[NodeRef.NORTHEAST]
            nodes[NodeRef.SOUTHWEST] = neighbor.nodes[NodeRef.NORTH]
            edges[EdgeRef.SOUTHWEST] = neighbor.edges[EdgeRef.NORTHEAST]
        elif neighbor_direction == Direction.WEST:
            nodes[NodeRef.NORTHWEST] = neighbor.nodes[NodeRef.NORTHEAST]
            nodes[NodeRef.SOUTHWEST] = neighbor.nodes[NodeRef.SOUTHEAST]
            edges[EdgeRef.WEST] = neighbor.edges[EdgeRef.EAST]
        elif neighbor_direction == Direction.NORTHWEST:
            nodes[NodeRef.NORTH] = neighbor.nodes[NodeRef.SOUTHEAST]
            nodes[NodeRef.NORTHWEST] = neighbor.nodes[NodeRef.SOUTH]
            edges[EdgeRef.NORTHWEST] = neighbor.edges[EdgeRef.SOUTHEAST]
        elif neighbor_direction == Direction.NORTHEAST:
            nodes[NodeRef.NORTH] = neighbor.nodes[NodeRef.SOUTHWEST]
            nodes[NodeRef.NORTHEAST] = neighbor.nodes[NodeRef.SOUTH]
            edges[EdgeRef.NORTHEAST] = neighbor.edges[EdgeRef.SOUTHWEST]
        else:
            raise Exception("Something went wrong")

    # Initializes new ones
    for noderef, value in nodes.items():
        if value is None:
            nodes[noderef] = node_autoinc
            node_autoinc += 1
    # Every node now has an ID; retain the original dictionary and its order.
    initialized_nodes = cast(dict[NodeRef, NodeId], nodes)
    for edgeref, edge_value in edges.items():
        if edge_value is None:
            a_noderef, b_noderef = get_edge_nodes(edgeref)
            edge_nodes = (initialized_nodes[a_noderef], initialized_nodes[b_noderef])
            edges[edgeref] = edge_nodes

    return initialized_nodes, cast(dict[EdgeRef, EdgeId], edges), node_autoinc


def get_edge_nodes(edge_ref: EdgeRef) -> tuple[NodeRef, NodeRef]:
    """returns pair of nodes at the "ends" of a given edge"""
    return {
        EdgeRef.EAST: (NodeRef.NORTHEAST, NodeRef.SOUTHEAST),
        EdgeRef.SOUTHEAST: (NodeRef.SOUTHEAST, NodeRef.SOUTH),
        EdgeRef.SOUTHWEST: (NodeRef.SOUTH, NodeRef.SOUTHWEST),
        EdgeRef.WEST: (NodeRef.SOUTHWEST, NodeRef.NORTHWEST),
        EdgeRef.NORTHWEST: (NodeRef.NORTHWEST, NodeRef.NORTH),
        EdgeRef.NORTHEAST: (NodeRef.NORTH, NodeRef.NORTHEAST),
    }[edge_ref]
