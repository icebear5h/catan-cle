"""Per-entity rows for the public-board contract and fact payloads."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypedDict

from cle.game_engine.models.map import PORT_DIRECTION_TO_NODEREFS, LandTile, Port
from cle.game_engine.models.player import Color
from cle.game_engine.observation import PlayerObservation
from cle.game_engine.public_board.identifiers import (
    canonical_edge,
    edge_id,
    node_id,
    port_id,
    tile_id,
)

ContractRow = dict[str, object]
Edge = tuple[int, int]


class FactTile(TypedDict):
    id: str
    cube: list[int]
    resource: str
    number: int | None
    robber: bool
    corners: dict[str, str]
    sides: dict[str, str]


class FactNode(TypedDict):
    id: str
    color: str | None
    building: str | None
    tiles: list[str]
    edges: list[str]
    ports: list[str]


class FactEdge(TypedDict):
    id: str
    nodes: list[str]
    road: str | None
    tiles: list[str]


class FactPort(TypedDict):
    id: str
    cube: list[int]
    direction: str
    resource: str
    ratio: str
    nodes: list[str]


def attached_nodes_by_port(ports: Sequence[Port]) -> dict[int, list[int]]:
    return {
        port.id: [
            port.nodes[node_ref]
            for node_ref in PORT_DIRECTION_TO_NODEREFS[port.direction]
        ]
        for port in ports
    }


def tile_rows(
    land_tiles: Sequence[LandTile],
    tile_coordinates: Mapping[int, tuple[int, ...]],
    robber_position: tuple[int, ...],
) -> tuple[list[ContractRow], list[FactTile]]:
    contract_tiles: list[ContractRow] = []
    fact_tiles: list[FactTile] = []
    for tile in land_tiles:
        coordinate = tile_coordinates[tile.id]
        resource = tile.resource
        contract_tiles.append(
            {
                "id": tile.id,
                "coord": list(coordinate),
                "resource": resource,
                "number": tile.number,
                "has_robber": coordinate == robber_position,
                "nodes": sorted(tile.nodes.values()),
                "edges": [
                    list(canonical_edge(tuple(edge)))
                    for edge in sorted(
                        tile.edges.values(),
                        key=lambda value: canonical_edge(tuple(value)),
                    )
                ],
            }
        )
        fact_tiles.append(
            {
                "id": tile_id(tile.id),
                "cube": list(coordinate),
                "resource": resource or "DESERT",
                "number": tile.number,
                "robber": coordinate == robber_position,
                "corners": {
                    node_ref.value: node_id(corner)
                    for node_ref, corner in sorted(
                        tile.nodes.items(), key=lambda item: item[0].value
                    )
                },
                "sides": {
                    edge_ref.value: edge_id(tuple(edge))
                    for edge_ref, edge in sorted(
                        tile.edges.items(), key=lambda item: item[0].value
                    )
                },
            }
        )
    return contract_tiles, fact_tiles


def node_rows(
    observation: PlayerObservation,
    node_count: int,
    all_edges: Sequence[Edge],
    port_ids_by_node: Mapping[int, Sequence[int]],
) -> tuple[list[ContractRow], list[FactNode]]:
    board_map = observation.board_map
    contract_nodes: list[ContractRow] = []
    fact_nodes: list[FactNode] = []
    for node in range(node_count):
        building = observation.buildings_dict.get(node)
        color = building[0].value if building else None
        building_type = str(building[1]) if building else None
        adjacent_tiles = sorted(
            tile.id for tile in board_map.adjacent_tiles.get(node, ())
        )
        adjacent_edges = [edge for edge in all_edges if node in edge]
        port_ids = sorted(port_ids_by_node.get(node, ()))
        contract_nodes.append(
            {
                "id": node,
                "color": color,
                "building": building_type,
                "adjacent_tiles": adjacent_tiles,
                "adjacent_edges": [list(edge) for edge in adjacent_edges],
                "port_ids": port_ids,
            }
        )
        fact_nodes.append(
            {
                "id": node_id(node),
                "color": color,
                "building": building_type,
                "tiles": [tile_id(adjacent) for adjacent in adjacent_tiles],
                "edges": [edge_id(edge) for edge in adjacent_edges],
                "ports": [port_id(port) for port in port_ids],
            }
        )
    return contract_nodes, fact_nodes


def edge_rows(
    all_edges: Sequence[Edge],
    road_owners: Mapping[Edge, str],
    edge_tiles: Mapping[Edge, Sequence[int]],
) -> tuple[list[ContractRow], list[FactEdge]]:
    contract_edges: list[ContractRow] = []
    fact_edges: list[FactEdge] = []
    for edge in all_edges:
        owner = road_owners.get(edge)
        adjacent_tile_ids = sorted(edge_tiles[edge])
        contract_edges.append(
            {
                "id": list(edge),
                "road_color": owner,
            }
        )
        fact_edges.append(
            {
                "id": edge_id(edge),
                "nodes": [node_id(endpoint) for endpoint in edge],
                "road": owner,
                "tiles": [tile_id(adjacent) for adjacent in adjacent_tile_ids],
            }
        )
    return contract_edges, fact_edges


def port_rows(
    ports: Sequence[Port],
    port_coordinates: Mapping[int, tuple[int, ...]],
    nodes_by_port: Mapping[int, Sequence[int]],
) -> tuple[list[ContractRow], list[FactPort]]:
    contract_ports: list[ContractRow] = []
    fact_ports: list[FactPort] = []
    for port in ports:
        coordinate = port_coordinates[port.id]
        attached_nodes = sorted(nodes_by_port[port.id])
        contract_ports.append(
            {
                "id": port.id,
                "coord": list(coordinate),
                "direction": port.direction.value,
                "kind": "generic" if port.resource is None else "resource",
                "ratio": "3:1" if port.resource is None else "2:1",
                "resource": port.resource,
                "attached_nodes": attached_nodes,
            }
        )
        fact_ports.append(
            {
                "id": port_id(port.id),
                "cube": list(coordinate),
                "direction": port.direction.value,
                "resource": port.resource or "GENERIC",
                "ratio": "3:1" if port.resource is None else "2:1",
                "nodes": [node_id(attached) for attached in attached_nodes],
            }
        )
    return contract_ports, fact_ports


def road_owners(
    road_groups: Mapping[Color, Sequence[Edge]],
) -> dict[Edge, str]:
    owners: dict[Edge, str] = {}
    for color, roads in road_groups.items():
        for road in roads:
            edge = canonical_edge(tuple(road))
            owner = color.value
            previous = owners.setdefault(edge, owner)
            if previous != owner:
                raise ValueError(f"Conflicting public road owners for {edge}")
    return owners
