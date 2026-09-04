"""Immutable public-board snapshots for model and evaluation presentations."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from cle.game_engine.models.map import (
    NUM_NODES,
    PORT_DIRECTION_TO_NODEREFS,
    LandTile,
    Port,
)
from cle.game_engine.observation import PlayerObservation


PUBLIC_BOARD_CONTRACT_SCHEMA = "catan_public_board_contract/v0"
PUBLIC_BOARD_FACTS_SCHEMA = "catan_full_public_graph/v1"
CANONICAL_BOARD_IDENTITY = "canonical_engine_ids"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_edge(edge: tuple[int, int]) -> tuple[int, int]:
    first, second = edge
    return (first, second) if first < second else (second, first)


def _tile_id(tile_id: int) -> str:
    return f"T{tile_id:02d}"


def _node_id(node_id: int) -> str:
    return f"N{node_id:02d}"


def _edge_id(edge: tuple[int, int]) -> str:
    first, second = _canonical_edge(edge)
    return f"E{first:02d}_{second:02d}"


def _port_id(port_id: int) -> str:
    return f"P{port_id:02d}"


@dataclass(frozen=True, slots=True)
class PublicBoardSnapshot:
    """Deep, canonical snapshot of facts visible on the physical board."""

    contract_json: str
    facts_json: str
    facts_sha256: str
    identity_space: str = CANONICAL_BOARD_IDENTITY

    def __post_init__(self) -> None:
        if self.identity_space != CANONICAL_BOARD_IDENTITY:
            raise ValueError("Runtime board snapshots must use canonical engine IDs")
        if _sha256_text(self.facts_json) != self.facts_sha256:
            raise ValueError("Public-board fact digest does not match its payload")
        contract = json.loads(self.contract_json)
        facts = json.loads(self.facts_json)
        if contract.get("schema") != PUBLIC_BOARD_CONTRACT_SCHEMA:
            raise ValueError("Unexpected public-board contract schema")
        if facts.get("schema") != PUBLIC_BOARD_FACTS_SCHEMA:
            raise ValueError("Unexpected public-board fact schema")

    def contract(self) -> dict[str, Any]:
        """Return a fresh mutable copy for a presentation renderer."""

        return json.loads(self.contract_json)

    def facts(self) -> dict[str, Any]:
        """Return a fresh mutable copy for a presentation renderer."""

        return json.loads(self.facts_json)


def snapshot_public_board(observation: PlayerObservation) -> PublicBoardSnapshot:
    """Snapshot only public board geometry and pieces from one observation."""

    board_map = observation.board_map
    land_tiles = sorted(board_map.tiles_by_id.values(), key=lambda tile: tile.id)
    ports = sorted(board_map.ports_by_id.values(), key=lambda port: port.id)
    tile_coordinates = {
        tile.id: tuple(coordinate)
        for coordinate, tile in board_map.tiles.items()
        if isinstance(tile, LandTile)
    }
    port_coordinates = {
        tile.id: tuple(coordinate)
        for coordinate, tile in board_map.tiles.items()
        if isinstance(tile, Port)
    }

    all_edges = sorted(
        {
            _canonical_edge(tuple(edge))
            for tile in land_tiles
            for edge in tile.edges.values()
        }
    )
    edge_tiles: dict[tuple[int, int], list[int]] = defaultdict(list)
    for tile in land_tiles:
        for edge in tile.edges.values():
            edge_tiles[_canonical_edge(tuple(edge))].append(tile.id)

    road_owners: dict[tuple[int, int], str] = {}
    road_groups = {
        observation.my_color: observation.my_roads,
        **observation.opponent_roads,
    }
    for color, roads in road_groups.items():
        for road in roads:
            edge = _canonical_edge(tuple(road))
            owner = color.value
            previous = road_owners.setdefault(edge, owner)
            if previous != owner:
                raise ValueError(f"Conflicting public road owners for {edge}")

    attached_nodes_by_port = {
        port.id: [
            port.nodes[node_ref]
            for node_ref in PORT_DIRECTION_TO_NODEREFS[port.direction]
        ]
        for port in ports
    }
    port_ids_by_node: dict[int, list[int]] = defaultdict(list)
    for port in ports:
        for node_id in attached_nodes_by_port[port.id]:
            port_ids_by_node[node_id].append(port.id)

    contract_tiles = []
    fact_tiles = []
    for tile in land_tiles:
        coordinate = tile_coordinates[tile.id]
        resource = tile.resource
        contract_tiles.append(
            {
                "id": tile.id,
                "coord": list(coordinate),
                "resource": resource,
                "number": tile.number,
                "has_robber": coordinate == tuple(observation.robber_position),
                "nodes": sorted(tile.nodes.values()),
                "edges": [
                    list(_canonical_edge(tuple(edge)))
                    for edge in sorted(
                        tile.edges.values(),
                        key=lambda value: _canonical_edge(tuple(value)),
                    )
                ],
            }
        )
        fact_tiles.append(
            {
                "id": _tile_id(tile.id),
                "cube": list(coordinate),
                "resource": resource or "DESERT",
                "number": tile.number,
                "robber": coordinate == tuple(observation.robber_position),
                "corners": {
                    node_ref.value: _node_id(node_id)
                    for node_ref, node_id in sorted(
                        tile.nodes.items(), key=lambda item: item[0].value
                    )
                },
                "sides": {
                    edge_ref.value: _edge_id(tuple(edge))
                    for edge_ref, edge in sorted(
                        tile.edges.items(), key=lambda item: item[0].value
                    )
                },
            }
        )

    contract_nodes = []
    fact_nodes = []
    for node_id in range(NUM_NODES):
        building = observation.buildings_dict.get(node_id)
        color = building[0].value if building else None
        building_type = str(building[1]) if building else None
        adjacent_tiles = sorted(
            tile.id for tile in board_map.adjacent_tiles.get(node_id, ())
        )
        adjacent_edges = [edge for edge in all_edges if node_id in edge]
        port_ids = sorted(port_ids_by_node.get(node_id, ()))
        contract_nodes.append(
            {
                "id": node_id,
                "color": color,
                "building": building_type,
                "adjacent_tiles": adjacent_tiles,
                "adjacent_edges": [list(edge) for edge in adjacent_edges],
                "port_ids": port_ids,
            }
        )
        fact_nodes.append(
            {
                "id": _node_id(node_id),
                "color": color,
                "building": building_type,
                "tiles": [_tile_id(tile_id) for tile_id in adjacent_tiles],
                "edges": [_edge_id(edge) for edge in adjacent_edges],
                "ports": [_port_id(port_id) for port_id in port_ids],
            }
        )

    contract_edges = []
    fact_edges = []
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
                "id": _edge_id(edge),
                "nodes": [_node_id(node_id) for node_id in edge],
                "road": owner,
                "tiles": [_tile_id(tile_id) for tile_id in adjacent_tile_ids],
            }
        )

    contract_ports = []
    fact_ports = []
    for port in ports:
        coordinate = port_coordinates[port.id]
        attached_nodes = sorted(attached_nodes_by_port[port.id])
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
                "id": _port_id(port.id),
                "cube": list(coordinate),
                "direction": port.direction.value,
                "resource": port.resource or "GENERIC",
                "ratio": "3:1" if port.resource is None else "2:1",
                "nodes": [_node_id(node_id) for node_id in attached_nodes],
            }
        )

    player_colors = sorted(
        {
            observation.my_color.value,
            *(color.value for color in observation.opponent_vps),
        }
    )
    robber_tile = next(tile for tile in contract_tiles if tile["has_robber"])
    contract = {
        "schema": PUBLIC_BOARD_CONTRACT_SCHEMA,
        "sample": {},
        "source": {},
        "current": {
            "current_color": observation.turn_player_color.value,
            "current_prompt": observation.current_phase,
            "num_completed_turns": observation.current_turn,
            "is_initial_build_phase": observation.current_phase
            == "initial_placement",
        },
        "players": [{"color": color} for color in player_colors],
        "tiles": contract_tiles,
        "nodes": contract_nodes,
        "edges": contract_edges,
        "ports": contract_ports,
        "robber": {
            "tile_id": robber_tile["id"],
            "coord": list(observation.robber_position),
        },
        "achievements": {
            "longest_road": {
                "holder": (
                    observation.longest_road_holder.value
                    if observation.longest_road_holder
                    else None
                )
            },
            "largest_army": {
                "holder": (
                    observation.largest_army_holder.value
                    if observation.largest_army_holder
                    else None
                )
            },
        },
    }
    facts = {
        "schema": PUBLIC_BOARD_FACTS_SCHEMA,
        "tiles": sorted(fact_tiles, key=lambda item: item["id"]),
        "nodes": sorted(fact_nodes, key=lambda item: item["id"]),
        "edges": sorted(fact_edges, key=lambda item: item["id"]),
        "ports": sorted(fact_ports, key=lambda item: item["id"]),
    }
    contract_json = _canonical_json(contract)
    facts_json = _canonical_json(facts)
    return PublicBoardSnapshot(
        contract_json=contract_json,
        facts_json=facts_json,
        facts_sha256=_sha256_text(facts_json),
    )
