"""Immutable public-board snapshots for model and evaluation presentations."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import TypeAlias

from cle.game_engine.models.map import NUM_NODES, LandTile, Port
from cle.game_engine.observation import PlayerObservation
from cle.game_engine.public_board.identifiers import (
    CANONICAL_BOARD_IDENTITY,
    PUBLIC_BOARD_CONTRACT_SCHEMA,
    PUBLIC_BOARD_FACTS_SCHEMA,
    canonical_edge,
    canonical_json,
    sha256_text,
)
from cle.game_engine.public_board.rows import (
    Edge,
    attached_nodes_by_port,
    edge_rows,
    node_rows,
    port_rows,
    road_owners,
    tile_rows,
)

__all__ = [
    "CANONICAL_BOARD_IDENTITY",
    "PUBLIC_BOARD_CONTRACT_SCHEMA",
    "PUBLIC_BOARD_FACTS_SCHEMA",
    "JsonValue",
    "PublicBoardSnapshot",
    "snapshot_public_board",
]

JsonValue: TypeAlias = (
    None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
)


def _load_json_object(text: str) -> dict[str, JsonValue]:
    loaded: object = json.loads(text)
    if not isinstance(loaded, dict):
        raise ValueError("Public-board payloads must be JSON objects")
    return loaded


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
        if sha256_text(self.facts_json) != self.facts_sha256:
            raise ValueError("Public-board fact digest does not match its payload")
        contract = json.loads(self.contract_json)
        facts = json.loads(self.facts_json)
        if contract.get("schema") != PUBLIC_BOARD_CONTRACT_SCHEMA:
            raise ValueError("Unexpected public-board contract schema")
        if facts.get("schema") != PUBLIC_BOARD_FACTS_SCHEMA:
            raise ValueError("Unexpected public-board fact schema")

    def contract(self) -> dict[str, JsonValue]:
        """Return a fresh mutable copy for a presentation renderer."""

        return _load_json_object(self.contract_json)

    def facts(self) -> dict[str, JsonValue]:
        """Return a fresh mutable copy for a presentation renderer."""

        return _load_json_object(self.facts_json)


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
            canonical_edge(tuple(edge))
            for tile in land_tiles
            for edge in tile.edges.values()
        }
    )
    edge_tiles: dict[Edge, list[int]] = defaultdict(list)
    for tile in land_tiles:
        for edge in tile.edges.values():
            edge_tiles[canonical_edge(tuple(edge))].append(tile.id)

    owners = road_owners(
        {
            observation.my_color: observation.my_roads,
            **observation.opponent_roads,
        }
    )

    nodes_by_port = attached_nodes_by_port(ports)
    port_ids_by_node: dict[int, list[int]] = defaultdict(list)
    for port in ports:
        for node_id in nodes_by_port[port.id]:
            port_ids_by_node[node_id].append(port.id)

    robber_position = tuple(observation.robber_position)
    contract_tiles, fact_tiles = tile_rows(land_tiles, tile_coordinates, robber_position)
    contract_nodes, fact_nodes = node_rows(observation, NUM_NODES, all_edges, port_ids_by_node)
    contract_edges, fact_edges = edge_rows(all_edges, owners, edge_tiles)
    contract_ports, fact_ports = port_rows(ports, port_coordinates, nodes_by_port)

    player_colors = sorted(
        {
            observation.my_color.value,
            *(color.value for color in observation.opponent_vps),
        }
    )
    robber_tile = next(tile for tile in contract_tiles if tile["has_robber"])
    contract: dict[str, object] = {
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
    facts: dict[str, object] = {
        "schema": PUBLIC_BOARD_FACTS_SCHEMA,
        "tiles": sorted(fact_tiles, key=lambda item: item["id"]),
        "nodes": sorted(fact_nodes, key=lambda item: item["id"]),
        "edges": sorted(fact_edges, key=lambda item: item["id"]),
        "ports": sorted(fact_ports, key=lambda item: item["id"]),
    }
    contract_json = canonical_json(contract)
    facts_json = canonical_json(facts)
    return PublicBoardSnapshot(
        contract_json=contract_json,
        facts_json=facts_json,
        facts_sha256=sha256_text(facts_json),
    )
