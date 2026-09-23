"""Conversion from a public-board contract to the HexBoard render state."""

from __future__ import annotations

from typing import NotRequired, TypedDict

from evals.catan_board_bench.annotations.geometry import _atlas_render_refs
from evals.catan_board_bench.tokens import canonical_edge
from evals.json_types import (
    JsonDict,
    JsonList,
    JsonValue,
    as_dict,
    as_dicts,
    as_int,
    as_list,
    as_str,
)


class RenderTile(TypedDict):
    """One HexBoard tile payload: land (``RESOURCE_TILE``/``DESERT``) or ``PORT``."""

    id: int
    type: str
    resource: NotRequired[str | None]
    number: NotRequired[int | None]
    direction: NotRequired[str]
    emoji: NotRequired[str]
    port_nodes: NotRequired[list[int]]


class PlacedTile(TypedDict):
    coordinate: list[int]
    tile: RenderTile


class RenderNode(TypedDict):
    id: int
    tile_coordinate: list[int]
    direction: str
    building: str | None
    color: str | None


class RenderEdge(TypedDict):
    id: list[int]
    tile_coordinate: list[int]
    direction: str
    color: str | None


class RenderState(TypedDict):
    """The minimal HexBoard ``GameState`` shape the Python renderer consumes."""

    tiles: list[PlacedTile]
    nodes: dict[str, RenderNode]
    edges: list[RenderEdge]
    robber_coordinate: list[int] | None
    adjacent_tiles: JsonDict
    colors: list[str]
    current_color: str
    bot_colors: list[str]
    player_state: JsonDict
    longest_roads_by_player: JsonDict
    played_knights_by_player: JsonDict
    is_initial_build_phase: JsonValue
    current_prompt: JsonValue
    current_playable_actions: JsonList
    actions: JsonList
    winning_color: None
    state_index: JsonValue
    achievements: JsonValue


def _int_list(value: JsonValue, label: str) -> list[int]:
    return [as_int(item, f"{label} entry") for item in as_list(value, label)]


def _optional_str(value: JsonValue, label: str) -> str | None:
    return None if value is None else as_str(value, label)


def _optional_int(value: JsonValue, label: str) -> int | None:
    return None if value is None else as_int(value, label)


def contract_to_render_state(contract: JsonDict) -> RenderState:
    """Convert a public board contract into the minimal HexBoard GameState."""

    refs = _atlas_render_refs()
    placed_tiles: list[PlacedTile] = []

    for tile in as_dicts(contract.get("tiles", []), "contract tiles"):
        resource = _optional_str(tile.get("resource"), "tile resource")
        tile_payload: RenderTile
        if resource is None:
            tile_payload = {"id": as_int(tile["id"], "tile id"), "type": "DESERT"}
        else:
            tile_payload = {
                "id": as_int(tile["id"], "tile id"),
                "type": "RESOURCE_TILE",
                "resource": resource,
                "number": _optional_int(tile.get("number"), "tile number"),
            }
        placed_tiles.append(
            {"coordinate": _int_list(tile["coord"], "tile coord"), "tile": tile_payload}
        )

    for port in as_dicts(contract.get("ports", []), "contract ports"):
        resource = _optional_str(port.get("resource"), "port resource")
        placed_tiles.append(
            {
                "coordinate": _int_list(port["coord"], "port coord"),
                "tile": {
                    "id": as_int(port["id"], "port id"),
                    "type": "PORT",
                    "direction": as_str(port["direction"], "port direction"),
                    "resource": resource,
                    "emoji": "3:1" if resource is None else resource,
                    "port_nodes": _int_list(port["attached_nodes"], "port attached_nodes"),
                },
            }
        )

    nodes: dict[str, RenderNode] = {}
    for node in as_dicts(contract.get("nodes", []), "contract nodes"):
        node_id = as_int(node["id"], "node id")
        node_ref = refs["nodes"].get(node_id)
        if not node_ref:
            continue
        nodes[str(node_id)] = {
            "id": node_id,
            "tile_coordinate": node_ref["tile_coordinate"],
            "direction": node_ref["direction"],
            "building": _optional_str(node.get("building"), "node building"),
            "color": _optional_str(node.get("color"), "node color"),
        }

    edges: list[RenderEdge] = []
    for edge in as_dicts(contract.get("edges", []), "contract edges"):
        first, second = _int_list(edge["id"], "edge id")
        edge_id = tuple(canonical_edge((first, second)))
        edge_ref = refs["edges"].get(edge_id)
        if not edge_ref:
            continue
        edges.append(
            {
                "id": list(edge_id),
                "tile_coordinate": edge_ref["tile_coordinate"],
                "direction": edge_ref["direction"],
                "color": _optional_str(edge.get("road_color"), "edge road_color"),
            }
        )

    players = [
        as_str(player["color"], "player color")
        for player in as_dicts(contract.get("players", []), "contract players")
    ]
    current = as_dict(contract.get("current", {}), "contract current")
    robber_coord = as_dict(contract.get("robber", {}), "contract robber").get("coord")
    current_color = current.get("current_color")

    return {
        "tiles": placed_tiles,
        "nodes": nodes,
        "edges": edges,
        "robber_coordinate": (
            None if robber_coord is None else _int_list(robber_coord, "robber coord")
        ),
        "adjacent_tiles": {},
        "colors": players,
        "current_color": (
            as_str(current_color, "current_color")
            if current_color
            else (players[0] if players else "RED")
        ),
        "bot_colors": [],
        "player_state": {},
        "longest_roads_by_player": {},
        "played_knights_by_player": {},
        "is_initial_build_phase": current.get("is_initial_build_phase", False),
        "current_prompt": current.get("current_prompt", ""),
        "current_playable_actions": [],
        "actions": [],
        "winning_color": None,
        "state_index": current.get("num_completed_turns", 0),
        "achievements": contract.get("achievements", {}),
    }
