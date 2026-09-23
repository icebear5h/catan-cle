"""Engine-contract validation and resource/production solvers."""

from __future__ import annotations

from typing import cast

from cle.game_engine.models.enums import CITY, RESOURCES, SETTLEMENT
from cle.game_engine.public_board import PUBLIC_BOARD_CONTRACT_SCHEMA
from sft.board.spatial_tasks._topology import COLORS, RESOURCE_KEYS, _topology, node_tile_tokens
from sft.json_types import JsonDict, as_dict, as_int, as_str


def _indexed_rows(contract: JsonDict, key: str, ids: set[int]) -> dict[int, JsonDict]:
    rows = contract.get(key)
    if not isinstance(rows, list):
        raise ValueError(f"contract {key} must be a complete list")
    indexed: dict[int, JsonDict] = {}
    for entry in rows:
        if not isinstance(entry, dict) or type(entry.get("id")) is not int:
            raise ValueError(f"invalid contract {key} row: {entry!r}")
        row = entry
        row_id = cast("int", row["id"])
        if row_id not in ids or row_id in indexed:
            raise ValueError(f"invalid or duplicate contract {key} id: {row_id!r}")
        indexed[row_id] = row
    if set(indexed) != ids:
        raise ValueError(f"contract {key} must cover the canonical atlas exactly")
    return indexed


def _check_ids(value: object, expected: set[int], label: str) -> None:
    if (
        not isinstance(value, list)
        or any(type(item) is not int for item in value)
        or len(value) != len(expected)
        or set(value) != expected
    ):
        raise ValueError(f"noncanonical {label}: {value!r}")


def _check_resource_number(resource: object, number: object) -> None:
    if not isinstance(resource, str) or resource not in (*RESOURCE_KEYS, "desert"):
        raise ValueError(f"invalid resource: {resource!r}")
    if resource == "desert":
        if number is not None:
            raise ValueError("desert must have a null number")
    elif type(number) is not int or not 2 <= number <= 12 or number == 7:
        raise ValueError(f"invalid tile number: {number!r}")


def _contract_tiles(contract: JsonDict) -> dict[int, JsonDict]:
    if not isinstance(contract, dict) or contract.get("schema") != PUBLIC_BOARD_CONTRACT_SCHEMA:
        raise ValueError(f"expected {PUBLIC_BOARD_CONTRACT_SCHEMA} contract")
    atlas_tiles = _topology()[2]
    tiles = _indexed_rows(contract, "tiles", set(atlas_tiles))
    for tile_id, tile in tiles.items():
        atlas_tile = atlas_tiles[tile_id]
        if not {"resource", "number", "has_robber"} <= tile.keys():
            raise ValueError(f"missing tile facts for {tile_id}")
        if "token" in tile and tile["token"] != atlas_tile["token"]:
            raise ValueError(f"noncanonical tile token for {tile_id}")
        _check_ids(tile.get("nodes"),
                   {as_int(v) for v in as_dict(atlas_tile["nodes"]).values()}, "tile nodes")
        coord = tile.get("coord")
        if (
            not isinstance(coord, list)
            or any(type(value) is not int for value in coord)
            or coord != atlas_tile["coord"]
        ):
            raise ValueError(f"noncanonical tile coordinate for {tile_id}")
        resource = tile["resource"]
        if resource is not None and (not isinstance(resource, str) or resource not in RESOURCES):
            raise ValueError(f"invalid contract resource: {resource!r}")
        _check_resource_number("desert" if resource is None else resource.lower(), tile["number"])
        if type(tile["has_robber"]) is not bool:
            raise ValueError(f"invalid robber flag for {tile_id}")
    robber = contract.get("robber")
    if (
        not isinstance(robber, dict)
        or type(robber.get("tile_id")) is not int
        or robber["tile_id"] not in tiles
    ):
        raise ValueError("invalid contract robber tile")
    robber_id = cast("int", robber["tile_id"])
    if [tile_id for tile_id, tile in tiles.items() if tile["has_robber"]] != [robber_id]:
        raise ValueError("contract robber flags disagree with robber tile")
    coord = robber.get("coord")
    if (
        not isinstance(coord, list)
        or any(type(value) is not int for value in coord)
        or coord != tiles[robber_id]["coord"]
    ):
        raise ValueError("contract robber coordinate disagrees with robber tile")
    if "tile_token" in robber and robber["tile_token"] != atlas_tiles[robber_id]["token"]:
        raise ValueError("contract robber token disagrees with robber tile")
    return tiles


def local_node_tiles(contract: JsonDict, node: str) -> dict[str, JsonDict]:
    """Read a node's resource/number neighborhood from an engine contract."""

    tokens = node_tile_tokens(node)
    tiles = _contract_tiles(contract)
    result = {}
    for token in tokens:
        tile = tiles[int(token[2:-1])]
        result[token] = {
            "resource": "desert" if tile["resource"] is None
            else as_str(tile["resource"]).lower(),
            "number": tile["number"],
        }
    return result


def _check_color_roll(color: object, roll: object) -> None:
    if not isinstance(color, str) or color not in COLORS:
        raise ValueError(f"invalid canonical color: {color!r}")
    if type(roll) is not int or not 2 <= roll <= 12:
        raise ValueError(f"invalid dice roll: {roll!r}")


def dice_production(contract: JsonDict, color: str, roll: int) -> dict[str, int]:
    """Count settlement=1/city=2 production, blocked by the robber, ignoring bank supply."""

    _check_color_roll(color, roll)
    tiles = _contract_tiles(contract)
    players = contract.get("players")
    if not isinstance(players, list) or not players:
        raise ValueError("contract must identify its players")
    colors = set()
    for player in players:
        owner = player.get("color") if isinstance(player, dict) else None
        if not isinstance(owner, str) or owner not in COLORS or owner in colors:
            raise ValueError(f"invalid or duplicate contract player: {owner!r}")
        colors.add(owner)
    if color not in colors:
        raise ValueError(f"color {color!r} is not a contract player")
    nodes = _indexed_rows(contract, "nodes", {int(token[2:-1]) for token in _topology()[0]})
    result = dict.fromkeys(RESOURCE_KEYS, 0)
    for node_id, node in nodes.items():
        token = f"<N{node_id:02d}>"
        if "token" in node and node["token"] != token:
            raise ValueError(f"noncanonical node token for {node_id}")
        touching = {int(tile[2:-1]) for tile in node_tile_tokens(token)}
        _check_ids(node.get("adjacent_tiles"), touching, "node adjacent_tiles")
        if not {"building", "color"} <= node.keys():
            raise ValueError(f"missing building facts for {token}")
        building, owner = node["building"], node["color"]
        if building is None and owner is None:
            continue
        if building not in (SETTLEMENT, CITY) or not isinstance(owner, str) or owner not in colors:
            raise ValueError(f"invalid building/owner for {token}")
        if owner == color:
            for tile_id in touching:
                tile = tiles[tile_id]
                if (
                    tile["resource"] is not None
                    and tile["number"] == roll
                    and not tile["has_robber"]
                ):
                    result[as_str(tile["resource"]).lower()] += 2 if building == CITY else 1
    return result
