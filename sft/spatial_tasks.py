"""Canonical atlas solvers and strict, task-specific spatial answer scoring."""

from __future__ import annotations

import json
from collections import deque
from functools import lru_cache

from cle.game_engine.models.enums import CITY, RESOURCES, SETTLEMENT
from cle.game_engine.models.player import Color
from cle.game_engine.public_board import PUBLIC_BOARD_CONTRACT_SCHEMA
from evals.catan_board_bench.tokens import atlas_metadata


TASK_TYPES = ("node_tiles", "shortest_node_path", "local_node_tiles", "dice_production")
RESOURCE_KEYS = tuple(resource.lower() for resource in RESOURCES)
COLORS = tuple(color.value for color in Color)


@lru_cache(maxsize=1)
def _topology() -> tuple[dict[str, set[str]], dict[str, list[str]], dict[int, dict]]:
    atlas = atlas_metadata()
    nodes = {node["id"]: node["token"] for node in atlas["nodes"]}
    graph = {token: set() for token in nodes.values()}
    for edge in atlas["edges"]:
        a, b = (nodes[node_id] for node_id in edge["id"])
        graph[a].add(b)
        graph[b].add(a)
    touching = {
        token: sorted(tile["token"] for tile in atlas["tiles"] if node_id in tile["nodes"].values())
        for node_id, token in nodes.items()
    }
    return graph, touching, {tile["id"]: tile for tile in atlas["tiles"]}


def atlas_node_graph() -> dict[str, set[str]]:
    """Return a detached copy of the canonical 54-node, 72-edge land graph."""

    return {node: set(neighbors) for node, neighbors in _topology()[0].items()}


def node_tile_tokens(node: str) -> list[str]:
    """Return the sorted canonical land-tile tokens touching a valid node."""

    touching = _topology()[1]
    if not isinstance(node, str) or node not in touching:
        raise ValueError(f"invalid canonical node: {node!r}")
    return list(touching[node])


def shortest_node_path(start: str, end: str) -> list[str]:
    """Return sorted-neighbor BFS's route, including both endpoints."""

    node_tile_tokens(start)
    node_tile_tokens(end)
    graph = _topology()[0]
    parents: dict[str, str | None] = {start: None}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        if node == end:
            path = []
            while node is not None:
                path.append(node)
                node = parents[node]
            return path[::-1]
        for neighbor in sorted(graph[node]):
            if neighbor not in parents:
                parents[neighbor] = node
                queue.append(neighbor)
    raise ValueError(f"no canonical path from {start!r} to {end!r}")


def _indexed_rows(contract: dict, key: str, ids: set[int]) -> dict[int, dict]:
    rows = contract.get(key)
    if not isinstance(rows, list):
        raise ValueError(f"contract {key} must be a complete list")
    indexed = {}
    for row in rows:
        if not isinstance(row, dict) or type(row.get("id")) is not int:
            raise ValueError(f"invalid contract {key} row: {row!r}")
        row_id = row["id"]
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


def _contract_tiles(contract: dict) -> dict[int, dict]:
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
        _check_ids(tile.get("nodes"), set(atlas_tile["nodes"].values()), "tile nodes")
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
    robber_id = robber["tile_id"]
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


def local_node_tiles(contract: dict, node: str) -> dict[str, dict]:
    """Read a node's resource/number neighborhood from an engine contract."""

    tokens = node_tile_tokens(node)
    tiles = _contract_tiles(contract)
    result = {}
    for token in tokens:
        tile = tiles[int(token[2:-1])]
        result[token] = {
            "resource": "desert" if tile["resource"] is None else tile["resource"].lower(),
            "number": tile["number"],
        }
    return result


def _check_color_roll(color: object, roll: object) -> None:
    if not isinstance(color, str) or color not in COLORS:
        raise ValueError(f"invalid canonical color: {color!r}")
    if type(roll) is not int or not 2 <= roll <= 12:
        raise ValueError(f"invalid dice roll: {roll!r}")


def dice_production(contract: dict, color: str, roll: int) -> dict[str, int]:
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
                    result[tile["resource"].lower()] += 2 if building == CITY else 1
    return result


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_json_number(value: str) -> None:
    raise ValueError(f"non-integer JSON number: {value}")


def _json_answer(text: str, task: str, keys: set[str]) -> dict:
    if not isinstance(text, str):
        raise ValueError("answer must be JSON text")
    answer = json.loads(
        text,
        object_pairs_hook=_unique_object,
        parse_float=_reject_json_number,
        parse_constant=_reject_json_number,
    )
    if not isinstance(answer, dict) or set(answer) != keys:
        raise ValueError("answer must be an object with exactly the required keys")
    for value in answer.values():
        if task == "dice_production":
            if type(value) is not int or value < 0:
                raise ValueError("production counts must be nonnegative integers")
        else:
            if not isinstance(value, dict) or set(value) != {"resource", "number"}:
                raise ValueError("tile answer must contain exactly resource and number")
            _check_resource_number(value["resource"], value["number"])
    return answer


def _token_answer(text: str, vocabulary: set[str]) -> list[str]:
    if not isinstance(text, str):
        raise ValueError("answer must be token text")
    tokens = text.split()
    if not tokens or len(tokens) != len(set(tokens)) or not set(tokens) <= vocabulary:
        raise ValueError("answer must contain distinct canonical tokens only")
    return tokens


def _valid_path(path: list[str], start: str, end: str, length: int) -> bool:
    graph = _topology()[0]
    return (
        len(path) == length
        and path[0] == start
        and path[-1] == end
        and all(b in graph[a] for a, b in zip(path, path[1:]))
    )


def score_spatial_task(expected: str, response: str, metadata: dict) -> dict | None:
    """Score the four named tasks; malformed gold/targets raise, other tasks return None.

    Direct callers get whitespace tolerance only. The evaluator applies its existing
    chat-wrapper normalization first. Dynamic gold comes from the contract solvers;
    this boundary validates its complete answer shape, not an inferred text prefix.
    """

    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a dictionary")
    task = metadata.get("task_type")
    if task is not None and not isinstance(task, str):
        raise ValueError("task_type must be a string")
    if task not in TASK_TYPES:
        return None
    target = metadata.get("target")
    target_keys = (
        {"start", "end"}
        if task == "shortest_node_path"
        else {"color", "roll"}
        if task == "dice_production"
        else {"node"}
    )
    if not isinstance(target, dict) or set(target) != target_keys:
        raise ValueError(f"invalid target for {task}: {target!r}")
    if task == "shortest_node_path":
        start, end = target["start"], target["end"]
        length = len(shortest_node_path(start, end))
        vocabulary = set(_topology()[0])
        gold = _token_answer(expected, vocabulary)
        if not _valid_path(gold, start, end, length):
            raise ValueError("gold is not a shortest path for its target")
    elif task == "node_tiles":
        keys = set(node_tile_tokens(target["node"]))
        vocabulary = {tile["token"] for tile in _topology()[2].values()}
        gold = _token_answer(expected, vocabulary)
        if set(gold) != keys:
            raise ValueError("gold tiles disagree with target node")
    else:
        if task == "dice_production":
            _check_color_roll(target["color"], target["roll"])
            keys = set(RESOURCE_KEYS)
        else:
            keys = set(node_tile_tokens(target["node"]))
        gold = _json_answer(expected, task, keys)
        if task == "dice_production" and target["roll"] == 7 and any(gold.values()):
            raise ValueError("a roll of seven cannot produce resources")

    response_norm = response.strip() if isinstance(response, str) else response
    try:
        if task in ("node_tiles", "shortest_node_path"):
            answer = _token_answer(response, vocabulary)
            correct = (
                set(answer) == set(gold)
                if task == "node_tiles"
                else _valid_path(answer, start, end, length)
            )
            response_norm = " ".join(sorted(answer) if task == "node_tiles" else answer)
        else:
            answer = _json_answer(response, task, keys)
            correct = answer == gold
            response_norm = json.dumps(answer, sort_keys=True)
    except ValueError:
        correct = False
    expected_norm = (
        json.dumps(gold, sort_keys=True)
        if isinstance(gold, dict)
        else " ".join(sorted(gold) if task == "node_tiles" else gold)
    )
    return {
        "correct": correct,
        "scoring": task,
        "expected_normalized": expected_norm,
        "response_normalized": response_norm,
    }
