"""Deterministic balanced question targets and their graph-derived answers."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from typing import TypeVar

from cle.game_engine.models.player import Color
from evals.catan_board_bench.ascii_variations.codec import _token_or_none
from evals.catan_board_bench.ascii_variations.facts import (
    AsciiBoard,
    FullEdge,
    FullFacts,
    FullNode,
    FullPort,
)
from evals.json_types import JsonDict, as_str

_Item = TypeVar("_Item")


def _find_board_item(
    boards: Sequence[AsciiBoard],
    *,
    start: int,
    collection: str,
    items: Callable[[FullFacts], Sequence[_Item]],
    predicate: Callable[[_Item, FullFacts], bool],
) -> tuple[AsciiBoard, _Item]:
    """Pick a deterministic matching item from ``items(facts)``; ``collection`` labels it."""

    for offset in range(len(boards)):
        board = boards[(start + offset) % len(boards)]
        matches = [item for item in items(board["facts"]) if predicate(item, board["facts"])]
        if matches:
            return board, matches[(start * 7 + offset) % len(matches)]
    raise ValueError(f"no matching {collection} item")


def _fact_nodes(facts: FullFacts) -> list[FullNode]:
    return facts["nodes"]


def _fact_edges(facts: FullFacts) -> list[FullEdge]:
    return facts["edges"]


def _fact_ports(facts: FullFacts) -> list[FullPort]:
    return facts["ports"]


def _port_occupants(port: FullPort, facts: FullFacts) -> list[JsonDict]:
    nodes = {node["id"]: node for node in facts.get("nodes", [])}
    occupants: list[JsonDict] = []
    for node_id in port["nodes"]:
        node = nodes.get(node_id)
        if node and node["building"] is not None:
            occupants.append(
                {
                    "node": node_id,
                    "color": _token_or_none(node["color"]),
                    "building": _token_or_none(node["building"]),
                }
            )
    return sorted(occupants, key=lambda item: as_str(item["node"], "occupant node"))


def _production_candidates(
    boards: Sequence[AsciiBoard],
) -> list[tuple[AsciiBoard, int, list[JsonDict]]]:
    rolls = (2, 3, 4, 5, 6, 8, 9, 10, 11, 12)
    selected: list[tuple[AsciiBoard, int, list[JsonDict]]] = []
    used_boards: set[str] = set()
    used_rolls: dict[bool, set[int]] = {True: set(), False: set()}
    for index in range(6):
        want_payouts = index % 2 == 0
        found: tuple[AsciiBoard, int, list[JsonDict]] | None = None
        for offset in range(len(boards)):
            board = boards[(index + offset) % len(boards)]
            if board["sample_id"] in used_boards:
                continue
            candidates: list[tuple[AsciiBoard, int, list[JsonDict]]] = []
            for roll in rolls:
                payouts = _roll_payouts(board["facts"], roll)
                if bool(payouts) == want_payouts:
                    candidates.append((board, roll, payouts))
            if candidates:
                fresh = [
                    candidate
                    for candidate in candidates
                    if candidate[1] not in used_rolls[want_payouts]
                ]
                choices = fresh or candidates
                found = choices[(index * 3) % len(choices)]
                break
        if found is None:
            raise ValueError("production candidates are not balanced")
        selected.append(found)
        used_boards.add(found[0]["sample_id"])
        used_rolls[want_payouts].add(found[1])
    return selected


def _roll_payouts(facts: FullFacts, roll: int) -> list[JsonDict]:
    nodes = {node["id"]: node for node in facts["nodes"]}
    totals: Counter[tuple[str | None, str]] = Counter()
    for tile in facts["tiles"]:
        if tile["number"] != roll or tile["robber"] or tile["resource"] == "DESERT":
            continue
        for node_id in tile["corners"].values():
            node = nodes[node_id]
            if node["building"] is None:
                continue
            amount = 2 if node["building"] == "CITY" else 1
            totals[(node["color"], tile["resource"])] += amount
    return [
        {
            "color": _token_or_none(color),
            "resource": _token_or_none(resource),
            "count": count,
        }
        for (color, resource), count in sorted(totals.items())
    ]


def _building_count_candidates(
    boards: Sequence[AsciiBoard],
) -> list[tuple[AsciiBoard, str, int, int]]:
    selected: list[tuple[AsciiBoard, str, int, int]] = []
    used_boards: set[str] = set()
    for index in range(6):
        want_city = index % 2 == 0
        found: tuple[AsciiBoard, str, int, int] | None = None
        for offset in range(len(boards)):
            board = boards[(index + offset) % len(boards)]
            if board["sample_id"] in used_boards:
                continue
            colors = sorted(
                {color for node in board["facts"]["nodes"] if (color := node["color"])}
            )
            candidates: list[tuple[AsciiBoard, str, int, int]] = []
            for color in colors:
                settlements = sum(
                    node["color"] == color and node["building"] == "SETTLEMENT"
                    for node in board["facts"]["nodes"]
                )
                cities = sum(
                    node["color"] == color and node["building"] == "CITY"
                    for node in board["facts"]["nodes"]
                )
                if bool(cities) == want_city:
                    candidates.append((board, color, settlements, cities))
            if candidates:
                found = candidates[index % len(candidates)]
                break
        if found is None:
            raise ValueError("building count candidates are not balanced")
        selected.append(found)
        used_boards.add(found[0]["sample_id"])
    return selected


def _road_inventory_candidates(
    boards: Sequence[AsciiBoard],
) -> list[tuple[AsciiBoard, str, list[str]]]:
    colors = [color.value for color in Color]
    selected: list[tuple[AsciiBoard, str, list[str]]] = []
    used_boards: set[str] = set()
    for index in range(6):
        want_roads = index % 2 == 0
        found: tuple[AsciiBoard, str, list[str]] | None = None
        for offset in range(len(boards)):
            board = boards[(index + offset) % len(boards)]
            if board["sample_id"] in used_boards:
                continue
            candidates: list[tuple[AsciiBoard, str, list[str]]] = []
            for color in colors:
                edges = sorted(
                    edge["id"] for edge in board["facts"]["edges"] if edge["road"] == color
                )
                if bool(edges) == want_roads:
                    candidates.append((board, color, edges))
            if candidates:
                found = candidates[index % len(candidates)]
                break
        if found is None:
            raise ValueError("road inventory candidates are not balanced")
        selected.append(found)
        used_boards.add(found[0]["sample_id"])
    return selected


def _disconnected_node_pair(facts: FullFacts, offset: int) -> tuple[str, str]:
    connected = {tuple(sorted(edge["nodes"])) for edge in facts["edges"]}
    node_ids = [node["id"] for node in facts["nodes"]]
    candidates = [
        (left, right)
        for left_index, left in enumerate(node_ids)
        for right in node_ids[left_index + 1 :]
        if tuple(sorted((left, right))) not in connected
    ]
    return candidates[offset % len(candidates)]
