"""Fact schema, canonical ordering, digests, and shared scalar codecs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Iterable, Sequence, TypedDict

from evals.json_types import JsonDict as JsonDict
from evals.json_types import JsonValue, as_dicts, as_int, as_list, as_str

REPRESENTATION_NAMES = (
    "verbose_json",
    "compact_json",
    "graph_dsl",
    "spatial_ascii",
)
FACT_SCHEMA = "catan_public_board_facts/v1"


class BoardFactTile(TypedDict):
    token: str
    coord: list[int]
    resource: str
    number: int | None
    has_robber: bool


class BoardFactNode(TypedDict):
    token: str
    color: str
    building: str


class BoardFactEdge(TypedDict):
    token: str
    nodes: list[str]
    road_color: str


class BoardFactPort(TypedDict):
    token: str
    coord: list[int]
    direction: str
    resource: str
    ratio: str
    nodes: list[str]


class BoardFacts(TypedDict):
    """The target-neutral public fact set every representation round-trips."""

    schema: str
    defaults: dict[str, str]
    tiles: list[BoardFactTile]
    nodes: list[BoardFactNode]
    edges: list[BoardFactEdge]
    ports: list[BoardFactPort]


def _ints(value: JsonValue, label: str) -> list[int]:
    return [as_int(item, label) for item in as_list(value, label)]


def _strs(value: JsonValue, label: str) -> list[str]:
    return [as_str(item, label) for item in as_list(value, label)]


def _optional_int(value: JsonValue, label: str) -> int | None:
    return None if value is None else as_int(value, label)


def public_board_facts(contract: Mapping[str, JsonValue]) -> BoardFacts:
    """Normalize the common fact set needed by the visual question suite.

    Nodes and edges are sparse dynamic state. Explicit defaults make omission
    equivalent to EMPTY while avoiding full fixed-atlas topology in only one
    representation.
    """
    tiles: list[BoardFactTile] = [
        {
            "token": as_str(tile["token"], "tile token"),
            "coord": _ints(tile["coord"], "tile coord"),
            "resource": as_str(tile.get("resource") or "DESERT", "tile resource"),
            "number": _optional_int(tile.get("number"), "tile number"),
            "has_robber": bool(tile.get("has_robber")),
        }
        for tile in as_dicts(contract["tiles"], "contract tiles")
    ]
    nodes: list[BoardFactNode] = [
        {
            "token": as_str(node["token"], "node token"),
            "color": as_str(node["color"], "node color"),
            "building": as_str(node["building"], "node building"),
        }
        for node in as_dicts(contract["nodes"], "contract nodes")
        if node.get("building") is not None
    ]
    edges: list[BoardFactEdge] = [
        {
            "token": as_str(edge["token"], "edge token"),
            "nodes": _strs(edge["node_tokens"], "edge node_tokens"),
            "road_color": as_str(edge["road_color"], "edge road_color"),
        }
        for edge in as_dicts(contract["edges"], "contract edges")
        if edge.get("road_color") is not None
    ]
    ports: list[BoardFactPort] = [
        {
            "token": as_str(port["token"], "port token"),
            "coord": _ints(port["coord"], "port coord"),
            "direction": as_str(port["direction"], "port direction"),
            "resource": as_str(port.get("resource") or "GENERIC", "port resource"),
            "ratio": as_str(port["ratio"], "port ratio"),
            "nodes": _strs(port["attached_node_tokens"], "port attached_node_tokens"),
        }
        for port in as_dicts(contract["ports"], "contract ports")
    ]
    return canonicalize_facts(
        {
            "schema": FACT_SCHEMA,
            "defaults": {"node": "EMPTY", "edge": "EMPTY"},
            "tiles": tiles,
            "nodes": nodes,
            "edges": edges,
            "ports": ports,
        }
    )


def canonicalize_facts(facts: BoardFacts) -> BoardFacts:
    return {
        "schema": facts["schema"],
        "defaults": {
            "node": facts["defaults"]["node"],
            "edge": facts["defaults"]["edge"],
        },
        "tiles": sorted(facts["tiles"], key=lambda item: item["token"]),
        "nodes": sorted(facts["nodes"], key=lambda item: item["token"]),
        "edges": sorted(facts["edges"], key=lambda item: item["token"]),
        "ports": sorted(facts["ports"], key=lambda item: item["token"]),
    }


def fact_digest(facts: BoardFacts) -> str:
    encoded = json.dumps(
        canonicalize_facts(facts),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()



def _coord(values: Sequence[int]) -> str:
    return "(" + ",".join(str(value) for value in values) + ")"


def _parse_coord(value: str) -> list[int]:
    return [int(part) for part in value.strip("()").split(",")]


def _join(values: Iterable[str]) -> str:
    return ",".join(values)

