"""Typed shapes of the canonical full public graph and their JSON narrowing guards."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TypedDict, TypeGuard, TypeVar

from evals.json_types import JsonDict

_Item = TypeVar("_Item")


class FullTile(TypedDict):
    id: str
    cube: list[int]
    resource: str
    number: int | None
    robber: bool
    corners: dict[str, str]
    sides: dict[str, str]


class FullNode(TypedDict):
    id: str
    color: str | None
    building: str | None
    tiles: list[str]
    edges: list[str]
    ports: list[str]


class FullEdge(TypedDict):
    id: str
    nodes: list[str]
    road: str | None
    tiles: list[str]


class FullPort(TypedDict):
    id: str
    cube: list[int]
    direction: str
    resource: str
    ratio: str
    nodes: list[str]


class FullFacts(TypedDict):
    schema: str
    tiles: list[FullTile]
    nodes: list[FullNode]
    edges: list[FullEdge]
    ports: list[FullPort]


class MinimalNode(TypedDict):
    id: str
    color: str | None
    building: str | None


class MinimalEdge(TypedDict):
    id: str
    nodes: list[str]
    road: str | None


class MinimalGraph(TypedDict):
    schema: str
    tiles: list[FullTile]
    nodes: list[MinimalNode]
    edges: list[MinimalEdge]
    ports: list[FullPort]


class AliasMap(TypedDict):
    sample_id: str
    seed: int
    tiles: dict[str, str]
    nodes: dict[str, str]
    edges: dict[str, str]
    ports: dict[str, str]


class AsciiBoard(TypedDict):
    """One selected source board of the twelve-board smoke probe."""

    sample_id: str
    facts: FullFacts
    contract: JsonDict
    contract_path: str
    digest: str


def _has(value: object, *keys: str) -> TypeGuard[Mapping[str, object]]:
    return isinstance(value, dict) and all(key in value for key in keys)


def _is_str_list(value: object) -> TypeGuard[list[str]]:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _is_int_list(value: object) -> TypeGuard[list[int]]:
    return isinstance(value, list) and all(isinstance(item, int) for item in value)


def _is_str_map(value: object) -> TypeGuard[dict[str, str]]:
    return isinstance(value, dict) and all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    )


def _is_optional_str(value: object) -> bool:
    return value is None or isinstance(value, str)


def is_full_tile(value: object) -> TypeGuard[FullTile]:
    return (
        _has(value, "id", "cube", "resource", "number", "robber", "corners", "sides")
        and isinstance(value["id"], str)
        and _is_int_list(value["cube"])
        and isinstance(value["resource"], str)
        and (value["number"] is None or isinstance(value["number"], int))
        and isinstance(value["robber"], bool)
        and _is_str_map(value["corners"])
        and _is_str_map(value["sides"])
    )


def is_minimal_node(value: object) -> TypeGuard[MinimalNode]:
    return (
        _has(value, "id", "color", "building")
        and isinstance(value["id"], str)
        and _is_optional_str(value["color"])
        and _is_optional_str(value["building"])
    )


def is_full_node(value: object) -> TypeGuard[FullNode]:
    return (
        is_minimal_node(value)
        and _has(value, "tiles", "edges", "ports")
        and _is_str_list(value["tiles"])
        and _is_str_list(value["edges"])
        and _is_str_list(value["ports"])
    )


def is_minimal_edge(value: object) -> TypeGuard[MinimalEdge]:
    return (
        _has(value, "id", "nodes", "road")
        and isinstance(value["id"], str)
        and _is_str_list(value["nodes"])
        and _is_optional_str(value["road"])
    )


def is_full_edge(value: object) -> TypeGuard[FullEdge]:
    return is_minimal_edge(value) and _has(value, "tiles") and _is_str_list(value["tiles"])


def is_full_port(value: object) -> TypeGuard[FullPort]:
    return (
        _has(value, "id", "cube", "direction", "resource", "ratio", "nodes")
        and isinstance(value["id"], str)
        and _is_int_list(value["cube"])
        and isinstance(value["direction"], str)
        and isinstance(value["resource"], str)
        and isinstance(value["ratio"], str)
        and _is_str_list(value["nodes"])
    )


def typed_items(
    value: object,
    guard: Callable[[object], TypeGuard[_Item]],
    label: str,
) -> list[_Item]:
    """Narrow a list whose every item must satisfy ``guard``; items are not copied."""

    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    items = [item for item in value if guard(item)]
    if len(items) != len(value):
        raise ValueError(f"invalid {label} item")
    return items


def full_facts_from_json(value: object) -> FullFacts:
    """Narrow a decoded ``catan_full_public_graph/v1`` fact object."""

    if not _has(value, "schema", "tiles", "nodes", "edges", "ports"):
        raise ValueError("full public graph facts must be an object with all collections")
    schema = value["schema"]
    if not isinstance(schema, str):
        raise ValueError("full public graph schema must be a string")
    return {
        "schema": schema,
        "tiles": typed_items(value["tiles"], is_full_tile, "tiles"),
        "nodes": typed_items(value["nodes"], is_full_node, "nodes"),
        "edges": typed_items(value["edges"], is_full_edge, "edges"),
        "ports": typed_items(value["ports"], is_full_port, "ports"),
    }
