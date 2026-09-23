"""Task taxonomies, compass geometry, and strict JSON helpers."""

from __future__ import annotations

import json

from cle.game_engine.models.enums import RESOURCES
from cle.game_engine.models.player import Color
from sft.json_types import JsonDict, JsonValue

STATIC_TASKS = frozenset({
    "symbolic_direction", "symbolic_direction_choice", "symbolic_neighbors",
    "symbolic_incidence", "symbolic_oriented_step",
})
TRAIN_TASKS = STATIC_TASKS | frozenset({
    "symbolic_owned_nodes", "symbolic_owned_roads", "symbolic_piece_owner",
    "symbolic_owned_incident_roads", "symbolic_reachable", "symbolic_shortest_route",
    "symbolic_near_nodes", "symbolic_near", "symbolic_local_constraint",
    "symbolic_scene_tiles",
})
TRANSFER_TASKS = frozenset({
    "symbolic_settlement_locations", "symbolic_longest_lengths",
    "symbolic_longest_leaders", "symbolic_longest_award",
})
SYMBOLIC_TASKS = TRAIN_TASKS | TRANSFER_TASKS
COLORS = frozenset(c.value for c in Color)
RESOURCES_LOWER = frozenset(r.lower() for r in RESOURCES) | {"desert"}
OFFSETS = dict(zip(
    ("NORTH", "NORTHEAST", "SOUTHEAST", "SOUTH", "SOUTHWEST", "NORTHWEST"),
    ((0, -2), (1, -1), (1, 1), (0, 2), (-1, 1), (-1, -1)), strict=True,
))
DIRECTIONS = ("left", "right", "above", "below")
SET_FORMAT = "Output only the exact unordered set, separated by spaces; NONE if empty. No duplicates or explanation."
ROUTE_RULES = (
    "Use only existing roads of that color. Opponent buildings may be endpoints, "
    "including the start, but never interior transit vertices. "
)
TRAIL_RULES = (
    "A trail uses only one player's existing roads. Count edges, with no undirected "
    "edge reused; cycles and revisited vertices are allowed. Opponent buildings block "
    "interior transit but may be endpoints, including the start. "
)
SETTLEMENT_RULES = (
    "A board-legal settlement location is empty and has no building of any color at "
    "any adjacent node. In setup no road connection is required; in normal play at "
    "least one incident existing road must be owned by the queried player. Ignore "
    "hand, piece supply, and whose turn it is. "
)


class PhysicalStateError(ValueError):
    """Validly encoded source fails material supply or building-distance admission."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _keys(value: object, keys: set[str], label: str) -> None:
    _require(isinstance(value, dict) and set(value) == keys, f"invalid {label} keys")


def _unique_object(pairs: list[tuple[str, JsonValue]]) -> JsonDict:
    result: JsonDict = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _bad_number(value: str) -> None:
    raise ValueError(f"non-integer JSON number: {value}")


def strict_json(text: str) -> object:
    """JSON without duplicate keys, floating numbers, NaN, or trailing prose."""
    _require(isinstance(text, str), "expected JSON text")
    return json.loads(text, object_pairs_hook=_unique_object,
                      parse_float=_bad_number, parse_constant=_bad_number)
