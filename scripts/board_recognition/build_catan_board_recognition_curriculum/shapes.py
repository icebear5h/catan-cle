"""Concrete JSON shapes, class vocabularies, and narrowing helpers."""

from __future__ import annotations

from collections.abc import Iterable

from cle.players.data import JsonValue
from sft.scripts.builders.build_node_factor_dataset import AtlasIndices

JsonDict = dict[str, JsonValue]

DATASET_SCHEMA = "catan_board_recognition_dataset/v1"
SAMPLE_SCHEMA = "catan_board_recognition_sample/v1"
LABEL_SCHEMA = "catan_board_recognition_dense_labels/v1"
ENTITY_TYPES = ("tile", "node", "edge", "port")
RESOURCE_CLASSES = ("DESERT", "WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")
NUMBER_CLASSES = (None, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12)
PORT_CLASSES = (
    "THREE_TO_ONE",
    "TWO_TO_ONE_WOOD",
    "TWO_TO_ONE_BRICK",
    "TWO_TO_ONE_SHEEP",
    "TWO_TO_ONE_WHEAT",
    "TWO_TO_ONE_ORE",
)

__all__ = [
    "DATASET_SCHEMA",
    "ENTITY_TYPES",
    "LABEL_SCHEMA",
    "NUMBER_CLASSES",
    "PORT_CLASSES",
    "RESOURCE_CLASSES",
    "SAMPLE_SCHEMA",
    "JsonDict",
    "BoardIndices",
    "edge_ids_from",
    "edge_pair",
    "integer",
    "number",
    "obj",
    "objs",
    "strings",
    "text",
    "values",
]


BoardIndices = AtlasIndices


def edge_ids_from(indices: BoardIndices) -> list[tuple[int, int]]:
    """Read the atlas edge ids out of the builder index structure."""
    raw = indices["edges"]
    if not isinstance(raw, Iterable):
        raise ValueError("indices edges is not iterable")
    pairs: list[tuple[int, int]] = []
    for item in raw:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError("indices edges must hold node-id pairs")
        first, second = item
        if not isinstance(first, int) or not isinstance(second, int):
            raise ValueError("indices edges must hold node-id pairs")
        pairs.append((first, second))
    return sorted(pairs)


def number(value: JsonValue, label: str) -> float:
    """Require a JSON number at ``label``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} is not a number")
    return float(value)


def obj(value: JsonValue, label: str) -> JsonDict:
    """Require a JSON object at ``label``."""
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    return value


def values(value: JsonValue, label: str) -> list[JsonValue]:
    """Require a JSON list at ``label``."""
    if not isinstance(value, list):
        raise ValueError(f"{label} is not a list")
    return value


def objs(value: JsonValue, label: str) -> list[JsonDict]:
    """Require a JSON list of objects at ``label``."""
    return [obj(item, f"{label} entry") for item in values(value, label)]


def strings(value: JsonValue, label: str) -> list[str]:
    """Require a JSON list of strings at ``label``."""
    return [text(item, f"{label} entry") for item in values(value, label)]


def text(value: JsonValue, label: str) -> str:
    """Require a JSON string at ``label``."""
    if not isinstance(value, str):
        raise ValueError(f"{label} is not a string")
    return value


def integer(value: JsonValue, label: str) -> int:
    """Require a JSON integer at ``label``."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} is not an integer")
    return value


def edge_pair(value: JsonValue, label: str) -> tuple[int, int]:
    """Require a two-element integer edge id at ``label``."""
    pair = values(value, label)
    if len(pair) != 2:
        raise ValueError(f"{label} must have two endpoints")
    return integer(pair[0], label), integer(pair[1], label)
