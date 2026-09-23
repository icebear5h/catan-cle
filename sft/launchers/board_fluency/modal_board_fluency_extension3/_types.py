"""Typed resource/limit shapes and JSON narrowing shared by the r03/r04 extensions."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping
from decimal import Decimal
from typing import NotRequired, TypedDict

from sft.json_types import JsonDict, JsonList, JsonValue, as_dict, as_int


class Resource(TypedDict):
    cpu: int
    memory_gib: int
    gpu: NotRequired[str]


class Limits(TypedDict):
    stage_seconds: dict[str, int]
    startup_seconds: int
    coordinator_seconds: int
    absolute_seconds: int
    reservation_seconds: int
    resources: dict[str, Resource]
    retries: int
    max_containers: int
    scaledown_seconds: int


def dig(value: JsonValue, *keys: str) -> JsonValue:
    """Index nested JSON objects, raising `TypeError` where a level is not an object."""
    for key in keys:
        value = as_dict(value)[key]
    return value


def json_list(items: Iterable[JsonValue]) -> JsonList:
    """Copy already-JSON items (for example `list[str]`) into an invariant JSON array."""
    return list(items)


def json_dict(items: Mapping[str, JsonValue]) -> JsonDict:
    """Copy an already-JSON mapping (for example a `Counter`) into a JSON object."""
    return dict(items)


def counts(value: JsonValue) -> Counter[str]:
    """Narrow a JSON object of integer tallies to a `Counter`."""
    return Counter({key: as_int(count) for key, count in as_dict(value).items()})


def decimal_receipt(text: str) -> dict[str, object]:
    """Decode a cost receipt with exact `Decimal` floats; it must be a JSON object."""
    receipt: object = json.loads(text, parse_float=Decimal)
    if not isinstance(receipt, dict):
        raise TypeError(f"Expected a JSON object receipt, got {type(receipt).__name__}")
    return receipt


def receipt_number(receipt: dict[str, object], *keys: str) -> float:
    """Read a nested numeric receipt field (parsed as `Decimal`) as a float."""
    value: object = receipt
    for key in keys:
        if not isinstance(value, dict):
            raise TypeError(f"Expected a JSON object at {key!r}, got {type(value).__name__}")
        value = value[key]
    if isinstance(value, bool) or not isinstance(value, Decimal | int | float):
        raise TypeError(f"Expected a JSON number, got {type(value).__name__}")
    return float(value)


def recorded_app_id(value: object) -> str:
    """Narrow a recorded Modal app ID, failing exactly as `stop_app` does for a non-string."""
    if not isinstance(value, str):
        raise ValueError("a recorded Modal app ID is required to stop this run")
    return value
