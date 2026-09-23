"""Typed shapes for the extension launchers' fixed configuration and JSON reads."""

from __future__ import annotations

from typing import NotRequired, TypedDict

from sft.json_types import JsonValue, as_dict, as_list


class Resources(TypedDict):
    """Per-stage container shape; CPU-only stages omit `gpu`."""

    cpu: int
    memory_gib: int
    gpu: NotRequired[str]


class Limits(TypedDict):
    """Launch caps recorded verbatim in every plan."""

    stage_seconds: dict[str, int]
    startup_seconds: int
    coordinator_seconds: int
    absolute_seconds: int
    reservation_seconds: int
    resources: dict[str, Resources]
    retries: int
    max_containers: int
    scaledown_seconds: int


def at(value: JsonValue, *keys: str | int) -> JsonValue:
    """Follow `keys` through nested JSON objects (str keys) and arrays (int keys).

    Raises `TypeError` when a level is not the container its key implies, the
    same failure a plain subscript chain on decoded JSON would have raised.
    """
    for key in keys:
        value = as_list(value)[key] if isinstance(key, int) else as_dict(value)[key]
    return value


def as_number(value: object) -> float:
    """Narrow a JSON number without converting it, so an `int` stays an `int`."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"Expected a JSON number, got {type(value).__name__}")
    return value
