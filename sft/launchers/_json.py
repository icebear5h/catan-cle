"""Nested JSON receipt access for the spatial launchers.

Plans, receipts and checkpoint sidecars are decoded JSON. `at` walks a path of
object keys / array indices, narrowing each hop with `sft.json_types`, so a
malformed receipt fails with the same `KeyError`/`IndexError` (or a `TypeError`
where the old code would have hit one a line later) instead of being unchecked.
"""

from __future__ import annotations

from sft.json_types import JsonDict, JsonList, JsonValue, as_dict, as_float, as_int, as_list, as_str

__all__ = ["at", "at_dict", "at_float", "at_int", "at_list", "at_str"]


def _step(value: object, key: str | int) -> JsonValue:
    return as_list(value)[key] if isinstance(key, int) else as_dict(value)[key]


def at(value: object, key: str | int, *keys: str | int) -> JsonValue:
    """Return `value[key][keys[0]]...`, narrowing every container on the way."""
    current = _step(value, key)
    for next_key in keys:
        current = _step(current, next_key)
    return current


def at_dict(value: object, key: str | int, *keys: str | int) -> JsonDict:
    return as_dict(at(value, key, *keys))


def at_list(value: object, key: str | int, *keys: str | int) -> JsonList:
    return as_list(at(value, key, *keys))


def at_str(value: object, key: str | int, *keys: str | int) -> str:
    return as_str(at(value, key, *keys))


def at_int(value: object, key: str | int, *keys: str | int) -> int:
    return as_int(at(value, key, *keys))


def at_float(value: object, key: str | int, *keys: str | int) -> float:
    return as_float(at(value, key, *keys))
