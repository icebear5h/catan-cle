"""Concrete JSON aliases shared by evaluation modules (no ``Any``)."""

from __future__ import annotations

from typing import TypeAlias

JsonValue: TypeAlias = (
    None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
)
JsonDict: TypeAlias = dict[str, JsonValue]
JsonList: TypeAlias = list[JsonValue]


def as_dict(value: JsonValue, label: str) -> JsonDict:
    """Require a JSON object at ``label``."""
    if not isinstance(value, dict):
        raise TypeError(f"{label} is not a JSON object: {type(value).__name__}")
    return value


def as_list(value: JsonValue, label: str) -> JsonList:
    """Require a JSON list at ``label``."""
    if not isinstance(value, list):
        raise TypeError(f"{label} is not a JSON list: {type(value).__name__}")
    return value


def as_dicts(value: JsonValue, label: str) -> list[JsonDict]:
    """Require a JSON list of objects at ``label``."""
    return [as_dict(item, f"{label} entry") for item in as_list(value, label)]


def dict_or_empty(value: JsonValue) -> JsonDict:
    """Return ``value`` when it is an object, else an empty object."""
    return value if isinstance(value, dict) else {}


def list_or_empty(value: JsonValue) -> JsonList:
    """Return ``value`` when it is a list, else an empty list."""
    return value if isinstance(value, list) else []


def as_str(value: JsonValue, label: str) -> str:
    """Require a JSON string at ``label``."""
    if not isinstance(value, str):
        raise TypeError(f"{label} is not a JSON string: {type(value).__name__}")
    return value


def as_int(value: JsonValue, label: str) -> int:
    """Require a JSON integer (bools included, as ``int()`` would accept them)."""
    if not isinstance(value, int):
        raise TypeError(f"{label} is not a JSON integer: {type(value).__name__}")
    return value


def as_number(value: JsonValue, label: str) -> int | float:
    """Require a JSON number without widening integers to floats."""
    if not isinstance(value, (int, float)):
        raise TypeError(f"{label} is not a JSON number: {type(value).__name__}")
    return value


__all__ = [
    "JsonDict",
    "JsonList",
    "JsonValue",
    "as_dict",
    "as_dicts",
    "as_int",
    "as_list",
    "as_number",
    "as_str",
    "dict_or_empty",
    "list_or_empty",
]
