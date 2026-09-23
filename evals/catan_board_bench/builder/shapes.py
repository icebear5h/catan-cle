"""Narrowing reads over public-board contracts and QA rows (plain JSON objects)."""

from __future__ import annotations

from collections.abc import Iterable

from evals.json_types import JsonDict, JsonList, JsonValue, as_dict, as_dicts, as_list, as_str


def rows(record: JsonDict, key: str) -> list[JsonDict]:
    """The JSON-object list stored at ``record[key]``."""
    return as_dicts(record[key], key)


def member(record: JsonDict, key: str) -> JsonDict:
    """The JSON object stored at ``record[key]``."""
    return as_dict(record[key], key)


def text(value: JsonValue, label: str) -> str:
    """A JSON string used as answer text."""
    return as_str(value, label)


def texts(value: JsonValue, label: str) -> list[str]:
    """A JSON list of strings, such as a token list."""
    return [as_str(item, f"{label} entry") for item in as_list(value, label)]


def whole_number(value: JsonValue, label: str) -> int:
    """``int(value)`` for the scalar JSON values ``int()`` accepts."""
    if isinstance(value, (int, float, str)):
        return int(value)
    raise TypeError(f"{label} is not an integer-like JSON scalar: {type(value).__name__}")


def json_list(items: Iterable[JsonValue]) -> JsonList:
    """Copy concretely typed items into a JSON list with identical contents."""
    return list(items)


__all__ = ["json_list", "member", "rows", "text", "texts", "whole_number"]
