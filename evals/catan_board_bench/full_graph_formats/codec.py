"""Shared scalar and static-record codecs for lossless graph formats."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeVar

_Value = TypeVar("_Value")


def _record_fields(line: str) -> dict[str, str]:
    parts = line.split("|")
    result = {"id": parts[1]}
    for part in parts[2:]:
        key, value = part.split("=", 1)
        if key in result:
            raise ValueError(f"duplicate integrated record field: {key}")
        result[key] = value
    return result


def _insert_unique(
    mapping: dict[str, _Value],
    key: str,
    value: _Value,
    label: str,
) -> None:
    if key in mapping:
        raise ValueError(f"duplicate {label}: {key}")
    mapping[key] = value


def _mapping(mapping: Mapping[str, object], order: Sequence[str]) -> str:
    return ",".join(f"{key}:{mapping[key]}" for key in order)


def _parse_mapping(value: str) -> dict[str, str]:
    return dict(item.split(":", 1) for item in value.split(","))


def _csv(values: Sequence[object]) -> str:
    return "-" if not values else ",".join(str(value) for value in values)


def _parse_csv(value: str) -> list[str]:
    return [] if value == "-" else value.split(",")


def _parse_int_csv(value: str) -> list[int]:
    return [int(item) for item in value.split(",")]


def _null_marker(value: object) -> str:
    return "-" if value is None else str(value)


def _parse_null_marker(value: str) -> str | None:
    return None if value == "-" else value


def _parse_optional_int(value: str) -> int | None:
    return None if value == "-" else int(value)


def _parse_binary(value: str) -> bool:
    if value not in {"0", "1"}:
        raise ValueError(f"expected binary value, found {value!r}")
    return value == "1"


def _parse_boolean_atom(value: object) -> bool:
    if value not in {"true", "false"}:
        raise ValueError(f"expected Datalog boolean atom, found {value!r}")
    return value == "true"
