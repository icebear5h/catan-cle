"""Typed field readers for rebuilding config dataclasses from decoded JSON.

`SomeConfig(**payload)` on a `JsonDict` cannot be type-checked: every value is a
`JsonValue`. Each reader narrows one field to its declared type. Absent fields
take the caller's default (exactly what omitting the keyword did), and numbers
pass through unchanged so `asdict` digests stay byte-identical.
"""

from __future__ import annotations

from collections.abc import Iterable

from sft.json_types import JsonDict

__all__ = [
    "bool_field",
    "float_field",
    "int_field",
    "opt_int_field",
    "opt_str_field",
    "reject_dropped",
    "reject_unknown",
    "str_field",
]


def _bad(name: str, value: object, expected: str) -> TypeError:
    return TypeError(f"config field {name!r} must be {expected}, got {type(value).__name__}")


def str_field(payload: JsonDict, name: str, default: str) -> str:
    value = payload.get(name, default)
    if not isinstance(value, str):
        raise _bad(name, value, "a string")
    return value


def opt_str_field(payload: JsonDict, name: str, default: str | None) -> str | None:
    value = payload.get(name, default)
    if value is not None and not isinstance(value, str):
        raise _bad(name, value, "a string or null")
    return value


def bool_field(payload: JsonDict, name: str, default: bool) -> bool:
    value = payload.get(name, default)
    if not isinstance(value, bool):
        raise _bad(name, value, "a boolean")
    return value


def int_field(payload: JsonDict, name: str, default: int) -> int:
    value = payload.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise _bad(name, value, "an integer")
    return value


def opt_int_field(payload: JsonDict, name: str, default: int | None) -> int | None:
    value = payload.get(name, default)
    if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
        raise _bad(name, value, "an integer or null")
    return value


def float_field(payload: JsonDict, name: str, default: float) -> float:
    value = payload.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _bad(name, value, "a number")
    return value


def reject_unknown(payload: JsonDict, allowed: Iterable[str], owner: str) -> None:
    """Raise `TypeError` for keys the dataclass constructor would reject."""
    unknown = sorted(set(payload) - set(allowed))
    if unknown:
        raise TypeError(f"{owner} got unexpected fields {unknown}")


def reject_dropped(config: object, payload: JsonDict, owner: str) -> None:
    """Guard an explicit field list against its dataclass growing a field."""
    dropped = sorted(k for k in payload if getattr(config, k) != payload[k])
    if dropped:
        raise TypeError(f"{owner} builder does not carry fields {dropped}")
