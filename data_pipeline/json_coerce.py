"""Narrow decoded JSON values to the concrete field types the API client uses.

Every coercer raises `TypeError` on a value it cannot honour. The parsing loops
already wrap each item in `try/except Exception`, so a malformed row is skipped
with the existing warning instead of populating a wrongly typed dataclass.
"""

from data_pipeline.json_types import JsonDict, JsonList, JsonValue


def as_int(value: JsonValue) -> int:
    """Return a JSON number as an int, accepting the numeric-string form too."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError(f"expected an integer, got {type(value).__name__}")
    return int(value)


def as_int_or_none(value: JsonValue) -> int | None:
    return None if value is None else as_int(value)


def as_float_or_none(value: JsonValue) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError(f"expected a number, got {type(value).__name__}")
    return float(value)


def as_float(value: JsonValue) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError(f"expected a number, got {type(value).__name__}")
    return float(value)


def as_str(value: JsonValue) -> str:
    if not isinstance(value, str):
        raise TypeError(f"expected a string, got {type(value).__name__}")
    return value


def as_dict(value: JsonValue) -> JsonDict:
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object, got {type(value).__name__}")
    return value


def as_list(value: JsonValue) -> JsonList:
    if not isinstance(value, list):
        raise TypeError(f"expected a JSON array, got {type(value).__name__}")
    return value


def as_dict_list(value: JsonValue) -> list[JsonDict]:
    return [as_dict(item) for item in as_list(value)]


def as_str_list(value: JsonValue) -> list[str]:
    return [as_str(item) for item in as_list(value)]


__all__ = [
    "as_dict",
    "as_dict_list",
    "as_float",
    "as_float_or_none",
    "as_int",
    "as_int_or_none",
    "as_list",
    "as_str",
    "as_str_list",
]
