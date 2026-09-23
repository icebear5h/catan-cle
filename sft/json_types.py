"""Recursive JSON value typing and narrowing helpers shared across SFT modules.

`json.load`/`json.loads` hand back `Any`, which strict mypy rejects under
`disallow_any_explicit`. Annotating decoded payloads as `JsonValue` keeps the
shape honest; the `as_*` helpers narrow a `JsonValue` to the concrete type a
call site already assumed. They raise `TypeError` when the payload disagrees,
which is the same failure mode the unannotated code had (an `AttributeError`
or `TypeError` a line or two later), only earlier and with a better message.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TypeAlias

JsonValue: TypeAlias = (
    None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
)
JsonDict: TypeAlias = dict[str, JsonValue]
JsonList: TypeAlias = list[JsonValue]

# `JsonValue` is what `json.load` hands back, so its containers are invariant:
# a `list[int]` is not a `list[JsonValue]`. `JsonLike` is the write-side twin
# built from covariant `Sequence`/`Mapping`, so already-typed payloads
# (`list[int]`, `dict[str, str]`, ...) can be stored without copying them.
JsonLike: TypeAlias = (
    None
    | bool
    | int
    | float
    | str
    | Sequence["JsonLike"]
    | Mapping[str, "JsonLike"]
)
JsonLikeDict: TypeAlias = dict[str, JsonLike]

__all__ = [
    "JsonDict",
    "JsonLike",
    "JsonLikeDict",
    "JsonList",
    "JsonValue",
    "as_bool",
    "as_dict",
    "as_float",
    "as_int",
    "as_list",
    "as_str",
    "dict_items",
    "dump_json",
    "load_json",
    "load_json_dict",
    "load_json_list",
    "loads_json",
    "json_dict",
    "json_list",
    "json_path",
    "opt_dict",
    "opt_float",
    "opt_str",
    "str_keys",
]


def _reject(value: object, expected: str) -> TypeError:
    return TypeError(f"Expected a JSON {expected}, got {type(value).__name__}")


def as_dict(value: object) -> JsonDict:
    """Narrow `value` to a JSON object."""
    if not isinstance(value, dict):
        raise _reject(value, "object")
    return value


def as_list(value: object) -> JsonList:
    """Narrow `value` to a JSON array."""
    if not isinstance(value, list):
        raise _reject(value, "array")
    return value


def as_str(value: object) -> str:
    """Narrow `value` to a JSON string."""
    if not isinstance(value, str):
        raise _reject(value, "string")
    return value


def as_int(value: object) -> int:
    """Narrow `value` to a JSON integer (JSON has no separate int type)."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _reject(value, "number")
    return int(value)


def as_float(value: object) -> float:
    """Narrow `value` to a JSON number."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _reject(value, "number")
    return float(value)


def as_bool(value: object) -> bool:
    """Narrow `value` to a JSON boolean, accepting truthy JSON scalars."""
    if not isinstance(value, bool):
        raise _reject(value, "boolean")
    return value


def opt_dict(value: object) -> JsonDict | None:
    """Narrow `value` to a JSON object, passing `None` through."""
    if value is None:
        return None
    return as_dict(value)


def opt_str(value: object) -> str | None:
    """Narrow `value` to a JSON string, passing `None` through."""
    if value is None:
        return None
    return as_str(value)


def opt_float(value: object) -> float | None:
    """Narrow `value` to a JSON number, passing `None` through."""
    if value is None:
        return None
    return as_float(value)


def json_dict(values: Mapping[str, JsonValue]) -> JsonDict:
    """Copy an already-typed mapping (`dict[str, int]`, `Counter`, ...) into a `JsonDict`."""
    return dict(values)


def json_list(values: Iterable[JsonValue]) -> JsonList:
    """Copy already-typed JSON values into a `JsonList`.

    `list` is invariant, so a `list[str]` is not a `JsonList`; `Iterable` is
    covariant, so this accepts any iterable of JSON values.
    """
    return list(values)


def json_path(value: object, key: str, *keys: str) -> JsonValue:
    """Follow nested object keys, narrowing each level to a JSON object.

    A missing key raises `KeyError`, exactly like chained subscripts would.
    """
    current = as_dict(value)[key]
    for next_key in keys:
        current = as_dict(current)[next_key]
    return current


def str_keys(value: object) -> dict[str, str]:
    """Narrow `value` to a JSON object whose values are all strings."""
    return {key: as_str(item) for key, item in as_dict(value).items()}


def dict_items(value: object) -> list[tuple[str, JsonDict]]:
    """Narrow `value` to a JSON object of JSON objects, as an item list."""
    return [(key, as_dict(item)) for key, item in as_dict(value).items()]


def loads_json(text: str | bytes) -> JsonValue:
    """Decode JSON text into a `JsonValue`."""
    decoded: JsonValue = json.loads(text)
    return decoded


def load_json(path: str | Path) -> JsonValue:
    """Read and decode a JSON file into a `JsonValue`."""
    return loads_json(Path(path).read_text(encoding="utf-8"))


def load_json_dict(path: str | Path) -> JsonDict:
    """Read a JSON file that must contain an object."""
    return as_dict(load_json(path))


def load_json_list(path: str | Path) -> JsonList:
    """Read a JSON file that must contain an array."""
    return as_list(load_json(path))


def dump_json(value: JsonLike, *, indent: int | None = 2) -> str:
    """Serialize a `JsonValue` with the project's default formatting."""
    return json.dumps(value, indent=indent)
