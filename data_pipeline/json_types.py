"""Concrete JSON type aliases shared across the data pipeline.

`JsonDict` used to be spelled `dict[str, Any]` in a dozen modules. The strict
type gate rejects explicit `Any`, so the alias now names the real JSON value
domain and callers narrow before using a value as a number or a string.
"""

from typing import TypeAlias

JsonValue: TypeAlias = (
    None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
)
JsonDict: TypeAlias = dict[str, JsonValue]
JsonList: TypeAlias = list[JsonValue]

__all__ = ["JsonDict", "JsonList", "JsonValue"]
