"""The single failure type raised when model output cannot select an action."""

from __future__ import annotations


class PlayerResponseParseError(ValueError):
    """Raised when model output cannot select an authorized action."""


def parse_json_integer(value: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise PlayerResponseParseError("JSON integer is too large.") from exc
