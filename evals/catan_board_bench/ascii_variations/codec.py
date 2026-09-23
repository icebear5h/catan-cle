"""Scalar record encoding and deterministic JSON serialization."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from evals.json_types import JsonList, JsonValue


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _json_list(values: Iterable[JsonValue]) -> JsonList:
    """Copy typed JSON items into a plain JSON list (same items, same order)."""

    return list(values)


def _token_or_none(value: str | None) -> str | None:
    return None if value is None else f"<{value}>"


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


def _nullable(value: object) -> str:
    return "-" if value is None else str(value)


def _parse_nullable(value: str) -> str | None:
    return None if value == "-" else value


def _parse_nullable_int(value: str) -> int | None:
    return None if value == "-" else int(value)


def _json_digest(value: object) -> str:
    payload = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()
