"""Typed JSON artifact helpers shared by the Pi completion scripts."""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

from evals.json_types import JsonDict, JsonValue

__all__ = [
    "append_jsonl",
    "object_list",
    "read_json",
    "read_jsonl",
    "string_list",
    "utc_now",
    "write_json",
    "write_jsonl",
]


def read_json(path: Path) -> JsonDict:
    """Load one JSON object, rejecting any other top-level shape."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def read_jsonl(path: Path) -> list[JsonDict]:
    """Load JSONL rows, treating a missing file as an empty artifact."""
    if not path.exists():
        return []
    rows: list[JsonDict] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        rows.append(row)
    return rows


def write_json(path: Path, payload: JsonValue) -> None:
    """Write indented JSON through a temporary file for atomic replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[JsonDict]) -> None:
    """Rewrite a JSONL artifact through a temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def append_jsonl(path: Path, rows: Iterable[JsonDict]) -> None:
    """Append rows to a JSONL artifact, flushing before returning."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()


def utc_now() -> str:
    """Return the current UTC timestamp in ISO 8601 form."""
    return datetime.now(timezone.utc).isoformat()


def object_list(payload: JsonDict, key: str, source: str) -> list[JsonDict]:
    """Read ``key`` from ``payload`` as a list of JSON objects."""
    records = payload[key]
    if not isinstance(records, list):
        raise ValueError(f"{source} is missing a {key} list")
    rows: list[JsonDict] = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"{source} {key} entries must be JSON objects")
        rows.append(record)
    return rows


def string_list(payload: JsonDict, key: str, source: str) -> list[str]:
    """Read ``key`` from ``payload`` as a list of strings."""
    records = payload[key]
    if not isinstance(records, list):
        raise ValueError(f"{source} is missing a {key} list")
    values: list[str] = []
    for record in records:
        if not isinstance(record, str):
            raise ValueError(f"{source} {key} entries must be strings")
        values.append(record)
    return values
