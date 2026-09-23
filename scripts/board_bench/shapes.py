"""Concrete JSON shapes and narrowing helpers shared by the board-bench scripts."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path

from cle.players.data import JsonValue

JsonDict = dict[str, JsonValue]

__all__ = [
    "JsonDict",
    "integer",
    "markdown_table",
    "membership",
    "number",
    "numeric",
    "obj",
    "objs",
    "read_json",
    "read_json_object",
    "read_jsonl",
    "strings",
    "text",
    "values",
    "write_json",
    "write_jsonl",
    "write_text",
]


def numeric(value: JsonValue, label: str) -> int | float:
    """Read a JSON number without widening integers to floats."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value
    raise ValueError(f"{label} is not a number")


def obj(value: JsonValue, label: str) -> JsonDict:
    """Require a JSON object at ``label``."""
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    return value


def values(value: JsonValue, label: str) -> list[JsonValue]:
    """Require a JSON list at ``label``."""
    if not isinstance(value, list):
        raise ValueError(f"{label} is not a list")
    return value


def objs(value: JsonValue, label: str) -> list[JsonDict]:
    """Require a JSON list of objects at ``label``."""
    return [obj(item, f"{label} entry") for item in values(value, label)]


def membership(value: JsonValue, label: str) -> frozenset[str]:
    """Read a JSON list or object as the set its ``in`` test would match."""
    if isinstance(value, dict):
        return frozenset(value)
    if isinstance(value, list):
        return frozenset(str(item) for item in value)
    raise ValueError(f"{label} is not a list or object")


def strings(value: JsonValue, label: str) -> list[str]:
    """Require a JSON list of strings at ``label``."""
    return [text(item, f"{label} entry") for item in values(value, label)]


def text(value: JsonValue, label: str) -> str:
    """Require a JSON string at ``label``."""
    if not isinstance(value, str):
        raise ValueError(f"{label} is not a string")
    return value


def integer(value: JsonValue, label: str) -> int:
    """Require a JSON integer at ``label``."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} is not an integer")
    return value


def number(value: JsonValue, label: str) -> float:
    """Require a JSON number at ``label``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} is not a number")
    return float(value)


def read_json(path: Path) -> JsonValue:
    """Load any JSON document."""
    with path.open() as handle:
        payload: JsonValue = json.load(handle)
    return payload


def read_json_object(path: Path) -> JsonDict:
    """Load a JSON document that must be an object."""
    return obj(read_json(path), str(path))


def read_jsonl(path: Path) -> list[JsonDict]:
    """Load JSONL rows, skipping blank lines."""
    rows: list[JsonDict] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(obj(json.loads(stripped), f"{path}:{line_number}"))
    return rows


def write_json(path: Path, payload: JsonValue) -> None:
    """Write indented, key-sorted JSON with a trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: Path, rows: Iterable[JsonDict]) -> None:
    """Write JSONL rows with sorted keys."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_text(path: Path, content: str) -> None:
    """Write a text artifact, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """Render one GitHub-flavoured markdown table."""
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)
