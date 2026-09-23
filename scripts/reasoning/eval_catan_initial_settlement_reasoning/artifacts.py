"""JSON, hashing, and path helpers for the initial-settlement reasoning probe."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

from cle.harness.models import ModelMessage
from cle.players.data import JsonValue

JsonDict = dict[str, JsonValue]

__all__ = [
    "JsonDict",
    "canonical_sha256",
    "existing_traces",
    "json_bool",
    "json_float",
    "json_int",
    "json_object",
    "json_payload",
    "json_text",
    "markdown_fence",
    "message_payload",
    "model_key",
    "read_json",
    "trace_path",
    "utc_now",
    "write_json",
    "write_json_once",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_sha256(value: JsonValue) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def json_payload(value: object) -> JsonValue:
    """Narrow an opaque provider payload to JSON, rejecting other values."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [json_payload(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_payload(item) for key, item in value.items()}
    raise TypeError(f"Provider payload is not JSON-serializable: {type(value)!r}")


def json_object(value: JsonValue, label: str) -> JsonDict:
    """Require a JSON object at ``label``."""
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    return value


def json_text(value: JsonValue, label: str) -> str:
    """Require a JSON string at ``label``."""
    if not isinstance(value, str):
        raise ValueError(f"{label} is not a string")
    return value


def json_int(value: JsonValue, label: str) -> int:
    """Require a JSON integer at ``label``."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{label} is not an integer")
    return value


def json_float(value: JsonValue, label: str) -> float:
    """Require a JSON number at ``label``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} is not a number")
    return float(value)


def json_bool(value: JsonValue, label: str) -> bool:
    """Require a JSON boolean at ``label``."""
    if not isinstance(value, bool):
        raise ValueError(f"{label} is not a boolean")
    return value


def model_key(model_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", model_id.lower()).strip("_")


def read_json(path: Path) -> JsonDict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def write_json(path: Path, value: Mapping[str, JsonValue]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_json_once(path: Path, value: Mapping[str, JsonValue]) -> None:
    if path.exists():
        if read_json(path) != value:
            raise RuntimeError(f"Immutable JSON artifact changed: {path}")
        return
    write_json(path, value)


def message_payload(messages: Sequence[ModelMessage]) -> list[JsonDict]:
    return [{"role": message.role, "content": message.content} for message in messages]


def trace_path(output_dir: Path, seed: int, model_id: str) -> Path:
    return output_dir / "traces" / f"seed_{seed}" / f"{model_key(model_id)}.json"


def existing_traces(output_dir: Path) -> list[JsonDict]:
    traces = []
    for path in sorted((output_dir / "traces").glob("seed_*/*.json")):
        traces.append(read_json(path))
    return traces


def markdown_fence(value: str, language: str = "text") -> str:
    return f"~~~~{language}\n{value}\n~~~~"
