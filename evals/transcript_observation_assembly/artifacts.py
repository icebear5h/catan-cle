"""Timestamps, hashing, and atomic JSON/JSONL artifact helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

from evals.json_types import JsonDict, JsonValue
from evals.transcript_observation_assembly.job import ObservationAssemblyError


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> JsonDict:
    if not path.exists():
        raise ObservationAssemblyError(f"Missing artifact: {path}")
    payload: JsonValue = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ObservationAssemblyError(f"Artifact is not an object: {path}")
    return payload


def _read_jsonl(path: Path) -> List[JsonDict]:
    if not path.exists():
        return []
    rows: List[JsonDict] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row: JsonValue = json.loads(line)
        if not isinstance(row, dict):
            raise ObservationAssemblyError(f"{path}:{line_number} is not an object")
        rows.append(row)
    return rows


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def _append_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()

