"""Atomic JSON and JSONL artifact readers and writers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Dict, Iterable, List

from evals.json_types import JsonDict, JsonValue
from evals.transcript_reasoning.job import NarratorReasoningError


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


def _read_json(path: Path) -> JsonDict:
    if not path.exists():
        raise NarratorReasoningError(f"Missing artifact: {path}")
    data: JsonValue = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise NarratorReasoningError(f"Artifact is not an object: {path}")
    return data


def _read_jsonl(path: Path) -> List[JsonDict]:
    if not path.exists():
        return []
    rows: List[JsonDict] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row: JsonValue = json.loads(line)
        if not isinstance(row, dict):
            raise NarratorReasoningError(f"{path}:{line_number} is not an object")
        rows.append(row)
    return rows


def _plan_identity(plan: Mapping[str, object]) -> Dict[str, object]:
    return {
        key: value
        for key, value in plan.items()
        if key not in {"created_at"}
    }

