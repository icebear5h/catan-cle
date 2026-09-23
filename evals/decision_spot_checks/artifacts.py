"""Strict JSON and JSONL readers for decision-evaluation artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from evals.decision_spot_checks.config import (
    DecisionEvalArtifactError,
    DecisionEvalRunConfig,
)
from evals.json_types import JsonDict


def _run_files_available(config: DecisionEvalRunConfig) -> bool:
    required = (
        config.artifact_dir / "plan.json",
        config.artifact_dir / "decision_manifest.jsonl",
        config.bucket_index_path,
    )
    return all(path.exists() for path in required)


def _read_json(path: Path) -> JsonDict:
    if not path.exists():
        raise DecisionEvalArtifactError(f"Missing decision-eval artifact: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DecisionEvalArtifactError(f"Could not read {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise DecisionEvalArtifactError(f"{path.name} must contain a JSON object")
    return value


def _read_optional_json(path: Path) -> JsonDict | None:
    if not path.exists():
        return None
    return _read_json(path)


def _read_jsonl(path: Path, *, required: bool = True) -> list[JsonDict]:
    if not path.exists():
        if required:
            raise DecisionEvalArtifactError(f"Missing decision-eval artifact: {path}")
        return []
    rows: list[JsonDict] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise DecisionEvalArtifactError(f"Could not read {path.name}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DecisionEvalArtifactError(
                f"Could not read {path.name}:{line_number}: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise DecisionEvalArtifactError(
                f"{path.name}:{line_number} must contain a JSON object"
            )
        rows.append(row)
    return rows
