"""Reading the persisted run files, and the narrow casts validation earns."""

import json
from collections.abc import Iterable
from pathlib import Path
from typing import cast

from .schema import NarratorReasoningArtifactError

__all__ = ["_listed", "_number", "_read_json", "_read_jsonl", "_require_equal"]


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise NarratorReasoningArtifactError(
            f"Missing narrator-reasoning artifact: {path.name}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise NarratorReasoningArtifactError(
            f"Could not read narrator-reasoning artifact {path.name}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise NarratorReasoningArtifactError(
            f"Narrator-reasoning artifact {path.name} is not an object"
        )
    return payload


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise NarratorReasoningArtifactError(
            f"Could not read narrator-reasoning artifact {path.name}: {exc}"
        ) from exc

    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise NarratorReasoningArtifactError(
                f"Could not read {path.name}:{line_number}: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise NarratorReasoningArtifactError(
                f"{path.name}:{line_number} is not an object"
            )
        rows.append(row)
    return rows


def _require_equal(label: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise NarratorReasoningArtifactError(
            f"Narrator-reasoning {label} mismatch: "
            f"expected {expected!r}, got {actual!r}"
        )


def _number(value: object) -> float:
    """Read a field the surrounding validation has already proved numeric."""
    return cast(float, value)


def _listed(value: object) -> list[object]:
    """Copy a JSON list field whose shape the artifact contract guarantees."""
    return list(cast(Iterable[object], value))
