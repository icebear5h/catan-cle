"""Reading the persisted artifact files, tolerating only a live-appended tail."""

import json
from pathlib import Path

from .schema import ModelTraceArtifactError

__all__ = ["_read_json", "_read_jsonl", "_require_equal"]


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise ModelTraceArtifactError(f"Missing model-trace artifact: {path.name}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ModelTraceArtifactError(
            f"Could not read model-trace artifact {path.name}: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise ModelTraceArtifactError(f"Model-trace artifact {path.name} is not an object")
    return data


def _read_jsonl(
    path: Path,
    *,
    required: bool = True,
    tolerate_incomplete_final_line: bool = False,
) -> tuple[list[dict[str, object]], bool]:
    """Read JSONL, optionally ignoring only a concurrently appended tail."""
    if not path.exists():
        if required:
            raise ModelTraceArtifactError(f"Missing model-trace artifact: {path.name}")
        return [], False

    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ModelTraceArtifactError(
            f"Could not read model-trace artifact {path.name}: {exc}"
        ) from exc

    rows: list[dict[str, object]] = []
    lines = text.splitlines(keepends=True)
    for line_index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            is_incomplete_tail = (
                tolerate_incomplete_final_line
                and line_index == len(lines) - 1
                and not line.endswith(("\n", "\r"))
            )
            if is_incomplete_tail:
                return rows, True
            raise ModelTraceArtifactError(
                f"Could not read model-trace artifact {path.name}: "
                f"invalid JSON on line {line_index + 1}: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise ModelTraceArtifactError(f"{path.name}:{line_index + 1} is not an object")
        rows.append(row)
    return rows, False


def _require_equal(label: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise ModelTraceArtifactError(
            f"Model-trace {label} mismatch: expected {expected!r}, got {actual!r}"
        )
