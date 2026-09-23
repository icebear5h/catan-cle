"""Deterministic writers, stable ranking, and the training row record."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

from data_pipeline.json_types import JsonDict, JsonValue


def _write_json(path: Path, payload: JsonValue) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: Iterable[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _stable_rank(*parts: object) -> int:
    text = ":".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16)


def _training_row(prompt: str, answer: str, image_name: str, stage: str) -> JsonDict:
    return {
        "curriculum_stage": stage,
        "images": [image_name],
        "messages": [
            {"role": "user", "content": f"<image>\n{prompt}"},
            {"role": "assistant", "content": answer},
        ],
    }

