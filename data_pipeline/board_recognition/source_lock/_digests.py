"""Stable file and canonical-JSON digests, plus repository-relative paths."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from data_pipeline.board_recognition.source_lock._config import PROJECT_ROOT
from data_pipeline.json_types import JsonValue


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def repository_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def normalize_game_id(path: Path) -> str:
    return path.stem.removesuffix("_sample")


def _safe_json(value: object) -> JsonValue:
    decoded: JsonValue = json.loads(json.dumps(value, default=str, sort_keys=True))
    return decoded


__all__ = [
    "_safe_json",
    "canonical_sha256",
    "file_sha256",
    "normalize_game_id",
    "repository_relative",
]
