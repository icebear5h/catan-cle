"""Strict JSON reading and small path/hash helpers shared by the builder."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from data_pipeline.board_recognition.sources import file_sha256
from sft.json_types import JsonDict, JsonValue


def read_json(path: Path) -> JsonDict:
    # Source metadata contains non-integer audit measurements; duplicate keys still fail.
    return cast("JsonDict", json.loads(path.read_text(), object_pairs_hook=_unique_pairs))


def _unique_pairs(pairs: list[tuple[str, JsonValue]]) -> JsonDict:
    result: JsonDict = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_jsonl(path: Path) -> list[JsonDict]:
    with path.open() as handle:
        return [cast("JsonDict", json.loads(line, object_pairs_hook=_unique_pairs))
                for line in handle if line.strip()]


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _asset(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    _check(path.is_relative_to(root) and path.is_file(), f"invalid source asset path: {relative}")
    return path


def _hash(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": file_sha256(path)}
