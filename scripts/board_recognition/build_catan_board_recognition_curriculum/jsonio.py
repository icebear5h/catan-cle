"""JSON/JSONL artifact IO, digests, and the leakage ledger."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Sequence
from pathlib import Path

from cle.players.data import JsonValue
from scripts.board_recognition.build_catan_board_recognition_curriculum.shapes import JsonDict

__all__ = [
    "canonical_json",
    "file_sha256",
    "json_digest",
    "load_leakage_ledger",
    "prepare_output_dir",
    "read_json_object",
    "read_jsonl",
    "write_json",
    "write_jsonl",
]


def canonical_json(value: JsonValue) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def json_digest(value: JsonValue) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number}: expected a JSON object")
                rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Sequence[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_json(path: Path, value: JsonValue) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def read_json_object(path: Path) -> JsonDict:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def prepare_output_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output directory is not empty: {path}; pass --overwrite")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def load_leakage_ledger(path: Path) -> tuple[JsonDict, set[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"required benchmark leakage ledger is missing: {path}")
    payload = read_json_object(path)
    game_ids = payload.get("benchmark_game_ids")
    if not isinstance(game_ids, list) or not game_ids:
        raise ValueError("benchmark leakage ledger has no game IDs")
    return payload, {str(game_id) for game_id in game_ids}
