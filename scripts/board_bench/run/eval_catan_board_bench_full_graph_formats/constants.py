"""Endpoint, system prompt, run locking, and small shared helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv

from cle.harness.board_surface import TextBoardPresentation
from cle.players.data import JsonValue
from scripts.board_bench.shapes import JsonDict, obj, text

load_dotenv()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
SYSTEM_PROMPT = """You are answering strict engine-scored questions about an authoritative public Catan board graph serialized as text.

Rules:
- Use only the supplied board graph. Entity IDs are opaque and board-local.
- Every one of the 19 tiles, 54 nodes, 72 edges, and 9 ports is represented. Native null, NULL, none, or dash values mean absent/empty.
- Pointy-top cube direction deltas are:
  LEFT=(-1,+1,0), RIGHT=(+1,-1,0),
  UP-LEFT=(0,+1,-1), UP-RIGHT=(+1,0,-1),
  DOWN-LEFT=(-1,0,+1), DOWN-RIGHT=(0,-1,+1).
- For nominal roll production: a settlement produces 1, a city produces 2, and a robber blocks its tile. Aggregate by color and resource.
- Return exactly one JSON object matching the requested shape. No markdown, prose, or extra keys.
- Use null for absent scalar values and [] for empty lists.
- Sort entity-ID lists lexicographically. Preserve tuple associations in lists of objects.
- Wrap color, resource, and building values in angle brackets, for example <BLUE>, <WOOD>, and <CITY>. Board-local entity IDs such as T03, N14, and E27 do not use angle brackets.
"""

__all__ = [
    "OPENROUTER_URL",
    "SYSTEM_PROMPT",
    "FormatJob",
    "acquire_run_lock",
    "extract_message_text",
    "file_sha256",
    "job_key",
    "json_digest",
    "model_supports_reasoning_control",
    "percentile",
    "read_jsonl",
    "release_run_lock",
    "split_csv",
    "usage_reasoning_tokens",
    "write_json",
]


class FormatJob(TypedDict):
    """One question rendered in one serialization format."""

    format: str
    qa: JsonDict
    representation_path: str
    representation_sha256: str
    board_presentation: TextBoardPresentation
    prompt: str


def usage_reasoning_tokens(usage: JsonDict) -> int:
    details = usage.get("completion_tokens_details") or {}
    if not isinstance(details, dict):
        return 0
    return int(str(details.get("reasoning_tokens", 0) or 0))


def job_key(job: FormatJob) -> tuple[str, str]:
    return (job["format"], text(job["qa"]["id"], "question id"))


def model_supports_reasoning_control(model_id: str) -> bool:
    value = model_id.lower()
    return any(family in value for family in ("qwen3.5", "qwen3.6", "qwen3.7", "qwen3.8"))


def extract_message_text(message: JsonDict) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ).strip()
    return str(content).strip()


def percentile(values: Sequence[int], fraction: float) -> int | None:
    if not values:
        return None
    index = max(0, math.ceil(fraction * len(values)) - 1)
    return values[index]


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def read_jsonl(path: Path) -> list[JsonDict]:
    if not path.exists():
        return []
    return [obj(json.loads(line), str(path)) for line in path.read_text().splitlines() if line]


def write_json(path: Path, payload: JsonDict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_digest(value: JsonValue) -> str:
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def acquire_run_lock(output_dir: Path) -> tuple[Path, int]:
    lock_path = output_dir / ".evaluation.lock"
    try:
        file_descriptor = os.open(
            lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
    except FileExistsError as exc:
        raise SystemExit(f"Evaluation output is already locked: {lock_path}") from exc
    os.write(file_descriptor, f"pid={os.getpid()}\n".encode())
    return lock_path, file_descriptor


def release_run_lock(lock_path: Path, file_descriptor: int) -> None:
    os.close(file_descriptor)
    lock_path.unlink(missing_ok=True)
