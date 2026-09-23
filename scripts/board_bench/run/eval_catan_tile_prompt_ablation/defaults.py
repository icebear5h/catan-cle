"""Endpoints, path defaults, IO helpers, and run locking."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv

from scripts.board_bench.run.eval_catan_tile_prompt_ablation import (
    TileTruth,
    sha256_bytes,
)
from scripts.board_bench.shapes import JsonDict, obj

load_dotenv()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_QA_PATH = Path(
    "artifacts/runs/catan_board_bench/piece_recognition/"
    "qwen3_8_27b_piece_full_20260816/qa_snapshot.jsonl"
)
DEFAULT_IMAGE_ROOT = Path("artifacts/generated/catan_board_bench/piece_recognition")
DEFAULT_MODEL = "qwen/qwen3.8-27b"
DEFAULT_PROVIDER = "AkashML"

__all__ = [
    "DEFAULT_IMAGE_ROOT",
    "DEFAULT_MODEL",
    "DEFAULT_PROVIDER",
    "DEFAULT_QA_PATH",
    "OPENROUTER_URL",
    "TileJob",
    "TileQuestion",
    "acquire_run_lock",
    "file_sha256",
    "has_explicit_reasoning_usage",
    "job_key",
    "model_supports_reasoning_control",
    "percentile",
    "read_jsonl",
    "release_run_lock",
    "reported_reasoning_tokens",
    "split_csv",
    "usage_reasoning_tokens",
    "validate_request_contract",
]


class TileQuestion(TypedDict):
    """One isolated-tile question locked to its image bytes."""

    id: str
    sample_id: str
    category: str
    image_path: str
    image_sha256: str
    source_row_sha256: str
    truth: TileTruth


class TileJob(TypedDict):
    """One question rendered under one prompt condition."""

    condition: str
    qa: TileQuestion
    prompt: str
    prompt_sha256: str
    expected_response: str


def split_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def read_jsonl(path: Path) -> list[JsonDict]:
    return [
        obj(json.loads(line), str(path))
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def percentile(values: Sequence[int], probability: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[math.ceil(probability * len(ordered)) - 1]


def reported_reasoning_tokens(usage: JsonDict) -> int | None:
    details = usage.get("completion_tokens_details")
    if not isinstance(details, dict) or "reasoning_tokens" not in details:
        return None
    value = details["reasoning_tokens"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or not float(value).is_integer():
        return None
    return int(value)


def has_explicit_reasoning_usage(usage: JsonDict) -> bool:
    return reported_reasoning_tokens(usage) is not None


def usage_reasoning_tokens(usage: JsonDict) -> int:
    value = reported_reasoning_tokens(usage)
    if value is None:
        raise ValueError("reasoning-token usage field is invalid")
    return value


def job_key(job: TileJob) -> tuple[str, str]:
    return (job["condition"], job["qa"]["id"])


def model_supports_reasoning_control(model_id: str) -> bool:
    return model_id == DEFAULT_MODEL


def validate_request_contract(model_id: str, provider_order: Sequence[str]) -> None:
    if len(provider_order) != 1:
        raise SystemExit("Exactly one OpenRouter provider is required")
    if not model_supports_reasoning_control(model_id):
        raise SystemExit(f"No explicit reasoning-disable control for model: {model_id}")


def acquire_run_lock(output_dir: Path) -> tuple[Path, int]:
    lock_path = output_dir / ".evaluation.lock"
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise SystemExit(f"Output directory is already locked: {lock_path}") from exc
    os.write(fd, f"pid={os.getpid()}\n".encode())
    return lock_path, fd


def release_run_lock(lock_path: Path, lock_fd: int) -> None:
    os.close(lock_fd)
    lock_path.unlink(missing_ok=True)
