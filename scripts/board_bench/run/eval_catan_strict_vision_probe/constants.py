"""Schemas, the system prompt, run locking, and small shared helpers."""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv

from cle.harness.board_surface import ImageBoardPresentation
from scripts.board_bench.builders.render_catan_strict_vision_probe import (
    DEFAULT_OUTPUT_DIR as DEFAULT_DATASET_DIR,
)
from scripts.board_bench.builders.render_catan_strict_vision_probe import (
    OUTPUT_SCHEMA as DATASET_SCHEMA,
)
from scripts.board_bench.shapes import JsonDict

load_dotenv()

EVAL_SCHEMA = "catan_strict_vision_eval/v2"
SUITE_NAME = "strict_vision_probe_60"
SYSTEM_PROMPT = """You are answering strict engine-scored questions about an ordinary public Catan board screenshot.

Rules:
- Use only the raw screenshot and supplied fixed engine-atlas context.
- The screenshot is not annotated. JSON engine state is used only by the scorer and is not supplied to you.
- Txx, Nxx, Exx_yy, and Pxx are canonical engine identifiers with fixed board geometry.
- Read resources, dice numbers, robber state, road colors, buildings, and colors from the screenshot.
- Pointy-top directions are LEFT, RIGHT, UP-LEFT, UP-RIGHT, DOWN-LEFT, and DOWN-RIGHT.
- For nominal roll production: a settlement produces 1, a city produces 2, and a robber blocks its tile. Aggregate by color and resource.
- Return exactly one JSON object matching the requested shape. No markdown, prose, or extra keys.
- Use null for absent scalar values and [] for empty lists.
- Sort entity-ID lists lexicographically. Preserve tuple associations in lists of objects.
- Wrap color, resource, and building values in angle brackets, for example <BLUE>, <WOOD>, and <CITY>. Canonical entity IDs do not use angle brackets in JSON.
"""

__all__ = [
    "DATASET_SCHEMA",
    "DEFAULT_DATASET_DIR",
    "EVAL_SCHEMA",
    "SUITE_NAME",
    "SYSTEM_PROMPT",
    "VisionJob",
    "acquire_run_lock",
    "percentile",
    "release_run_lock",
    "split_csv",
    "usage_reasoning_tokens",
]


class VisionJob(TypedDict):
    """One screenshot question with its locked artifacts and prompt."""

    qa: JsonDict
    image_path: str
    image_sha256: str
    contract_path: str
    contract_sha256: str
    board_presentation: ImageBoardPresentation
    prompt: str


def usage_reasoning_tokens(usage: JsonDict) -> int:
    details = usage.get("completion_tokens_details") or {}
    if not isinstance(details, dict):
        return 0
    return int(str(details.get("reasoning_tokens", 0) or 0))


def acquire_run_lock(output_dir: Path) -> tuple[Path, int]:
    lock_path = output_dir / ".evaluation.lock"
    try:
        file_descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise SystemExit(f"Evaluation output is already locked: {lock_path}") from exc
    os.write(file_descriptor, f"pid={os.getpid()}\n".encode())
    return lock_path, file_descriptor


def release_run_lock(lock_path: Path, file_descriptor: int) -> None:
    os.close(file_descriptor)
    lock_path.unlink(missing_ok=True)


def percentile(values: Sequence[int], fraction: float) -> int | None:
    if not values:
        return None
    return values[max(0, math.ceil(fraction * len(values)) - 1)]


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]
