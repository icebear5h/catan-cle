"""Dataset defaults, the system prompt, and small shared helpers."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv

from scripts.board_bench.shapes import JsonDict, obj, text

load_dotenv()

DEFAULT_DATASET_DIR = Path("evals/catan_board_bench/datasets/ascii_variation_probe")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
SYSTEM_PROMPT = """You are answering strict engine-scored questions about an authoritative public Catan board graph encoded as ASCII text.

Rules:
- Use only the supplied board graph. Entity IDs are opaque and board-local.
- Every tile, node, edge, and port record is explicit. A dash (-) means absent/empty.
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
    "DEFAULT_DATASET_DIR",
    "OPENROUTER_URL",
    "SYSTEM_PROMPT",
    "extract_message_text",
    "job_key",
    "model_supports_reasoning_control",
    "percentile",
    "split_csv",
    "usage_reasoning_tokens",
]


def usage_reasoning_tokens(usage: JsonDict) -> int:
    details = usage.get("completion_tokens_details") or {}
    if not isinstance(details, dict):
        return 0
    return int(str(details.get("reasoning_tokens", 0) or 0))


def job_key(job: JsonDict) -> tuple[str, str]:
    return (
        text(job["variant"], "job variant"),
        text(obj(job["qa"], "job qa")["id"], "question id"),
    )


def model_supports_reasoning_control(model_id: str) -> bool:
    value = model_id.lower()
    return any(
        family in value
        for family in (
            "qwen3.5",
            "qwen3.6",
            "qwen3.7",
            "qwen3.8",
        )
    )


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
