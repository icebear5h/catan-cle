"""Model registry and prompts for the local VLM board benchmark."""

from __future__ import annotations

from typing import TypedDict


class ModelResult(TypedDict, total=False):
    """One model's benchmark reply, or the reason there is none."""

    error: str
    response: str
    latency_ms: int
    model: str
    usage: dict[str, object]


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODELS: dict[str, tuple[str, str | None]] = {
    # Free tier
    "nemotron-12b-free": ("openrouter", "nvidia/nemotron-nano-12b-v2-vl:free"),
    "gemma3-27b-free": ("openrouter", "google/gemma-3-27b-it:free"),
    "gemma3-12b-free": ("openrouter", "google/gemma-3-12b-it:free"),
    "gemma3-4b-free": ("openrouter", "google/gemma-3-4b-it:free"),

    # Cheap
    "llama-3.2-11b": ("openrouter", "meta-llama/llama-3.2-11b-vision-instruct"),
    "qwen3-vl-8b": ("openrouter", "qwen/qwen3-vl-8b-instruct"),
    "qwen3-vl-8b-thinking": ("openrouter", "qwen/qwen3-vl-8b-thinking"),
    "qwen3-vl-30b-a3b": ("openrouter", "qwen/qwen3-vl-30b-a3b-instruct"),
    "qwen3-vl-32b": ("openrouter", "qwen/qwen3-vl-32b-instruct"),
    "ui-tars-7b": ("openrouter", "bytedance/ui-tars-1.5-7b"),
    "mistral-small-24b": ("openrouter", "mistralai/mistral-small-3.1-24b-instruct"),

    # Ceiling tests
    "gemini-2.5-flash": ("openrouter", "google/gemini-2.5-flash"),
    "gpt-4o-mini": ("openrouter", "openai/gpt-4o-mini"),
    "gpt-5.4-nano": ("openrouter", "openai/gpt-5.4-nano"),

    # Self-hosted
    "glm-4.1v": ("modal", None),
}

DEFAULT_MODELS: list[str] = [
    "gemma3-12b-free",
    "gemma3-27b-free",
    "nemotron-12b-free",
    "qwen3-vl-8b",
]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a visual analyzer for a Settlers of Catan board game screenshot.

Rules:
- ONLY describe what you can literally see in the image.
- If you cannot read a number or color clearly, say "unclear" instead of guessing.
- Do not use any prior knowledge about Catan boards to fill in gaps.
- Do not describe what a Catan board "typically" looks like.
- Be specific: use exact numbers, exact colors, exact positions.
- If you are unsure about something, skip it entirely."""

PROMPTS: dict[str, str] = {
    "count_hexes": "How many hexagonal land tiles are on this board? Just give the number.",

    "tile_reading": """For each hex tile on this board, state the resource type and the number on it.
Use these resource names only: wood, brick, sheep, wheat, ore, desert.
Format as one per line: RESOURCE NUMBER
Example: wheat 6
List all tiles you can see.""",

    "building_reading": """List every settlement and city on the board.
For each one, state the color and whether it's a settlement (small) or city (large).
Format: COLOR TYPE
Example: red settlement
List all buildings, nothing else.""",

    "numbers_only": """List every number token you can see on the hex tiles.
Format: one number per line, nothing else.""",
}
