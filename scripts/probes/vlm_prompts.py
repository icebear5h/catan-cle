"""Prompts and the OpenRouter model registry for the Modal VLM board benchmark.

Kept beside the Modal app rather than imported from ``vlm_benchmark`` so the
container only has to load stdlib-only source: the benchmark images carry
torch/httpx, not this repository's runtime dependencies.
"""

from __future__ import annotations

OPENROUTER_MODELS: dict[str, str] = {
    "nemotron-12b-free": "nvidia/nemotron-nano-12b-v2-vl:free",
    "gemma3-27b-free": "google/gemma-3-27b-it:free",
    "gemma3-12b-free": "google/gemma-3-12b-it:free",
    "gemma3-4b-free": "google/gemma-3-4b-it:free",
    "qwen3-vl-8b": "qwen/qwen3-vl-8b-instruct",
    "qwen3-vl-8b-thinking": "qwen/qwen3-vl-8b-thinking",
    "qwen3-vl-30b-a3b": "qwen/qwen3-vl-30b-a3b-instruct",
    "qwen3-vl-32b": "qwen/qwen3-vl-32b-instruct",
    "ui-tars-7b": "bytedance/ui-tars-1.5-7b",
    "llama-3.2-11b-vision": "meta-llama/llama-3.2-11b-vision-instruct",
    "mistral-small-3.1-24b": "mistralai/mistral-small-3.1-24b-instruct",
    "gemini-2.5-flash": "google/gemini-2.5-flash",
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "gpt-5.4-nano": "openai/gpt-5.4-nano",
}

SYSTEM_PROMPT = """You are a visual analyzer for a Settlers of Catan board game screenshot.

Rules:
- ONLY describe what you can literally see in the image.
- If you cannot read a number or color clearly, say "unclear" instead of guessing.
- Do not use any prior knowledge about Catan boards to fill in gaps.
- Do not describe what a Catan board "typically" looks like.
- Be specific: use exact numbers, exact colors, exact positions.
- If you are unsure about something, skip it entirely."""

BENCHMARK_PROMPTS: dict[str, str] = {
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

    "count_hexes": "How many hexagonal land tiles are on this board? Just give the number.",
}
