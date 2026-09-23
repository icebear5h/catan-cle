"""Paths, provider endpoints, model catalog, prompts, and category aliases."""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from cle.game_engine.models.player import Color

load_dotenv()


BENCH_DIR = Path("evals/catan_board_bench/datasets/catan_board_bench_100")
QUESTION_DIR = BENCH_DIR / "questions"
RUNS_DIR = Path("artifacts/runs/catan_board_bench")
PROBE_DIR = Path("artifacts/generated/catan_board_bench/piece_recognition")
PROBE_QUESTION_DIR = PROBE_DIR / "questions"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
NOVITA_URL = "https://api.novita.ai/openai/v1/chat/completions"
MOONDREAM_URL = "https://api.moondream.ai/v1/query"
MOONDREAM_RESOLVE_IPS = ["104.21.54.37", "172.67.223.31"]
MOONDREAM_HOST = "api.moondream.ai"


SMALL_VLM_MODELS: dict[str, str] = {
    "gemma3-4b": "google/gemma-3-4b-it",
    "gemma3-12b": "google/gemma-3-12b-it",
    "gemma3-27b": "google/gemma-3-27b-it",
    "gemma4-26b-a4b-free": "google/gemma-4-26b-a4b-it:free",
    "gemma4-31b-free": "google/gemma-4-31b-it:free",
    "nemotron-3-nano-free": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "nemotron-12b-free": "nvidia/nemotron-nano-12b-v2-vl:free",
    "llama-3.2-11b": "meta-llama/llama-3.2-11b-vision-instruct",
    "qwen3-vl-8b": "qwen/qwen3-vl-8b-instruct",
    "qwen3-vl-8b-thinking": "qwen/qwen3-vl-8b-thinking",
    "qwen3.5-9b": "qwen/qwen3.5-9b",
    "qwen3.5-flash": "qwen/qwen3.5-flash-02-23",
    "moondream2": "moondream2",
    "ui-tars-7b": "bytedance/ui-tars-1.5-7b",
    "ministral-3b": "mistralai/ministral-3b-2512",
    "ministral-8b": "mistralai/ministral-8b-2512",
    "mistral-small-3.2-24b": "mistralai/mistral-small-3.2-24b-instruct",
    "mistral-small-24b": "mistralai/mistral-small-3.1-24b-instruct",
    "gemini-3.1-pro-preview": "google/gemini-3.1-pro-preview",
    "gemini-3.1-pro-preview-customtools": "google/gemini-3.1-pro-preview-customtools",
    "gemini-pro-latest": "~google/gemini-pro-latest",
}

DEFAULT_MODELS = [
    "gemma3-4b",
    "qwen3-vl-8b",
    "nemotron-12b-free",
    "ui-tars-7b",
]

DEFAULT_CATEGORIES = [
    "robber_tile",
    "robber_resource_number",
    "tile_resource_number",
    "tile_has_robber",
    "node_occupancy",
    "edge_road_owner",
    "color_road_locations",
    "port_trade_type",
    "port_occupancy",
    "nodes_connected",
    "edge_connects_nodes",
    "color_building_counts",
    "color_road_count",
]

SYSTEM_PROMPT = """You are answering engine-scored questions about a Catan board screenshot.

Rules:
- Use only visible public board information from the image plus the supplied fixed atlas context.
- Do not infer hidden hands or hidden development cards.
- Return only the final answer string, with no explanation.
- Preserve exact tokens such as <T07>, <N18>, <E03_17>, <RED>, and <WOOD>.
- If the answer is empty or absent, use the exact sentinel requested by the question context."""


COLOR_TOKEN_NAMES = [color.value for color in Color]
CATEGORY_ALIASES = {
    "isolated_tile_resource_number": "tile_resource_number",
    "local_patch_tile_resource_number": "tile_resource_number",
    "isolated_road_owner": "edge_road_owner",
    "local_patch_edge_road_owner": "edge_road_owner",
    "isolated_node_occupancy": "node_occupancy",
    "local_patch_node_occupancy": "node_occupancy",
    "isolated_port_trade_type": "port_trade_type",
    "local_patch_port_trade_type": "port_trade_type",
    "isolated_robber_presence": "robber_presence",
    "local_patch_robber_presence": "robber_presence",
}

PROVIDER_ENV = {
    "moondream": "MOONDREAM_API_KEY",
    "novita": "NOVITA_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

__all__ = [
    "BENCH_DIR",
    "CATEGORY_ALIASES",
    "COLOR_TOKEN_NAMES",
    "DEFAULT_CATEGORIES",
    "DEFAULT_MODELS",
    "MOONDREAM_HOST",
    "MOONDREAM_RESOLVE_IPS",
    "MOONDREAM_URL",
    "NOVITA_URL",
    "OPENROUTER_URL",
    "PROBE_DIR",
    "PROBE_QUESTION_DIR",
    "PROVIDER_ENV",
    "QUESTION_DIR",
    "RUNS_DIR",
    "SMALL_VLM_MODELS",
    "SYSTEM_PROMPT",
    "canonical_category",
    "resolve_model_specs",
    "sanitize_model_key",
    "split_csv",
    "timestamp_slug",
]


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def resolve_model_specs(keys_or_ids: Sequence[str]) -> dict[str, str]:
    specs: dict[str, str] = {}
    for value in keys_or_ids:
        if value in SMALL_VLM_MODELS:
            specs[value] = SMALL_VLM_MODELS[value]
        else:
            specs[sanitize_model_key(value)] = value
    return specs


def sanitize_model_key(model_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", model_id).strip("_")


def canonical_category(category: str) -> str:
    return CATEGORY_ALIASES.get(category, category)


def timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
