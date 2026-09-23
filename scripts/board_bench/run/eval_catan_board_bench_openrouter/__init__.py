"""Evaluate OpenRouter VLMs on CatanBoardBench engine-scored QA.

This is intended for cheap model screening before SFT. Constants, text
normalization, question selection, provider transport, scoring, summaries, the
plan, and the CLI live in sibling modules. Every pre-split name stays
importable at this path.

``httpx`` is re-exported because tests patch ``<this module>.httpx.Client``."""

from __future__ import annotations

import httpx

from scripts.board_bench.run.eval_catan_board_bench_openrouter.cli import main
from scripts.board_bench.run.eval_catan_board_bench_openrouter.constants import (
    BENCH_DIR,
    CATEGORY_ALIASES,
    COLOR_TOKEN_NAMES,
    DEFAULT_CATEGORIES,
    DEFAULT_MODELS,
    MOONDREAM_HOST,
    MOONDREAM_RESOLVE_IPS,
    MOONDREAM_URL,
    NOVITA_URL,
    OPENROUTER_URL,
    PROBE_DIR,
    PROBE_QUESTION_DIR,
    QUESTION_DIR,
    RUNS_DIR,
    SMALL_VLM_MODELS,
    SYSTEM_PROMPT,
    canonical_category,
    resolve_model_specs,
    sanitize_model_key,
    split_csv,
    timestamp_slug,
)
from scripts.board_bench.run.eval_catan_board_bench_openrouter.plan import build_plan, parse_args
from scripts.board_bench.run.eval_catan_board_bench_openrouter.questions import (
    attach_contract_if_available,
    build_prompt,
    find_by_token,
    local_atlas_context,
    resolve_contract_path,
    select_questions,
    sentinel_hint,
    tile_layout_text,
)
from scripts.board_bench.run.eval_catan_board_bench_openrouter.scoring import (
    building_token,
    color_token,
    component_score,
    resource_token,
    score_answer,
)
from scripts.board_bench.run.eval_catan_board_bench_openrouter.summary import (
    summarize,
    summarize_records,
)
from scripts.board_bench.run.eval_catan_board_bench_openrouter.text_norm import (
    contains_bare_count,
    contains_count,
    contains_labeled_count,
    contains_value,
    edge_tokens,
    extract_content,
    extract_message_text,
    model_supports_reasoning_control,
    normalize_text,
    number_tokens,
    repair_merged_tokens,
    repair_partial_tokens,
)
from scripts.board_bench.run.eval_catan_board_bench_openrouter.transport import (
    call_moondream,
    call_novita,
    call_openrouter,
    call_provider,
)
from scripts.board_bench.shapes import JsonDict, write_json

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
    "QUESTION_DIR",
    "RUNS_DIR",
    "SMALL_VLM_MODELS",
    "SYSTEM_PROMPT",
    "JsonDict",
    "attach_contract_if_available",
    "build_plan",
    "build_prompt",
    "building_token",
    "call_moondream",
    "call_novita",
    "call_openrouter",
    "call_provider",
    "canonical_category",
    "color_token",
    "component_score",
    "contains_bare_count",
    "contains_count",
    "contains_labeled_count",
    "contains_value",
    "edge_tokens",
    "extract_content",
    "extract_message_text",
    "find_by_token",
    "httpx",
    "local_atlas_context",
    "main",
    "model_supports_reasoning_control",
    "normalize_text",
    "number_tokens",
    "parse_args",
    "repair_merged_tokens",
    "repair_partial_tokens",
    "resolve_contract_path",
    "resolve_model_specs",
    "resource_token",
    "sanitize_model_key",
    "score_answer",
    "select_questions",
    "sentinel_hint",
    "split_csv",
    "summarize",
    "summarize_records",
    "tile_layout_text",
    "timestamp_slug",
    "write_json",
]
