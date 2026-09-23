"""Evaluate strict Catan full-graph ASCII variations through OpenRouter.

Constants, job/plan construction, records, transport, summaries, and the CLI
live in sibling modules. Every pre-split name stays importable at this path.

``httpx`` is re-exported because tests patch ``<this module>.httpx.Client``."""

from __future__ import annotations

import httpx

from scripts.board_bench.run.eval_catan_board_bench_ascii_variations.cli import main, parse_args
from scripts.board_bench.run.eval_catan_board_bench_ascii_variations.constants import (
    DEFAULT_DATASET_DIR,
    OPENROUTER_URL,
    SYSTEM_PROMPT,
    extract_message_text,
    job_key,
    model_supports_reasoning_control,
    percentile,
    split_csv,
    usage_reasoning_tokens,
)
from scripts.board_bench.run.eval_catan_board_bench_ascii_variations.jobs import (
    build_jobs,
    build_plan,
    build_prompt,
    validate_or_write_plan,
)
from scripts.board_bench.run.eval_catan_board_bench_ascii_variations.records import (
    latest_records,
    response_record,
    successful_record,
)
from scripts.board_bench.run.eval_catan_board_bench_ascii_variations.summary import (
    summarize,
    summarize_group,
)
from scripts.board_bench.run.eval_catan_board_bench_ascii_variations.transport import (
    call_openrouter_text,
)
from scripts.board_bench.shapes import JsonDict, read_jsonl, write_json

__all__ = [
    "DEFAULT_DATASET_DIR",
    "OPENROUTER_URL",
    "SYSTEM_PROMPT",
    "JsonDict",
    "build_jobs",
    "build_plan",
    "build_prompt",
    "call_openrouter_text",
    "extract_message_text",
    "httpx",
    "job_key",
    "latest_records",
    "main",
    "model_supports_reasoning_control",
    "parse_args",
    "percentile",
    "read_jsonl",
    "response_record",
    "split_csv",
    "successful_record",
    "summarize",
    "summarize_group",
    "usage_reasoning_tokens",
    "validate_or_write_plan",
    "write_json",
]
