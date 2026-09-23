"""Run the locked OpenRouter Catan VLM board/placement sweep.

Catalog validation, cost tracking, phase commands, the status artifact, and
the CLI live in sibling modules. Every pre-split name stays importable here."""

from __future__ import annotations

from scripts.board_bench.run.run_catan_openrouter_model_sweep.catalog import (
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT_DIR,
    OPENROUTER_MODELS_URL,
    fetch_catalog,
    model_key,
    read_json,
    validate_config,
    write_json,
)
from scripts.board_bench.run.run_catan_openrouter_model_sweep.cli import main, parse_args
from scripts.board_bench.run.run_catan_openrouter_model_sweep.commands import (
    board_command,
    reasoning_trace_command,
    run_command,
    strategy_reasoning_trace_command,
)
from scripts.board_bench.run.run_catan_openrouter_model_sweep.costs import (
    read_jsonl,
    response_cost_usd,
    track_cost,
)
from scripts.board_bench.run.run_catan_openrouter_model_sweep.status import update_status

__all__ = [
    "DEFAULT_CONFIG",
    "DEFAULT_OUTPUT_DIR",
    "OPENROUTER_MODELS_URL",
    "board_command",
    "fetch_catalog",
    "main",
    "model_key",
    "parse_args",
    "read_json",
    "read_jsonl",
    "reasoning_trace_command",
    "response_cost_usd",
    "run_command",
    "strategy_reasoning_trace_command",
    "track_cost",
    "update_status",
    "validate_config",
    "write_json",
]
