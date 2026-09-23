"""Capture native provider reasoning on one fresh first-settlement decision.

The helpers, seeded inputs, cost bounds, capture, reports, and CLI live in
sibling modules. Every pre-split name stays importable at this path."""

from __future__ import annotations

from scripts.reasoning.eval_catan_initial_settlement_reasoning.artifacts import (
    canonical_sha256,
    existing_traces,
    markdown_fence,
    message_payload,
    model_key,
    read_json,
    trace_path,
    utc_now,
    write_json,
    write_json_once,
)
from scripts.reasoning.eval_catan_initial_settlement_reasoning.capture import (
    SCHEMA,
    capture_trace,
)
from scripts.reasoning.eval_catan_initial_settlement_reasoning.cli import (
    PLAN_SCHEMA,
    main,
    parse_args,
)
from scripts.reasoning.eval_catan_initial_settlement_reasoning.costs import (
    catalog_cost_bounds,
    usage_cost_usd,
)
from scripts.reasoning.eval_catan_initial_settlement_reasoning.reports import (
    render_trace_markdown,
    write_indexes,
    write_trace,
)
from scripts.reasoning.eval_catan_initial_settlement_reasoning.seeds import (
    COLORS,
    SeedInput,
    build_seed_input,
    legal_action_payload,
    load_prompt_variant,
)

__all__ = [
    "COLORS",
    "PLAN_SCHEMA",
    "SCHEMA",
    "SeedInput",
    "build_seed_input",
    "canonical_sha256",
    "capture_trace",
    "catalog_cost_bounds",
    "existing_traces",
    "legal_action_payload",
    "load_prompt_variant",
    "main",
    "markdown_fence",
    "message_payload",
    "model_key",
    "parse_args",
    "read_json",
    "render_trace_markdown",
    "trace_path",
    "usage_cost_usd",
    "utc_now",
    "write_indexes",
    "write_json",
    "write_json_once",
    "write_trace",
]
