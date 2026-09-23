"""Probe Catan eval outputs for part-level failure diagnostics.

Normalization, failure inference, aggregation, reporting, and the CLI live in
sibling modules. Every pre-split name stays importable at this path."""

from __future__ import annotations

from scripts.probes.probe_catan_board_parts.cli import build_parser, main
from scripts.probes.probe_catan_board_parts.failures import infer_failure_tokens
from scripts.probes.probe_catan_board_parts.report import render_text_report
from scripts.probes.probe_catan_board_parts.stats import (
    Analysis,
    FailureExample,
    JsonDict,
    PartStats,
    Totals,
    analyze_records,
    top_failures_by_part,
)
from scripts.probes.probe_catan_board_parts.tokens import (
    CANONICAL_CATEGORY_MAP,
    CATEGORY_KEYS,
    CITY_RE,
    COLOR_TOKEN_RE,
    EDGE_TOKEN_RE,
    MODEL_PREFIX,
    NODE_TOKEN_RE,
    NUMBER_RE,
    RATIO_RE,
    ROAD_RE,
    SETTLEMENT_RE,
    TILE_RESOURCE_RE,
    canonical_category,
    classify_part,
    color_from_text,
    extract_edge_tokens,
    extract_numbers_for,
    is_empty_like,
    normalize_response,
    normalize_text,
    parse_int,
)

__all__ = [
    "CANONICAL_CATEGORY_MAP",
    "CATEGORY_KEYS",
    "CITY_RE",
    "COLOR_TOKEN_RE",
    "EDGE_TOKEN_RE",
    "MODEL_PREFIX",
    "NODE_TOKEN_RE",
    "NUMBER_RE",
    "RATIO_RE",
    "ROAD_RE",
    "SETTLEMENT_RE",
    "TILE_RESOURCE_RE",
    "Analysis",
    "FailureExample",
    "JsonDict",
    "PartStats",
    "Totals",
    "analyze_records",
    "build_parser",
    "canonical_category",
    "classify_part",
    "color_from_text",
    "extract_edge_tokens",
    "extract_numbers_for",
    "infer_failure_tokens",
    "is_empty_like",
    "main",
    "normalize_response",
    "normalize_text",
    "parse_int",
    "render_text_report",
    "top_failures_by_part",
]
