"""Probe a single board sample for atlas-part coverage and eval outcomes.

The shapes, inventory, scoring, report, and CLI live in sibling modules.
Every pre-split name stays importable at this path."""

from __future__ import annotations

from scripts.probes.probe_catan_board_coverage.cli import (
    build_parser,
    load_response_map,
    main,
    resolve_sample,
)
from scripts.probes.probe_catan_board_coverage.evaluate import (
    evaluate_part_token,
    summarize_part_payloads,
    summarize_probe,
    uncovered_for_part,
)
from scripts.probes.probe_catan_board_coverage.inventory import (
    build_expected_inventory,
    question_to_targets,
)
from scripts.probes.probe_catan_board_coverage.model import (
    CANONICAL_CATEGORY_MAP,
    ROBBER_PROBE_TYPES,
    TILE_MODE_PROBES,
    TILE_MODES,
    JsonDict,
    canonical_category,
    filter_probe_map,
    normalize_token,
    read_jsonl,
    read_manifest,
)
from scripts.probes.probe_catan_board_coverage.report import (
    build_report,
    filter_parts,
    render_text_report,
)

__all__ = [
    "CANONICAL_CATEGORY_MAP",
    "ROBBER_PROBE_TYPES",
    "TILE_MODES",
    "TILE_MODE_PROBES",
    "JsonDict",
    "build_expected_inventory",
    "build_parser",
    "build_report",
    "canonical_category",
    "evaluate_part_token",
    "filter_parts",
    "filter_probe_map",
    "load_response_map",
    "main",
    "normalize_token",
    "question_to_targets",
    "read_jsonl",
    "read_manifest",
    "render_text_report",
    "resolve_sample",
    "summarize_part_payloads",
    "summarize_probe",
    "uncovered_for_part",
]
