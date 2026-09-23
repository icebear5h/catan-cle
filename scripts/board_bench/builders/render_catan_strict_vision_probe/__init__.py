"""Project the locked strict 60-question text probe onto raw engine screenshots.

Constants, source validation, ID projection, rendering, and the CLI live in
sibling modules. Every pre-split name stays importable at this path."""

from __future__ import annotations

from scripts.board_bench.builders.render_catan_strict_vision_probe.cli import main, parse_args
from scripts.board_bench.builders.render_catan_strict_vision_probe.constants import (
    DEFAULT_BOARD_CANVAS_FRACTION,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SOURCE_DIR,
    DEFAULT_VIEW_PADDING_FACTOR,
    ENTITY_ID_PATTERN,
    OUTPUT_SCHEMA,
    file_sha256,
    json_digest,
)
from scripts.board_bench.builders.render_catan_strict_vision_probe.projection import (
    canonical_id_map,
    canonicalize_question,
    normalize_projected_answer,
    translate_json_ids,
    translate_text_ids,
    validate_canonical_questions,
)
from scripts.board_bench.builders.render_catan_strict_vision_probe.render import (
    render_strict_vision_probe,
)
from scripts.board_bench.builders.render_catan_strict_vision_probe.source import (
    validate_render_args,
    validate_source_dataset,
)
from scripts.board_bench.shapes import JsonDict, read_jsonl, write_json, write_jsonl

__all__ = [
    "DEFAULT_BOARD_CANVAS_FRACTION",
    "DEFAULT_IMAGE_SIZE",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_SOURCE_DIR",
    "DEFAULT_VIEW_PADDING_FACTOR",
    "ENTITY_ID_PATTERN",
    "OUTPUT_SCHEMA",
    "JsonDict",
    "canonical_id_map",
    "canonicalize_question",
    "file_sha256",
    "json_digest",
    "main",
    "normalize_projected_answer",
    "parse_args",
    "read_jsonl",
    "render_strict_vision_probe",
    "translate_json_ids",
    "translate_text_ids",
    "validate_canonical_questions",
    "validate_render_args",
    "validate_source_dataset",
    "write_json",
    "write_jsonl",
]
