"""Cursor-safe loading for GPT-reconstructed narrator reasoning paragraphs."""

from .artifact import load_narrator_reasoning_artifact
from .readers import (
    _read_json,
    _read_jsonl,
    _require_equal,
)
from .runs import (
    _artifact_error_collection,
    load_paired_narrator_reasoning,
)
from .schema import (
    _CURATED_REASONING_RUNS,
    ANCHOR_KINDS,
    PARAGRAPH_KINDS,
    PROJECT_ROOT,
    RESULT_SCHEMA,
    RUN_SCHEMA,
    WINDOW_SCHEMA,
    CuratedRun,
    NarratorReasoningArtifactError,
    curated_run,
)
from .validation import (
    _validate_paragraph,
    _validate_plan_anchors,
    _validate_plan_jobs,
)
from .window import build_paired_narrator_reasoning_window

__all__ = [
    "ANCHOR_KINDS",
    "CuratedRun",
    "NarratorReasoningArtifactError",
    "PARAGRAPH_KINDS",
    "PROJECT_ROOT",
    "RESULT_SCHEMA",
    "RUN_SCHEMA",
    "WINDOW_SCHEMA",
    "_CURATED_REASONING_RUNS",
    "_artifact_error_collection",
    "_read_json",
    "_read_jsonl",
    "_require_equal",
    "_validate_paragraph",
    "_validate_plan_anchors",
    "_validate_plan_jobs",
    "build_paired_narrator_reasoning_window",
    "curated_run",
    "load_narrator_reasoning_artifact",
    "load_paired_narrator_reasoning",
]
