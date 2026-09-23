"""Curated, cursor-safe model traces for paired replay transcripts."""

from .artifact import load_model_trace_artifact
from .indexing import _canonical_availability, _index_comparisons, _index_manifest
from .readers import _read_json, _read_jsonl, _require_equal
from .runs import (
    _artifact_error_collection,
    load_paired_model_traces,
)
from .schema import (
    _CURATED_TRACE_RUNS,
    MODEL_TRACE_SCHEMA,
    PROJECT_ROOT,
    CuratedTraceRun,
    ModelTraceArtifactError,
    curated_trace_run,
)
from .window import build_paired_model_trace_window

__all__ = [
    "CuratedTraceRun",
    "MODEL_TRACE_SCHEMA",
    "ModelTraceArtifactError",
    "PROJECT_ROOT",
    "_CURATED_TRACE_RUNS",
    "_artifact_error_collection",
    "_canonical_availability",
    "_index_comparisons",
    "_index_manifest",
    "_read_json",
    "_read_jsonl",
    "_require_equal",
    "build_paired_model_trace_window",
    "curated_trace_run",
    "load_model_trace_artifact",
    "load_paired_model_traces",
]
