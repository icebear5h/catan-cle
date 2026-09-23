"""CatanBoardBench human-verification routes.

The reasoning-run registry and ``_load_latest_eval_summaries`` are patched by
tests on this module, so they and every caller that must honour a patch live
here. The remaining routes register themselves when ``bench_api`` is imported.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from flask import Response, jsonify, request

from .bench_api import (
    INITIAL_SETTLEMENT_REASONING_ROOT,
    reasoning_trace_payload,
    reasoning_traces_payload,
)
from .bench_api.blueprint import bench_bp as bench_bp

# Compatibility re-exports: every name the single bench module used to define.
from .bench_api.board_bench import (
    get_catan_board_bench_annotations as get_catan_board_bench_annotations,
)
from .bench_api.board_bench import get_catan_board_bench_example as get_catan_board_bench_example
from .bench_api.board_bench import get_catan_board_bench_image as get_catan_board_bench_image
from .bench_api.board_bench import get_catan_board_bench_sample as get_catan_board_bench_sample
from .bench_api.board_bench import (
    list_catan_board_bench_examples as list_catan_board_bench_examples,
)
from .bench_api.common import (
    _coerce_category_payload,
    _coerce_float,
    _coerce_int,
    _map,
    _normalize_model_key,
    _parse_generated_at,
    _parse_requested_eval_models,
)
from .bench_api.common import _count_by as _count_by
from .bench_api.common import _int_arg as _int_arg
from .bench_api.common import _read_json as _read_json
from .bench_api.common import _read_jsonl as _read_jsonl
from .bench_api.common import _sha256_file as _sha256_file
from .bench_api.common import _unique_values as _unique_values
from .bench_api.paths import BENCH_DIR as BENCH_DIR
from .bench_api.paths import CANONICAL_MODELS, DEFAULT_EVAL_MODELS, EVAL_ROOT
from .bench_api.paths import PROJECT_ROOT as PROJECT_ROOT
from .bench_api.paths import QA_PATH as QA_PATH
from .bench_api.paths import QUESTION_DIR as QUESTION_DIR
from .bench_api.paths import SFT_BIDIRECTIONAL_ROOT as SFT_BIDIRECTIONAL_ROOT
from .bench_api.paths import SFT_CURRICULUM_STAGES as SFT_CURRICULUM_STAGES
from .bench_api.paths import SFT_EVAL_RECORDS_PATH as SFT_EVAL_RECORDS_PATH
from .bench_api.paths import SFT_EVAL_ROOT as SFT_EVAL_ROOT
from .bench_api.paths import SFT_EVAL_SUITE_PATH as SFT_EVAL_SUITE_PATH
from .bench_api.paths import SFT_EVAL_SUMMARY_PATH as SFT_EVAL_SUMMARY_PATH
from .bench_api.paths import SFT_FORWARD_ROOT as SFT_FORWARD_ROOT
from .bench_api.paths import SFT_IMAGE_ROOT as SFT_IMAGE_ROOT
from .bench_api.paths import SFT_REPLAY_ROOT as SFT_REPLAY_ROOT
from .bench_api.paths import SFT_SMOKE_REPORT_PATH as SFT_SMOKE_REPORT_PATH
from .bench_api.paths import (
    SFT_SPATIAL_LOCALIZATION_IMAGE_ROOT as SFT_SPATIAL_LOCALIZATION_IMAGE_ROOT,
)
from .bench_api.paths import SFT_SPATIAL_LOCALIZATION_ROOT as SFT_SPATIAL_LOCALIZATION_ROOT
from .bench_api.paths import SFT_SPATIAL_LOCALIZATION_STAGES as SFT_SPATIAL_LOCALIZATION_STAGES
from .bench_api.paths import SFT_SPATIAL_ROBBER_ROOT as SFT_SPATIAL_ROBBER_ROOT
from .bench_api.paths import SFT_SPLITS as SFT_SPLITS
from .bench_api.paths import SFT_TOPOLOGY_PATH as SFT_TOPOLOGY_PATH
from .bench_api.paths import SFT_VIEWER_DATASETS as SFT_VIEWER_DATASETS
from .bench_api.paths import TEXT_FORMAT_DEV_COMPARISON_PATH as TEXT_FORMAT_DEV_COMPARISON_PATH
from .bench_api.paths import TEXT_FORMAT_DEV_V3_PATH as TEXT_FORMAT_DEV_V3_PATH
from .bench_api.paths import TEXT_FORMAT_DIR as TEXT_FORMAT_DIR
from .bench_api.paths import TEXT_FORMAT_MANIFEST_PATH as TEXT_FORMAT_MANIFEST_PATH
from .bench_api.paths import TEXT_FORMAT_METADATA_PATH as TEXT_FORMAT_METADATA_PATH
from .bench_api.paths import TEXT_FORMAT_QA_PATH as TEXT_FORMAT_QA_PATH
from .bench_api.paths import TEXT_FORMAT_TRANSFER_DIR as TEXT_FORMAT_TRANSFER_DIR
from .bench_api.paths import _viewer_dataset as _viewer_dataset
from .bench_api.qa import _compact_qa as _compact_qa
from .bench_api.qa import _contract_context as _contract_context
from .bench_api.qa import _find_by_token as _find_by_token
from .bench_api.qa import _load_contract as _load_contract
from .bench_api.qa import _qa_rows as _qa_rows
from .bench_api.qa import _question_detail as _question_detail
from .bench_api.qa import _sample_context as _sample_context
from .bench_api.qa import _sample_ids as _sample_ids
from .bench_api.readiness import _sft_launch_readiness as _sft_launch_readiness
from .bench_api.reasoning import _confined_reasoning_trace_path as _confined_reasoning_trace_path
from .bench_api.reasoning import _reasoning_trace_render_state as _reasoning_trace_render_state
from .bench_api.sft import get_catan_sft_data as get_catan_sft_data
from .bench_api.sft import get_catan_sft_eval as get_catan_sft_eval
from .bench_api.sft import get_catan_sft_image as get_catan_sft_image
from .bench_api.sft_rows import _sft_data_rows as _sft_data_rows
from .bench_api.sft_rows import _sft_eval_rows as _sft_eval_rows
from .bench_api.sft_rows import _spatial_localization_rows as _spatial_localization_rows
from .bench_api.sft_rows import _viewer_row_count as _viewer_row_count
from .bench_api.spatial import (
    get_catan_spatial_localization_data as get_catan_spatial_localization_data,
)
from .bench_api.spatial import (
    get_catan_spatial_localization_image as get_catan_spatial_localization_image,
)
from .bench_api.text_format import _average_metric as _average_metric
from .bench_api.text_format import _format_result as _format_result
from .bench_api.text_format import _paired_transfer_result as _paired_transfer_result
from .bench_api.text_format import _text_format_qa_rows as _text_format_qa_rows
from .bench_api.text_format import _text_format_question as _text_format_question
from .bench_api.text_format import (
    get_catan_text_format_v3_overview as get_catan_text_format_v3_overview,
)
from .bench_api.text_format import (
    get_catan_text_format_v3_sample as get_catan_text_format_v3_sample,
)

INITIAL_SETTLEMENT_REASONING_RUNS: dict[str, dict[str, object]] = {
    "strategy-guided-v1": {
        "title": "Strategy-guided v1",
        "description": (
            "Pips and resource diversity are subordinate to a coherent "
            "two-settlement opening plan."
        ),
        "path": INITIAL_SETTLEMENT_REASONING_ROOT
        / "initial_settlement_reasoning_strategy_v1",
        "excluded_models": {
            "qwen/qwen3.5-27b": (
                "DQ: uncapped baseline reached provider length without a final response"
            ),
            "qwen/qwen3.5-35b-a3b": (
                "DQ: uncapped baseline reached provider length without a final response"
            ),
        },
    },
    "baseline-v1": {
        "title": "Baseline v1",
        "description": "Original v7 first-settlement guidance.",
        "path": INITIAL_SETTLEMENT_REASONING_ROOT
        / "initial_settlement_reasoning_traces",
        "excluded_models": {},
    },
}
DEFAULT_INITIAL_SETTLEMENT_REASONING_RUN = "strategy-guided-v1"


def _load_latest_eval_summaries() -> dict[str, dict[str, object]]:
    runs: dict[str, tuple[float, object]] = {}
    if not EVAL_ROOT.exists():
        return {}

    for path in sorted(EVAL_ROOT.glob("*/summary.json"), reverse=True):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue

        generated_at = _parse_generated_at(data.get("generated_at"))
        run_time = generated_at if generated_at is not None else path.stat().st_mtime
        for raw_model_key, model_data in _map(data.get("models", {})).items():
            canonical = _normalize_model_key(raw_model_key)
            if canonical not in CANONICAL_MODELS:
                continue
            current = runs.get(canonical)
            if current is None or run_time > current[0]:
                runs[canonical] = (run_time, model_data)

    return {key: {"data": value, "generated_at": ts} for key, (ts, value) in runs.items()}


def _reasoning_trace_run_catalog() -> list[dict[str, object]]:
    return [
        {
            "id": run_id,
            "title": config["title"],
            "description": config["description"],
            "available": (
                (cast(Path, config["path"]) / "plan.json").is_file()
                and (cast(Path, config["path"]) / "summary.json").is_file()
            ),
            "excluded_models": config["excluded_models"],
        }
        for run_id, config in INITIAL_SETTLEMENT_REASONING_RUNS.items()
    ]


@bench_bp.route("/api/catan-board-bench/eval-comparison", methods=["GET"])
def get_catan_board_bench_eval_comparison() -> Response | tuple[Response, int]:
    """Return latest benchmark summaries for requested models in one response."""

    requested_models = _parse_requested_eval_models(request.args.get("models"))
    model_summaries = _load_latest_eval_summaries()

    selected: Sequence[str] = requested_models or list(DEFAULT_EVAL_MODELS)
    models = []
    categories = set[str]()

    for model_key in selected:
        canonical = _normalize_model_key(model_key)
        summary_entry = model_summaries.get(canonical)
        if summary_entry is None:
            models.append(
                {
                    "model": canonical,
                    "display_name": CANONICAL_MODELS.get(canonical, canonical),
                    "attempted": 0,
                    "exact_accuracy": 0.0,
                    "component_accuracy": 0.0,
                    "avg_latency_ms": 0.0,
                    "errors": 0,
                    "categories": {},
                }
            )
            continue

        model_data = _map(summary_entry["data"])
        model_categories = model_data.get("categories", {})
        if isinstance(model_categories, dict):
            categories.update(model_categories.keys())

        models.append(
            {
                "model": canonical,
                "display_name": CANONICAL_MODELS.get(canonical, canonical),
                "attempted": _coerce_int(model_data.get("attempted")),
                "exact_accuracy": _coerce_float(model_data.get("exact_accuracy")),
                "component_accuracy": _coerce_float(
                    model_data.get("component_accuracy")
                ),
                "avg_latency_ms": _coerce_float(model_data.get("avg_latency_ms")),
                "errors": _coerce_int(model_data.get("errors")),
                "categories": _coerce_category_payload(model_categories),
            }
        )

    return jsonify(
        {
            "requested": selected,
            "categories": sorted(categories),
            "models": models,
        }
    )


@bench_bp.route("/api/catan-board-bench/reasoning-traces", methods=["GET"])
def get_initial_settlement_reasoning_traces() -> Response | tuple[Response, int]:
    """Return one allowlisted direct-trace run and the run catalog."""

    run_id = request.args.get(
        "run_id", DEFAULT_INITIAL_SETTLEMENT_REASONING_RUN
    )
    run_config = INITIAL_SETTLEMENT_REASONING_RUNS.get(run_id)
    if run_config is None:
        return jsonify({"error": f"Unknown reasoning-trace run: {run_id}"}), 404
    return reasoning_traces_payload(
        run_id, run_config, _reasoning_trace_run_catalog()
    )


@bench_bp.route("/api/catan-board-bench/reasoning-trace", methods=["GET"])
def get_initial_settlement_reasoning_trace() -> Response | tuple[Response, int]:
    """Return one allowlisted direct trace without accepting a file path."""

    run_id = request.args.get(
        "run_id", DEFAULT_INITIAL_SETTLEMENT_REASONING_RUN
    )
    run_config = INITIAL_SETTLEMENT_REASONING_RUNS.get(run_id)
    if run_config is None:
        return jsonify({"error": f"Unknown reasoning-trace run: {run_id}"}), 404
    return reasoning_trace_payload(run_id, run_config)

