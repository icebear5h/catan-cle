"""CatanBoardBench human-verification routes."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from flask import Blueprint, jsonify, request, send_file

from evals.catan_board_bench.annotations import (
    annotation_payload_for_contract,
    contract_to_render_state,
)
from evals.catan_board_bench.paths import DATASETS_DIR
from cle.game_engine.game import GameEngine
from cle.game_engine.models.player import Color
from cle.game_engine.public_board import snapshot_public_board


bench_bp = Blueprint("bench", __name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BENCH_DIR = DATASETS_DIR / "catan_board_bench_100"
QUESTION_DIR = BENCH_DIR / "questions"
QA_PATH = QUESTION_DIR / "qa.jsonl"
EVAL_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "catan_board_bench"
    / "catan_board_bench_100"
    / "openrouter"
)
TEXT_FORMAT_DIR = DATASETS_DIR / "text_format_optimization_probe"
TEXT_FORMAT_QA_PATH = TEXT_FORMAT_DIR / "qa.jsonl"
TEXT_FORMAT_MANIFEST_PATH = TEXT_FORMAT_DIR / "manifest.jsonl"
TEXT_FORMAT_METADATA_PATH = TEXT_FORMAT_DIR / "metadata.json"
TEXT_FORMAT_DEV_COMPARISON_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "catan_board_bench"
    / "text_format_optimization"
    / "openrouter"
    / "qwen3_8_27b_dev_full_20260824"
    / "summary.json"
)
TEXT_FORMAT_DEV_V3_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "catan_board_bench"
    / "text_format_optimization"
    / "openrouter"
    / "qwen3_8_27b_dev_v3_full_20260824"
    / "summary.json"
)
TEXT_FORMAT_TRANSFER_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "catan_board_bench"
    / "text_format_optimization"
    / "openrouter"
    / "qwen3_8_27b_transfer_20260824"
)
SFT_REPLAY_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "generated"
    / "board_recognition"
    / "replay_v1"
)
SFT_FORWARD_ROOT = SFT_REPLAY_ROOT / "ms_swift_semantic_v1"
SFT_BIDIRECTIONAL_ROOT = SFT_REPLAY_ROOT / "ms_swift_bidirectional_v1"
SFT_IMAGE_ROOT = SFT_REPLAY_ROOT / "images"
SFT_SPATIAL_LOCALIZATION_ROOT = SFT_REPLAY_ROOT / "spatial_localization_v1"
SFT_SPATIAL_LOCALIZATION_IMAGE_ROOT = SFT_SPATIAL_LOCALIZATION_ROOT / "images"
SFT_SPATIAL_LOCALIZATION_STAGES = {
    "stage1": ("train", "validation", "test"),
    "stage2": ("train", "validation", "test"),
    "probes": ("validation", "test"),
}
SFT_SPLITS = ("train", "validation", "test", "color_diagnostic")
SFT_TOPOLOGY_PATH = (
    PROJECT_ROOT / "artifacts" / "generated" / "sft" / "atlas_topology"
    / "catan_atlas_topology.jsonl"
)
SFT_SMOKE_REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "sft"
    / "2026-09-01-qwen38-spatial-robber-curriculum-smoke.json"
)
SFT_SPATIAL_ROBBER_ROOT = SFT_REPLAY_ROOT / "spatial_robber_v1"
SFT_CURRICULUM_STAGES = (
    "spatial_grounding",
    "clean_board_grounding",
    "pieces_and_colors",
    "real_game_distribution",
)
SFT_EVAL_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "catan_board_bench"
    / "sft"
    / "qwen38_spatial_sft_b32"
    / "final_validation_v1"
)
SFT_EVAL_SUMMARY_PATH = SFT_EVAL_ROOT / "summary.json"
SFT_EVAL_RECORDS_PATH = SFT_EVAL_ROOT / "records.jsonl"
SFT_EVAL_SUITE_PATH = SFT_REPLAY_ROOT / "evals" / "validation_v1.jsonl"
INITIAL_SETTLEMENT_REASONING_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "model_selection"
    / "openrouter_vlm_27b_72b_20260831"
)
INITIAL_SETTLEMENT_REASONING_RUNS: dict[str, dict[str, Any]] = {
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

CANONICAL_MODELS = {
    "qwen3-vl-8b": "Qwen3 (8B)",
    "qwen3.5-9b": "Qwen3.5 (9B)",
    "qwen3.6-flash": "Qwen3.6 Flash",
}

DEFAULT_EVAL_MODELS = ("qwen3-vl-8b", "qwen3.5-9b", "qwen3.6-flash")


@bench_bp.route("/api/catan-board-bench/examples", methods=["GET"])
def list_catan_board_bench_examples():
    """Return compact QA rows for picker controls."""

    category = request.args.get("category")
    sample_id = request.args.get("sample_id")
    limit = _int_arg("limit", 100)
    offset = _int_arg("offset", 0)

    rows = [
        _compact_qa(qa)
        for qa in _qa_rows()
        if (not category or qa["category"] == category)
        and (not sample_id or qa["sample_id"] == sample_id)
    ]
    categories = sorted({qa["category"] for qa in _qa_rows()})
    samples = sorted({qa["sample_id"] for qa in rows})
    sample_counts = {sample: 0 for sample in samples}
    for row in rows:
        sample_counts[row["sample_id"]] += 1

    return jsonify(
        {
            "examples": rows[offset : offset + limit],
            "total": len(rows),
            "offset": offset,
            "limit": limit,
            "categories": categories,
            "samples": samples,
            "sample_counts": sample_counts,
        }
    )


@bench_bp.route("/api/catan-board-bench/example", methods=["GET"])
def get_catan_board_bench_example():
    """Return one QA row with its engine contract context."""

    qa_id = request.args.get("id")
    index = _int_arg("index", 0)
    rows = _qa_rows()

    if qa_id:
        qa = next((row for row in rows if row["id"] == qa_id), None)
        if qa is None:
            return jsonify({"error": f"Unknown CatanBoardBench QA id: {qa_id}"}), 404
    else:
        if index < 0 or index >= len(rows):
            return jsonify({"error": f"Index must be in [0, {len(rows) - 1}]"}), 400
        qa = rows[index]

    contract = _load_contract(qa["contract_path"])

    return jsonify(
        {
            **qa,
            "image_url": f"/api/catan-board-bench/image/{qa['image_path']}",
            "annotations_url": f"/api/catan-board-bench/annotations?id={qa['id']}",
            "contract_context": _contract_context(contract, qa),
            "render_state": contract_to_render_state(contract),
            "total": len(rows),
        }
    )


@bench_bp.route("/api/catan-board-bench/sample", methods=["GET"])
def get_catan_board_bench_sample():
    """Return one benchmark image/contract with every QA row for that sample."""

    sample_id = request.args.get("sample_id")
    index = _int_arg("index", 0)
    rows = _qa_rows()
    sample_ids = _sample_ids(rows)

    if sample_id:
        if sample_id not in sample_ids:
            return jsonify({"error": f"Unknown CatanBoardBench sample id: {sample_id}"}), 404
    else:
        if index < 0 or index >= len(sample_ids):
            return jsonify({"error": f"Index must be in [0, {len(sample_ids) - 1}]"}), 400
        sample_id = sample_ids[index]

    sample_rows = [row for row in rows if row["sample_id"] == sample_id]
    if not sample_rows:
        return jsonify({"error": f"No QA rows for CatanBoardBench sample id: {sample_id}"}), 404

    first = sample_rows[0]
    contract = _load_contract(first["contract_path"])
    return jsonify(
        {
            "sample_id": sample_id,
            "sample_index": sample_ids.index(sample_id),
            "sample_count": len(sample_ids),
            "image_path": first["image_path"],
            "image_url": f"/api/catan-board-bench/image/{first['image_path']}",
            "contract_path": first["contract_path"],
            "annotations_url": f"/api/catan-board-bench/annotations?id={first['id']}",
            "render_state": contract_to_render_state(contract),
            "contract_context": _sample_context(contract),
            "questions": [_question_detail(contract, row) for row in sample_rows],
        }
    )


@bench_bp.route("/api/catan-board-bench/annotations", methods=["GET"])
def get_catan_board_bench_annotations():
    """Return frontend-aligned bbox/point annotations for one QA row."""

    qa_id = request.args.get("id")
    index = _int_arg("index", 0)
    image_size = _int_arg("size", 512)
    rows = _qa_rows()

    if qa_id:
        qa = next((row for row in rows if row["id"] == qa_id), None)
        if qa is None:
            return jsonify({"error": f"Unknown CatanBoardBench QA id: {qa_id}"}), 404
    else:
        if index < 0 or index >= len(rows):
            return jsonify({"error": f"Index must be in [0, {len(rows) - 1}]"}), 400
        qa = rows[index]

    contract = _load_contract(qa["contract_path"])
    payload = annotation_payload_for_contract(contract, image_size=image_size)
    return jsonify(
        {
            "qa_id": qa["id"],
            "sample_id": qa["sample_id"],
            "category": qa["category"],
            "question": qa["question"],
            "answer": qa["answer"],
            **payload,
        }
    )


@bench_bp.route("/api/catan-board-bench/image/<path:image_path>", methods=["GET"])
def get_catan_board_bench_image(image_path: str):
    """Serve benchmark board images from the generated CatanBoardBench directory."""

    path = (BENCH_DIR / image_path).resolve()
    try:
        path.relative_to(BENCH_DIR.resolve())
    except ValueError:
        return jsonify({"error": "Image path is outside the benchmark directory"}), 400
    if not path.exists():
        return jsonify({"error": f"Image not found: {image_path}"}), 404
    return send_file(path)


@bench_bp.route("/api/catan-board-bench/eval-comparison", methods=["GET"])
def get_catan_board_bench_eval_comparison():
    """Return latest benchmark summaries for requested models in one response."""

    requested_models = _parse_requested_eval_models(request.args.get("models"))
    model_summaries = _load_latest_eval_summaries()

    selected: list[str] = requested_models or list(DEFAULT_EVAL_MODELS)
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

        model_data = summary_entry["data"]
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
def get_initial_settlement_reasoning_traces():
    """Return one allowlisted direct-trace run and the run catalog."""

    run_id = request.args.get(
        "run_id", DEFAULT_INITIAL_SETTLEMENT_REASONING_RUN
    )
    run_config = INITIAL_SETTLEMENT_REASONING_RUNS.get(run_id)
    if run_config is None:
        return jsonify({"error": f"Unknown reasoning-trace run: {run_id}"}), 404
    run_dir = run_config["path"]
    plan_path = run_dir / "plan.json"
    summary_path = run_dir / "summary.json"
    missing = [
        str(path.relative_to(PROJECT_ROOT))
        for path in (plan_path, summary_path)
        if not path.is_file()
    ]
    if missing:
        return jsonify(
            {
                "error": "Initial-settlement reasoning artifacts are missing",
                "missing": missing,
            }
        ), 404

    plan = _read_json(plan_path)
    summary = _read_json(summary_path)
    models = [str(model_id) for model_id in plan.get("models", [])]
    seeds = [_coerce_int(seed) for seed in plan.get("seeds", [])]
    trace_rows = [
        row for row in summary.get("traces", []) if isinstance(row, dict)
    ]
    by_seed: dict[int, list[dict[str, Any]]] = {seed: [] for seed in seeds}
    for row in trace_rows:
        seed = _coerce_int(row.get("seed"))
        if seed in by_seed and row.get("model_id") in models:
            by_seed[seed].append(row)

    seed_status = []
    for seed in seeds:
        captured_models = {
            str(row.get("model_id")) for row in by_seed.get(seed, [])
        }
        captured = len(captured_models)
        if captured == 0:
            status = "not_started"
        elif captured == len(models):
            status = "complete"
        else:
            status = "partial"
        seed_status.append(
            {
                "seed": seed,
                "status": status,
                "captured": captured,
                "expected": len(models),
                "missing_models": [
                    model_id
                    for model_id in models
                    if model_id not in captured_models
                ],
            }
        )

    return jsonify(
        {
            "schema": "catan-initial-settlement-reasoning-ui/v1",
            "run_id": run_id,
            "run_title": run_config["title"],
            "run_description": run_config["description"],
            "available_runs": _reasoning_trace_run_catalog(),
            "excluded_models": run_config["excluded_models"],
            "complete": bool(summary.get("complete")),
            "captured_traces": len(trace_rows),
            "planned_traces": len(models) * len(seeds),
            "recorded_cost_usd": _coerce_float(
                summary.get("recorded_cost_usd")
            ),
            "models": models,
            "seeds": seed_status,
            "traces": trace_rows,
            "conditions": {
                "decision": plan.get("decision"),
                "actor": plan.get("actor"),
                "colors": plan.get("colors", []),
                "context_suite": plan.get("context_suite"),
                "board_surface": plan.get("board_surface"),
                "reasoning_request": plan.get("reasoning_request"),
                "temperature": plan.get("temperature"),
                "max_tokens_omitted": bool(plan.get("max_tokens_omitted")),
                "replay_input": bool(plan.get("replay_input")),
                "human_action_labels": bool(plan.get("human_action_labels")),
                "scheduling": plan.get("scheduling"),
                "prompt_variant": plan.get("prompt_variant"),
            },
        }
    )


@bench_bp.route("/api/catan-board-bench/reasoning-trace", methods=["GET"])
def get_initial_settlement_reasoning_trace():
    """Return one allowlisted direct trace without accepting a file path."""

    run_id = request.args.get(
        "run_id", DEFAULT_INITIAL_SETTLEMENT_REASONING_RUN
    )
    run_config = INITIAL_SETTLEMENT_REASONING_RUNS.get(run_id)
    if run_config is None:
        return jsonify({"error": f"Unknown reasoning-trace run: {run_id}"}), 404
    run_dir = run_config["path"]
    model_id = request.args.get("model_id", "")
    raw_seed = request.args.get("seed", "")
    if not model_id or not raw_seed:
        return jsonify({"error": "seed and model_id are required"}), 400
    try:
        seed = int(raw_seed)
    except ValueError:
        return jsonify({"error": "seed must be an integer"}), 400

    plan_path = run_dir / "plan.json"
    summary_path = run_dir / "summary.json"
    if not plan_path.is_file() or not summary_path.is_file():
        return jsonify({"error": "Initial-settlement reasoning artifacts are missing"}), 404
    plan = _read_json(plan_path)
    planned_models = [str(value) for value in plan.get("models", [])]
    planned_seeds = [_coerce_int(value) for value in plan.get("seeds", [])]
    if model_id not in planned_models:
        return jsonify({"error": f"Unknown reasoning-trace model: {model_id}"}), 404
    if seed not in planned_seeds:
        return jsonify({"error": f"Unknown reasoning-trace seed: {seed}"}), 404

    summary = _read_json(summary_path)
    row = next(
        (
            item
            for item in summary.get("traces", [])
            if isinstance(item, dict)
            and _coerce_int(item.get("seed")) == seed
            and item.get("model_id") == model_id
        ),
        None,
    )
    if row is None:
        return jsonify(
            {
                "error": "Reasoning trace has not been captured",
                "seed": seed,
                "model_id": model_id,
            }
        ), 404

    try:
        trace_path = _confined_reasoning_trace_path(
            row.get("json_path"), run_dir
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    if not trace_path.is_file():
        return jsonify({"error": "Indexed reasoning trace file is missing"}), 404
    trace = _read_json(trace_path)
    if (
        _coerce_int(trace.get("input", {}).get("seed")) != seed
        or trace.get("request", {}).get("requested_model") != model_id
    ):
        return jsonify({"error": "Reasoning trace identity does not match its index"}), 409
    try:
        render_state, board_sha256 = _reasoning_trace_render_state(trace)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 409
    trace["run_id"] = run_id
    trace["render_state"] = render_state
    trace["render_state_provenance"] = {
        "seed": seed,
        "board_sha256": board_sha256,
        "verified_against_trace": True,
        "renderer": "existing HexBoard contract",
    }
    return jsonify(trace)


@bench_bp.route("/api/catan-board-bench/text-format-v3", methods=["GET"])
def get_catan_text_format_v3_overview():
    """Return the frozen v3 text-format result and board inventory."""

    required = (
        TEXT_FORMAT_METADATA_PATH,
        TEXT_FORMAT_MANIFEST_PATH,
        TEXT_FORMAT_QA_PATH,
        TEXT_FORMAT_DEV_COMPARISON_PATH,
        TEXT_FORMAT_DEV_V3_PATH,
    )
    missing = [str(path.relative_to(PROJECT_ROOT)) for path in required if not path.exists()]
    if missing:
        return jsonify({"error": "Text-format v3 artifacts are missing", "missing": missing}), 404

    metadata = _read_json(TEXT_FORMAT_METADATA_PATH)
    manifest = _read_jsonl(TEXT_FORMAT_MANIFEST_PATH)
    questions = _text_format_qa_rows()
    comparison = _read_json(TEXT_FORMAT_DEV_COMPARISON_PATH)
    v3_summary = _read_json(TEXT_FORMAT_DEV_V3_PATH)
    baseline = _format_result(comparison, "tile_rows")
    winner = _format_result(v3_summary, "indexed_tile_rows")
    transfer = _paired_transfer_result(TEXT_FORMAT_TRANSFER_DIR)

    boards = []
    for row in manifest:
        sample_id = row["sample_id"]
        sample_questions = [item for item in questions if item["sample_id"] == sample_id]
        metrics = row.get("representation_metrics", {}).get("indexed_tile_rows", {})
        boards.append(
            {
                "sample_id": sample_id,
                "question_count": len(sample_questions),
                "categories": sorted({item["category"] for item in sample_questions}),
                "characters": _coerce_int(metrics.get("characters")),
                "lines": _coerce_int(metrics.get("lines")),
            }
        )

    source_qa_sha = metadata.get("source_lock", {}).get("qa.jsonl")
    current_qa_sha = _sha256_file(TEXT_FORMAT_QA_PATH)
    baseline_tokens = baseline["prompt_tokens"]
    winner_tokens = winner["prompt_tokens"]
    baseline_chars = _average_metric(manifest, "tile_rows", "characters")
    winner_chars = _average_metric(manifest, "indexed_tile_rows", "characters")

    return jsonify(
        {
            "format_id": "indexed_tile_rows_v3",
            "display_name": "Indexed Tile Rows v3",
            "model": v3_summary.get("plan", {}).get("model", "qwen/qwen3.8-27b"),
            "dataset_schema": metadata.get("schema"),
            "question_count": metadata.get("question_count", len(questions)),
            "board_count": metadata.get("board_count", len(boards)),
            "categories": sorted(metadata.get("categories", {}).keys()),
            "boards": boards,
            "question_lock": {
                "byte_identical_to_source": bool(source_qa_sha and source_qa_sha == current_qa_sha),
                "sha256": current_qa_sha,
                "source_sha256": source_qa_sha,
            },
            "development": {
                "baseline": baseline,
                "winner": winner,
                "prompt_token_multiplier": round(winner_tokens / baseline_tokens, 3)
                if baseline_tokens
                else None,
                "character_multiplier": round(winner_chars / baseline_chars, 3)
                if baseline_chars
                else None,
            },
            "transfer": transfer,
            "index_families": metadata.get("query_indexes", []),
            "interpretation": (
                "Questions and targets are unchanged. The input is question-independent but "
                "task-aware: it materializes reusable Catan joins rather than testing unaided "
                "spatial reasoning."
            ),
        }
    )


@bench_bp.route("/api/catan-board-bench/text-format-v3/sample", methods=["GET"])
def get_catan_text_format_v3_sample():
    """Return one exact v3 serialization with its unchanged question rows."""

    sample_id = request.args.get("sample_id", "")
    manifest = _read_jsonl(TEXT_FORMAT_MANIFEST_PATH) if TEXT_FORMAT_MANIFEST_PATH.exists() else []
    sample_rows = {row["sample_id"]: row for row in manifest}
    if sample_id not in sample_rows:
        return jsonify({"error": f"Unknown text-format v3 sample id: {sample_id}"}), 404

    representation_path = (
        TEXT_FORMAT_DIR / "representations" / sample_id / "indexed_tile_rows.txt"
    ).resolve()
    representation_path.relative_to(TEXT_FORMAT_DIR.resolve())
    if not representation_path.exists():
        return jsonify({"error": f"Missing v3 representation for {sample_id}"}), 404

    text = representation_path.read_text()
    questions = [
        _text_format_question(row)
        for row in _text_format_qa_rows()
        if row["sample_id"] == sample_id
    ]
    query_lines = [line for line in text.splitlines() if line.startswith("QI|")]
    query_counts: dict[str, int] = {}
    for line in query_lines:
        family = line.split("|", 3)[1]
        query_counts[family] = query_counts.get(family, 0) + 1

    return jsonify(
        {
            "sample_id": sample_id,
            "format": "indexed_tile_rows",
            "version": "v3",
            "text": text,
            "characters": len(text.rstrip("\n")),
            "lines": len(text.splitlines()),
            "sha256": _sha256_file(representation_path),
            "query_index_counts": query_counts,
            "questions": questions,
        }
    )


@bench_bp.route("/api/catan-board-bench/sft-data", methods=["GET"])
def get_catan_sft_data():
    """Return a filterable window over the active bidirectional vision-SFT rows."""

    split = request.args.get("split", "train")
    if split not in SFT_SPLITS:
        return jsonify({"error": f"Unknown SFT split: {split}"}), 400

    required = (
        SFT_BIDIRECTIONAL_ROOT / "metadata.json",
        SFT_BIDIRECTIONAL_ROOT / "trainable_tokens.json",
        SFT_BIDIRECTIONAL_ROOT / "mixed" / f"{split}.jsonl",
        SFT_BIDIRECTIONAL_ROOT / "mixed_index" / f"{split}.jsonl",
        SFT_BIDIRECTIONAL_ROOT / "audit" / f"{split}.jsonl",
        SFT_FORWARD_ROOT / "audit" / f"{split}.jsonl",
    )
    missing = [str(path.relative_to(PROJECT_ROOT)) for path in required if not path.is_file()]
    if missing:
        return jsonify({"error": "Vision-SFT artifacts are missing", "missing": missing}), 404

    rows = list(_sft_data_rows(split))
    row_kind = request.args.get("row_kind", "all")
    entity_type = request.args.get("entity_type", "all")
    density_bin = request.args.get("density_bin", "all")
    description_style = request.args.get("description_style", "all")
    query = request.args.get("query", "").strip().lower()

    filtered = [
        row
        for row in rows
        if (row_kind == "all" or row["row_kind"] == row_kind)
        and (entity_type == "all" or row["entity_type"] == entity_type)
        and (density_bin == "all" or row["density_bin"] == density_bin)
        and (description_style == "all" or row["description_style"] == description_style)
        and (
            not query
            or query
            in " ".join(
                str(row.get(field, ""))
                for field in (
                    "query_id",
                    "state_id",
                    "prompt",
                    "answer",
                    "description",
                    "head",
                )
            ).lower()
        )
    ]

    offset = max(0, _int_arg("offset", 0))
    limit = min(200, max(1, _int_arg("limit", 80)))
    metadata = _read_json(SFT_BIDIRECTIONAL_ROOT / "metadata.json")
    inventory = _read_json(SFT_BIDIRECTIONAL_ROOT / "trainable_tokens.json")
    direction_counts = _count_by(rows, "row_kind")
    state_count = len({row["state_id"] for row in rows})

    return jsonify(
        {
            "schema": "catan_board_recognition_sft_datavis/v2",
            "split": split,
            "rows": filtered[offset : offset + limit],
            "total": len(filtered),
            "offset": offset,
            "limit": limit,
            "summary": {
                "row_count": len(rows),
                "state_count": state_count,
                "rows_per_state": metadata.get("mixed_rows_per_state"),
                "direction_counts": direction_counts,
                "direction_ratio": (
                    f"{direction_counts.get('forward', 0) // max(direction_counts.get('inverse', 1), 1)}:1"
                ),
                "trainable_tokens": len(inventory.get("tokens", [])),
                "target_coverage": len(metadata.get("target_token_counts", {})),
                "source_schema": metadata.get("schema"),
                "source_sha256": metadata.get("files", {})
                .get(split, {})
                .get("mixed_annotations_sha256"),
            },
            "distributions": {
                "direction": direction_counts,
                "entity": _count_by(rows, "entity_type"),
                "density": _count_by(rows, "density_bin"),
                "description_style": _count_by(rows, "description_style"),
            },
            "launch_readiness": _sft_launch_readiness(rows),
            "facets": {
                "splits": list(SFT_SPLITS),
                "row_kinds": _unique_values(rows, "row_kind"),
                "entity_types": _unique_values(rows, "entity_type"),
                "density_bins": _unique_values(rows, "density_bin"),
                "description_styles": _unique_values(rows, "description_style"),
            },
        }
    )


@bench_bp.route("/api/catan-board-bench/sft-eval", methods=["GET"])
def get_catan_sft_eval():
    """Return the held-out generation eval for the published SFT checkpoint."""

    required = (
        SFT_EVAL_SUMMARY_PATH,
        SFT_EVAL_RECORDS_PATH,
        SFT_EVAL_SUITE_PATH,
    )
    missing = [str(path.relative_to(PROJECT_ROOT)) for path in required if not path.is_file()]
    if missing:
        return jsonify({"error": "SFT eval artifacts are missing", "missing": missing}), 404

    rows = list(_sft_eval_rows())
    category = request.args.get("category", "all")
    density_bin = request.args.get("density_bin", "all")
    row_kind = request.args.get("row_kind", "all")
    suite = request.args.get("suite", "all")
    correctness = request.args.get("correctness", "all")
    query = request.args.get("query", "").strip().lower()
    if correctness not in {"all", "correct", "incorrect"}:
        return jsonify({"error": f"Unknown correctness filter: {correctness}"}), 400

    filtered = [
        row
        for row in rows
        if (category == "all" or row["category"] == category)
        and (density_bin == "all" or row["density_bin"] == density_bin)
        and (row_kind == "all" or row["row_kind"] == row_kind)
        and (suite == "all" or row["suite"] == suite)
        and (
            correctness == "all"
            or (correctness == "correct" and row["correct"])
            or (correctness == "incorrect" and not row["correct"])
        )
        and (
            not query
            or query
            in " ".join(
                str(row.get(field, ""))
                for field in (
                    "id",
                    "state_id",
                    "prompt",
                    "expected",
                    "response",
                    "category",
                    "relationship",
                )
            ).lower()
        )
    ]

    offset = max(0, _int_arg("offset", 0))
    limit = min(200, max(1, _int_arg("limit", 80)))
    raw_summary = _read_json(SFT_EVAL_SUMMARY_PATH)
    spatial_token_rows = [
        row
        for row in rows
        if row["category"] == "spatial_grounding" and row["polarity"] == "token_return"
    ]
    spatial_token_correct = sum(row["correct"] for row in spatial_token_rows)
    return jsonify(
        {
            "schema": "catan_qwen38_spatial_sft_eval_datavis/v1",
            "rows": filtered[offset : offset + limit],
            "total": len(filtered),
            "offset": offset,
            "limit": limit,
            "summary": {
                "attempted": raw_summary.get("attempted", len(rows)),
                "correct": raw_summary.get("correct", 0),
                "exact_accuracy": raw_summary.get("exact_accuracy", 0.0),
                "model_id": raw_summary.get("model_id"),
                "bits": raw_summary.get("bits"),
                "batch_size": raw_summary.get("batch_size"),
                "generated_at": raw_summary.get("generated_at"),
                "by_category": raw_summary.get("by_category", {}),
                "by_density_bin": raw_summary.get("by_density_bin", {}),
                "by_polarity": raw_summary.get("by_polarity", {}),
                "by_relationship": raw_summary.get("by_relationship", {}),
                "by_row_kind": raw_summary.get("by_row_kind", {}),
                "by_suite": raw_summary.get("by_suite", {}),
                "spatial_token_return": {
                    "correct": spatial_token_correct,
                    "total": len(spatial_token_rows),
                    "exact_accuracy": (
                        spatial_token_correct / len(spatial_token_rows)
                        if spatial_token_rows
                        else 0.0
                    ),
                },
            },
            "checkpoint": {
                "name": "catan-qwen3.8-27b-spatial-sft",
                "hub_url": "https://huggingface.co/icebear5h/catan-qwen3.8-27b-spatial-sft",
                "modal_url": "https://modal.com/apps/tetracorp/main/ap-G7ZApUsiwktgqJra0gEUG8",
            },
            "facets": {
                "categories": _unique_values(rows, "category"),
                "density_bins": _unique_values(rows, "density_bin"),
                "row_kinds": _unique_values(rows, "row_kind"),
                "suites": _unique_values(rows, "suite"),
            },
        }
    )


@bench_bp.route("/api/catan-board-bench/sft-image/<path:image_name>", methods=["GET"])
def get_catan_sft_image(image_name: str):
    """Serve one generated SFT board image without exposing arbitrary files."""

    path = (SFT_IMAGE_ROOT / image_name).resolve()
    try:
        path.relative_to(SFT_IMAGE_ROOT.resolve())
    except ValueError:
        return jsonify({"error": "Image path is outside the SFT image directory"}), 400
    if not path.is_file():
        return jsonify({"error": f"Unknown SFT image: {image_name}"}), 404
    return send_file(path)


@bench_bp.route(
    "/api/catan-board-bench/spatial-localization-data", methods=["GET"]
)
def get_catan_spatial_localization_data():
    """Return the staged empty-board spatial-localization SFT corpus."""

    stage = request.args.get("stage", "stage1")
    if stage not in SFT_SPATIAL_LOCALIZATION_STAGES:
        return jsonify({"error": f"Unknown spatial-localization stage: {stage}"}), 400
    split = request.args.get("split", SFT_SPATIAL_LOCALIZATION_STAGES[stage][0])
    if split not in SFT_SPATIAL_LOCALIZATION_STAGES[stage]:
        return jsonify(
            {"error": f"Split {split} is unavailable for spatial-localization {stage}"}
        ), 400

    data_path = SFT_SPATIAL_LOCALIZATION_ROOT / stage / f"{split}.jsonl"
    metadata_path = SFT_SPATIAL_LOCALIZATION_ROOT / "metadata.json"
    missing = [
        str(path.relative_to(PROJECT_ROOT))
        for path in (data_path, metadata_path)
        if not path.is_file()
    ]
    if missing:
        return jsonify(
            {"error": "Spatial-localization artifacts are missing", "missing": missing}
        ), 404

    rows = list(_spatial_localization_rows(stage, split))
    task_type = request.args.get("task_type", "all")
    entity_type = request.args.get("entity_type", "all")
    relationship = request.args.get("relationship", "all")
    polarity = request.args.get("polarity", "all")
    query = request.args.get("query", "").strip().lower()
    filtered = [
        row
        for row in rows
        if (task_type == "all" or row["task_type"] == task_type)
        and (entity_type == "all" or row["entity_type"] == entity_type)
        and (relationship == "all" or row["relationship"] == relationship)
        and (polarity == "all" or row["polarity"] == polarity)
        and (
            not query
            or query
            in " ".join(
                str(row.get(field, ""))
                for field in (
                    "row_id",
                    "state_id",
                    "prompt",
                    "answer",
                    "target_token",
                    "tokens",
                    "marker",
                )
            ).lower()
        )
    ]

    offset = max(0, _int_arg("offset", 0))
    limit = min(200, max(1, _int_arg("limit", 80)))
    metadata = _read_json(metadata_path)
    file_key = f"{stage}/{split}.jsonl"
    file_metadata = metadata.get("files", {}).get(file_key, {})
    stage_catalog = []
    for stage_name, stage_splits in SFT_SPATIAL_LOCALIZATION_STAGES.items():
        split_counts = {
            split_name: metadata.get("files", {})
            .get(f"{stage_name}/{split_name}.jsonl", {})
            .get("rows", 0)
            for split_name in stage_splits
        }
        stage_catalog.append(
            {
                "id": stage_name,
                "splits": split_counts,
                "train_rows": split_counts.get("train", 0),
            }
        )

    return jsonify(
        {
            "schema": "catan_spatial_localization_datavis/v1",
            "stage": stage,
            "split": split,
            "rows": filtered[offset : offset + limit],
            "total": len(filtered),
            "offset": offset,
            "limit": limit,
            "summary": {
                "row_count": len(rows),
                "state_count": len({row["state_id"] for row in rows}),
                "image_count": len({row["image_name"] for row in rows}),
                "source_schema": metadata.get("schema"),
                "source_sha256": file_metadata.get("sha256"),
                "atlas_counts": metadata.get("atlas_counts", {}),
                "atlas_tokens": sum(metadata.get("atlas_counts", {}).values()),
                "marker_groups_per_board": metadata.get("marker_groups_per_board"),
                "node_edge_sampling_multiplier": metadata.get(
                    "node_edge_sampling_multiplier"
                ),
                "stage2_marker_replay_fraction": metadata.get(
                    "stage2_marker_replay_fraction"
                ),
            },
            "stages": stage_catalog,
            "distributions": {
                "task_type": _count_by(rows, "task_type"),
                "entity": _count_by(rows, "entity_type"),
                "relationship": _count_by(rows, "relationship"),
                "polarity": _count_by(rows, "polarity"),
            },
            "facets": {
                "stages": list(SFT_SPATIAL_LOCALIZATION_STAGES),
                "splits": list(SFT_SPATIAL_LOCALIZATION_STAGES[stage]),
                "task_types": _unique_values(rows, "task_type"),
                "entity_types": _unique_values(rows, "entity_type"),
                "relationships": _unique_values(rows, "relationship"),
                "polarities": _unique_values(rows, "polarity"),
            },
        }
    )


@bench_bp.route(
    "/api/catan-board-bench/spatial-localization-image/<path:image_name>",
    methods=["GET"],
)
def get_catan_spatial_localization_image(image_name: str):
    """Serve one generated localization image without exposing arbitrary files."""

    path = (SFT_SPATIAL_LOCALIZATION_IMAGE_ROOT / image_name).resolve()
    try:
        path.relative_to(SFT_SPATIAL_LOCALIZATION_IMAGE_ROOT.resolve())
    except ValueError:
        return jsonify({"error": "Image path is outside the localization directory"}), 400
    if not path.is_file():
        return jsonify({"error": f"Unknown localization image: {image_name}"}), 404
    return send_file(path)


def _int_arg(name: str, default: int) -> int:
    try:
        return int(request.args.get(name, default))
    except ValueError:
        return default


def _parse_requested_eval_models(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return tuple()
    requested = tuple(_normalize_model_key(item) for item in raw.split(",") if item.strip())
    # De-duplicate while preserving user order
    out: list[str] = []
    seen: set[str] = set()
    for item in requested:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return tuple(out)


def _load_latest_eval_summaries() -> dict[str, dict[str, Any]]:
    runs: dict[str, tuple[float, dict[str, Any]]] = {}
    if not EVAL_ROOT.exists():
        return {}

    for path in sorted(EVAL_ROOT.glob("*/summary.json"), reverse=True):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue

        generated_at = _parse_generated_at(data.get("generated_at"))
        run_time = generated_at if generated_at is not None else path.stat().st_mtime
        for raw_model_key, model_data in data.get("models", {}).items():
            canonical = _normalize_model_key(raw_model_key)
            if canonical not in CANONICAL_MODELS:
                continue
            current = runs.get(canonical)
            if current is None or run_time > current[0]:
                runs[canonical] = (run_time, model_data)

    return {key: {"data": value, "generated_at": ts} for key, (ts, value) in runs.items()}


def _coerce_category_payload(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}

    out: dict[str, Any] = {}
    for category, payload in raw.items():
        if not isinstance(payload, dict):
            continue
        out[category] = {
            "attempted": _coerce_int(payload.get("attempted")),
            "requests": _coerce_int(payload.get("requests")),
            "errors": _coerce_int(payload.get("errors")),
            "exact_accuracy": _coerce_float(payload.get("exact_accuracy")),
            "component_accuracy": _coerce_float(payload.get("component_accuracy")),
            "avg_latency_ms": _coerce_float(payload.get("avg_latency_ms")),
        }
    return out


def _coerce_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _coerce_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _normalize_model_key(raw: str) -> str:
    value = raw.strip().lower().replace("_", "-")
    if value.startswith("qwen-qwen"):
        value = "qwen" + value[len("qwen-qwen") :]
    if value in {"qwen3", "qwen3-vl", "qwen-3", "qwen3vl"}:
        value = "qwen3-vl-8b"
    if value in {"qwen3.5", "qwen-3.5", "qwen3.5-vl", "qwen-3.5-vl"}:
        value = "qwen3.5-9b"
    if value in {"qwen3.6", "qwen-qwen3.6-flash"}:
        value = "qwen3.6-flash"
    return value


def _parse_generated_at(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    for fmt in ("%Y%m%dT%H%M%SZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%f%z"):
        try:
            return datetime.strptime(value, fmt).timestamp()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None
    return None


@lru_cache(maxsize=1)
def _qa_rows() -> tuple[dict[str, Any], ...]:
    if not QA_PATH.exists():
        return tuple()
    return tuple(json.loads(line) for line in QA_PATH.read_text().splitlines() if line.strip())


def _sample_ids(rows: tuple[dict[str, Any], ...]) -> list[str]:
    return sorted({row["sample_id"] for row in rows})


@lru_cache(maxsize=128)
def _load_contract(contract_path: str) -> dict[str, Any]:
    path = (BENCH_DIR / contract_path).resolve()
    path.relative_to(BENCH_DIR.resolve())
    return json.loads(path.read_text())


def _compact_qa(qa: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": qa["id"],
        "sample_id": qa["sample_id"],
        "category": qa["category"],
        "question": qa["question"],
        "answer": qa["answer"],
    }


def _question_detail(contract: dict[str, Any], qa: dict[str, Any]) -> dict[str, Any]:
    return {
        **_compact_qa(qa),
        "target": qa.get("target", {}),
        "scoring": qa.get("scoring"),
        "contract_context": _contract_context(contract, qa),
    }


def _sample_context(contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample": contract.get("sample", {}),
        "source": contract.get("source", {}),
        "current": contract.get("current", {}),
        "robber": contract.get("robber", {}),
        "achievements": contract.get("achievements", {}),
    }


def _contract_context(contract: dict[str, Any], qa: dict[str, Any]) -> dict[str, Any]:
    target = qa.get("target", {})
    context: dict[str, Any] = {
        "sample": contract.get("sample", {}),
        "source": contract.get("source", {}),
        "current": contract.get("current", {}),
        "target": target,
        "robber": contract.get("robber", {}),
        "achievements": contract.get("achievements", {}),
    }

    tile_token = target.get("tile_token")
    node_token = target.get("node_token")
    edge_token = target.get("edge_token")
    port_token = target.get("port_token")

    if tile_token:
        context["tile"] = _find_by_token(contract.get("tiles", []), tile_token)
    if node_token:
        context["node"] = _find_by_token(contract.get("nodes", []), node_token)
    if edge_token:
        context["edge"] = _find_by_token(contract.get("edges", []), edge_token)
    if port_token:
        context["port"] = _find_by_token(contract.get("ports", []), port_token)

    if qa["category"] == "port_type_nodes" and "port" not in context:
        context["ports"] = contract.get("ports", [])
    if qa["category"] == "color_road_locations":
        color = target.get("color")
        context["roads_for_color"] = [
            edge for edge in contract.get("edges", []) if edge.get("road_color") == color
        ]
    if qa["category"] == "color_building_locations":
        color = target.get("color")
        context["buildings_for_color"] = [
            node for node in contract.get("nodes", []) if node.get("color") == color
        ]

    return context


def _find_by_token(items: list[dict[str, Any]], token: str) -> dict[str, Any] | None:
    return next((item for item in items if item.get("token") == token), None)


@lru_cache(maxsize=1)
def _text_format_qa_rows() -> tuple[dict[str, Any], ...]:
    if not TEXT_FORMAT_QA_PATH.exists():
        return tuple()
    return tuple(_read_jsonl(TEXT_FORMAT_QA_PATH))


def _text_format_question(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "category": row["category"],
        "question": row["question"],
        "answer": row["answer"],
        "answer_text": row["answer_text"],
        "output_schema": row["output_schema"],
        "scoring": row["scoring"],
        "target": row["target"],
    }


def _format_result(summary: dict[str, Any], format_name: str) -> dict[str, Any]:
    payload = summary.get("formats", {}).get(format_name, {})
    return {
        "format": format_name,
        "exact": _coerce_int(payload.get("exact")),
        "requests": _coerce_int(payload.get("requests")),
        "exact_accuracy": _coerce_float(payload.get("exact_accuracy")),
        "prompt_tokens": _coerce_int(payload.get("prompt_tokens")),
        "cost": _coerce_float(payload.get("cost")),
    }


def _paired_transfer_result(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    responses_path = run_dir / "responses.jsonl"
    if not summary_path.exists() or not responses_path.exists():
        return {"available": False}

    summary = _read_json(summary_path)
    accepted: dict[tuple[str, str], dict[str, Any]] = {}
    for row in _read_jsonl(responses_path):
        if row.get("error") or not row.get("response"):
            continue
        accepted[(str(row.get("format")), str(row.get("question_id")))] = row

    baseline_ids = {
        question_id for format_name, question_id in accepted if format_name == "tile_rows"
    }
    winner_ids = {
        question_id
        for format_name, question_id in accepted
        if format_name == "indexed_tile_rows"
    }
    paired_ids = baseline_ids & winner_ids

    def paired_score(format_name: str) -> dict[str, Any]:
        exact = sum(
            bool(accepted[(format_name, question_id)].get("score", {}).get("correct"))
            for question_id in paired_ids
        )
        return {
            "format": format_name,
            "exact": exact,
            "requests": len(paired_ids),
            "exact_accuracy": exact / len(paired_ids) if paired_ids else 0.0,
        }

    planned = _coerce_int(summary.get("plan", {}).get("request_count"))
    accepted_count = _coerce_int(summary.get("overall", {}).get("requests"))
    return {
        "available": True,
        "complete": bool(summary.get("complete")),
        "paired_questions": len(paired_ids),
        "baseline": paired_score("tile_rows"),
        "winner": paired_score("indexed_tile_rows"),
        "accepted_requests": accepted_count,
        "planned_requests": planned,
        "remaining_requests": max(0, planned - accepted_count),
        "note": "Credit-truncated; no tuning used transfer outcomes."
        if not summary.get("complete")
        else None,
    }


def _average_metric(manifest: list[dict[str, Any]], format_name: str, metric: str) -> float:
    values = [
        _coerce_float(row.get("representation_metrics", {}).get(format_name, {}).get(metric))
        for row in manifest
    ]
    values = [value for value in values if value > 0]
    return sum(values) / len(values) if values else 0.0


def _reasoning_trace_run_catalog() -> list[dict[str, Any]]:
    return [
        {
            "id": run_id,
            "title": config["title"],
            "description": config["description"],
            "available": (
                (config["path"] / "plan.json").is_file()
                and (config["path"] / "summary.json").is_file()
            ),
            "excluded_models": config["excluded_models"],
        }
        for run_id, config in INITIAL_SETTLEMENT_REASONING_RUNS.items()
    ]


def _reasoning_trace_render_state(
    trace: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    trace_input = trace.get("input")
    if not isinstance(trace_input, dict):
        raise ValueError("Reasoning trace has no input contract")
    seed = trace_input.get("seed")
    raw_colors = trace_input.get("colors")
    actor_name = trace_input.get("actor")
    board = trace_input.get("board_presentation")
    if not isinstance(seed, int):
        raise ValueError("Reasoning trace seed is invalid")
    if not isinstance(raw_colors, list) or len(raw_colors) != 4:
        raise ValueError("Reasoning trace color order is invalid")
    try:
        colors = tuple(Color[str(value)] for value in raw_colors)
        actor = Color[str(actor_name)]
    except KeyError as exc:
        raise ValueError("Reasoning trace contains an unknown color") from exc
    if actor != colors[0]:
        raise ValueError("Reasoning trace actor is not first in player order")
    if not isinstance(board, dict) or not isinstance(board.get("board_sha256"), str):
        raise ValueError("Reasoning trace has no board SHA-256")

    engine = GameEngine(colors, seed=seed, shuffle_players=False)
    snapshot = snapshot_public_board(engine.observe(actor))
    expected_sha256 = board["board_sha256"]
    if snapshot.facts_sha256 != expected_sha256:
        raise ValueError("Reconstructed board does not match reasoning trace SHA-256")
    return contract_to_render_state(snapshot.contract()), snapshot.facts_sha256


def _confined_reasoning_trace_path(raw_path: Any, run_dir: Path) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("Reasoning trace index has no JSON path")
    relative = Path(raw_path)
    if relative.is_absolute() or relative.suffix != ".json":
        raise ValueError("Reasoning trace index contains an invalid JSON path")
    path = (run_dir / relative).resolve()
    try:
        path.relative_to(run_dir.resolve())
    except ValueError as exc:
        raise ValueError("Reasoning trace path is outside the run directory") from exc
    return path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@lru_cache(maxsize=8)
def _spatial_localization_rows(
    stage: str, split: str
) -> tuple[dict[str, Any], ...]:
    source_rows = _read_jsonl(
        SFT_SPATIAL_LOCALIZATION_ROOT / stage / f"{split}.jsonl"
    )
    combined: list[dict[str, Any]] = []
    for position, source in enumerate(source_rows):
        messages = source.get("messages", [])
        images = source.get("images", [])
        if len(messages) != 2 or len(images) != 1:
            raise ValueError(
                f"Invalid spatial-localization transport row: {source.get('row_id')}"
            )
        image_name = str(images[0])
        spatial_targets = source.get("spatial_targets", [])
        spatial_target = spatial_targets[0] if spatial_targets else None
        tokens = [str(token) for token in source.get("tokens", [])]
        combined.append(
            {
                "index": position,
                "record_id": f"{source['row_id']}@{position}",
                "row_id": source["row_id"],
                "state_id": source["state_id"],
                "image_name": image_name,
                "image_url": (
                    "/api/catan-board-bench/spatial-localization-image/"
                    f"{image_name}"
                ),
                "prompt": messages[0].get("content", ""),
                "answer": messages[1].get("content", ""),
                "curriculum_stage": source.get("curriculum_stage"),
                "grounding_stage": source.get("grounding_stage", "unknown"),
                "task_type": source.get("task_type", "unknown"),
                "entity_type": source.get("entity_type", "unknown"),
                "target_token": source.get("target_token"),
                "tokens": tokens,
                "marker": source.get("marker"),
                "marker_style": source.get("marker_style"),
                "marker_group": source.get("marker_group", []),
                "relationship": source.get("relationship", "unknown"),
                "polarity": source.get("polarity", "unknown"),
                "sampling_repeat": source.get("sampling_repeat"),
                "replay_source": source.get("replay_source"),
                "probe_style": source.get("probe_style"),
                "eval_variant": source.get("eval_variant"),
                "spatial_target": spatial_target,
            }
        )
    return tuple(combined)


@lru_cache(maxsize=1)
def _sft_eval_rows() -> tuple[dict[str, Any], ...]:
    suite_rows = {row["id"]: row for row in _read_jsonl(SFT_EVAL_SUITE_PATH)}
    records = _read_jsonl(SFT_EVAL_RECORDS_PATH)
    if set(suite_rows) != {row["id"] for row in records}:
        raise ValueError("SFT eval records do not match the frozen validation suite")

    combined: list[dict[str, Any]] = []
    for position, record in enumerate(records):
        source = suite_rows[record["id"]]
        metadata = {**source.get("metadata", {}), **record.get("metadata", {})}
        messages = source.get("messages", [])
        images = source.get("images", [])
        if len(messages) != 2 or len(images) != 1:
            raise ValueError(f"Invalid SFT eval transport row: {record['id']}")
        image_name = str(images[0])
        score = record.get("score", {})
        combined.append(
            {
                "index": position,
                "id": record["id"],
                "state_id": metadata.get("state_id", "unknown"),
                "image_name": image_name,
                "image_url": f"/api/catan-board-bench/sft-image/{image_name}",
                "prompt": messages[0].get("content", ""),
                "expected": score.get("expected_normalized", record.get("expected", "")),
                "response": score.get("response_normalized", record.get("response", "")),
                "raw_response": record.get("response", ""),
                "correct": bool(score.get("correct")),
                "scoring": score.get("scoring", "exact"),
                "category": metadata.get("category", "unknown"),
                "density_bin": metadata.get("density_bin", "unknown"),
                "entity_type": metadata.get("entity_type", "unknown"),
                "row_kind": metadata.get("row_kind", "unknown"),
                "suite": metadata.get("suite", "unknown"),
                "relationship": metadata.get("relationship"),
                "polarity": metadata.get("polarity"),
                "task_type": metadata.get("task_type"),
                "curriculum_stage": metadata.get("curriculum_stage"),
            }
        )
    return tuple(combined)


@lru_cache(maxsize=len(SFT_SPLITS))
def _sft_data_rows(split: str) -> tuple[dict[str, Any], ...]:
    mixed_rows = _read_jsonl(SFT_BIDIRECTIONAL_ROOT / "mixed" / f"{split}.jsonl")
    mixed_index = _read_jsonl(
        SFT_BIDIRECTIONAL_ROOT / "mixed_index" / f"{split}.jsonl"
    )
    forward_audit = {
        row["query_id"]: row
        for row in _read_jsonl(SFT_FORWARD_ROOT / "audit" / f"{split}.jsonl")
    }
    inverse_audit = {
        row["query_id"]: row
        for row in _read_jsonl(SFT_BIDIRECTIONAL_ROOT / "audit" / f"{split}.jsonl")
    }
    if len(mixed_rows) != len(mixed_index):
        raise ValueError(f"SFT mixed rows and index disagree for split {split}")

    combined: list[dict[str, Any]] = []
    for position, (training_row, index_row) in enumerate(zip(mixed_rows, mixed_index)):
        row_kind = index_row["row_kind"]
        query_id = index_row["query_id"]
        audit = (forward_audit if row_kind == "forward" else inverse_audit).get(query_id)
        if audit is None:
            raise ValueError(f"Missing {row_kind} SFT audit row: {query_id}")
        messages = training_row.get("messages", [])
        images = training_row.get("images", [])
        if len(messages) != 2 or len(images) != 1:
            raise ValueError(f"Invalid multimodal SFT transport row: {query_id}")
        image_name = str(images[0])
        description_style = (
            audit.get("description_style")
            or audit.get("head")
            or "unspecified"
        )
        target_token = audit.get("target_token") or audit.get("slot")
        combined.append(
            {
                "index": position,
                "query_id": query_id,
                "state_id": index_row["state_id"],
                "row_kind": row_kind,
                "image_name": image_name,
                "image_url": f"/api/catan-board-bench/sft-image/{image_name}",
                "prompt": messages[0].get("content", ""),
                "answer": messages[1].get("content", ""),
                "entity_type": audit.get("entity_type") or "unspecified",
                "density_bin": audit.get("density_bin") or "unspecified",
                "description_style": description_style,
                "description": audit.get("description") or audit.get("head") or "",
                "head": audit.get("head"),
                "attribute": audit.get("attribute"),
                "target_token": target_token,
                "visual_qualifier": audit.get("visual_qualifier"),
                "source_kind": audit.get("source_kind"),
                "curriculum_stage": (
                    training_row.get("curriculum_stage")
                    or index_row.get("curriculum_stage")
                    or audit.get("curriculum_stage")
                ),
                "piece_count": audit.get("piece_count"),
                "road_count": audit.get("road_count"),
                "building_count": audit.get("building_count"),
            }
        )
    return tuple(combined)


def _sft_launch_readiness(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Expose what is real today and what still blocks the staged production run."""

    density_counts = _count_by(rows, "density_bin")
    supplement_metadata_path = SFT_SPATIAL_ROBBER_ROOT / "metadata.json"
    supplement = (
        _read_json(supplement_metadata_path)
        if supplement_metadata_path.is_file()
        else None
    )
    stage_counts = dict((supplement or {}).get("stage_counts", {}).get("train", {}))
    missing_stages = [
        stage for stage in SFT_CURRICULUM_STAGES if stage not in stage_counts
    ]
    directional_terms = (" above ", " below ", " left of ", " right of ")
    source_directional_rows = sum(
        any(term in f" {row.get('prompt', '').lower()} " for term in directional_terms)
        for row in rows
    )
    supplement_task_counts = dict(
        (supplement or {}).get("task_counts", {}).get("train", {})
    )
    directional_rows = supplement_task_counts.get(
        "spatial_grounding", source_directional_rows
    )
    robber_rows = supplement_task_counts.get("robber", 0)
    topology_rows = _read_jsonl(SFT_TOPOLOGY_PATH) if SFT_TOPOLOGY_PATH.is_file() else []
    smoke = _read_json(SFT_SMOKE_REPORT_PATH) if SFT_SMOKE_REPORT_PATH.is_file() else None
    effective_batch_size = 8
    projected_steps = (len(rows) + effective_batch_size - 1) // effective_batch_size
    step_seconds = float((smoke or {}).get("step_wall_seconds") or 0)
    runtime_seconds = float((smoke or {}).get("train_runtime_seconds") or 0)
    smoke_steps = max(_coerce_int((smoke or {}).get("optimizer_steps")), 1)
    runtime_per_step = runtime_seconds / smoke_steps

    stages = [
        {
            "id": "spatial_grounding",
            "title": "Pure spatial grounding",
            "rows": directional_rows,
            "status": "blocked" if directional_rows == 0 else "available",
            "note": "above, below, left, right, adjacency, connectivity, balanced hard negatives",
        },
        {
            "id": "clean_board_grounding",
            "title": "Clean board grounding",
            "rows": (
                density_counts.get("empty", 0)
                + density_counts.get("setup", 0)
                + stage_counts.get("clean_board_grounding", 0)
            ),
            "status": "unstaged",
            "note": (
                f"{density_counts.get('empty', 0) + density_counts.get('setup', 0):,} "
                f"source rows + {stage_counts.get('clean_board_grounding', 0):,} robber rows"
            ),
        },
        {
            "id": "pieces_and_colors",
            "title": "Pieces and colors",
            "rows": (
                density_counts.get("sparse", 0)
                + stage_counts.get("pieces_and_colors", 0)
            ),
            "status": "unstaged",
            "note": (
                f"{density_counts.get('sparse', 0):,} sparse source rows + "
                f"{stage_counts.get('pieces_and_colors', 0):,} robber rows"
            ),
        },
        {
            "id": "real_game_distribution",
            "title": "Real-game distribution",
            "rows": (
                density_counts.get("dense", 0)
                + stage_counts.get("real_game_distribution", 0)
            ),
            "status": "unstaged",
            "note": (
                f"{density_counts.get('dense', 0):,} dense source rows + "
                f"{stage_counts.get('real_game_distribution', 0):,} robber rows"
            ),
        },
    ]

    return {
        "status": "blocked",
        "reason": (
            "Spatial and robber supplements are materialized; the 12,288-row source "
            "still needs composition into one ordered production curriculum."
        ),
        "required_stages": list(SFT_CURRICULUM_STAGES),
        "stage_counts": stage_counts,
        "labeled_rows": sum(stage_counts.values()),
        "missing_stages": missing_stages,
        "directional_rows": directional_rows,
        "robber_rows": robber_rows,
        "topology_rows": len(topology_rows),
        "supplement": supplement,
        "stages": stages,
        "projection": {
            "effective_batch_size": effective_batch_size,
            "optimizer_steps": projected_steps,
            "steady_state_hours": round(projected_steps * step_seconds / 3600, 1),
            "smoke_inclusive_hours": round(projected_steps * runtime_per_step / 3600, 1),
        },
        "smoke": smoke,
    }


def _count_by(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(field) or "unspecified")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _unique_values(rows: list[dict[str, Any]], field: str) -> list[str]:
    return sorted({str(row.get(field) or "unspecified") for row in rows})


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
