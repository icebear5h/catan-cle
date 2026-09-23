"""The SFT dataset, evaluation, and image endpoints."""

from __future__ import annotations

from flask import Response, jsonify, request, send_file

from .blueprint import bench_bp
from .common import (
    _count_by,
    _int_arg,
    _map,
    _read_json,
    _seq,
    _unique_values,
)
from .paths import (
    PROJECT_ROOT,
    SFT_BIDIRECTIONAL_ROOT,
    SFT_EVAL_RECORDS_PATH,
    SFT_EVAL_SUITE_PATH,
    SFT_EVAL_SUMMARY_PATH,
    SFT_FORWARD_ROOT,
    SFT_IMAGE_ROOT,
    SFT_SPLITS,
)
from .readiness import _sft_launch_readiness
from .sft_rows import _sft_data_rows, _sft_eval_rows


@bench_bp.route("/api/catan-board-bench/sft-data", methods=["GET"])
def get_catan_sft_data() -> Response | tuple[Response, int]:
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
                "trainable_tokens": len(_seq(inventory.get("tokens", []))),
                "target_coverage": len(_map(metadata.get("target_token_counts", {}))),
                "source_schema": metadata.get("schema"),
                "source_sha256": _map(_map(metadata.get("files", {})).get(split, {}))
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
def get_catan_sft_eval() -> Response | tuple[Response, int]:
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
    spatial_token_correct = sum(bool(row["correct"]) for row in spatial_token_rows)
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
def get_catan_sft_image(image_name: str) -> Response | tuple[Response, int]:
    """Serve one generated SFT board image without exposing arbitrary files."""

    path = (SFT_IMAGE_ROOT / image_name).resolve()
    try:
        path.relative_to(SFT_IMAGE_ROOT.resolve())
    except ValueError:
        return jsonify({"error": "Image path is outside the SFT image directory"}), 400
    if not path.is_file():
        return jsonify({"error": f"Unknown SFT image: {image_name}"}), 404
    return send_file(path)
