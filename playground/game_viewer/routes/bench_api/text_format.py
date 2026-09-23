"""The text-format v3 overview and one sampled question."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path

from flask import Response, jsonify, request

from .blueprint import bench_bp
from .common import (
    _coerce_float,
    _coerce_int,
    _map,
    _read_json,
    _read_jsonl,
    _sha256_file,
)
from .paths import (
    PROJECT_ROOT,
    TEXT_FORMAT_DEV_COMPARISON_PATH,
    TEXT_FORMAT_DEV_V3_PATH,
    TEXT_FORMAT_DIR,
    TEXT_FORMAT_MANIFEST_PATH,
    TEXT_FORMAT_METADATA_PATH,
    TEXT_FORMAT_QA_PATH,
    TEXT_FORMAT_TRANSFER_DIR,
)


@lru_cache(maxsize=1)
def _text_format_qa_rows() -> tuple[dict[str, object], ...]:
    if not TEXT_FORMAT_QA_PATH.exists():
        return tuple()
    return tuple(_read_jsonl(TEXT_FORMAT_QA_PATH))


def _text_format_question(row: dict[str, object]) -> dict[str, object]:
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


def _format_result(summary: dict[str, object], format_name: str) -> dict[str, object]:
    payload = _map(_map(summary.get("formats", {})).get(format_name, {}))
    return {
        "format": format_name,
        "exact": _coerce_int(payload.get("exact")),
        "requests": _coerce_int(payload.get("requests")),
        "exact_accuracy": _coerce_float(payload.get("exact_accuracy")),
        "prompt_tokens": _coerce_int(payload.get("prompt_tokens")),
        "cost": _coerce_float(payload.get("cost")),
    }


def _paired_transfer_result(run_dir: Path) -> dict[str, object]:
    summary_path = run_dir / "summary.json"
    responses_path = run_dir / "responses.jsonl"
    if not summary_path.exists() or not responses_path.exists():
        return {"available": False}

    summary = _read_json(summary_path)
    accepted: dict[tuple[str, str], dict[str, object]] = {}
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

    def paired_score(format_name: str) -> dict[str, object]:
        exact = sum(
            bool(_map(accepted[(format_name, question_id)].get("score", {})).get("correct"))
            for question_id in paired_ids
        )
        return {
            "format": format_name,
            "exact": exact,
            "requests": len(paired_ids),
            "exact_accuracy": exact / len(paired_ids) if paired_ids else 0.0,
        }

    planned = _coerce_int(_map(summary.get("plan", {})).get("request_count"))
    accepted_count = _coerce_int(_map(summary.get("overall", {})).get("requests"))
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


def _average_metric(manifest: Sequence[Mapping[str, object]], format_name: str, metric: str) -> float:
    values = [
        _coerce_float(
            _map(_map(row.get("representation_metrics", {})).get(format_name, {})).get(metric)
        )
        for row in manifest
    ]
    values = [value for value in values if value > 0]
    return sum(values) / len(values) if values else 0.0


@bench_bp.route("/api/catan-board-bench/text-format-v3", methods=["GET"])
def get_catan_text_format_v3_overview() -> Response | tuple[Response, int]:
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
        metrics = _map(_map(row.get("representation_metrics", {})).get("indexed_tile_rows", {}))
        boards.append(
            {
                "sample_id": sample_id,
                "question_count": len(sample_questions),
                "categories": sorted(
                    {str(item["category"]) for item in sample_questions}
                ),
                "characters": _coerce_int(metrics.get("characters")),
                "lines": _coerce_int(metrics.get("lines")),
            }
        )

    source_qa_sha = _map(metadata.get("source_lock", {})).get("qa.jsonl")
    current_qa_sha = _sha256_file(TEXT_FORMAT_QA_PATH)
    baseline_tokens = _coerce_float(baseline["prompt_tokens"])
    winner_tokens = _coerce_float(winner["prompt_tokens"])
    baseline_chars = _average_metric(manifest, "tile_rows", "characters")
    winner_chars = _average_metric(manifest, "indexed_tile_rows", "characters")

    return jsonify(
        {
            "format_id": "indexed_tile_rows_v3",
            "display_name": "Indexed Tile Rows v3",
            "model": _map(v3_summary.get("plan", {})).get("model", "qwen/qwen3.8-27b"),
            "dataset_schema": metadata.get("schema"),
            "question_count": metadata.get("question_count", len(questions)),
            "board_count": metadata.get("board_count", len(boards)),
            "categories": sorted(_map(metadata.get("categories", {})).keys()),
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
def get_catan_text_format_v3_sample() -> Response | tuple[Response, int]:
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
