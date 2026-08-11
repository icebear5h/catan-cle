"""CatanBench human-verification routes."""

from __future__ import annotations

from datetime import datetime
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from flask import Blueprint, jsonify, request, send_file

from catanbench.annotations import (
    annotation_payload_for_contract,
    contract_to_render_state,
)


bench_bp = Blueprint("bench", __name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BENCH_DIR = PROJECT_ROOT / "data_pipeline" / "catanbench" / "datasets" / "catanbench_100"
QUESTION_DIR = BENCH_DIR / "questions"
QA_PATH = QUESTION_DIR / "qa.jsonl"
EVAL_ROOT = BENCH_DIR / "openrouter_eval"

CANONICAL_MODELS = {
    "qwen3-vl-8b": "Qwen3 (8B)",
    "qwen3.5-9b": "Qwen3.5 (9B)",
    "qwen3.6-flash": "Qwen3.6 Flash",
}

DEFAULT_EVAL_MODELS = ("qwen3-vl-8b", "qwen3.5-9b", "qwen3.6-flash")

@bench_bp.route("/api/catanbench/examples", methods=["GET"])
def list_catanbench_examples():
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


@bench_bp.route("/api/catanbench/example", methods=["GET"])
def get_catanbench_example():
    """Return one QA row with its engine contract context."""

    qa_id = request.args.get("id")
    index = _int_arg("index", 0)
    rows = _qa_rows()

    if qa_id:
        qa = next((row for row in rows if row["id"] == qa_id), None)
        if qa is None:
            return jsonify({"error": f"Unknown CatanBench QA id: {qa_id}"}), 404
    else:
        if index < 0 or index >= len(rows):
            return jsonify({"error": f"Index must be in [0, {len(rows) - 1}]"}), 400
        qa = rows[index]

    contract = _load_contract(qa["contract_path"])

    return jsonify(
        {
            **qa,
            "image_url": f"/api/catanbench/image/{qa['image_path']}",
            "annotations_url": f"/api/catanbench/annotations?id={qa['id']}",
            "contract_context": _contract_context(contract, qa),
            "render_state": contract_to_render_state(contract),
            "total": len(rows),
        }
    )


@bench_bp.route("/api/catanbench/sample", methods=["GET"])
def get_catanbench_sample():
    """Return one benchmark image/contract with every QA row for that sample."""

    sample_id = request.args.get("sample_id")
    index = _int_arg("index", 0)
    rows = _qa_rows()
    sample_ids = _sample_ids(rows)

    if sample_id:
        if sample_id not in sample_ids:
            return jsonify({"error": f"Unknown CatanBench sample id: {sample_id}"}), 404
    else:
        if index < 0 or index >= len(sample_ids):
            return jsonify({"error": f"Index must be in [0, {len(sample_ids) - 1}]"}), 400
        sample_id = sample_ids[index]

    sample_rows = [row for row in rows if row["sample_id"] == sample_id]
    if not sample_rows:
        return jsonify({"error": f"No QA rows for CatanBench sample id: {sample_id}"}), 404

    first = sample_rows[0]
    contract = _load_contract(first["contract_path"])
    return jsonify(
        {
            "sample_id": sample_id,
            "sample_index": sample_ids.index(sample_id),
            "sample_count": len(sample_ids),
            "image_path": first["image_path"],
            "image_url": f"/api/catanbench/image/{first['image_path']}",
            "contract_path": first["contract_path"],
            "annotations_url": f"/api/catanbench/annotations?id={first['id']}",
            "render_state": contract_to_render_state(contract),
            "contract_context": _sample_context(contract),
            "questions": [_question_detail(contract, row) for row in sample_rows],
        }
    )


@bench_bp.route("/api/catanbench/annotations", methods=["GET"])
def get_catanbench_annotations():
    """Return frontend-aligned bbox/point annotations for one QA row."""

    qa_id = request.args.get("id")
    index = _int_arg("index", 0)
    image_size = _int_arg("size", 512)
    rows = _qa_rows()

    if qa_id:
        qa = next((row for row in rows if row["id"] == qa_id), None)
        if qa is None:
            return jsonify({"error": f"Unknown CatanBench QA id: {qa_id}"}), 404
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


@bench_bp.route("/api/catanbench/image/<path:image_path>", methods=["GET"])
def get_catanbench_image(image_path: str):
    """Serve benchmark board images from the generated CatanBench directory."""

    path = (BENCH_DIR / image_path).resolve()
    try:
        path.relative_to(BENCH_DIR.resolve())
    except ValueError:
        return jsonify({"error": "Image path is outside the benchmark directory"}), 400
    if not path.exists():
        return jsonify({"error": f"Image not found: {image_path}"}), 404
    return send_file(path)


@bench_bp.route("/api/catanbench/eval-comparison", methods=["GET"])
def get_catanbench_eval_comparison():
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
                "attempted": int(model_data.get("attempted", 0)),
                "exact_accuracy": float(model_data.get("exact_accuracy", 0.0)),
                "component_accuracy": float(model_data.get("component_accuracy", 0.0)),
                "avg_latency_ms": float(model_data.get("avg_latency_ms", 0.0)),
                "errors": int(model_data.get("errors", 0)),
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


def _int_arg(name: str, default: int) -> int:
    try:
        return int(request.args.get(name, default))
    except ValueError:
        return default


def _parse_requested_eval_models(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return tuple()
    requested = tuple(
        _normalize_model_key(item)
        for item in raw.split(",")
        if item.strip()
    )
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
            "attempted": int(payload.get("attempted", 0)),
            "requests": int(payload.get("requests", 0)),
            "errors": int(payload.get("errors", 0)),
            "exact_accuracy": float(payload.get("exact_accuracy", 0.0)),
            "component_accuracy": float(payload.get("component_accuracy", 0.0)),
            "avg_latency_ms": float(payload.get("avg_latency_ms", 0.0)),
        }
    return out


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
            edge
            for edge in contract.get("edges", [])
            if edge.get("road_color") == color
        ]
    if qa["category"] == "color_building_locations":
        color = target.get("color")
        context["buildings_for_color"] = [
            node
            for node in contract.get("nodes", [])
            if node.get("color") == color
        ]

    return context


def _find_by_token(items: list[dict[str, Any]], token: str) -> dict[str, Any] | None:
    return next((item for item in items if item.get("token") == token), None)
