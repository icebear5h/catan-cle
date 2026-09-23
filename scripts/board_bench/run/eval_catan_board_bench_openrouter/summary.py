"""Per-model and per-category accuracy rollups."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime, timezone

from scripts.board_bench.shapes import JsonDict, numeric, obj, text

__all__ = ["summarize", "summarize_records"]


def summarize(records: Sequence[JsonDict], plan: JsonDict) -> JsonDict:
    by_model: dict[str, list[JsonDict]] = defaultdict(list)
    by_model_category: dict[tuple[str, str], list[JsonDict]] = defaultdict(list)
    for record in records:
        model_key = text(record["model_key"], "model_key")
        by_model[model_key].append(record)
        by_model_category[(model_key, text(record["category"], "category"))].append(record)

    model_summary: JsonDict = {}
    for model_key, model_records in by_model.items():
        summary = summarize_records(model_records)
        summary["categories"] = {
            category: summarize_records(category_records)
            for (key, category), category_records in by_model_category.items()
            if key == model_key
        }
        model_summary[model_key] = summary

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "plan": plan,
        "models": model_summary,
    }


def summarize_records(records: Sequence[JsonDict]) -> JsonDict:
    attempted = [record for record in records if not record.get("error")]
    scores = [obj(record["score"], "record score") for record in attempted]
    errors = len(records) - len(attempted)
    exact = sum(1 for score in scores if score["correct"])
    component_correct = sum(
        numeric(score["component_correct"], "component_correct") for score in scores
    )
    component_total = sum(
        numeric(score["component_total"], "component_total") for score in scores
    )
    latencies = [
        numeric(record["latency_ms"], "latency_ms")
        for record in attempted
        if record.get("latency_ms") is not None
    ]
    return {
        "requests": len(records),
        "attempted": len(attempted),
        "errors": errors,
        "exact_accuracy": exact / len(attempted) if attempted else 0.0,
        "component_accuracy": component_correct / component_total if component_total else 0.0,
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else None,
    }
