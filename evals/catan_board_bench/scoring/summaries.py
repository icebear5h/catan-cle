"""Run-level and per-model aggregation of scored benchmark records."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Sequence, Tuple

from evals.catan_board_bench.scoring.categories import JsonDict
from evals.json_types import as_dict, as_number, as_str


def summarize(records: Sequence[JsonDict], plan: JsonDict) -> JsonDict:
    by_model: Dict[str, List[JsonDict]] = defaultdict(list)
    by_model_category: Dict[Tuple[str, str], List[JsonDict]] = defaultdict(list)
    for record in records:
        model_key = as_str(record["model_key"], "record model_key")
        by_model[model_key].append(record)
        by_model_category[(model_key, as_str(record["category"], "record category"))].append(
            record
        )

    model_summary: JsonDict = {}
    for model_key, model_records in by_model.items():
        summary = summarize_records(model_records)
        model_summary[model_key] = summary
        summary["categories"] = {
            category: summarize_records(category_records)
            for (key, category), category_records in by_model_category.items()
            if key == model_key
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "plan": plan,
        "models": model_summary,
    }


def _score(record: JsonDict) -> JsonDict:
    return as_dict(record["score"], "record score")


def summarize_records(records: Sequence[JsonDict]) -> JsonDict:
    attempted = [record for record in records if not record.get("error")]
    errors = len(records) - len(attempted)
    exact = sum(1 for record in attempted if _score(record)["correct"])
    component_correct = sum(
        as_number(_score(record)["component_correct"], "component_correct")
        for record in attempted
    )
    component_total = sum(
        as_number(_score(record)["component_total"], "component_total") for record in attempted
    )
    latencies = [
        as_number(record["latency_ms"], "latency_ms")
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
