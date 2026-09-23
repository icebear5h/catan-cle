"""Per-representation and per-category accuracy rollups."""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime, timezone

from cle.players.data import JsonValue
from scripts.board_bench.shapes import JsonDict, obj, text

__all__ = ["summarize", "summarize_group"]

USAGE_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens", "cost")


def _numeric(value: JsonValue, label: str) -> int | float:
    """Read a JSON number without widening integers to floats."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value
    raise ValueError(f"{label} is not a number")


def summarize(records: Sequence[JsonDict], plan: JsonDict) -> JsonDict:
    by_representation: dict[str, list[JsonDict]] = defaultdict(list)
    for record in records:
        by_representation[text(record["representation"], "representation")].append(record)
    representations: JsonDict = {}
    for name, rows in by_representation.items():
        group = summarize_group(rows)
        by_category: dict[str, list[JsonDict]] = defaultdict(list)
        for row in rows:
            by_category[text(row["category"], "category")].append(row)
        group["categories"] = {
            category: summarize_group(category_rows)
            for category, category_rows in sorted(by_category.items())
        }
        representations[name] = group
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "plan": plan,
        "representations": representations,
    }


def _usage_total(records: Sequence[JsonDict], key: str) -> int | float:
    total: int | float = 0
    for record in records:
        usage = record.get("usage") or {}
        if not isinstance(usage, dict):
            continue
        value = usage.get(key, 0) or 0
        total += _numeric(value, f"usage {key}")
    return total


def summarize_group(records: Sequence[JsonDict]) -> JsonDict:
    attempted = [record for record in records if not record.get("error")]
    scores = [obj(record["score"], "record score") for record in attempted]
    component_correct = sum(
        _numeric(score["component_correct"], "component_correct") for score in scores
    )
    component_total = sum(_numeric(score["component_total"], "component_total") for score in scores)
    latencies = [
        _numeric(record["latency_ms"], "latency_ms")
        for record in attempted
        if record.get("latency_ms") is not None
    ]
    usage: JsonDict = {key: _usage_total(records, key) for key in USAGE_KEYS}
    exact = sum(1 for score in scores if score["correct"])
    return {
        "requests": len(records),
        "attempted": len(attempted),
        "errors": len(records) - len(attempted),
        "exact": exact,
        "exact_accuracy": (exact / len(attempted) if attempted else 0.0),
        "component_correct": component_correct,
        "component_total": component_total,
        "component_accuracy": (component_correct / component_total if component_total else 0.0),
        "avg_latency_ms": statistics.mean(latencies) if latencies else None,
        "median_latency_ms": statistics.median(latencies) if latencies else None,
        "usage": usage,
        "prompt_characters": sum(
            _numeric(record["prompt_characters"], "prompt_characters") for record in records
        ),
    }
