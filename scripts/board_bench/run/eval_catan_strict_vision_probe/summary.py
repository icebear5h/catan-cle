"""Overall and per-category rollups for the vision probe."""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import datetime, timezone

from evals.catan_board_bench.ascii_variations import score_strict_json_answer
from scripts.board_bench.run.eval_catan_strict_vision_probe.constants import (
    percentile,
    usage_reasoning_tokens,
)
from scripts.board_bench.shapes import JsonDict, integer, numeric, obj, text

__all__ = ["summarize", "summarize_group"]


def summarize(records: Sequence[JsonDict], plan: JsonDict) -> JsonDict:
    normalized: list[JsonDict] = []
    for record in records:
        copy = dict(record)
        copy["score"] = score_strict_json_answer(
            obj(record["expected"], "expected answer"), text(record["response"], "response")
        )
        normalized.append(copy)
    by_category: dict[str, list[JsonDict]] = defaultdict(list)
    for record in normalized:
        by_category[text(record["category"], "category")].append(record)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "plan": plan,
        "records": len(normalized),
        "complete": len(normalized) == plan["request_count"],
        "overall": summarize_group(normalized),
        "categories": {
            category: summarize_group(rows) for category, rows in sorted(by_category.items())
        },
    }


def summarize_group(records: Sequence[JsonDict]) -> JsonDict:
    successful = [row for row in records if not row.get("error")]
    latencies = sorted(
        integer(row["latency_ms"], "latency_ms")
        for row in successful
        if row.get("latency_ms") is not None
    )
    usage: list[JsonDict] = []
    for row in successful:
        entry = row.get("usage") or {}
        usage.append(entry if isinstance(entry, dict) else {})
    scores = [obj(row["score"], "record score") for row in successful]
    exact = sum(numeric(score["correct"], "score correct") for score in scores)
    json_valid = sum(numeric(score["json_valid"], "score json_valid") for score in scores)
    protocol_exact = sum(
        numeric(score["protocol_exact"], "score protocol_exact") for score in scores
    )
    return {
        "requests": len(records),
        "successful": len(successful),
        "errors": len(records) - len(successful),
        "exact": exact,
        "exact_accuracy": exact / len(successful) if successful else 0.0,
        "json_valid": json_valid,
        "json_valid_accuracy": json_valid / len(successful) if successful else 0.0,
        "protocol_exact": protocol_exact,
        "protocol_exact_accuracy": protocol_exact / len(successful) if successful else 0.0,
        "prompt_tokens": sum(
            numeric(item.get("prompt_tokens", 0) or 0, "prompt_tokens") for item in usage
        ),
        "completion_tokens": sum(
            numeric(item.get("completion_tokens", 0) or 0, "completion_tokens") for item in usage
        ),
        "reasoning_tokens": sum(usage_reasoning_tokens(item) for item in usage),
        "providers": {
            provider: count
            for provider, count in Counter(
                str(row["provider"]) for row in successful
            ).items()
        },
        "median_latency_ms": statistics.median(latencies) if latencies else None,
        "p95_latency_ms": percentile(latencies, 0.95),
    }
