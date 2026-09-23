"""Per-variant and per-category rollups."""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import datetime, timezone

from scripts.board_bench.run.eval_catan_board_bench_ascii_variations.constants import (
    percentile,
    usage_reasoning_tokens,
)
from scripts.board_bench.shapes import JsonDict, integer, numeric, obj, text

__all__ = ["summarize", "summarize_group"]


def summarize(records: Sequence[JsonDict], plan: JsonDict) -> JsonDict:
    by_variant: dict[str, list[JsonDict]] = defaultdict(list)
    for record in records:
        by_variant[text(record["variant"], "record variant")].append(record)
    variants: JsonDict = {}
    for variant, rows in sorted(by_variant.items()):
        result = summarize_group(rows)
        by_category: dict[str, list[JsonDict]] = defaultdict(list)
        for row in rows:
            by_category[text(row["category"], "record category")].append(row)
        result["categories"] = {
            category: summarize_group(category_rows)
            for category, category_rows in sorted(by_category.items())
        }
        variants[variant] = result
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "plan": plan,
        "records": len(records),
        "complete": len(records) == plan["request_count"],
        "overall": summarize_group(records),
        "variants": variants,
    }


def _usage_rows(successful: Sequence[JsonDict]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for row in successful:
        usage = row.get("usage") or {}
        rows.append(usage if isinstance(usage, dict) else {})
    return rows


def summarize_group(records: Sequence[JsonDict]) -> JsonDict:
    successful = [row for row in records if not row.get("error")]
    latencies = sorted(
        integer(row["latency_ms"], "latency_ms")
        for row in successful
        if row.get("latency_ms") is not None
    )
    usage = _usage_rows(successful)
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
        "protocol_exact_accuracy": (protocol_exact / len(successful) if successful else 0.0),
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
                text(row["provider"], "record provider")
                for row in successful
                if row.get("provider") is not None
            ).items()
        },
        "cost": sum(numeric(item.get("cost", 0.0) or 0.0, "cost") for item in usage),
        "median_latency_ms": statistics.median(latencies) if latencies else None,
        "p95_latency_ms": percentile(latencies, 0.95) if latencies else None,
    }
