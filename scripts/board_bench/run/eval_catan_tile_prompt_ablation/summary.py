"""Per-condition metrics, paired effects, and the factorial rollup."""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import datetime, timezone

from scripts.board_bench.run.eval_catan_tile_prompt_ablation import (
    ANGLE_CONDITIONS,
    CONDITIONS,
    DESCRIBED_CONDITIONS,
    RESOURCE_CLASSES,
)
from scripts.board_bench.run.eval_catan_tile_prompt_ablation.defaults import (
    TileJob,
    percentile,
    usage_reasoning_tokens,
)
from scripts.board_bench.shapes import JsonDict, integer, numeric, obj, strings, text

__all__ = ["factorial_summary", "metric_summary", "paired_effect", "summarize"]


def _score(row: JsonDict) -> JsonDict:
    return obj(row["score"], "record score")


def _truth(row: JsonDict) -> JsonDict:
    return obj(row["truth"], "record truth")


def _usage(row: JsonDict) -> JsonDict:
    usage = row.get("usage") or {}
    return usage if isinstance(usage, dict) else {}


def _flag(row: JsonDict, key: str) -> bool:
    return bool(_score(row)[key])


def _nested_confusion(
    confusion: Counter[tuple[str, str]],
) -> dict[str, Counter[str]]:
    nested: dict[str, Counter[str]] = defaultdict(Counter)
    for (expected, predicted), count in confusion.items():
        nested[expected][predicted] += count
    return dict(sorted(nested.items()))


def _predicted_resource(row: JsonDict) -> str:
    parsed = _score(row)["parsed"]
    if not isinstance(parsed, dict):
        return "INVALID"
    return str(parsed.get("resource", "INVALID"))


def metric_summary(records: Sequence[JsonDict]) -> JsonDict:
    numbered = [row for row in records if _truth(row)["number"] is not None]
    latencies = [
        integer(row["latency_ms"], "latency_ms")
        for row in records
        if row.get("latency_ms") is not None
    ]
    confusion = Counter(
        (text(_truth(row)["resource"], "truth resource"), _predicted_resource(row))
        for row in records
    )
    pair_exact = sum(_flag(row, "pair_correct") for row in records)
    resource_exact = sum(_flag(row, "resource_correct") for row in records)
    number_exact = sum(_score(row)["number_correct"] is True for row in numbered)
    return {
        "requests": len(records),
        "pair_exact": pair_exact,
        "pair_accuracy": (pair_exact / len(records) if records else None),
        "resource_exact": resource_exact,
        "resource_accuracy": (resource_exact / len(records) if records else None),
        "number_exact": number_exact,
        "number_requests": len(numbered),
        "number_accuracy": (number_exact / len(numbered) if numbered else None),
        "protocol_valid": sum(_flag(row, "protocol_valid") for row in records),
        "protocol_exact": sum(_flag(row, "protocol_exact") for row in records),
        "wrapper_rescued": sum(
            _flag(row, "wrapper_rescued_content_correct") for row in records
        ),
        "prompt_tokens": sum(
            int(numeric(_usage(row).get("prompt_tokens", 0) or 0, "prompt_tokens"))
            for row in records
        ),
        "completion_tokens": sum(
            int(numeric(_usage(row).get("completion_tokens", 0) or 0, "completion_tokens"))
            for row in records
        ),
        "reasoning_tokens": sum(usage_reasoning_tokens(_usage(row)) for row in records),
        "cost": sum(
            float(numeric(_usage(row).get("cost", 0) or 0, "cost")) for row in records
        ),
        "median_latency_ms": statistics.median(latencies) if latencies else None,
        "p95_latency_ms": percentile(latencies, 0.95),
        "resource_confusion": {
            expected: {name: count for name, count in sorted(predicted.items())}
            for expected, predicted in _nested_confusion(confusion).items()
        },
    }


def paired_effect(
    by_key: dict[tuple[str, str], JsonDict],
    first: str,
    second: str,
    metric: str,
) -> JsonDict:
    question_ids = sorted(
        question_id
        for condition, question_id in by_key
        if condition == first and (second, question_id) in by_key
    )
    values_pairs = []
    for question_id in question_ids:
        first_value = _score(by_key[(first, question_id)])[metric]
        second_value = _score(by_key[(second, question_id)])[metric]
        if first_value is None or second_value is None:
            continue
        values_pairs.append((bool(first_value), bool(second_value)))
    first_only = sum(a and not b for a, b in values_pairs)
    second_only = sum(b and not a for a, b in values_pairs)
    first_exact = sum(a for a, _ in values_pairs)
    second_exact = sum(b for _, b in values_pairs)
    return {
        "first": first,
        "second": second,
        "metric": metric,
        "pairs": len(values_pairs),
        "first_exact": first_exact,
        "second_exact": second_exact,
        "first_only": first_only,
        "second_only": second_only,
        "effect_percentage_points": (
            100 * (first_exact - second_exact) / len(values_pairs)
            if values_pairs
            else None
        ),
    }


def factorial_summary(by_key: dict[tuple[str, str], JsonDict]) -> JsonDict:
    question_sets = {
        condition: {question_id for name, question_id in by_key if name == condition}
        for condition in CONDITIONS
    }
    if (
        len(by_key) != 408
        or any(len(question_sets[condition]) != 102 for condition in CONDITIONS)
        or len({frozenset(ids) for ids in question_sets.values()}) != 1
    ):
        return {"complete": False}
    condition_rates: dict[str, float | None] = {}
    for condition in CONDITIONS:
        rows = [row for (name, _), row in by_key.items() if name == condition]
        condition_rates[condition] = (
            sum(_flag(row, "pair_correct") for row in rows) / len(rows) if rows else None
        )
    if any(condition_rates[name] is None for name in CONDITIONS):
        return {"complete": False}
    rates = {name: value for name, value in condition_rates.items() if value is not None}
    angle_rate = statistics.mean(rates[name] for name in CONDITIONS if name in ANGLE_CONDITIONS)
    plain_rate = statistics.mean(
        rates[name] for name in CONDITIONS if name not in ANGLE_CONDITIONS
    )
    described_rate = statistics.mean(
        rates[name] for name in CONDITIONS if name in DESCRIBED_CONDITIONS
    )
    labels_rate = statistics.mean(
        rates[name] for name in CONDITIONS if name not in DESCRIBED_CONDITIONS
    )
    angle_description_gain = rates["angle_described"] - rates["angle_labels"]
    plain_description_gain = rates["plain_described"] - rates["plain_labels"]
    return {
        "complete": True,
        "condition_pair_accuracy": {name: value for name, value in condition_rates.items()},
        "angle_minus_plain_percentage_points": 100 * (angle_rate - plain_rate),
        "described_minus_labels_percentage_points": 100 * (described_rate - labels_rate),
        "interaction_percentage_points": 100
        * (angle_description_gain - plain_description_gain),
    }


def summarize(
    accepted: dict[tuple[str, str], JsonDict],
    *,
    jobs: Sequence[TileJob],
    plan: JsonDict,
) -> JsonDict:
    plan_conditions = strings(plan["conditions"], "plan conditions")
    by_condition: JsonDict = {}
    for condition in plan_conditions:
        rows = [row for (name, _), row in accepted.items() if name == condition]
        per_resource: JsonDict = {
            resource: metric_summary(
                [row for row in rows if _truth(row)["resource"] == resource]
            )
            for resource in RESOURCE_CLASSES
        }
        by_condition[condition] = {
            **metric_summary(rows),
            "per_resource": per_resource,
        }
    comparisons: JsonDict = {}
    if set(plan_conditions) == set(CONDITIONS):
        for metric in ("pair_correct", "resource_correct", "number_correct"):
            for first, second in (
                ("angle_labels", "plain_labels"),
                ("angle_described", "plain_described"),
                ("angle_described", "angle_labels"),
                ("plain_described", "plain_labels"),
            ):
                key = f"{metric}:{first}_vs_{second}"
                comparisons[key] = paired_effect(accepted, first, second, metric)
    return {
        "complete": len(accepted) == len(jobs),
        "records": len(accepted),
        "planned_requests": len(jobs),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall": metric_summary(list(accepted.values())),
        "conditions": by_condition,
        "paired_comparisons": comparisons,
        "factorial_pair_effects": (
            factorial_summary(accepted)
            if set(plan_conditions) == set(CONDITIONS)
            else {"complete": False}
        ),
        "plan": plan,
    }
