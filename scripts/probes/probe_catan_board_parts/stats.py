"""Typed aggregation of part-level accuracy and failure counts."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import TypedDict

from cle.players.data import JsonValue
from scripts.probes.probe_catan_board_parts.failures import infer_failure_tokens
from scripts.probes.probe_catan_board_parts.tokens import (
    MODEL_PREFIX,
    canonical_category,
    classify_part,
)

JsonDict = dict[str, JsonValue]
SampleKey = tuple[str, str, JsonValue, JsonValue, JsonValue]

__all__ = [
    "Analysis",
    "FailureExample",
    "JsonDict",
    "PartStats",
    "SampleKey",
    "Totals",
    "analyze_records",
    "top_failures_by_part",
]


class FailureExample(TypedDict):
    """One mismatching answer kept for the diagnostics report."""

    sample_id: JsonValue
    question_id: JsonValue
    category: str
    model_key: JsonValue
    expected: str
    response: str
    failures: list[str]


@dataclass
class Totals:
    """Run-wide counters."""

    attempted: int = 0
    exact: int = 0
    component: int = 0
    component_total: int = 0
    errors: int = 0

    @property
    def component_accuracy(self) -> float:
        return self.component / self.component_total if self.component_total else 0.0

    @property
    def exact_accuracy(self) -> float:
        return self.exact / self.attempted if self.attempted else 0.0

    def as_json(self) -> JsonDict:
        return {
            "attempted": self.attempted,
            "exact": self.exact,
            "component": self.component,
            "component_total": self.component_total,
            "errors": self.errors,
            "component_accuracy": self.component_accuracy,
            "exact_accuracy": self.exact_accuracy,
        }


@dataclass
class PartStats:
    """Counters, failure tallies, and examples for one board part."""

    attempted: int = 0
    exact: int = 0
    component: int = 0
    component_total: int = 0
    errors: int = 0
    failure_counts: Counter[str] = field(default_factory=Counter)
    samples: list[SampleKey] = field(default_factory=list)
    examples: list[FailureExample] = field(default_factory=list)

    @property
    def component_accuracy(self) -> float:
        return self.component / self.component_total if self.component_total else 0.0

    @property
    def exact_accuracy(self) -> float:
        return self.exact / self.attempted if self.attempted else 0.0


@dataclass
class Analysis:
    """The full probe result."""

    totals: Totals
    parts: dict[str, PartStats]


def _count(value: JsonValue) -> int:
    """Mirror ``int(value or 0)`` without accepting non-numeric JSON."""
    if not value:
        return 0
    if isinstance(value, (bool, int, float, str)):
        return int(value)
    raise TypeError("score counters must be numeric")


def _score_of(row: JsonDict) -> JsonDict:
    score = row.get("score") or {}
    if not isinstance(score, dict):
        raise TypeError("score must be a JSON object")
    return score


def analyze_records(
    records: list[JsonDict], model_filter: str | None = None
) -> Analysis:
    selected = [
        r for r in records if model_filter is None or r.get(MODEL_PREFIX) == model_filter
    ]
    by_part: dict[str, PartStats] = {}
    totals = Totals()

    for row in selected:
        if row.get("error"):
            totals.errors += 1
            continue

        category = canonical_category(str(row.get("category", "")))
        part = classify_part(category)
        model_key = row.get(MODEL_PREFIX, "")
        score = _score_of(row)
        component_correct = _count(score.get("component_correct", 0))
        component_total = _count(score.get("component_total", 0))
        exact_ok = bool(score.get("correct", False))
        expected = str(row.get("expected", ""))
        response = str(row.get("response", ""))
        failures = infer_failure_tokens(category, expected, response)

        totals.attempted += 1
        totals.exact += 1 if exact_ok else 0
        totals.component += component_correct
        totals.component_total += component_total

        agg = by_part.setdefault(part, PartStats())
        agg.attempted += 1
        agg.exact += 1 if exact_ok else 0
        agg.component += component_correct
        agg.component_total += component_total
        agg.samples.append(
            (
                part,
                category,
                model_key,
                row.get("sample_id", ""),
                row.get("question_id", ""),
            )
        )
        if failures:
            for failure in failures:
                agg.failure_counts[failure] += 1
            agg.examples.append(
                FailureExample(
                    sample_id=row.get("sample_id"),
                    question_id=row.get("question_id"),
                    category=category,
                    model_key=model_key,
                    expected=expected,
                    response=response,
                    failures=failures,
                )
            )
            totals.errors += 1
        agg.errors += 1 if failures else 0

    return Analysis(totals=totals, parts=by_part)


def _sample_sort_key(item: FailureExample) -> str:
    return str(item["sample_id"])


def top_failures_by_part(
    part_stats: dict[str, PartStats], top_n: int
) -> dict[str, list[FailureExample]]:
    out: dict[str, list[FailureExample]] = {}
    for part, agg in part_stats.items():
        out[part] = sorted(agg.examples, key=_sample_sort_key)[:top_n]
    return out
