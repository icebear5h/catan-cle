#!/usr/bin/env python
"""Build a reproducible comparison artifact from CatanBoardBench response runs."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


JsonDict = dict[str, Any]


def iter_jsonl(path: Path) -> Iterable[JsonDict]:
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc


def parse_named_path(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label.strip() or not raw_path.strip():
        raise ValueError(f"expected LABEL=RUN_DIR, got {value!r}")
    return label.strip(), Path(raw_path.strip())


def load_run(run_dir: Path) -> list[JsonDict]:
    response_path = run_dir / "responses.jsonl"
    if not response_path.exists():
        raise FileNotFoundError(response_path)
    records = list(iter_jsonl(response_path))
    question_ids = [record["question_id"] for record in records]
    duplicates = sorted(
        {question_id for question_id in question_ids if question_ids.count(question_id) > 1}
    )
    if duplicates:
        raise ValueError(f"duplicate question ids in {response_path}: {duplicates[:5]}")
    return records


def nested_usage_value(usage: JsonDict, section: str, key: str) -> int:
    details = usage.get(section) or {}
    value = details.get(key, 0) if isinstance(details, dict) else 0
    return int(value or 0)


def summarize_records(records: Sequence[JsonDict]) -> JsonDict:
    successful = [record for record in records if not record.get("error")]
    exact_correct = sum(bool(record["score"]["correct"]) for record in successful)
    component_correct = sum(int(record["score"]["component_correct"]) for record in successful)
    component_total = sum(int(record["score"]["component_total"]) for record in successful)
    latencies = [
        int(record["latency_ms"]) for record in successful if record.get("latency_ms") is not None
    ]

    usage_totals = defaultdict(int)
    for record in successful:
        usage = record.get("usage") or {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            usage_totals[key] += int(usage.get(key, 0) or 0)
        usage_totals["image_tokens"] += nested_usage_value(
            usage, "prompt_tokens_details", "image_tokens"
        )
        usage_totals["reasoning_tokens"] += nested_usage_value(
            usage, "completion_tokens_details", "reasoning_tokens"
        )

    categories: dict[str, JsonDict] = {}
    for category in sorted({record["category"] for record in records}):
        category_records = [record for record in successful if record["category"] == category]
        category_exact = sum(bool(record["score"]["correct"]) for record in category_records)
        category_component_correct = sum(
            int(record["score"]["component_correct"]) for record in category_records
        )
        category_component_total = sum(
            int(record["score"]["component_total"]) for record in category_records
        )
        categories[category] = {
            "attempted": len(category_records),
            "exact_correct": category_exact,
            "exact_accuracy": category_exact / len(category_records) if category_records else None,
            "component_correct": category_component_correct,
            "component_total": category_component_total,
            "component_accuracy": (
                category_component_correct / category_component_total
                if category_component_total
                else None
            ),
        }

    return {
        "requests": len(records),
        "attempted": len(successful),
        "errors": len(records) - len(successful),
        "exact_correct": exact_correct,
        "exact_accuracy": exact_correct / len(successful) if successful else None,
        "component_correct": component_correct,
        "component_total": component_total,
        "component_accuracy": component_correct / component_total if component_total else None,
        "latency_ms": {
            "mean": statistics.mean(latencies) if latencies else None,
            "median": statistics.median(latencies) if latencies else None,
        },
        "usage": dict(sorted(usage_totals.items())),
        "models": sorted({record.get("model_id") for record in records}),
        "served_models": sorted({record.get("served_model") for record in successful}),
        "providers": sorted({record.get("provider") for record in successful}),
        "categories": categories,
    }


def exact_two_sided_binomial_p(first_only: int, second_only: int) -> float | None:
    discordant = first_only + second_only
    if discordant == 0:
        return None
    tail = sum(math.comb(discordant, index) for index in range(min(first_only, second_only) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def paired_exact(
    first_label: str,
    first_records: Sequence[JsonDict],
    second_label: str,
    second_records: Sequence[JsonDict],
) -> JsonDict:
    first_by_id = {
        record["question_id"]: record for record in first_records if not record.get("error")
    }
    second_by_id = {
        record["question_id"]: record for record in second_records if not record.get("error")
    }
    question_ids = sorted(set(first_by_id) & set(second_by_id))
    pairs = []
    for question_id in question_ids:
        first = first_by_id[question_id]
        second = second_by_id[question_id]
        if first["expected"] != second["expected"] or first["category"] != second["category"]:
            raise ValueError(f"paired question contract differs for {question_id}")
        pairs.append((bool(first["score"]["correct"]), bool(second["score"]["correct"])))

    both_correct = sum(first and second for first, second in pairs)
    first_only = sum(first and not second for first, second in pairs)
    second_only = sum(second and not first for first, second in pairs)
    both_wrong = sum(not first and not second for first, second in pairs)
    return {
        "first": first_label,
        "second": second_label,
        "pairs": len(pairs),
        "both_correct": both_correct,
        "first_only": first_only,
        "second_only": second_only,
        "both_wrong": both_wrong,
        "first_exact": both_correct + first_only,
        "second_exact": both_correct + second_only,
        "second_minus_first_percentage_points": (
            100.0 * (second_only - first_only) / len(pairs) if pairs else None
        ),
        "exact_mcnemar_binomial_p": exact_two_sided_binomial_p(first_only, second_only),
    }


def build_comparison(
    named_runs: Sequence[tuple[str, Path]], pairs: Sequence[tuple[str, str]]
) -> JsonDict:
    if len({label for label, _ in named_runs}) != len(named_runs):
        raise ValueError("run labels must be unique")
    records = {label: load_run(path) for label, path in named_runs}
    unknown_pair_labels = sorted(
        {label for pair in pairs for label in pair if label not in records}
    )
    if unknown_pair_labels:
        raise ValueError(f"unknown pair labels: {unknown_pair_labels}")
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runs": {
            label: {"run_dir": str(path), **summarize_records(records[label])}
            for label, path in named_runs
        },
        "paired_exact": {
            f"{first}_vs_{second}": paired_exact(first, records[first], second, records[second])
            for first, second in pairs
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="LABEL=RUN_DIR")
    parser.add_argument(
        "--pair",
        action="append",
        default=[],
        help="FIRST_LABEL,SECOND_LABEL for a paired exact comparison",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    named_runs = [parse_named_path(value) for value in args.run]
    pairs = []
    for value in args.pair:
        labels = [label.strip() for label in value.split(",")]
        if len(labels) != 2 or not all(labels):
            raise SystemExit(f"expected FIRST_LABEL,SECOND_LABEL, got {value!r}")
        pairs.append((labels[0], labels[1]))
    comparison = build_comparison(named_runs, pairs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
