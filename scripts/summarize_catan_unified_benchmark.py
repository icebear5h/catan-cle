#!/usr/bin/env python
"""Unify current hosted-model perception and strict image/text benchmark runs."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from evals.catan_board_bench.ascii_variations import score_strict_json_answer
from scripts.render_catan_strict_vision_probe import json_digest, read_jsonl, write_json
from scripts.summarize_catan_board_bench_runs import (
    load_run,
    parse_named_path,
    summarize_records,
)


JsonDict = dict[str, Any]
SCHEMA = "catan_unified_hosted_benchmark/v1"
CURRENT_QWEN_MODEL = "qwen/qwen3.8-max"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--perception-run",
        action="append",
        required=True,
        help="LABEL=RUN_DIR for a current 110-question run",
    )
    parser.add_argument(
        "--strict-image-run",
        action="append",
        required=True,
        help="LABEL=RUN_DIR for a current strict 60 image run",
    )
    parser.add_argument(
        "--strict-text-run",
        required=True,
        help="LABEL=RUN_DIR for the current Qwen Max strict 60 text run",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    comparison = build_unified_comparison(
        perception_runs=[parse_named_path(value) for value in args.perception_run],
        strict_image_runs=[parse_named_path(value) for value in args.strict_image_run],
        strict_text_run=parse_named_path(args.strict_text_run),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, comparison)
    print(args.output)
    return 0


def build_unified_comparison(
    *,
    perception_runs: Sequence[tuple[str, Path]],
    strict_image_runs: Sequence[tuple[str, Path]],
    strict_text_run: tuple[str, Path],
) -> JsonDict:
    labels = [
        *(label for label, _path in perception_runs),
        *(label for label, _path in strict_image_runs),
        strict_text_run[0],
    ]
    if len(labels) != len(set(labels)):
        raise ValueError("all unified run labels must be unique")

    perception = {}
    for label, run_dir in perception_runs:
        records = load_run(run_dir)
        validate_current_model_records(records)
        if len(records) != 110:
            raise ValueError(f"perception run {label} must contain 110 records")
        perception[label] = {
            "run_dir": str(run_dir),
            **summarize_records(records),
        }

    strict_image = {}
    strict_image_records = {}
    for label, run_dir in strict_image_runs:
        records = load_strict_run(run_dir, expected_records=60)
        validate_current_model_records(records)
        strict_image_records[label] = records
        strict_image[label] = summarize_strict_run(run_dir, records)

    text_label, text_run_dir = strict_text_run
    text_records = load_strict_run(text_run_dir, expected_records=60)
    validate_current_model_records(text_records)
    text_models = {row["model_id"] for row in text_records}
    if text_models != {CURRENT_QWEN_MODEL}:
        raise ValueError(f"strict text run must use {CURRENT_QWEN_MODEL}, got {text_models}")
    strict_text = {text_label: summarize_strict_run(text_run_dir, text_records)}

    qwen_image_labels = [
        label
        for label, records in strict_image_records.items()
        if {row["model_id"] for row in records} == {CURRENT_QWEN_MODEL}
    ]
    if len(qwen_image_labels) != 1:
        raise ValueError("exactly one current Qwen Max strict image run is required")
    qwen_image_label = qwen_image_labels[0]
    matched_qwen = paired_modalities(
        image_label=qwen_image_label,
        image_records=strict_image_records[qwen_image_label],
        text_label=text_label,
        text_records=text_records,
    )

    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reporting_policy": {
            "current_models_only": True,
            "old_qwen_checkpoints_excluded": True,
            "historical_artifacts_deleted": False,
            "combined_numerator_reported": False,
            "reason": (
                "The 110 component-presence perception diagnostic and strict 60 "
                "typed-JSON cohort measure different contracts and remain separate panels."
            ),
        },
        "cohorts": {
            "perception_110": {
                "questions": 110,
                "scoring": "CatanBoardBench exact plus component-presence",
                "models": perception,
            },
            "strict_raw_vision_60": {
                "questions": 60,
                "scoring": "strict_typed_json/v2",
                "image_annotation": None,
                "identity_projection": "canonical_engine_ids",
                "models": strict_image,
            },
            "strict_text_60": {
                "questions": 60,
                "scoring": "strict_typed_json/v2",
                "format": "indexed_tile_rows",
                "models": strict_text,
            },
        },
        "matched_qwen_max_image_vs_text": matched_qwen,
        "artifact_digest": json_digest(
            {
                "perception": perception,
                "strict_image": strict_image,
                "strict_text": strict_text,
                "matched_qwen": matched_qwen,
            }
        ),
    }


def load_strict_run(run_dir: Path, *, expected_records: int) -> list[JsonDict]:
    response_path = run_dir / "responses.jsonl"
    records = read_jsonl(response_path)
    question_ids = [row["question_id"] for row in records]
    if len(records) != expected_records or len(set(question_ids)) != expected_records:
        raise ValueError(
            f"strict run {run_dir} must have {expected_records} unique records, "
            f"got {len(records)}/{len(set(question_ids))}"
        )
    errors = [row for row in records if row.get("error")]
    if errors:
        raise ValueError(f"strict run {run_dir} contains {len(errors)} errors")
    normalized = []
    for row in records:
        copy = dict(row)
        copy["score"] = score_strict_json_answer(row["expected"], row["response"])
        normalized.append(copy)
    return normalized


def summarize_strict_run(run_dir: Path, records: Sequence[JsonDict]) -> JsonDict:
    summary = json.loads((run_dir / "summary.json").read_text())
    exact = sum(row["score"]["correct"] for row in records)
    json_valid = sum(row["score"]["json_valid"] for row in records)
    if summary["overall"]["exact"] != exact or summary["overall"]["json_valid"] != json_valid:
        raise ValueError(f"stored strict summary is stale: {run_dir}")
    return {
        "run_dir": str(run_dir),
        "model": next(iter({row["model_id"] for row in records})),
        "requests": len(records),
        "errors": 0,
        "exact": exact,
        "exact_accuracy": exact / len(records),
        "json_valid": json_valid,
        "json_valid_accuracy": json_valid / len(records),
        "protocol_exact": summary["overall"]["protocol_exact"],
        "prompt_tokens": summary["overall"]["prompt_tokens"],
        "completion_tokens": summary["overall"]["completion_tokens"],
        "reasoning_tokens": summary["overall"]["reasoning_tokens"],
        "median_latency_ms": summary["overall"]["median_latency_ms"],
        "providers": summary["overall"]["providers"],
        "categories": {
            category: {
                "requests": values["requests"],
                "exact": values["exact"],
                "exact_accuracy": values["exact_accuracy"],
                "json_valid": values["json_valid"],
            }
            for category, values in sorted(summary["categories"].items())
        },
    }


def validate_current_model_records(records: Sequence[JsonDict]) -> None:
    for model_id in {row.get("model_id", "") for row in records}:
        lowered = model_id.lower()
        if "qwen" in lowered and model_id != CURRENT_QWEN_MODEL:
            raise ValueError(f"old Qwen checkpoint is excluded from active scorecard: {model_id}")


def paired_modalities(
    *,
    image_label: str,
    image_records: Sequence[JsonDict],
    text_label: str,
    text_records: Sequence[JsonDict],
) -> JsonDict:
    image_by_id = {row["question_id"]: row for row in image_records}
    text_by_id = {row["question_id"]: row for row in text_records}
    question_ids = sorted(set(image_by_id) & set(text_by_id))
    if len(question_ids) != 60:
        raise ValueError(f"matched modality comparison requires 60 pairs, got {len(question_ids)}")
    pairs = []
    for question_id in question_ids:
        image = image_by_id[question_id]
        text = text_by_id[question_id]
        if image["category"] != text["category"]:
            raise ValueError(f"category differs across modalities: {question_id}")
        pairs.append((bool(image["score"]["correct"]), bool(text["score"]["correct"])))
    both_correct = sum(image and text for image, text in pairs)
    image_only = sum(image and not text for image, text in pairs)
    text_only = sum(text and not image for image, text in pairs)
    both_wrong = sum(not image and not text for image, text in pairs)
    image_prompt_tokens = sum(
        int((row.get("usage") or {}).get("prompt_tokens", 0) or 0) for row in image_records
    )
    text_prompt_tokens = sum(
        int((row.get("usage") or {}).get("prompt_tokens", 0) or 0) for row in text_records
    )
    return {
        "image": image_label,
        "text": text_label,
        "pairs": len(pairs),
        "both_correct": both_correct,
        "image_only": image_only,
        "text_only": text_only,
        "both_wrong": both_wrong,
        "image_exact": both_correct + image_only,
        "text_exact": both_correct + text_only,
        "text_minus_image_percentage_points": 100 * (text_only - image_only) / len(pairs),
        "exact_mcnemar_binomial_p": exact_two_sided_binomial_p(image_only, text_only),
        "prompt_tokens": {
            "image": image_prompt_tokens,
            "text": text_prompt_tokens,
            "text_over_image": text_prompt_tokens / image_prompt_tokens,
        },
    }


def exact_two_sided_binomial_p(first_only: int, second_only: int) -> float | None:
    discordant = first_only + second_only
    if discordant == 0:
        return None
    tail = sum(math.comb(discordant, index) for index in range(min(first_only, second_only) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


if __name__ == "__main__":
    raise SystemExit(main())
