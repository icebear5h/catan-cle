#!/usr/bin/env python
"""Rescore piece-recognition outputs under explicit semantic alias policies."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from evals.catan_board_bench.scoring import score_answer
from scripts.board_bench.shapes import JsonDict, numeric, obj, text

RESOURCE_SYNONYMS = {
    "<FOREST>": "<WOOD>",
    "<WOOL>": "<SHEEP>",
    "<GRAIN>": "<WHEAT>",
}
BLUE_COLOR_COLLAPSE = {"<MYSTIC_BLUE>": "<BLUE>"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--qa", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    response_rows = read_jsonl(args.responses)
    qa_by_id = {row["id"]: row for row in read_jsonl(args.qa)}
    policies = {
        "resource_synonyms": RESOURCE_SYNONYMS,
        "resource_synonyms_plus_blue_color_collapse_diagnostic": {
            **RESOURCE_SYNONYMS,
            **BLUE_COLOR_COLLAPSE,
        },
    }

    rescored_rows: list[JsonDict] = []
    policy_rows: dict[str, list[JsonDict]] = defaultdict(list)
    for record in response_rows:
        qa = qa_by_id[record["question_id"]]
        policy_scores: JsonDict = {}
        policy_responses: JsonDict = {}
        for policy_name, substitutions in policies.items():
            transformed = apply_substitutions(text(record["response"], "response"), substitutions)
            score = score_answer(qa, transformed)
            policy_responses[policy_name] = transformed
            policy_scores[policy_name] = score
            policy_rows[policy_name].append(
                {
                    **record,
                    "policy_response": transformed,
                    "policy_score": score,
                }
            )
        rescored_rows.append(
            {
                "question_id": record["question_id"],
                "category": record["category"],
                "original_response": record["response"],
                "original_score": record["score"],
                "policy_responses": policy_responses,
                "policy_scores": policy_scores,
            }
        )

    policy_summaries: JsonDict = {}
    summary: JsonDict = {
        "schema": "catan_board_bench_piece_semantic_rescore/v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(args.responses),
        "source_sha256": hashlib.sha256(args.responses.read_bytes()).hexdigest(),
        "qa": str(args.qa),
        "policies": policy_summaries,
        "notes": {
            "resource_synonyms": (
                "Lexical diagnostic only; canonical Catan output remains strict."
            ),
            "resource_synonyms_plus_blue_color_collapse_diagnostic": (
                "MYSTIC_BLUE and BLUE are distinct engine colors. This policy is "
                "reported only to diagnose apparent color-family recognition."
            ),
        },
    }
    for policy_name, substitutions in policies.items():
        rows = policy_rows[policy_name]
        policy_summaries[policy_name] = {
            "substitutions": {key: value for key, value in substitutions.items()},
            "overall": summarize(rows),
            "groups": {
                "isolated": summarize(
                    [row for row in rows if text(row["category"], "category").startswith("isolated_")]
                ),
                "local_patch": summarize(
                    [row for row in rows if text(row["category"], "category").startswith("local_patch_")]
                ),
            },
            "categories": {
                category: summarize(
                    [row for row in rows if row["category"] == category]
                )
                for category in sorted({text(row["category"], "category") for row in rows})
            },
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "summary_semantic_aliases_v2.json", summary)
    write_jsonl(args.output_dir / "responses_semantic_aliases_v2.jsonl", rescored_rows)
    print(json.dumps(policy_summaries, indent=2, sort_keys=True))


def apply_substitutions(value: str, substitutions: dict[str, str]) -> str:
    result = value
    for source, target in substitutions.items():
        result = result.replace(source, target)
    return result


def summarize(records: Sequence[JsonDict]) -> JsonDict:
    scores = [obj(record["policy_score"], "policy score") for record in records]
    exact = sum(bool(score["correct"]) for score in scores)
    component_correct = sum(
        numeric(score["component_correct"], "component_correct") for score in scores
    )
    component_total = sum(
        numeric(score["component_total"], "component_total") for score in scores
    )
    return {
        "requests": len(records),
        "exact": exact,
        "exact_accuracy": exact / len(records) if records else 0.0,
        "component_correct": component_correct,
        "component_total": component_total,
        "component_accuracy": (
            component_correct / component_total if component_total else 0.0
        ),
    }


def read_jsonl(path: Path) -> list[JsonDict]:
    return [obj(json.loads(line), str(path)) for line in path.read_text().splitlines()]


def write_json(path: Path, payload: JsonDict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Sequence[JsonDict]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
