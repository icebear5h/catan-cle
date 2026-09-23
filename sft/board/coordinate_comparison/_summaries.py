"""Paired atlas/coordinate outcome summaries over rescored comparison records."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import cast

from sft.board.coordinate_comparison._constants import (
    REPRESENTATIONS,
    SCHEMA,
    PairArm,
)
from sft.board.coordinate_comparison._mapping import _require, _same
from sft.board.coordinate_comparison._scoring import (
    _pair_identity,
    score_coordinate_comparison,
)
from sft.json_types import JsonDict, JsonLikeDict, as_dict, as_str


def _summarize_pairs(pairs: dict[str, dict[str, PairArm]]) -> JsonLikeDict:
    outcomes: dict[str, list[str]] = {
        name: [] for name in ("atlas_only", "coordinates_only", "both", "neither")}
    correct_counts: Counter[str] = Counter()
    valid_counts: Counter[str] = Counter()
    invalid_ids: dict[str, list[str]] = {rep: [] for rep in REPRESENTATIONS}
    for pair_id, arms in pairs.items():
        a, c = (arms[rep]["score"]["correct"] for rep in REPRESENTATIONS)
        outcome = "both" if a and c else "atlas_only" if a else "coordinates_only" if c else "neither"
        outcomes[outcome].append(pair_id)
        for rep, item in arms.items():
            score = item["score"]
            correct_counts[rep] += 1 if score["correct"] else 0
            valid_counts[rep] += 1 if score["format_valid"] else 0
            if not score["format_valid"]:
                invalid_ids[rep].append(item["id"])
    by_representation: JsonLikeDict = {
        rep: {"rows": len(pairs), "correct": correct_counts[rep],
              "format_valid": valid_counts[rep], "format_invalid_ids": invalid_ids[rep],
              "accuracy": correct_counts[rep] / len(pairs) if pairs else None,
              "format_invalid": len(pairs) - valid_counts[rep],
              "format_valid_rate": valid_counts[rep] / len(pairs) if pairs else None}
        for rep in REPRESENTATIONS
    }
    return {
        "pairs": len(pairs), "rows": 2 * len(pairs), "pair_ids": list(pairs),
        **{name: len(values) for name, values in outcomes.items()},
        "outcome_pair_ids": outcomes, "by_representation": by_representation,
        "coordinates_minus_atlas_accuracy": ((len(outcomes["coordinates_only"]) - len(outcomes["atlas_only"])) / len(pairs)
                                              if pairs else None),
    }


def paired_summary(records: list[JsonDict]) -> JsonLikeDict:
    """Require complete pairs (subsets allowed), rescore raw outputs, retain outcome IDs."""
    _require(isinstance(records, list), "records must be a list")
    pairs: dict[str, dict[str, PairArm]] = {}
    seen: set[object] = set()
    for record in records:
        _require(isinstance(record, dict) and {"id", "metadata", "expected", "response"} <= record.keys(),
                 "incomplete comparison record")
        metadata = as_dict(record["metadata"])
        score = score_coordinate_comparison(cast("str", record["expected"]),
                                            cast("str", record["response"]), metadata)
        _require(score is not None, "noncomparison record in paired summary")
        pair_id = as_str(metadata["pair_id"])
        representation = as_str(metadata["representation"])
        _require(record["id"] == f"{pair_id}/{representation}" and record["id"] not in seen, "duplicate/mismatched record ID")
        seen.add(record["id"])
        arms = pairs.setdefault(pair_id, {})
        _require(representation not in arms, "duplicate pair arm")
        arms[representation] = PairArm(id=as_str(record["id"]), metadata=metadata,
                                       score=cast("JsonLikeDict", score))
    by_task: defaultdict[str, dict[str, dict[str, PairArm]]] = defaultdict(dict)
    by_area: defaultdict[str, dict[str, dict[str, PairArm]]] = defaultdict(dict)
    for pair_id, arms in pairs.items():
        _require(set(arms) == set(REPRESENTATIONS), f"incomplete pair: {pair_id}")
        atlas, coordinates = (arms[rep]["metadata"] for rep in REPRESENTATIONS)
        _same(_pair_identity(atlas), _pair_identity(coordinates), "paired source/provenance")
        _require(coordinates["comparison_position"] == cast("int", atlas["comparison_position"]) + 1,
                 "paired original positions mismatch")
        by_task[as_str(atlas["task_type"])][pair_id] = arms
        by_area[as_str(atlas["area"])][pair_id] = arms
    return {
        "schema": SCHEMA, "rescored_from_raw": True, **_summarize_pairs(pairs),
        "by_task": {task: _summarize_pairs(group) for task, group in sorted(by_task.items())},
        "by_area": {area: _summarize_pairs(group) for area, group in sorted(by_area.items())},
    }
