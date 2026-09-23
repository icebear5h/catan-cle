"""Bucket counts, response recency, and unique-index helpers."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable, Mapping, Sequence

from evals.decision_buckets import DecisionBucketSuite
from evals.decision_spot_checks.config import DecisionEvalArtifactError
from evals.decision_spot_checks.shapes import dict_or_empty
from evals.json_types import JsonDict, JsonValue, as_list


def _bucket_counts(
    decisions: Sequence[Mapping[str, JsonValue]],
    suite: DecisionBucketSuite,
) -> list[JsonDict]:
    decision_counts: Counter[str] = Counter()
    response_counts: Counter[str] = Counter()
    disagreement_counts: Counter[str] = Counter()
    episodes: defaultdict[str, set[str]] = defaultdict(set)
    for decision in decisions:
        for bucket_id in as_list(decision.get("bucket_ids", []), "bucket_ids"):
            # Only the suite's string ids are ever read back from the counters.
            if not isinstance(bucket_id, str):
                continue
            decision_counts[bucket_id] += 1
            episode_id = dict_or_empty(decision.get("episode_ids"), "episode_ids").get(
                bucket_id
            )
            if episode_id:
                episodes[bucket_id].add(str(episode_id))
            model = dict_or_empty(decision.get("model"), "decision model")
            if model.get("response_present"):
                response_counts[bucket_id] += 1
                if model.get("agreement") is False:
                    disagreement_counts[bucket_id] += 1
    return [
        {
            "bucket_id": bucket.id,
            "decision_count": decision_counts[bucket.id],
            "episode_count": len(episodes[bucket.id]),
            "response_count": response_counts[bucket.id],
            "disagreement_count": disagreement_counts[bucket.id],
        }
        for bucket in suite.buckets
    ]


def _latest_responses(
    rows: Iterable[Mapping[str, JsonValue]],
    model_id: str,
) -> dict[str, JsonDict]:
    latest: dict[str, JsonDict] = {}
    for row in rows:
        if row.get("model_id") != model_id:
            continue
        decision_id = row.get("decision_id")
        if isinstance(decision_id, str) and decision_id:
            latest[decision_id] = dict(row)
    return latest


def _unique_index(
    rows: Iterable[Mapping[str, JsonValue]],
    label: str,
) -> dict[str, JsonDict]:
    indexed: dict[str, JsonDict] = {}
    for row in rows:
        decision_id = row.get("decision_id")
        if not isinstance(decision_id, str) or not decision_id:
            raise DecisionEvalArtifactError(f"{label} row is missing decision_id")
        if decision_id in indexed:
            raise DecisionEvalArtifactError(f"Duplicate {label} decision_id: {decision_id}")
        indexed[decision_id] = dict(row)
    return indexed
