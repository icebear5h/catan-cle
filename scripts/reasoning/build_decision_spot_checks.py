#!/usr/bin/env python3
"""Build provider-free bucket evidence for a replay decision-eval artifact."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from evals.decision_buckets import load_decision_bucket_suite
from evals.decision_spot_checks.shapes import dict_or_empty
from evals.json_types import JsonDict, as_dicts, as_list, as_str
from evals.replay_action_diff import scan_replay
from scripts.reasoning.jsonio import utc_now, write_json, write_jsonl

INDEX_SCHEMA = "decision-bucket-index-v1"


def build_bucket_index(
    game_id: str,
    target_player: str,
    output_path: Path,
) -> JsonDict:
    suite = load_decision_bucket_suite()
    scan = scan_replay(game_id, target_player)
    semantic_errors = scan["semantic_errors"]
    if semantic_errors:
        raise RuntimeError(
            f"Replay scan found {len(as_list(semantic_errors, 'semantic_errors'))} "
            "semantic errors"
        )

    rows: list[JsonDict] = []
    stage_counts: Counter[str] = Counter()
    bucket_counts: Counter[str] = Counter()
    episode_ids: dict[str, set[str]] = {
        bucket.id: set() for bucket in suite.buckets
    }
    for record in as_dicts(scan["records"], "scan records"):
        assignment = dict_or_empty(record.get("bucket_assignment"), "bucket_assignment")
        stage_counts[str(assignment.get("stage") or "unknown")] += 1
        for bucket_value in as_list(assignment.get("bucket_ids", []), "bucket_ids"):
            bucket_id = as_str(bucket_value, "bucket_id")
            bucket_counts[bucket_id] += 1
            episode_id = dict_or_empty(
                assignment.get("episode_ids"), "episode_ids"
            ).get(bucket_id)
            if episode_id:
                episode_ids[bucket_id].add(str(episode_id))
        rows.append(
            {
                "schema": INDEX_SCHEMA,
                "decision_id": record["decision_id"],
                "game_id": record["game_id"],
                "replay_index": record["replay_index"],
                "source_replay_index": record.get("source_replay_index"),
                "classification": record.get("classification"),
                "state_features": record.get("state_features"),
                "bucket_assignment": assignment,
            }
        )

    write_jsonl(output_path, rows)
    summary: JsonDict = {
        "schema": INDEX_SCHEMA,
        "generated_at": utc_now(),
        "suite_id": suite.id,
        "suite_version": suite.version,
        "game_id": game_id,
        "target_player": target_player,
        "target_player_id": scan["target_player_id"],
        "target_engine_color": scan["target_engine_color"],
        "record_count": len(rows),
        "stage_counts": {key: count for key, count in sorted(stage_counts.items())},
        "bucket_decision_counts": {
            bucket.id: bucket_counts[bucket.id] for bucket in suite.buckets
        },
        "bucket_episode_counts": {
            bucket.id: len(episode_ids[bucket.id]) for bucket in suite.buckets
        },
        "index_file": output_path.name,
    }
    summary_path = output_path.with_suffix(".summary.json")
    write_json(summary_path, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-id", required=True)
    parser.add_argument("--target-player", default="captured")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_bucket_index(
        game_id=args.game_id,
        target_player=args.target_player,
        output_path=args.output,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
