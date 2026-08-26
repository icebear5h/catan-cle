#!/usr/bin/env python3
"""Build provider-free bucket evidence for a replay decision-eval artifact."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from evals.decision_buckets import load_decision_bucket_suite
from evals.replay_action_diff import scan_replay


INDEX_SCHEMA = "decision-bucket-index-v1"


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def build_bucket_index(
    game_id: str,
    target_player: str,
    output_path: Path,
) -> dict[str, Any]:
    suite = load_decision_bucket_suite()
    scan = scan_replay(game_id, target_player)
    if scan["semantic_errors"]:
        raise RuntimeError(
            f"Replay scan found {len(scan['semantic_errors'])} semantic errors"
        )

    rows = []
    stage_counts: Counter[str] = Counter()
    bucket_counts: Counter[str] = Counter()
    episode_ids: dict[str, set[str]] = {
        bucket.id: set() for bucket in suite.buckets
    }
    for record in scan["records"]:
        assignment = record.get("bucket_assignment") or {}
        stage_counts[str(assignment.get("stage") or "unknown")] += 1
        for bucket_id in assignment.get("bucket_ids", []):
            bucket_counts[bucket_id] += 1
            episode_id = (assignment.get("episode_ids") or {}).get(bucket_id)
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

    _write_jsonl(output_path, rows)
    summary = {
        "schema": INDEX_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "suite_id": suite.id,
        "suite_version": suite.version,
        "game_id": game_id,
        "target_player": target_player,
        "target_player_id": scan["target_player_id"],
        "target_engine_color": scan["target_engine_color"],
        "record_count": len(rows),
        "stage_counts": dict(sorted(stage_counts.items())),
        "bucket_decision_counts": {
            bucket.id: bucket_counts[bucket.id] for bucket in suite.buckets
        },
        "bucket_episode_counts": {
            bucket.id: len(episode_ids[bucket.id]) for bucket in suite.buckets
        },
        "index_file": output_path.name,
    }
    summary_path = output_path.with_suffix(".summary.json")
    _write_json(summary_path, summary)
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
