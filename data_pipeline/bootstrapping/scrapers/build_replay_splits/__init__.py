#!/usr/bin/env python3
"""Build leakage-safe Colonist replay split manifests.

This script operates on lightweight game-index JSON files from
``scrape_top_players.py --mode index``. It does not download replays. It removes
CatanBoardBench holdout games, deduplicates game IDs, preserves color-balance metadata,
and writes deterministic game-level splits plus step-fraction plans.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from data_pipeline.bootstrapping.scrapers.build_replay_splits._config import (
    COLONIST_COLOR_NAMES,
    DEFAULT_EXCLUDE_IDS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RAW_REPLAY_DIR,
    EARLY_MID_STEP_PLAN,
    LATE_HARD_STEP_PLAN,
    ROOT,
)
from data_pipeline.bootstrapping.scrapers.build_replay_splits._models import (
    GameEntry,
    Manifest,
    SourceRecord,
    SplitSummary,
    StepPlan,
    Summary,
)
from data_pipeline.bootstrapping.scrapers.build_replay_splits._records import (
    load_excluded_game_ids,
    load_index_records,
    load_json,
    norm_game_id,
    parse_modes,
    planned_steps,
    portable_path,
    replay_event_count,
    write_json,
)
from data_pipeline.bootstrapping.scrapers.build_replay_splits._report import write_markdown
from data_pipeline.bootstrapping.scrapers.build_replay_splits._splits import (
    split_balanced,
    summarize,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-files", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--name", default="colonist_replay_splits")
    parser.add_argument("--exclude-game-ids", type=Path, default=DEFAULT_EXCLUDE_IDS)
    parser.add_argument("--raw-replay-dir", type=Path, default=DEFAULT_RAW_REPLAY_DIR)
    parser.add_argument("--modes", default="Classic4P,Tournament")
    parser.add_argument("--seed", type=int, default=20260515)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--max-games", type=int)
    parser.add_argument("--min-turn-count", type=int, default=20)
    return parser.parse_args()


def _enriched_games(
    grouped: dict[str, GameEntry],
    excluded_ids: set[str],
    raw_replay_dir: Path,
    min_turn_count: int,
) -> tuple[list[GameEntry], int, int]:
    """Drop holdout and too-short games, then attach replay and step-plan facts."""

    games: list[GameEntry] = []
    excluded_count = 0
    short_count = 0
    for game_id, grouped_entry in grouped.items():
        if game_id in excluded_ids:
            excluded_count += 1
            continue
        turn_count = grouped_entry.get("turnCount")
        if turn_count is not None:
            if not isinstance(turn_count, (int, float, str)):
                raise TypeError(f"{game_id}: turnCount must be a JSON number or string")
            if int(turn_count) < min_turn_count:
                short_count += 1
                continue

        event_count = replay_event_count(raw_replay_dir, game_id)
        entry = grouped_entry.copy()
        entry["raw_replay_present"] = event_count is not None
        entry["raw_replay_path"] = (
            str((raw_replay_dir / f"{game_id}.json").relative_to(ROOT))
            if event_count is not None
            else None
        )
        entry["event_count"] = event_count
        entry["step_plans"] = {
            "early_mid": planned_steps(EARLY_MID_STEP_PLAN, event_count),
            "late_hard": planned_steps(LATE_HARD_STEP_PLAN, event_count),
        }
        games.append(entry)
    return games, excluded_count, short_count


def main() -> int:
    args = parse_args()
    modes = parse_modes(args.modes)
    excluded_ids = load_excluded_game_ids(args.exclude_game_ids)
    grouped = load_index_records(args.index_files, modes=modes)

    games, excluded_count, short_count = _enriched_games(
        grouped, excluded_ids, args.raw_replay_dir, args.min_turn_count
    )

    splits = split_balanced(
        games,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        max_games=args.max_games,
    )
    manifest: Manifest = {
        "schema": "colonist_replay_splits/v0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_index_files": [portable_path(path) for path in args.index_files],
        "exclude_game_ids": portable_path(args.exclude_game_ids),
        "excluded_game_count": excluded_count,
        "short_game_count": short_count,
        "modes": sorted(modes) if modes else "all",
        "min_turn_count": args.min_turn_count,
        "seed": args.seed,
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": round(1.0 - args.train_ratio - args.val_ratio, 6),
        },
        "step_plans": {
            "early_mid": EARLY_MID_STEP_PLAN,
            "late_hard": LATE_HARD_STEP_PLAN,
        },
        "summary": summarize(splits),
        "splits": splits,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_json = args.output_dir / f"{args.name}.json"
    output_md = args.output_dir / f"{args.name}.md"
    write_json(output_json, manifest)
    write_markdown(output_md, manifest)

    print(f"wrote={output_json}")
    print(f"wrote={output_md}")
    print(json.dumps(manifest["summary"], indent=2, sort_keys=True))
    return 0


__all__ = [
    "COLONIST_COLOR_NAMES",
    "DEFAULT_EXCLUDE_IDS",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_RAW_REPLAY_DIR",
    "EARLY_MID_STEP_PLAN",
    "LATE_HARD_STEP_PLAN",
    "ROOT",
    "GameEntry",
    "Manifest",
    "SourceRecord",
    "SplitSummary",
    "StepPlan",
    "Summary",
    "load_excluded_game_ids",
    "load_index_records",
    "load_json",
    "main",
    "norm_game_id",
    "parse_args",
    "parse_modes",
    "planned_steps",
    "portable_path",
    "replay_event_count",
    "split_balanced",
    "summarize",
    "write_json",
    "write_markdown",
]


if __name__ == "__main__":
    raise SystemExit(main())
