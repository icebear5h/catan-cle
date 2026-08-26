#!/usr/bin/env python3
"""Build leakage-safe Colonist replay split manifests.

This script operates on lightweight game-index JSON files from
``scrape_top_players.py --mode index``. It does not download replays. It removes
CatanBoardBench holdout games, deduplicates game IDs, preserves color-balance metadata,
and writes deterministic game-level splits plus step-fraction plans.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EXCLUDE_IDS = (
    ROOT / "data_pipeline/catan_board_bench/datasets/catan_board_bench_100/leakage/benchmark_game_ids.json"
)
DEFAULT_RAW_REPLAY_DIR = ROOT / "artifacts" / "raw" / "colonist" / "replays"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "manifests" / "colonist" / "splits"

COLONIST_COLOR_NAMES = {
    1: "RED",
    2: "BLUE",
    3: "ORANGE",
    4: "GREEN",
    5: "BLACK",
    6: "BRONZE",
    7: "SILVER",
    8: "GOLD",
    9: "WHITE",
    10: "PINK",
    11: "MYSTIC_BLUE",
}

EARLY_MID_STEP_PLAN = {
    "early": [0.08, 0.18, 0.28],
    "mid": [0.38, 0.50],
}
LATE_HARD_STEP_PLAN = {
    "late_hard": [0.62, 0.74, 0.86],
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def portable_path(value: str | Path) -> str:
    """Return a repository-relative path when the target is inside the repo."""

    path = Path(value).expanduser()
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(path)


def norm_game_id(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text.removesuffix(".json").removesuffix("_sample")


def load_excluded_game_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    payload = load_json(path)
    ids = payload.get("benchmark_game_ids", payload if isinstance(payload, list) else [])
    return {game_id for game_id in (norm_game_id(value) for value in ids) if game_id}


def parse_modes(raw: str | None) -> set[str] | None:
    if not raw or raw.lower() == "all":
        return None
    return {part.strip() for part in raw.split(",") if part.strip()}


def load_index_records(paths: list[Path], modes: set[str] | None) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for path in paths:
        rows = load_json(path)
        if not isinstance(rows, list):
            raise ValueError(f"{path} must contain a JSON list")

        for row in rows:
            game_id = norm_game_id(row.get("game_id") or row.get("id"))
            if not game_id:
                continue

            mode = row.get("mode")
            if modes is not None and mode is not None and mode not in modes:
                continue

            player_color = int(row.get("player_color", 0) or 0)
            entry = grouped.setdefault(
                game_id,
                {
                    "game_id": game_id,
                    "source_index_files": [],
                    "source_records": [],
                    "indexed_player_color_ids": set(),
                    "turnCount": row.get("turnCount"),
                    "mode": mode,
                    "date": row.get("date"),
                    "replay_url": row.get("replay_url"),
                },
            )
            entry["source_index_files"].append(portable_path(path))
            entry["source_records"].append(
                {
                    "username": row.get("username"),
                    "player_rank": row.get("player_rank"),
                    "player_rating": row.get("player_rating"),
                    "player_color": player_color,
                    "player_color_name": COLONIST_COLOR_NAMES.get(player_color, f"COLOR_{player_color}"),
                    "result": row.get("result"),
                    "mode": mode,
                    "turnCount": row.get("turnCount"),
                    "date": row.get("date"),
                    "source_index": portable_path(row.get("source_index") or path),
                }
            )
            if player_color:
                entry["indexed_player_color_ids"].add(player_color)

            if entry.get("turnCount") is None and row.get("turnCount") is not None:
                entry["turnCount"] = row.get("turnCount")
            if entry.get("mode") is None and mode is not None:
                entry["mode"] = mode
            if entry.get("date") is None and row.get("date") is not None:
                entry["date"] = row.get("date")
            if entry.get("replay_url") is None and row.get("replay_url") is not None:
                entry["replay_url"] = row.get("replay_url")

    color_counter = Counter()
    for entry in grouped.values():
        color_counter.update(entry["indexed_player_color_ids"] or {0})

    for entry in grouped.values():
        color_ids = sorted(entry["indexed_player_color_ids"] or {0})
        balance_color = min(color_ids, key=lambda color_id: (color_counter[color_id], color_id))
        entry["indexed_player_color_ids"] = color_ids
        entry["indexed_player_color_names"] = [
            COLONIST_COLOR_NAMES.get(color_id, f"COLOR_{color_id}") for color_id in color_ids
        ]
        entry["balance_color_id"] = balance_color
        entry["balance_color_name"] = COLONIST_COLOR_NAMES.get(balance_color, f"COLOR_{balance_color}")
        entry["source_index_files"] = sorted(set(entry["source_index_files"]))

    return grouped


def replay_event_count(raw_replay_dir: Path, game_id: str) -> int | None:
    path = raw_replay_dir / f"{game_id}.json"
    if not path.exists():
        return None
    payload = load_json(path)
    data = payload.get("data", payload)
    event_history = data.get("eventHistory", data)
    events = event_history.get("events")
    return len(events) if isinstance(events, list) else None


def planned_steps(fractions: dict[str, list[float]], event_count: int | None) -> dict[str, Any]:
    payload: dict[str, Any] = {"fractions": fractions}
    if event_count:
        payload["event_indices"] = {
            band: sorted({max(0, min(event_count - 1, round(frac * (event_count - 1)))) for frac in values})
            for band, values in fractions.items()
        }
    return payload


def split_balanced(
    games: list[dict[str, Any]],
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    max_games: int | None,
) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    by_color: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for game in games:
        by_color[int(game["balance_color_id"])].append(game)

    splits = {"train": [], "val": [], "test": []}
    for color_id in sorted(by_color):
        color_games = by_color[color_id]
        rng.shuffle(color_games)
        n = len(color_games)
        train_n = int(n * train_ratio)
        val_n = int(n * val_ratio)
        for index, game in enumerate(color_games):
            if index < train_n:
                split = "train"
            elif index < train_n + val_n:
                split = "val"
            else:
                split = "test"
            game = dict(game)
            game["split"] = split
            splits[split].append(game)

    for split_games in splits.values():
        rng.shuffle(split_games)

    if max_games is not None:
        all_games = splits["train"] + splits["val"] + splits["test"]
        all_games = all_games[:max_games]
        splits = {"train": [], "val": [], "test": []}
        for game in all_games:
            splits[game["split"]].append(game)

    return splits


def summarize(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    summary = {"total_games": 0, "splits": {}}
    for split, games in splits.items():
        color_counts = Counter(game["balance_color_name"] for game in games)
        mode_counts = Counter(str(game.get("mode")) for game in games)
        raw_count = sum(1 for game in games if game.get("raw_replay_present"))
        summary["splits"][split] = {
            "games": len(games),
            "raw_replays_present": raw_count,
            "balance_color_counts": dict(sorted(color_counts.items())),
            "mode_counts": dict(sorted(mode_counts.items())),
        }
        summary["total_games"] += len(games)
    return summary


def write_markdown(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Colonist Replay Splits",
        "",
        f"Generated: {manifest['generated_at']}",
        f"Total games: {manifest['summary']['total_games']}",
        "",
        "## Split Counts",
        "",
        "| Split | Games | Raw replays present |",
        "| --- | ---: | ---: |",
    ]
    for split, data in manifest["summary"]["splits"].items():
        lines.append(f"| {split} | {data['games']} | {data['raw_replays_present']} |")

    lines.extend(["", "## Balance Color Counts", ""])
    for split, data in manifest["summary"]["splits"].items():
        lines.append(f"### {split}")
        lines.append("")
        lines.append("| Color | Games |")
        lines.append("| --- | ---: |")
        for color, count in data["balance_color_counts"].items():
            lines.append(f"| {color} | {count} |")
        lines.append("")

    lines.extend(
        [
            "## Step Plans",
            "",
            "`early_mid` is the current training target. `late_hard` is kept separate for a harder later dataset.",
            "",
            "```json",
            json.dumps(manifest["step_plans"], indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines))


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


def main() -> int:
    args = parse_args()
    modes = parse_modes(args.modes)
    excluded_ids = load_excluded_game_ids(args.exclude_game_ids)
    grouped = load_index_records(args.index_files, modes=modes)

    games = []
    excluded_count = 0
    short_count = 0
    for game_id, entry in grouped.items():
        if game_id in excluded_ids:
            excluded_count += 1
            continue
        turn_count = entry.get("turnCount")
        if turn_count is not None and int(turn_count) < args.min_turn_count:
            short_count += 1
            continue

        event_count = replay_event_count(args.raw_replay_dir, game_id)
        entry = dict(entry)
        entry["raw_replay_present"] = event_count is not None
        entry["raw_replay_path"] = (
            str((args.raw_replay_dir / f"{game_id}.json").relative_to(ROOT))
            if event_count is not None
            else None
        )
        entry["event_count"] = event_count
        entry["step_plans"] = {
            "early_mid": planned_steps(EARLY_MID_STEP_PLAN, event_count),
            "late_hard": planned_steps(LATE_HARD_STEP_PLAN, event_count),
        }
        games.append(entry)

    splits = split_balanced(
        games,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        max_games=args.max_games,
    )
    manifest = {
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


if __name__ == "__main__":
    raise SystemExit(main())
