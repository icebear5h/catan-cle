"""Colour-balanced, deterministic game-level split assignment and summaries."""

import random
from collections import Counter, defaultdict

from data_pipeline.bootstrapping.scrapers.build_replay_splits._models import (
    GameEntry,
    SplitSummary,
    Summary,
)


def split_balanced(
    games: list[GameEntry],
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    max_games: int | None,
) -> dict[str, list[GameEntry]]:
    """Assign games to train/val/test, balancing each balance colour separately."""

    rng = random.Random(seed)
    by_color: dict[int, list[GameEntry]] = defaultdict(list)
    for game in games:
        by_color[int(game["balance_color_id"])].append(game)

    splits: dict[str, list[GameEntry]] = {"train": [], "val": [], "test": []}
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
            assigned = game.copy()
            assigned["split"] = split
            splits[split].append(assigned)

    for split_games in splits.values():
        rng.shuffle(split_games)

    if max_games is not None:
        all_games = splits["train"] + splits["val"] + splits["test"]
        all_games = all_games[:max_games]
        splits = {"train": [], "val": [], "test": []}
        for game in all_games:
            splits[game["split"]].append(game)

    return splits


def summarize(splits: dict[str, list[GameEntry]]) -> Summary:
    """Count games, present replays, colours, and modes for each split."""

    summary: Summary = {"total_games": 0, "splits": {}}
    for split, games in splits.items():
        color_counts = Counter(game["balance_color_name"] for game in games)
        mode_counts = Counter(str(game.get("mode")) for game in games)
        raw_count = sum(1 for game in games if game.get("raw_replay_present"))
        split_summary: SplitSummary = {
            "games": len(games),
            "raw_replays_present": raw_count,
            "balance_color_counts": dict(sorted(color_counts.items())),
            "mode_counts": dict(sorted(mode_counts.items())),
        }
        summary["splits"][split] = split_summary
        summary["total_games"] += len(games)
    return summary


__all__ = ["split_balanced", "summarize"]
