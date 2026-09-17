#!/usr/bin/env python3
"""Keep capturing Colonist replays in bounded batches until told to stop.

Each round rebuilds a deduplicated candidate index (skipping raw, staging, and
rejected captures), runs one bounded scraper batch through the logged-in Chrome
over CDP, and cools down. Colonist returned 429 after ~50 replay requests in
~35 minutes (2026-09-17), so batches default to 45 with an hour between them.
One rate limit triggers a longer cooldown and a retry; a second consecutive one
stops the loop. It also halts on a batch that captures nothing (dead session or
exhausted index) or at the batch cap. Never run two scrapers at once; this waits
for any running scraper to exit first.
"""

import argparse
import json
import logging
import re
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CANDIDATES = PROJECT_ROOT / "artifacts" / "raw" / "colonist" / "indexes" / "4p_games_training_candidates.json"
STAGING_DIR = PROJECT_ROOT / "artifacts" / "staging" / "colonist" / "replays"
RAW_DIR = PROJECT_ROOT / "artifacts" / "raw" / "colonist" / "replays"
LOG_DIR = PROJECT_ROOT / "logs"
SCRAPER_MODULE = "data_pipeline.bootstrapping.scrapers.replay_playwright_scraper"
ANNOTATE_MODULE = "data_pipeline.bootstrapping.scrapers.annotate_seat_ratings"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
logger = logging.getLogger(__name__)


def captured_game_ids() -> set[str]:
    ids: set[str] = set()
    for pattern in (RAW_DIR.glob("*.json"), STAGING_DIR.glob("*.json"), STAGING_DIR.glob("*/*.json")):
        ids.update(path.stem for path in pattern)
    return ids


def build_index(candidates: Path, output: Path) -> int:
    games = json.loads(candidates.read_text(encoding="utf-8"))
    if isinstance(games, dict):
        games = games["games"]
    have = captured_game_ids()
    seen: set[str] = set()
    per_player: dict[str, list] = {}
    for game in games:
        game_id = str(game["game_id"])
        if game_id in have or game_id in seen:
            continue
        seen.add(game_id)
        per_player.setdefault(str(game.get("username", "")), []).append(game)
    # The candidates index is grouped by player, so a sequential batch samples one or
    # two accounts. Round-robin across players (most-played first, newest game first)
    # so every batch spreads over many chat styles.
    queues = sorted(per_player.values(), key=len, reverse=True)
    for queue in queues:
        queue.sort(key=lambda game: str(game.get("date", "")), reverse=True)
    remaining = []
    while queues:
        queues = [queue for queue in queues if queue]
        for queue in queues:
            remaining.append(queue.pop(0))
    output.write_text(json.dumps(remaining), encoding="utf-8")
    return len(remaining)


def scraper_running() -> bool:
    result = subprocess.run(["pgrep", "-f", SCRAPER_MODULE], capture_output=True, text=True)
    return result.returncode == 0


def run_batch(index: Path, cdp_url: str, batch_size: int, delay_seconds: float) -> dict[str, int]:
    log_path = LOG_DIR / f"colonist_scrape_{time.strftime('%Y%m%d-%H%M')}.log"
    command = [
        sys.executable, "-m", SCRAPER_MODULE,
        "--cdp-url", cdp_url,
        "--index-file", str(index),
        "--max-games", str(batch_size),
        "--max-attempts", str(int(batch_size * 1.5)),
        "--delay-seconds", str(delay_seconds),
        "--expected-player-count", "4",
        "--expected-mode-setting", "0",
    ]
    logger.info("Batch log: %s", log_path)
    with log_path.open("w", encoding="utf-8") as log_file:
        subprocess.run(command, cwd=PROJECT_ROOT, stdout=log_file, stderr=subprocess.STDOUT)
    text = log_path.read_text(encoding="utf-8")
    stats = {}
    for key in ("Success", "Failed", "Incompatible", "Rate limited", "Skipped"):
        match = re.search(rf"^\s*{key}: (\d+)$", text, re.M)
        stats[key.lower().replace(" ", "_")] = int(match.group(1)) if match else 0
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cdp-url", default="http://127.0.0.1:9222")
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--batch-size", type=int, default=45)
    parser.add_argument("--delay-seconds", type=float, default=40.0)
    parser.add_argument("--cooldown-seconds", type=float, default=3600.0)
    parser.add_argument("--rate-limit-cooldown-seconds", type=float, default=5400.0)
    parser.add_argument("--initial-delay-seconds", type=float, default=0.0)
    parser.add_argument("--max-batches", type=int, default=20)
    args = parser.parse_args()

    LOG_DIR.mkdir(exist_ok=True)
    index = LOG_DIR / "colonist_pull_loop_index.json"
    total = 0
    if args.initial_delay_seconds > 0:
        logger.info("Initial cooldown %.0fs before the first batch", args.initial_delay_seconds)
        time.sleep(args.initial_delay_seconds)
    rate_limited_last = False
    for batch in range(1, args.max_batches + 1):
        while scraper_running():
            logger.info("A scraper is already running; waiting 60s before starting batch %d", batch)
            time.sleep(60)
        remaining = build_index(args.candidates, index)
        logger.info("Batch %d/%d: %d uncaptured candidates", batch, args.max_batches, remaining)
        if remaining == 0:
            logger.info("Candidate index exhausted; stopping")
            break
        stats = run_batch(index, args.cdp_url, args.batch_size, args.delay_seconds)
        total += stats["success"]
        logger.info("Batch %d result: %s (total captured this run: %d)", batch, stats, total)
        if stats["rate_limited"]:
            if rate_limited_last:
                logger.error("Rate limited twice in a row; stopping. Restart manually after a long cooldown.")
                return 2
            rate_limited_last = True
            logger.warning("Rate limited after %d captures; cooling down %.0fs before one retry",
                           stats["success"], args.rate_limit_cooldown_seconds)
            time.sleep(args.rate_limit_cooldown_seconds)
            continue
        rate_limited_last = False
        if stats["success"] == 0:
            logger.error("Batch captured nothing (dead session or nothing left); stopping.")
            return 1
        # Ratings come from the leaderboard endpoint, which is separate from the
        # replay quota; the annotator reuses today's snapshot if one exists.
        subprocess.run([sys.executable, "-m", ANNOTATE_MODULE], cwd=PROJECT_ROOT)
        if batch < args.max_batches:
            logger.info("Cooling down %.0fs before the next batch", args.cooldown_seconds)
            time.sleep(args.cooldown_seconds)
    index.unlink(missing_ok=True)
    logger.info("Done: %d replays captured", total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
