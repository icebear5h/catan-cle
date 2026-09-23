#!/usr/bin/env python3
"""
Colonist.io replay scraper that captures replay JSON from a real browser session.

The direct replay API can return an initial 403 until Colonist/Cloudflare browser
state is ready. This scraper opens the replay page in a persistent Playwright
profile and records the successful /api/replay/data-from-game-id JSON response.
"""

import argparse
import asyncio
from pathlib import Path

from data_pipeline.bootstrapping.scrapers.replay_playwright_scraper._browser import (
    PlaywrightReplayScraper,
)
from data_pipeline.bootstrapping.scrapers.replay_playwright_scraper._config import (
    BASE_URL,
    DEFAULT_INDEX_FILE,
    DEFAULT_PROFILE_DIR,
    DEFAULT_STAGING_DIR,
    DEFAULT_TIMEOUT_MS,
    PROJECT_ROOT,
    REPLAY_DATA_PATH,
    ReplayRateLimitedError,
    ScrapeStats,
    logger,
)
from data_pipeline.bootstrapping.scrapers.replay_playwright_scraper._payload import (
    has_expected_mode_setting,
    has_expected_player_count,
    is_matching_replay_response,
    is_valid_replay_payload,
    replay_event_count,
    replay_mode_setting,
    replay_output_path,
    replay_page_url,
    replay_player_count,
    save_replay_json,
)
from data_pipeline.bootstrapping.scrapers.replay_playwright_scraper._scrape import (
    scrape_replays_from_index,
    scrape_single_replay,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture Colonist.io replay JSON with a persistent Playwright browser session"
    )
    parser.add_argument("--game-id", help="Single game ID to scrape")
    parser.add_argument("--player-color", type=int, default=0, help="Player color perspective")
    parser.add_argument("--index-file", type=Path, default=DEFAULT_INDEX_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_STAGING_DIR)
    parser.add_argument("--max-games", type=int, help="Maximum successful new games to scrape")
    parser.add_argument("--max-attempts", type=int, help="Maximum non-skipped games to try")
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=40.0,
        help="Seconds to wait between replay attempts (default: 40)",
    )
    parser.add_argument("--profile-dir", type=Path, default=Path(DEFAULT_PROFILE_DIR))
    parser.add_argument(
        "--expected-player-count",
        type=int,
        help="Only save replays with this many source players",
    )
    parser.add_argument(
        "--expected-mode-setting",
        type=int,
        help="Only save replays with this Colonist modeSetting value",
    )
    parser.add_argument(
        "--cdp-url",
        help=(
            "Attach to an already-running Chrome with remote debugging, e.g. "
            "http://127.0.0.1:9222. Uses that Chrome profile instead of --profile-dir."
        ),
    )
    parser.add_argument("--headless", action="store_true", help="Run the browser headlessly")
    parser.add_argument("--skip-existing", dest="skip_existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    return parser.parse_args()


async def main() -> int:
    args = parse_args()

    if args.game_id:
        success = await scrape_single_replay(
            game_id=args.game_id,
            player_color=args.player_color,
            output_dir=args.output_dir,
            profile_dir=args.profile_dir,
            headless=args.headless,
            skip_existing=args.skip_existing,
            cdp_url=args.cdp_url,
            expected_player_count=args.expected_player_count,
            expected_mode_setting=args.expected_mode_setting,
        )
        return 0 if success else 1

    stats = await scrape_replays_from_index(
        index_file=args.index_file,
        output_dir=args.output_dir,
        profile_dir=args.profile_dir,
        max_games=args.max_games,
        max_attempts=args.max_attempts,
        headless=args.headless,
        skip_existing=args.skip_existing,
        cdp_url=args.cdp_url,
        expected_player_count=args.expected_player_count,
        expected_mode_setting=args.expected_mode_setting,
        delay_seconds=args.delay_seconds,
    )
    print("\nScraping complete:")
    print(f"  Success: {stats['success']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Incompatible: {stats['incompatible']}")
    print(f"  Rate limited: {stats['rate_limited']}")
    print(f"  Skipped: {stats['skipped']}")
    if args.max_games is not None:
        return 0 if stats["success"] >= args.max_games else 1
    return 0 if stats["failed"] == 0 and stats["incompatible"] == 0 else 1


__all__ = [
    "BASE_URL",
    "DEFAULT_INDEX_FILE",
    "DEFAULT_PROFILE_DIR",
    "DEFAULT_STAGING_DIR",
    "DEFAULT_TIMEOUT_MS",
    "PROJECT_ROOT",
    "REPLAY_DATA_PATH",
    "PlaywrightReplayScraper",
    "ReplayRateLimitedError",
    "ScrapeStats",
    "has_expected_mode_setting",
    "has_expected_player_count",
    "is_matching_replay_response",
    "is_valid_replay_payload",
    "logger",
    "main",
    "parse_args",
    "replay_event_count",
    "replay_mode_setting",
    "replay_output_path",
    "replay_page_url",
    "replay_player_count",
    "save_replay_json",
    "scrape_replays_from_index",
    "scrape_single_replay",
]


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
