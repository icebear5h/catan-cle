#!/usr/bin/env python3
"""
Colonist.io Replay API Scraper

Uses the direct replay data API endpoint instead of WebSocket interception.
Much simpler and more reliable than browser-based scraping.

API Endpoint:
    GET /api/replay/data-from-game-id?gameId={id}&playerColor={color}

Returns full event history with structured state changes.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from data_pipeline.bootstrapping.scrapers.replay_api_scraper._batch import (
    scrape_replays_from_index,
    validate_jwt,
)
from data_pipeline.bootstrapping.scrapers.replay_api_scraper._client import ReplayAPIClient
from data_pipeline.bootstrapping.scrapers.replay_api_scraper._config import (
    ACTION_STATE,
    BUILDING_TYPE,
    DEFAULT_INDEX_FILE,
    DEFAULT_STAGING_DIR,
    EDGE_TYPE,
    LOG_TYPE,
    PROJECT_ROOT,
    RESOURCE_ENUM,
    logger,
)
from data_pipeline.bootstrapping.scrapers.replay_api_scraper._models import (
    ParsedReplay,
    ReplayEvent,
)
from data_pipeline.bootstrapping.scrapers.replay_api_scraper._parsing import (
    parse_replay,
    parse_replay_event,
)


async def main() -> None:
    """CLI entry point."""
    load_dotenv(dotenv_path=Path.cwd() / ".env")
    load_dotenv()

    parser = argparse.ArgumentParser(description="Scrape Colonist.io replays via API")
    parser.add_argument("--game-id", type=str, help="Single game ID to scrape")
    parser.add_argument(
        "--index-file", type=str, default=str(DEFAULT_INDEX_FILE), help="Game index file"
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(DEFAULT_STAGING_DIR), help="Output directory"
    )
    parser.add_argument("--max-games", type=int, help="Max successful new games to scrape")
    parser.add_argument("--max-attempts", type=int, help="Max non-skipped games to try")
    parser.add_argument("--player-color", type=int, default=0, help="Player color perspective")

    args = parser.parse_args()

    jwt_token = os.environ.get("COLONIST_JWT")
    if not jwt_token:
        print("ERROR: Set COLONIST_JWT environment variable")
        print("This direct API scraper is a debug fallback and may still 403 with only JWT.")
        print("For replay downloads, prefer replay_playwright_scraper.py.")
        print("Get JWT from DevTools > Application > Cookies > jwt_colonist.io")
        sys.exit(1)

    if not validate_jwt(jwt_token):
        sys.exit(1)

    if args.game_id:
        async with ReplayAPIClient(jwt_token=jwt_token) as client:
            raw_data = await client.get_replay_data(args.game_id, args.player_color)
        if not raw_data:
            print("Failed to scrape replay")
            sys.exit(1)

        replay = parse_replay(args.game_id, args.player_color, raw_data)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{args.game_id}.json"
        output_file.write_text(json.dumps(raw_data))
        print(f"\nGame {replay.game_id}: {replay.total_events} events")
        print(f"Saved raw payload to {output_file}")
    else:
        # Batch mode from index
        stats = await scrape_replays_from_index(
            index_file=args.index_file,
            output_dir=args.output_dir,
            jwt_token=jwt_token,
            max_games=args.max_games,
            max_attempts=args.max_attempts,
        )
        print("\nScraping complete:")
        print(f"  Success: {stats['success']}")
        print(f"  Failed: {stats['failed']}")
        print(f"  Skipped: {stats['skipped']}")


__all__ = [
    "ACTION_STATE",
    "BUILDING_TYPE",
    "DEFAULT_INDEX_FILE",
    "DEFAULT_STAGING_DIR",
    "EDGE_TYPE",
    "LOG_TYPE",
    "PROJECT_ROOT",
    "RESOURCE_ENUM",
    "ParsedReplay",
    "ReplayAPIClient",
    "ReplayEvent",
    "logger",
    "main",
    "parse_replay",
    "parse_replay_event",
    "scrape_replays_from_index",
    "validate_jwt",
]


if __name__ == "__main__":
    asyncio.run(main())
