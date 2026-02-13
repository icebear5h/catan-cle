#!/usr/bin/env python3
"""
Main script to scrape replays from top Colonist.io players.

Pipeline:
1. Fetch top players from leaderboard
2. Get their game history
3. Scrape replays from their winning games
4. Save structured data for training

Usage:
    python scrape_top_players.py --top 10 --games 20 --headless

Requirements:
    pip install playwright httpx msgpack
    playwright install chromium
"""

import asyncio
import argparse
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from colonist_api import ColonistAPI, GameHistoryEntry
from replay_scraper import ReplayScraper, scrape_multiple_replays
from colonist_schema import ReplayDataStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class TopPlayerReplayScraper:
    """
    Orchestrates scraping replays from top players.
    """

    def __init__(
        self,
        output_dir: str = "./data/replays",
        leaderboard_type: str = "Classic4P",
        jwt_token: Optional[str] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.leaderboard_type = leaderboard_type
        self.store = ReplayDataStore(str(output_dir))
        self.jwt_token = jwt_token

    async def run(
        self,
        top_n_players: int = 10,
        games_per_player: int = 20,
        wins_only: bool = True,
        headless: bool = True,
        max_total_games: Optional[int] = None,
    ):
        """
        Run the full scraping pipeline.

        Args:
            top_n_players: Number of top players to scrape
            games_per_player: Games to fetch per player
            wins_only: Only scrape winning games
            headless: Run browser in headless mode
            max_total_games: Maximum total games to scrape
        """
        logger.info(f"Starting scrape: top {top_n_players} players, {games_per_player} games each")

        # Step 1: Get replay URLs from top players
        async with ColonistAPI() as api:
            games = await api.get_top_player_replays(
                leaderboard_type=self.leaderboard_type,
                top_n_players=top_n_players,
                games_per_player=games_per_player,
                wins_only=wins_only,
            )

        if not games:
            logger.error("No games found to scrape")
            return

        logger.info(f"Found {len(games)} games to scrape")

        # Limit total if specified
        if max_total_games:
            games = games[:max_total_games]

        # Step 2: Filter out already-scraped games
        existing = set(self.store.list_replays())
        games_to_scrape = [g for g in games if g.game_id not in existing]

        logger.info(f"{len(games_to_scrape)} new games to scrape ({len(existing)} already in store)")

        # Step 3: Scrape each replay
        successful = 0
        failed = 0

        for i, game in enumerate(games_to_scrape):
            logger.info(f"[{i+1}/{len(games_to_scrape)}] Scraping game {game.game_id}...")

            try:
                scraper = ReplayScraper(headless=headless, jwt_token=self.jwt_token)
                replay = await scraper.scrape_replay(
                    game_id=game.game_id,
                    player_color=game.player_color,
                )

                if replay:
                    self.store.save_replay(replay)
                    successful += 1
                    logger.info(f"  Saved: {len(replay.steps)} steps captured")
                else:
                    failed += 1
                    logger.warning(f"  Failed: No data captured")

            except Exception as e:
                failed += 1
                logger.error(f"  Failed: {e}")

            # Rate limiting
            await asyncio.sleep(2)

        # Summary
        logger.info(f"\n{'='*50}")
        logger.info(f"Scraping complete!")
        logger.info(f"  Successful: {successful}")
        logger.info(f"  Failed: {failed}")
        logger.info(f"  Total in store: {len(self.store.list_replays())}")

    def export_training_data(
        self,
        output_file: str = "./data/training_examples.jsonl",
        winner_only: bool = True,
    ):
        """
        Export all scraped replays as training examples.

        Args:
            output_file: Output JSONL file path
            winner_only: Only include winning player's decisions
        """
        import json

        examples = self.store.get_all_training_examples(winner_only=winner_only)

        logger.info(f"Exporting {len(examples)} training examples to {output_file}")

        with open(output_file, "w") as f:
            for example in examples:
                f.write(json.dumps(example) + "\n")

        logger.info("Export complete!")


async def test_api_connection():
    """Test the API connection and endpoint discovery."""
    logger.info("Testing Colonist API connection...")

    async with ColonistAPI() as api:
        # Test leaderboard tabs
        try:
            tabs = await api.get_leaderboard_tabs()
            logger.info(f"Leaderboard tabs: {tabs}")
        except Exception as e:
            logger.error(f"Failed to get leaderboard tabs: {e}")

        # Test leaderboard
        try:
            entries = await api.get_leaderboard("Classic4P", start=1, end=5)
            logger.info(f"Top 5 Classic4P players:")
            for entry in entries:
                logger.info(f"  #{entry.rank}: {entry.username} (rating: {entry.rating})")
        except Exception as e:
            logger.error(f"Failed to get leaderboard: {e}")


async def test_single_replay(game_id: str, headless: bool = False, jwt_token: Optional[str] = None):
    """Test scraping a single replay."""
    logger.info(f"Testing replay scrape for game {game_id}...")

    scraper = ReplayScraper(headless=headless, jwt_token=jwt_token)
    replay = await scraper.scrape_replay(game_id)

    if replay:
        logger.info(f"Success! Captured {len(replay.steps)} steps")
        logger.info(f"Players: {[p['username'] for p in replay.players]}")
        logger.info(f"Winner: Player {replay.winner_index}")

        # Save it
        store = ReplayDataStore()
        store.save_replay(replay)
        logger.info(f"Saved to {store.base_dir}/{game_id}.json")
    else:
        logger.error("Failed to scrape replay")


def main():
    parser = argparse.ArgumentParser(
        description="Scrape replays from top Colonist.io players for training data"
    )

    parser.add_argument(
        "--mode",
        choices=["scrape", "test-api", "test-replay", "export"],
        default="scrape",
        help="Mode of operation",
    )

    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top players to scrape (default: 10)",
    )

    parser.add_argument(
        "--games",
        type=int,
        default=20,
        help="Games per player to fetch (default: 20)",
    )

    parser.add_argument(
        "--max-total",
        type=int,
        default=None,
        help="Maximum total games to scrape",
    )

    parser.add_argument(
        "--leaderboard",
        type=str,
        default="Classic4P",
        help="Leaderboard type (default: Classic4P)",
    )

    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run browser in headless mode",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/replays",
        help="Output directory for replays",
    )

    parser.add_argument(
        "--game-id",
        type=str,
        help="Game ID for test-replay mode",
    )

    parser.add_argument(
        "--all-games",
        action="store_true",
        help="Include losses as well as wins",
    )

    args = parser.parse_args()

    # Get JWT token from environment
    jwt_token = os.environ.get("COLONIST_JWT")
    if jwt_token and args.mode in ("scrape", "test-replay"):
        logger.info(f"Using JWT token: {jwt_token[:20]}...")
    elif args.mode in ("scrape", "test-replay"):
        logger.warning("No COLONIST_JWT env var set. Replay scraping requires authentication.")
        logger.warning("Get token from DevTools > Application > Cookies > jwt_colonist.io")

    if args.mode == "test-api":
        asyncio.run(test_api_connection())

    elif args.mode == "test-replay":
        if not args.game_id:
            print("Error: --game-id required for test-replay mode")
            return
        asyncio.run(test_single_replay(args.game_id, headless=args.headless, jwt_token=jwt_token))

    elif args.mode == "export":
        scraper = TopPlayerReplayScraper(output_dir=args.output_dir)
        scraper.export_training_data()

    else:  # scrape
        scraper = TopPlayerReplayScraper(
            output_dir=args.output_dir,
            leaderboard_type=args.leaderboard,
            jwt_token=jwt_token,
        )
        asyncio.run(scraper.run(
            top_n_players=args.top,
            games_per_player=args.games,
            wins_only=not args.all_games,
            headless=args.headless,
            max_total_games=args.max_total,
        ))


if __name__ == "__main__":
    main()
