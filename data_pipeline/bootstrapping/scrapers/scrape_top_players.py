#!/usr/bin/env python3
"""
Main script to scrape replays from top Colonist.io players.

Pipeline:
1. Fetch top players from leaderboard
2. Get their game history
3. Scrape replays from their winning games
4. Save structured data for training

Usage:
    python scrape_top_players.py --mode index --top 100 --games 100 --all-games
    python scrape_top_players.py --mode index --me --games 100 --all-games --game-modes all

Requirements:
    pip install playwright httpx msgpack
    playwright install chromium
"""

import asyncio
import argparse
import json
import logging
import os
from pathlib import Path
from typing import Optional

from colonist_api import ColonistAPI

try:
    from replay_scraper import ReplayScraper
except ImportError:
    ReplayScraper = None

try:
    from colonist_schema import ReplayDataStore
except ImportError:
    ReplayDataStore = None

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
        if ReplayScraper is None or ReplayDataStore is None:
            raise RuntimeError(
                "Browser replay scraping dependencies are unavailable. "
                "Use --mode index to build a game-id list, then replay_api_scraper.py "
                "to download raw replay JSON."
            )
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


async def build_game_index(
    output_file: str = "4p_games_top100.json",
    leaderboard_type: str = "Classic4P",
    top_n_players: int = 100,
    games_per_player: int = 100,
    wins_only: bool = False,
    max_total_games: Optional[int] = None,
    allow_duplicates: bool = False,
    game_modes: Optional[set[str]] = None,
    usernames: Optional[list[str]] = None,
    use_authenticated_user: bool = False,
    jwt_token: Optional[str] = None,
) -> list[dict]:
    """
    Build a replay candidate index from leaderboard players' public histories.

    This only discovers game IDs and lightweight metadata. It does not require
    replay API membership access and does not download replay event payloads.
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    seen_game_ids = set()

    async with ColonistAPI(jwt_token=jwt_token if use_authenticated_user else None) as api:
        if use_authenticated_user:
            username = await api.get_current_username()
            targets = [
                {
                    "username": username,
                    "rank": None,
                    "rating": None,
                    "source": "authenticated_user",
                }
            ]
            logger.info("Indexing authenticated user history for %s...", username)
        elif usernames:
            targets = [
                {
                    "username": username,
                    "rank": None,
                    "rating": None,
                    "source": "username",
                }
                for username in usernames
            ]
            logger.info("Indexing explicit user histories: %s", ", ".join(usernames))
        else:
            logger.info(
                "Fetching top %s players from %s leaderboard...",
                top_n_players,
                leaderboard_type,
            )
            leaderboard = []
            chunk_size = 100
            for start in range(1, top_n_players + 1, chunk_size):
                end = min(start + chunk_size - 1, top_n_players)
                leaderboard.extend(
                    await api.get_leaderboard(
                        leaderboard_type,
                        start=start,
                        end=end,
                    )
                )
                await asyncio.sleep(0.25)
            targets = [
                {
                    "username": player.username,
                    "rank": player.rank,
                    "rating": player.rating,
                    "source": "leaderboard",
                }
                for player in leaderboard
            ]

        for player_index, player in enumerate(targets, start=1):
            username = player["username"]
            logger.info(
                "[%s/%s] Fetching %s history (rank %s, rating %s)...",
                player_index,
                len(targets),
                username,
                player["rank"] if player["rank"] is not None else "n/a",
                player["rating"] if player["rating"] is not None else "n/a",
            )

            try:
                games = await api.get_player_games(username, limit=None)
            except Exception as exc:
                logger.warning("Failed to fetch games for %s: %s", username, exc)
                continue

            if game_modes:
                games = [game for game in games if game.mode in game_modes]

            if wins_only:
                games = [game for game in games if game.result == "win"]

            for game in games[:games_per_player]:
                if max_total_games is not None and len(records) >= max_total_games:
                    break

                if not allow_duplicates and game.game_id in seen_game_ids:
                    continue

                seen_game_ids.add(game.game_id)
                records.append(
                    {
                        "game_id": game.game_id,
                        "username": username,
                        "player_rank": player["rank"],
                        "player_rating": player["rating"],
                        "turnCount": game.turn_count,
                        "player_color": game.player_color,
                        "result": game.result,
                        "mode": game.mode,
                        "date": game.date.isoformat(),
                        "replay_url": game.replay_url,
                        "source": player["source"],
                    }
                )

            if max_total_games is not None and len(records) >= max_total_games:
                break

            await asyncio.sleep(0.5)

    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)

    logger.info(
        "Wrote %s game index records (%s unique game IDs) to %s",
        len(records),
        len({record["game_id"] for record in records}),
        output_path,
    )

    return records


def parse_game_modes(raw_modes: Optional[str]) -> Optional[set[str]]:
    """Parse a comma-separated mode list. Use 'all' to disable mode filtering."""
    if not raw_modes or raw_modes.lower() == "all":
        return None
    modes = {mode.strip() for mode in raw_modes.split(",") if mode.strip()}
    return modes or None


def parse_usernames(raw_usernames: Optional[str]) -> Optional[list[str]]:
    """Parse a comma-separated username list."""
    if not raw_usernames:
        return None
    usernames = [username.strip() for username in raw_usernames.split(",") if username.strip()]
    return usernames or None


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
    if ReplayScraper is None or ReplayDataStore is None:
        raise RuntimeError(
            "Browser replay scraper is unavailable. Use replay_api_scraper.py for direct API downloads."
        )

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
        choices=["index", "scrape", "test-api", "test-replay", "export"],
        default="index",
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
        "--username",
        type=str,
        help="Comma-separated Colonist username(s) to index instead of leaderboard players",
    )

    parser.add_argument(
        "--me",
        action="store_true",
        help="Index the authenticated JWT user's public history instead of leaderboard players",
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
        "--index-output",
        type=str,
        default="4p_games_top100.json",
        help="Output JSON file for index mode",
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

    parser.add_argument(
        "--game-modes",
        type=str,
        default="Classic4P,Tournament",
        help="Comma-separated history modes to index, or 'all' to disable filtering",
    )

    parser.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="Keep duplicate game IDs when multiple indexed players share a game",
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

    elif args.mode == "index":
        usernames = parse_usernames(args.username)
        if args.me and usernames:
            print("Error: use either --me or --username, not both")
            return
        if args.me and not jwt_token:
            print("Error: --me requires COLONIST_JWT")
            return

        asyncio.run(build_game_index(
            output_file=args.index_output,
            leaderboard_type=args.leaderboard,
            top_n_players=args.top,
            games_per_player=args.games,
            wins_only=not args.all_games,
            max_total_games=args.max_total,
            allow_duplicates=args.allow_duplicates,
            game_modes=parse_game_modes(args.game_modes),
            usernames=usernames,
            use_authenticated_user=args.me,
            jwt_token=jwt_token,
        ))

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
