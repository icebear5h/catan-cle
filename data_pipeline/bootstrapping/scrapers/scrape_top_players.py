#!/usr/bin/env python3
"""Build replay-candidate indexes from Colonist player histories.

This module discovers game IDs and lightweight metadata only. Replay payload
capture is owned by ``replay_playwright_scraper`` and defaults to staging.

Run from the repository root:

    python -m data_pipeline.bootstrapping.scrapers.scrape_top_players --mode index
"""

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Optional

from .colonist_api import ColonistAPI


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INDEX_OUTPUT = (
    PROJECT_ROOT / "artifacts" / "raw" / "colonist" / "indexes" / "4p_games_top100.json"
)

logger = logging.getLogger(__name__)


async def build_game_index(
    output_file: str | Path = DEFAULT_INDEX_OUTPUT,
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
            logger.info("Top 5 Classic4P players:")
            for entry in entries:
                logger.info(f"  #{entry.rank}: {entry.username} (rating: {entry.rating})")
        except Exception as e:
            logger.error(f"Failed to get leaderboard: {e}")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["index", "test-api"],
        default="index",
        help="Build an index or test public API connectivity",
    )
    parser.add_argument("--top", type=int, default=10, help="Top players to index")
    parser.add_argument("--games", type=int, default=20, help="Games per player")
    parser.add_argument("--max-total", type=int, help="Maximum index records")
    parser.add_argument("--leaderboard", default="Classic4P")
    parser.add_argument(
        "--username",
        help="Comma-separated Colonist usernames instead of leaderboard players",
    )
    parser.add_argument(
        "--me",
        action="store_true",
        help="Index the authenticated JWT user's public history",
    )
    parser.add_argument(
        "--index-output",
        type=Path,
        default=DEFAULT_INDEX_OUTPUT,
        help="Output JSON path",
    )
    parser.add_argument("--all-games", action="store_true", help="Include losses")
    parser.add_argument(
        "--game-modes",
        default="Classic4P,Tournament",
        help="Comma-separated history modes, or 'all'",
    )
    parser.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="Keep shared game IDs from multiple indexed players",
    )
    args = parser.parse_args()

    if args.mode == "test-api":
        asyncio.run(test_api_connection())
        return 0

    jwt_token = os.environ.get("COLONIST_JWT")
    usernames = parse_usernames(args.username)
    if args.me and usernames:
        parser.error("use either --me or --username, not both")
    if args.me and not jwt_token:
        parser.error("--me requires COLONIST_JWT")

    asyncio.run(
        build_game_index(
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
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
