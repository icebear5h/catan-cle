"""Batch scraping from a game index, plus JWT expiry validation."""

import asyncio
import base64
import json
from datetime import datetime
from pathlib import Path

from data_pipeline.bootstrapping.scrapers import replay_api_scraper
from data_pipeline.bootstrapping.scrapers.replay_api_scraper._config import logger
from data_pipeline.json_coerce import as_dict, as_dict_list, as_float, as_int
from data_pipeline.json_types import JsonValue

# The client and parser are resolved through the package at call time so that a
# test or caller replacing them on the module still takes effect here.


async def scrape_replays_from_index(
    index_file: str,
    output_dir: str,
    jwt_token: str,
    max_games: int | None = None,
    max_attempts: int | None = None,
    skip_existing: bool = True,
) -> dict[str, int]:
    """
    Scrape replays from the game index file.

    Args:
        index_file: Path to 4p_games_top100.json
        output_dir: Directory to save raw replay payloads
        jwt_token: Authentication token
        max_games: Maximum successful new games to scrape (None for all)
        max_attempts: Maximum non-skipped games to try before stopping
        skip_existing: Skip already-scraped games

    Returns:
        Stats dict with success/failure counts
    """
    # Load game index
    with open(index_file) as f:
        payload: JsonValue = json.load(f)
    games = as_dict_list(payload)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    stats = {"success": 0, "failed": 0, "skipped": 0}
    attempts = 0

    async with replay_api_scraper.ReplayAPIClient(jwt_token=jwt_token) as client:
        for i, game in enumerate(games):
            game_id = str(game["game_id"])
            player_color = as_int(game.get("player_color", 0))

            # Check if already scraped
            output_file = output_path / f"{game_id}.json"
            if skip_existing and output_file.exists():
                stats["skipped"] += 1
                continue

            if max_games is not None and stats["success"] >= max_games:
                break
            if max_attempts is not None and attempts >= max_attempts:
                break
            attempts += 1

            logger.info(f"[{i+1}/{len(games)}] Scraping {game_id} (player {game['username']})...")

            try:
                raw_data = await client.get_replay_data(game_id, player_color)
                if raw_data:
                    replay = replay_api_scraper.parse_replay(game_id, player_color, raw_data)

                    with open(output_file, "w") as f:
                        json.dump(raw_data, f)

                    stats["success"] += 1
                    logger.info(f"  Saved: {replay.total_events} events")
                else:
                    stats["failed"] += 1
                    logger.warning("  Failed: No data returned")

            except Exception as e:
                stats["failed"] += 1
                logger.error(f"  Failed: {e}")

            # Rate limiting
            await asyncio.sleep(0.5)

    return stats


def validate_jwt(token: str) -> bool:
    """Validate JWT token hasn't expired."""
    try:
        parts = token.split('.')
        payload = parts[1] + '=' * (4 - len(parts[1]) % 4)
        data = as_dict(json.loads(base64.urlsafe_b64decode(payload)))
        exp = datetime.fromtimestamp(as_float(data.get('exp', 0)))

        if datetime.now() > exp:
            logger.error(f"JWT token expired on {exp}")
            return False

        logger.info(f"JWT valid for user: {data.get('username')} (expires: {exp})")
        return True
    except Exception as e:
        logger.warning(f"Could not validate JWT: {e}")
        return True  # Proceed anyway


__all__ = ["scrape_replays_from_index", "validate_jwt"]
