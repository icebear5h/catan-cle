"""Single-replay and index-driven scraping loops."""

import asyncio
import json
from pathlib import Path

from data_pipeline.bootstrapping.scrapers import replay_playwright_scraper
from data_pipeline.bootstrapping.scrapers.replay_playwright_scraper._config import (
    ReplayRateLimitedError,
    ScrapeStats,
    logger,
)
from data_pipeline.bootstrapping.scrapers.replay_playwright_scraper._payload import (
    has_expected_mode_setting,
    has_expected_player_count,
    replay_event_count,
    replay_mode_setting,
    replay_output_path,
    replay_player_count,
    save_replay_json,
)
from data_pipeline.json_types import JsonValue

# The scraper class is resolved through the package at call time so that tests
# replacing `replay_playwright_scraper.PlaywrightReplayScraper` still take effect.


async def scrape_single_replay(
    game_id: str,
    player_color: int,
    output_dir: Path,
    profile_dir: Path,
    headless: bool,
    skip_existing: bool,
    cdp_url: str | None,
    expected_player_count: int | None = None,
    expected_mode_setting: int | None = None,
) -> bool:
    """Scrape one replay and save it to output_dir."""
    output_file = replay_output_path(output_dir, game_id)
    if skip_existing and output_file.exists():
        logger.info("Skipping existing replay %s: %s", game_id, output_file)
        return True

    try:
        async with replay_playwright_scraper.PlaywrightReplayScraper(
            profile_dir=profile_dir,
            headless=headless,
            cdp_url=cdp_url,
        ) as scraper:
            data = await scraper.capture_replay_data(game_id, player_color)
    except ReplayRateLimitedError as exc:
        logger.error("%s. Stop and cool down before retrying.", exc)
        return False

    if not data:
        return False
    if not has_expected_player_count(data, expected_player_count):
        logger.error(
            "Replay %s has %s players; expected %s. Not saving it.",
            game_id,
            replay_player_count(data),
            expected_player_count,
        )
        return False
    if not has_expected_mode_setting(data, expected_mode_setting):
        logger.error(
            "Replay %s has modeSetting=%s; expected %s. Not saving it.",
            game_id,
            replay_mode_setting(data),
            expected_mode_setting,
        )
        return False

    save_replay_json(output_file, data)
    event_count = replay_event_count(data)
    logger.info("Saved %s events to %s", event_count, output_file)
    return True


def _index_games(index_file: Path) -> list[dict[str, JsonValue]]:
    """Read the game index, which must be a JSON array of objects."""
    with index_file.open() as f:
        games: JsonValue = json.load(f)
    if not isinstance(games, list):
        raise TypeError(f"{index_file} must contain a JSON list")
    rows: list[dict[str, JsonValue]] = []
    for game in games:
        if not isinstance(game, dict):
            raise TypeError(f"{index_file} must contain a JSON list of objects")
        rows.append(game)
    return rows


async def scrape_replays_from_index(
    index_file: Path,
    output_dir: Path,
    profile_dir: Path,
    max_games: int | None,
    max_attempts: int | None,
    headless: bool,
    skip_existing: bool,
    cdp_url: str | None,
    expected_player_count: int | None = None,
    expected_mode_setting: int | None = None,
    delay_seconds: float = 0.0,
) -> dict[str, int]:
    """Scrape replay JSON files from a game index."""
    games = _index_games(index_file)

    stats = ScrapeStats()
    attempts = 0

    async with replay_playwright_scraper.PlaywrightReplayScraper(
        profile_dir=profile_dir,
        headless=headless,
        cdp_url=cdp_url,
    ) as scraper:
        for i, game in enumerate(games):
            if max_games is not None and stats.success >= max_games:
                break
            if max_attempts is not None and attempts >= max_attempts:
                break

            game_id = str(game["game_id"])
            raw_color = game.get("player_color", 0)
            player_color = int(raw_color) if isinstance(raw_color, (int, float, str)) else 0
            output_file = replay_output_path(output_dir, game_id)

            if skip_existing and output_file.exists():
                stats.skipped += 1
                continue

            attempts += 1
            logger.info(
                "[%s/%s] Capturing %s (playerColor=%s)",
                i + 1,
                len(games),
                game_id,
                player_color,
            )

            try:
                data = await scraper.capture_replay_data(
                    game_id,
                    player_color,
                )
            except ReplayRateLimitedError as exc:
                stats.rate_limited += 1
                logger.error(
                    "%s. Stopping the batch immediately; do not retry until "
                    "after a cooldown.",
                    exc,
                )
                break

            if data and not has_expected_player_count(
                data,
                expected_player_count,
            ):
                stats.incompatible += 1
                logger.warning(
                    "  Incompatible: %s players (expected %s); not saved",
                    replay_player_count(data),
                    expected_player_count,
                )
            elif data and not has_expected_mode_setting(
                data,
                expected_mode_setting,
            ):
                stats.incompatible += 1
                logger.warning(
                    "  Incompatible: modeSetting=%s (expected %s); not saved",
                    replay_mode_setting(data),
                    expected_mode_setting,
                )
            elif data:
                save_replay_json(output_file, data)
                event_count = replay_event_count(data)
                stats.success += 1
                logger.info("  Saved: %s events -> %s", event_count, output_file)
            else:
                stats.failed += 1
                logger.warning("  Failed: no replay data captured")

            if delay_seconds > 0:
                logger.info("  Pacing: sleeping %.1f seconds", delay_seconds)
                await asyncio.sleep(delay_seconds)

    return stats.to_dict()


__all__ = ["scrape_replays_from_index", "scrape_single_replay"]
