#!/usr/bin/env python3
"""
Colonist.io replay scraper that captures replay JSON from a real browser session.

The direct replay API can return an initial 403 until Colonist/Cloudflare browser
state is ready. This scraper opens the replay page in a persistent Playwright
profile and records the successful /api/replay/data-from-game-id JSON response.
"""

import argparse
import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlencode, urlparse

from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Response,
    TimeoutError as PlaywrightTimeoutError,
    async_playwright,
)

BASE_URL = "https://colonist.io"
REPLAY_DATA_PATH = "/api/replay/data-from-game-id"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROFILE_DIR = ".colonist-playwright-profile"
DEFAULT_INDEX_FILE = PROJECT_ROOT / "artifacts" / "raw" / "colonist" / "indexes" / "4p_games_top100.json"
DEFAULT_STAGING_DIR = PROJECT_ROOT / "artifacts" / "staging" / "colonist" / "replays"
DEFAULT_TIMEOUT_MS = 180_000

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ScrapeStats:
    success: int = 0
    failed: int = 0
    incompatible: int = 0
    rate_limited: int = 0
    skipped: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "success": self.success,
            "failed": self.failed,
            "incompatible": self.incompatible,
            "rate_limited": self.rate_limited,
            "skipped": self.skipped,
        }


class ReplayRateLimitedError(RuntimeError):
    """Raised when Colonist asks the scraper to stop making replay requests."""

    def __init__(self, game_id: str, retry_after: Optional[str]):
        self.game_id = game_id
        self.retry_after = retry_after
        retry_message = f"; Retry-After={retry_after}" if retry_after else ""
        super().__init__(f"Replay request rate-limited for {game_id}{retry_message}")


def replay_page_url(game_id: str, player_color: int) -> str:
    """Build the browser replay URL for a game and player perspective."""
    return f"{BASE_URL}/replay?{urlencode({'gameId': game_id, 'playerColor': player_color})}"


def replay_output_path(output_dir: Path, game_id: str) -> Path:
    """Return the canonical raw replay JSON path for a game."""
    return output_dir / f"{game_id}.json"


def is_matching_replay_response(url: str, game_id: str, player_color: int) -> bool:
    """Return True when a response URL is the replay data endpoint for the target game."""
    parsed = urlparse(url)
    if parsed.path != REPLAY_DATA_PATH:
        return False

    params = parse_qs(parsed.query)
    return (
        params.get("gameId") == [str(game_id)]
        and params.get("playerColor") == [str(player_color)]
    )


def _replay_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    """Return the replay payload whether the API response is wrapped in data or not."""
    nested = data.get("data")
    return nested if isinstance(nested, dict) else data


def is_valid_replay_payload(data: Any) -> bool:
    """Validate that an API response has the replay shape expected by the decoder."""
    if not isinstance(data, dict):
        return False

    payload = _replay_payload(data)
    event_history = payload.get("eventHistory")
    if not isinstance(event_history, dict):
        return False

    events = event_history.get("events")
    return isinstance(events, list) and len(events) > 0


def replay_player_count(data: Any) -> Optional[int]:
    """Return the source player count when replay metadata exposes it."""
    if not isinstance(data, dict):
        return None

    payload = _replay_payload(data)
    player_states = payload.get("playerUserStates")
    if isinstance(player_states, list):
        return len(player_states)

    play_order = payload.get("playOrder")
    if isinstance(play_order, list):
        return len(play_order)

    return None


def has_expected_player_count(data: Any, expected_player_count: Optional[int]) -> bool:
    """Return whether a replay matches an optional player-count constraint."""
    if expected_player_count is None:
        return True
    return replay_player_count(data) == expected_player_count


def replay_mode_setting(data: Any) -> Optional[int]:
    """Return Colonist's mode setting when replay metadata exposes it."""
    if not isinstance(data, dict):
        return None

    payload = _replay_payload(data)
    game_settings = payload.get("gameSettings")
    if not isinstance(game_settings, dict):
        return None

    mode_setting = game_settings.get("modeSetting")
    return mode_setting if isinstance(mode_setting, int) else None


def has_expected_mode_setting(data: Any, expected_mode_setting: Optional[int]) -> bool:
    """Return whether a replay matches an optional Colonist mode constraint."""
    if expected_mode_setting is None:
        return True
    return replay_mode_setting(data) == expected_mode_setting


def save_replay_json(output_file: Path, data: Dict[str, Any]) -> None:
    """Save replay JSON atomically enough to avoid leaving partial files on interruption."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file = output_file.with_suffix(output_file.suffix + ".tmp")
    with temp_file.open("w") as f:
        json.dump(data, f)
    temp_file.replace(output_file)


class PlaywrightReplayScraper:
    """Persistent-browser replay scraper."""

    def __init__(
        self,
        profile_dir: Path,
        headless: bool = False,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        cdp_url: Optional[str] = None,
    ):
        self.profile_dir = profile_dir
        self.headless = headless
        self.timeout_ms = timeout_ms
        self.cdp_url = cdp_url
        self._playwright = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None

    async def __aenter__(self) -> "PlaywrightReplayScraper":
        self._playwright = await async_playwright().start()
        if self.cdp_url:
            logger.info("Connecting to existing Chrome over CDP: %s", self.cdp_url)
            self.browser = await self._playwright.chromium.connect_over_cdp(self.cdp_url)
            if not self.browser.contexts:
                raise RuntimeError("Connected Chrome did not expose a browser context")
            self.context = self.browser.contexts[0]
            self.page = await self.context.new_page()
            return self

        self.profile_dir.mkdir(parents=True, exist_ok=True)
        self.context = await self._playwright.chromium.launch_persistent_context(
            user_data_dir=str(self.profile_dir),
            headless=self.headless,
            viewport={"width": 1440, "height": 900},
            accept_downloads=False,
        )
        self.page = self.context.pages[0] if self.context.pages else await self.context.new_page()
        return self

    async def __aexit__(self, *args) -> None:
        if self.context and not self.cdp_url:
            await self.context.close()
        if self._playwright:
            await self._playwright.stop()

    async def capture_replay_data(self, game_id: str, player_color: int) -> Optional[Dict[str, Any]]:
        """Open a replay page and return the first valid 200 replay data response."""
        if not self.page:
            raise RuntimeError("PlaywrightReplayScraper must be used as an async context manager")

        response_task = asyncio.create_task(
            self._wait_for_replay_response(self.page, game_id, player_color)
        )
        url = replay_page_url(game_id, player_color)
        logger.info("Opening replay %s (playerColor=%s)", game_id, player_color)

        try:
            await self.page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_ms)
            return await response_task
        except ReplayRateLimitedError:
            raise
        except (asyncio.TimeoutError, PlaywrightTimeoutError):
            logger.error(
                "Timed out waiting for replay data for %s. If the browser is showing login "
                "or a challenge, complete it and rerun the same command.",
                game_id,
            )
            return None
        except Exception as exc:
            logger.error("Failed to capture replay %s: %s", game_id, exc)
            return None
        finally:
            if not response_task.done():
                response_task.cancel()

    async def _wait_for_replay_response(
        self,
        page: Page,
        game_id: str,
        player_color: int,
    ) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Dict[str, Any]] = loop.create_future()

        async def read_successful_response(response: Response) -> None:
            try:
                data = await response.json()
                if not is_valid_replay_payload(data):
                    logger.warning(
                        "Ignoring 200 replay response for %s because it did not contain "
                        "eventHistory.events",
                        game_id,
                    )
                    return
                if not future.done():
                    future.set_result(data)
            except Exception as exc:
                logger.warning("Could not read replay response for %s: %s", game_id, exc)

        def on_response(response: Response) -> None:
            if not is_matching_replay_response(response.url, game_id, player_color):
                return

            status = response.status
            logger.info("Replay API response for %s: %s", game_id, status)
            if status == 200:
                asyncio.create_task(read_successful_response(response))
            elif status in {401, 403}:
                logger.info("Waiting for a later successful replay response after %s", status)
            elif status == 429 and not future.done():
                future.set_exception(
                    ReplayRateLimitedError(
                        game_id,
                        response.headers.get("retry-after"),
                    )
                )
            elif status == 404 and not future.done():
                future.set_exception(RuntimeError(f"Replay not found: {game_id}"))

        page.on("response", on_response)
        try:
            return await asyncio.wait_for(future, timeout=self.timeout_ms / 1000)
        finally:
            page.remove_listener("response", on_response)


async def scrape_single_replay(
    game_id: str,
    player_color: int,
    output_dir: Path,
    profile_dir: Path,
    headless: bool,
    skip_existing: bool,
    cdp_url: Optional[str],
    expected_player_count: Optional[int] = None,
    expected_mode_setting: Optional[int] = None,
) -> bool:
    """Scrape one replay and save it to output_dir."""
    output_file = replay_output_path(output_dir, game_id)
    if skip_existing and output_file.exists():
        logger.info("Skipping existing replay %s: %s", game_id, output_file)
        return True

    try:
        async with PlaywrightReplayScraper(
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
    event_count = len(_replay_payload(data)["eventHistory"]["events"])
    logger.info("Saved %s events to %s", event_count, output_file)
    return True


async def scrape_replays_from_index(
    index_file: Path,
    output_dir: Path,
    profile_dir: Path,
    max_games: Optional[int],
    max_attempts: Optional[int],
    headless: bool,
    skip_existing: bool,
    cdp_url: Optional[str],
    expected_player_count: Optional[int] = None,
    expected_mode_setting: Optional[int] = None,
    delay_seconds: float = 0.0,
) -> Dict[str, int]:
    """Scrape replay JSON files from a game index."""
    with index_file.open() as f:
        games = json.load(f)

    stats = ScrapeStats()
    attempts = 0

    async with PlaywrightReplayScraper(
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
            player_color = int(game.get("player_color", 0))
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
                event_count = len(_replay_payload(data)["eventHistory"]["events"])
                stats.success += 1
                logger.info("  Saved: %s events -> %s", event_count, output_file)
            else:
                stats.failed += 1
                logger.warning("  Failed: no replay data captured")

            if delay_seconds > 0:
                logger.info("  Pacing: sleeping %.1f seconds", delay_seconds)
                await asyncio.sleep(delay_seconds)

    return stats.to_dict()


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


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
