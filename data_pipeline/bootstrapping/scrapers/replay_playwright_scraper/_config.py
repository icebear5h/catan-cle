"""Endpoints, defaults, logging, and small shared types for the replay scraper."""

import logging
from dataclasses import dataclass
from pathlib import Path

BASE_URL = "https://colonist.io"
REPLAY_DATA_PATH = "/api/replay/data-from-game-id"
PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_PROFILE_DIR = ".colonist-playwright-profile"
DEFAULT_INDEX_FILE = PROJECT_ROOT / "artifacts" / "raw" / "colonist" / "indexes" / "4p_games_top100.json"
DEFAULT_STAGING_DIR = PROJECT_ROOT / "artifacts" / "staging" / "colonist" / "replays"
DEFAULT_TIMEOUT_MS = 180_000

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("data_pipeline.bootstrapping.scrapers.replay_playwright_scraper")


@dataclass
class ScrapeStats:
    success: int = 0
    failed: int = 0
    incompatible: int = 0
    rate_limited: int = 0
    skipped: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "success": self.success,
            "failed": self.failed,
            "incompatible": self.incompatible,
            "rate_limited": self.rate_limited,
            "skipped": self.skipped,
        }


class ReplayRateLimitedError(RuntimeError):
    """Raised when Colonist asks the scraper to stop making replay requests."""

    def __init__(self, game_id: str, retry_after: str | None) -> None:
        self.game_id = game_id
        self.retry_after = retry_after
        retry_message = f"; Retry-After={retry_after}" if retry_after else ""
        super().__init__(f"Replay request rate-limited for {game_id}{retry_message}")


__all__ = [
    "BASE_URL",
    "DEFAULT_INDEX_FILE",
    "DEFAULT_PROFILE_DIR",
    "DEFAULT_STAGING_DIR",
    "DEFAULT_TIMEOUT_MS",
    "PROJECT_ROOT",
    "REPLAY_DATA_PATH",
    "ReplayRateLimitedError",
    "ScrapeStats",
    "logger",
]
