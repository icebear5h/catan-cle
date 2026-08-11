"""
Colonist.io Replay Scraping Pipeline

Components:
- replay_playwright_scraper: Browser-session replay scraper (recommended)
- replay_api_scraper: Direct API scraper/debug fallback
- colonist_api: API client for leaderboards, profiles, game history
- colonist_schema: Data structures for game states and replays
- scrape_top_players: Build game index from leaderboards

Usage:
    # Capture replays through a persistent Playwright browser profile
    python replay_playwright_scraper.py --game-id 192418134 --player-color 2
    python replay_playwright_scraper.py --index-file 4p_games_top100.json --max-games 100

    # Build game index from leaderboards
    python scrape_top_players.py --mode index --top 100 --games 100 --all-games --game-modes Classic4P,Tournament
"""

from .colonist_api import ColonistAPI, LeaderboardEntry, GameHistoryEntry
from .colonist_schema import (
    ReplayData,
    ReplayStep,
    GameStateSnapshot,
    ActionRecord,
    ReplayDataStore,
)

__all__ = [
    "ColonistAPI",
    "LeaderboardEntry",
    "GameHistoryEntry",
    "ReplayData",
    "ReplayStep",
    "GameStateSnapshot",
    "ActionRecord",
    "ReplayDataStore",
]
