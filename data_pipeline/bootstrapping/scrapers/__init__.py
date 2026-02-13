"""
Colonist.io Replay Scraping Pipeline

Components:
- replay_api_scraper: Direct API scraper (recommended)
- colonist_api: API client for leaderboards, profiles, game history
- colonist_schema: Data structures for game states and replays
- scrape_top_players: Build game index from leaderboards

Usage:
    # Scrape replays via API (requires COLONIST_JWT env var)
    python replay_api_scraper.py --game-id 192418134
    python replay_api_scraper.py --max-games 100 --supabase

    # Build game index from leaderboards
    python scrape_top_players.py --mode all
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
