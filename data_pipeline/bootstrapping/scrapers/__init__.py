"""
Colonist.io Replay Scraping Pipeline

Components:
- replay_playwright_scraper: Browser-session replay scraper (recommended)
- replay_api_scraper: Direct API scraper/debug fallback
- colonist_api: API client for leaderboards, profiles, game history
- scrape_top_players: Build game index from leaderboards

Run the command modules from the repository root. Indexes default to
``artifacts/raw/colonist/indexes`` and captures default to
``artifacts/staging/colonist/replays``; see the bootstrapping README for bounded
examples and promotion gates.
"""

from .colonist_api import ColonistAPI, GameHistoryEntry, LeaderboardEntry

__all__ = ["ColonistAPI", "GameHistoryEntry", "LeaderboardEntry"]
