"""Shared configuration for the Reddit and BoardGameGeek forum scrapers."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
CACHE_DIR = PROJECT_ROOT / "artifacts" / "cache" / "pretraining" / "forums"

REDDIT_SUBREDDITS = ["catan", "Catan", "boardgames"]
REDDIT_SEARCH_QUERIES = [
    "strategy", "tips", "placement", "trading", "longest road",
    "development cards", "robber", "resource", "city", "settlement",
]

# Main Catan game ID on BGG is 13 (the original Settlers of Catan)
BGG_GAME_ID = 13
BGG_API_BASE = "https://boardgamegeek.com/xmlapi2"

__all__ = [
    "BGG_API_BASE",
    "BGG_GAME_ID",
    "CACHE_DIR",
    "PROJECT_ROOT",
    "REDDIT_SEARCH_QUERIES",
    "REDDIT_SUBREDDITS",
]
