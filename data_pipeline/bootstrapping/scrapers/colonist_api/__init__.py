"""
Colonist.io API client for scraping top player replays.

Endpoints discovered:
- GET /api/leaderboards-tabs/ - List leaderboard categories
- GET /api/leaderboards/{type} - Get rankings for a category
- GET /api/profile/{username} - Get player profile
- GET /api/profile/{username}/games - Get player's game history
"""

from data_pipeline.bootstrapping.scrapers.colonist_api._client import ColonistAPI
from data_pipeline.bootstrapping.scrapers.colonist_api._discovery import (
    discover_api_endpoints,
)
from data_pipeline.bootstrapping.scrapers.colonist_api._models import (
    GameHistoryEntry,
    LeaderboardEntry,
    logger,
)
from data_pipeline.bootstrapping.scrapers.colonist_api._parsing import (
    GAME_TYPES,
    parse_games_response,
    parse_leaderboard_response,
)

__all__ = [
    "GAME_TYPES",
    "ColonistAPI",
    "GameHistoryEntry",
    "LeaderboardEntry",
    "discover_api_endpoints",
    "logger",
    "parse_games_response",
    "parse_leaderboard_response",
]
