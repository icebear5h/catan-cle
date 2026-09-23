"""Dataclasses and logging for the Colonist.io API client."""

import logging
from dataclasses import dataclass
from datetime import datetime

from data_pipeline.json_types import JsonDict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("data_pipeline.bootstrapping.scrapers.colonist_api")


@dataclass
class LeaderboardEntry:
    """A player on the leaderboard."""
    rank: int
    username: str
    rating: int
    games_played: int
    win_rate: float | None = None


@dataclass
class GameHistoryEntry:
    """A game from a player's history."""
    game_id: str
    date: datetime
    mode: str  # "Classic4P", "CitiesAndKnights4P", etc.
    player_color: int
    result: str  # "win", "loss", "draw"
    players: list[JsonDict]
    replay_url: str
    turn_count: int | None = None


__all__ = ["GameHistoryEntry", "LeaderboardEntry", "logger"]
