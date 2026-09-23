"""Shared fixtures for spatial task graph, scoring, and dispatch contracts."""

import pytest

from cle.game_engine.game import GameEngine
from cle.game_engine.models.player import Color
from cle.game_engine.public_board import JsonValue, snapshot_public_board


@pytest.fixture
def game() -> GameEngine:
    return GameEngine(
        [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE],
        seed=123,
        shuffle_players=False,
    )


@pytest.fixture
def contract(game: GameEngine) -> dict[str, JsonValue]:
    return snapshot_public_board(game.observe(Color.RED)).contract()
