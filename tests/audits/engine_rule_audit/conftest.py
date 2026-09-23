"""Shared fixtures for local correctness evidence for engine rules, not a production policy."""

import pytest

from cle.game_engine.game import GameEngine
from cle.game_engine.models.decks import (
    CITY_COST_FREQDECK,
    DEVELOPMENT_CARD_COST_FREQDECK,
)
from cle.game_engine.models.enums import (
    VICTORY_POINT,
    Action,
    ActionType,
)
from cle.game_engine.models.player import Color

from .support import _assert_inventory, _fund, _position


@pytest.fixture
def terminal_game() -> GameEngine:
    nodes = (0, 2, 4, 7)
    game = _position(tuple((Color.RED, node) for node in nodes))
    for node in nodes:
        _fund(game, Color.RED, CITY_COST_FREQDECK)
        game.step(Action(Color.RED, ActionType.BUILD_CITY, node))
    for _ in range(2):
        _fund(game, Color.RED, DEVELOPMENT_CARD_COST_FREQDECK)
        game.step(Action(Color.RED, ActionType.BUY_DEVELOPMENT_CARD, VICTORY_POINT), force=True)
    _assert_inventory(game)
    assert game.winning_color() == Color.RED
    return game
