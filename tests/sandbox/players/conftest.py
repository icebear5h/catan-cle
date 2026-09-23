"""Shared fixtures for agent player context assembly, parsing, and receipt detachment."""

from dataclasses import replace

import pytest

from cle.game_engine.game import GameEngine
from cle.game_engine.models.enums import Action, ActionType
from cle.game_engine.models.player import Color
from cle.players.contracts import PlayerContext

from .support import COLORS, _context


@pytest.fixture
def trade_context() -> PlayerContext:
    engine = GameEngine(COLORS, seed=9, shuffle_players=False)
    return replace(
        _context(engine),
        legal_actions=(Action(Color.RED, ActionType.OFFER_TRADE, "parameterized trade"),),
    )
