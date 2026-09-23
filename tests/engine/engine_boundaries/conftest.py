"""Shared fixtures for isolated engine-boundary regressions using reduced, bank-balanced positions."""

import pytest

from cle.game_engine.game import GameEngine
from cle.game_engine.models.actions import generate_playable_actions
from cle.game_engine.models.enums import (
    CITY,
    ROAD,
    SETTLEMENT,
    Action,
    ActionPrompt,
    ActionType,
)
from cle.game_engine.models.player import Color
from cle.game_engine.trading import TradeOffer

from .support import COLORS


@pytest.fixture
def engine() -> GameEngine:
    game = GameEngine(COLORS, seed=7, shuffle_players=False, capture_history=True)
    state = game.state
    state.is_initial_build_phase = False
    state.current_prompt = ActionPrompt.PLAY_TURN
    state.player_state["P0_HAS_ROLLED"] = True
    state.player_state["P0_WOOD_IN_HAND"] = 5
    state.player_state["P0_BRICK_IN_HAND"] = 4
    state.player_state["P1_ORE_IN_HAND"] = 3
    state.resource_freqdeck = [14, 15, 19, 19, 16]
    # Populate lazy building caches before checking read-boundary mutations.
    for color in COLORS:
        for kind in (SETTLEMENT, CITY, ROAD):
            state.buildings_by_color[color][kind] = []
    state.playable_actions = generate_playable_actions(state)
    return game


@pytest.fixture
def offer_action() -> Action:
    return Action(
        Color.RED,
        ActionType.OFFER_TRADE,
        TradeOffer(Color.RED, frozenset({Color.BLUE}), (1, 0, 0, 0, 0), (0, 0, 0, 0, 1)),
    )
