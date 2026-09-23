"""Shared helpers for trade-window identity, negotiation, and preflight contracts."""

from copy import deepcopy

import pytest

from cle.game_engine.game import GameEngine
from cle.game_engine.models.actions import generate_playable_actions
from cle.game_engine.models.enums import Action, ActionPrompt, ActionType
from cle.game_engine.models.player import Color
from cle.game_engine.models.trade import ResourceBundle
from cle.game_engine.trading import (
    TradeLimits,
    TradeOffer,
    TradeWindow,
)

ZERO = (0, 0, 0, 0, 0)


WOOD = (1, 0, 0, 0, 0)


TWO_WOOD = (2, 0, 0, 0, 0)


BRICK = (0, 1, 0, 0, 0)


ORE = (0, 0, 0, 0, 1)


COLORS = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)


def _window(**limits: int) -> TradeWindow:
    return TradeWindow(
        id="trade-window-1",
        turn_player=Color.RED,
        participants=COLORS,
        limits=TradeLimits(**limits) if limits else TradeLimits(),
    )


def _offer(
    offered_by: Color = Color.RED,
    audience: tuple[Color, ...] = COLORS[1:],
    give: ResourceBundle = WOOD,
    receive: ResourceBundle = ORE,
    *,
    parent_offer_id: str | None = None,
    give_any: int = 0,
    receive_any: int = 0,
) -> TradeOffer:
    return TradeOffer(
        offered_by=offered_by,
        audience=frozenset(audience),
        give=give,
        receive=receive,
        give_any=give_any,
        receive_any=receive_any,
        parent_offer_id=parent_offer_id,
    )


def _trade_engine(**limits: int) -> GameEngine:
    engine = GameEngine(
        COLORS,
        seed=3,
        shuffle_players=False,
        capture_history=True,
        trade_limits=TradeLimits(**limits),
    )
    state = engine.state
    state.is_initial_build_phase = False
    state.current_prompt = ActionPrompt.PLAY_TURN
    state.player_state["P0_HAS_ROLLED"] = True
    for player in ("P0", "P1"):
        for resource in ("WOOD", "BRICK", "ORE"):
            state.player_state[f"{player}_{resource}_IN_HAND"] = 3
    state.resource_freqdeck = [13, 13, 19, 19, 13]
    state.playable_actions = generate_playable_actions(state)
    return engine


def _assert_invalid_trade(
    engine: GameEngine, offer: TradeOffer, action_type: ActionType, message: str
) -> None:
    action = Action(offer.offered_by, action_type, offer)
    before = engine.snapshot()
    proposed = deepcopy(offer)
    with pytest.raises(ValueError, match=message):
        engine.state.trade_window.validate_offer(offer)
    assert engine.is_action_valid(action) is False
    assert engine.is_action_valid(action) is False
    with pytest.raises(ValueError, match="not playable right now"):
        engine.step(action)
    assert offer == proposed
    assert engine.state.trade_window == before.state.trade_window
    assert engine.state.player_state == before.state.player_state
    assert engine.state.resource_freqdeck == before.state.resource_freqdeck
    assert engine.state.playable_actions == before.state.playable_actions
    assert engine.state.actions == before.state.actions
    assert tuple(engine.events) == before.events
    assert len(engine.history) == len(before.history)
    assert engine.rng.getstate() == before.state.rng.getstate()
