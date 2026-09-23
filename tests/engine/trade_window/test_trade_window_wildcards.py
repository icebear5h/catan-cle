"""Wildcard offers and exact-counter execution."""
from typing import Any

import pytest

from cle.game_engine.models.actions import trade_response_actions
from cle.game_engine.models.enums import Action, ActionType
from cle.game_engine.models.player import Color
from cle.game_engine.models.trade import ResourceBundle
from cle.game_engine.state import apply_action
from cle.game_engine.state_functions import get_player_freqdeck
from cle.game_engine.trading import (
    TradeCandidate,
    TradeOfferStatus,
    TradeWindowStatus,
)

from .support import COLORS, ORE, WOOD, ZERO, _offer, _trade_engine


def test_invalid_or_empty_offer_fails_closed() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        _offer(give=ZERO)
    with pytest.raises(ValueError, match="same resource"):
        _offer(receive=WOOD)


@pytest.mark.parametrize("counter", [False, True], ids=["root", "counter"])
@pytest.mark.parametrize(
    "give, receive, give_any, receive_any",
    [
        (ZERO, ORE, 1, 0),
        (WOOD, ZERO, 0, 1),
        (WOOD, ORE, 1, 0),
        (WOOD, ORE, 0, 1),
        (WOOD, ORE, 1, 1),
        (ZERO, ZERO, 1, 1),
    ],
    ids=["give-any", "receive-any", "mixed-give", "mixed-receive", "mixed-both", "all-any"],
)
def test_wildcards_require_an_exact_counter_before_execution(
    counter: bool,
    give: ResourceBundle,
    receive: ResourceBundle,
    give_any: int,
    receive_any: int,
) -> None:
    engine = _trade_engine()
    parent_id = None
    if counter:
        parent_id = engine.step(
            Action(Color.RED, ActionType.OFFER_TRADE, _offer())
        ).resolved_action.value.id
    proposal = _offer(
        offered_by=Color.BLUE if counter else Color.RED,
        audience=(Color.RED,) if counter else COLORS[1:],
        give=give,
        receive=receive,
        give_any=give_any,
        receive_any=receive_any,
        parent_offer_id=parent_id,
    )
    action = Action(
        proposal.offered_by,
        ActionType.COUNTER_OFFER if counter else ActionType.OFFER_TRADE,
        proposal,
    )
    assert engine.is_action_valid(action) is True
    offer: Any = engine.step(action).resolved_action.value
    window: Any = engine.state.trade_window
    if not counter:
        responses = trade_response_actions(engine.state, Color.BLUE)
        assert any(action.action_type == ActionType.COUNTER_OFFER for action in responses)
        accept = Action(Color.BLUE, ActionType.ACCEPT_TRADE, offer.id)
        assert accept in responses
        engine.step(accept)
        assert window.offers[offer.id].willing_by == {Color.BLUE}

    candidate: Any = TradeCandidate(offer.id, Color.RED, Color.BLUE)
    confirm = Action(Color.RED, ActionType.CONFIRM_TRADE, candidate)
    before = engine.snapshot()
    assert window.executable_candidates() == ()
    assert confirm not in engine.state.playable_actions
    assert engine.is_action_valid(confirm) is False
    with pytest.raises(ValueError, match="not playable right now"):
        engine.step(confirm)
    with pytest.raises(ValueError, match="not executable"):
        window.select(Color.RED, candidate)
    with pytest.raises(ValueError, match="not executable"):
        apply_action(engine.state, confirm, force=True)
    assert window == before.state.trade_window
    assert engine.state.player_state == before.state.player_state
    assert engine.state.resource_freqdeck == before.state.resource_freqdeck
    assert engine.state.actions == before.state.actions
    assert tuple(engine.events) == before.events
    assert len(engine.history) == len(before.history)

    exact: Any = engine.step(
        Action(
            Color.BLUE,
            ActionType.COUNTER_OFFER,
            _offer(
                offered_by=Color.BLUE,
                audience=(Color.RED,),
                give=ORE,
                receive=WOOD,
                parent_offer_id=parent_id if counter else offer.id,
            ),
        )
    ).resolved_action.value
    confirm_exact = Action(
        Color.RED,
        ActionType.CONFIRM_TRADE,
        TradeCandidate(exact.id, Color.RED, Color.BLUE),
    )
    assert confirm_exact in engine.state.playable_actions
    assert engine.is_action_valid(confirm_exact) is True
    engine.step(confirm_exact)
    assert get_player_freqdeck(engine.state, Color.RED) == [2, 3, 0, 0, 4]
    assert get_player_freqdeck(engine.state, Color.BLUE) == [4, 3, 0, 0, 2]
    assert engine.state.resource_freqdeck == before.state.resource_freqdeck
    assert window.offers[exact.id].status == TradeOfferStatus.EXECUTED
    assert window.status == TradeWindowStatus.CLOSED


def test_exact_root_trade_still_executes() -> None:
    engine = _trade_engine()
    offer = engine.step(
        Action(Color.RED, ActionType.OFFER_TRADE, _offer(audience=(Color.BLUE,)))
    ).resolved_action.value
    engine.step(Action(Color.BLUE, ActionType.ACCEPT_TRADE, offer.id))
    confirm = Action(
        Color.RED, ActionType.CONFIRM_TRADE, TradeCandidate(offer.id, Color.RED, Color.BLUE)
    )
    assert confirm in engine.state.playable_actions
    assert engine.is_action_valid(confirm) is True
    engine.step(confirm)
    assert get_player_freqdeck(engine.state, Color.RED) == [2, 3, 0, 0, 4]
    assert get_player_freqdeck(engine.state, Color.BLUE) == [4, 3, 0, 0, 2]
