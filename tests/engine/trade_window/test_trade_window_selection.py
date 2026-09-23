"""Supersession, selection, and round-limit contracts."""
from copy import deepcopy
from typing import Any

import pytest

from cle.game_engine.game import GameEngine
from cle.game_engine.models.actions import trade_response_actions
from cle.game_engine.models.enums import Action, ActionType
from cle.game_engine.models.player import Color
from cle.game_engine.trading import (
    TradeOfferStatus,
    TradeWindowStatus,
)

from .support import (
    BRICK,
    COLORS,
    ORE,
    TWO_WOOD,
    WOOD,
    _assert_invalid_trade,
    _offer,
    _trade_engine,
    _window,
)


def test_counteroffer_cannot_be_countered() -> None:
    window = _window(
        max_active_root_offers=1,
        max_active_counteroffers=1,
        max_offers_per_player=2,
        max_negotiation_rounds=2,
        max_operations_per_player_round=1,
    )
    root = window.create_offer(_offer())
    counter = window.create_offer(
        _offer(
            offered_by=Color.BLUE,
            audience=(Color.RED,),
            give=ORE,
            receive=BRICK,
            parent_offer_id=root.id,
        )
    )

    with pytest.raises(ValueError, match="counteroffer cannot be countered"):
        window.create_offer(
            _offer(
                offered_by=Color.WHITE,
                audience=(Color.RED,),
                give=BRICK,
                receive=TWO_WOOD,
                parent_offer_id=counter.id,
            )
        )
    assert window.cap_hits == 0


def test_counter_action_is_not_generated_for_a_counteroffer() -> None:
    engine = GameEngine(COLORS, seed=3, shuffle_players=False)
    window = _window()
    root = window.create_offer(_offer())
    window.create_offer(
        _offer(
            offered_by=Color.BLUE,
            audience=(Color.RED,),
            give=ORE,
            receive=BRICK,
            parent_offer_id=root.id,
        )
    )
    engine.state.trade_window = window
    engine.state.player_state["P0_WOOD_IN_HAND"] = 1

    actions = trade_response_actions(engine.state, Color.RED)

    assert all(
        action.action_type != ActionType.COUNTER_OFFER
        for action in actions
    )


def test_atomic_supersession_replaces_at_root_capacity() -> None:
    window = _window(
        max_active_root_offers=1,
        max_active_counteroffers=2,
        max_offers_per_player=1,
        max_negotiation_rounds=2,
        max_operations_per_player_round=1,
    )
    old = window.create_offer(_offer())
    proposal = _offer(give=TWO_WOOD, receive=BRICK)
    before = deepcopy(window)
    assert window.validate_offer(proposal, supersedes_offer_id=old.id) is None
    assert window == before
    new = window.create_offer(
        proposal,
        supersedes_offer_id=old.id,
    )

    assert old.status == TradeOfferStatus.WITHDRAWN
    assert window.active_offers == (new,)


def test_selection_and_execution_close_all_remaining_offers() -> None:
    window: Any = _window()
    selected: Any = window.create_offer(_offer())
    other = window.create_offer(_offer(give=BRICK, receive=WOOD))
    window.signal_willingness(selected.id, Color.BLUE)

    window.select(Color.RED, window.executable_candidates()[0])
    window.mark_executed()

    assert window.status == TradeWindowStatus.CLOSED
    assert selected.status == TradeOfferStatus.EXECUTED
    assert other.status == TradeOfferStatus.EXPIRED


def test_negotiation_round_limit_blocks_new_offers_but_keeps_candidates() -> None:
    window = _window(
        max_active_root_offers=4,
        max_active_counteroffers=4,
        max_offers_per_player=3,
        max_negotiation_rounds=1,
        max_operations_per_player_round=2,
    )
    offer: Any = window.create_offer(_offer())
    window.signal_willingness(offer.id, Color.BLUE)
    window.advance_round()

    with pytest.raises(ValueError, match="maximum negotiation rounds"):
        window.create_offer(_offer(give=BRICK, receive=WOOD))
    assert window.executable_candidates()


def test_empty_open_window_does_not_bypass_round_limit_in_menu() -> None:
    engine: Any = _trade_engine(max_negotiation_rounds=1)
    root = engine.step(
        Action(Color.RED, ActionType.OFFER_TRADE, _offer())
    ).resolved_action.value
    engine.state.trade_window.advance_round()
    engine.step(Action(Color.RED, ActionType.CANCEL_TRADE, root.id))
    assert not engine.state.trade_window.active_offers
    assert all(action.action_type != ActionType.OFFER_TRADE for action in engine.state.playable_actions)
    _assert_invalid_trade(
        engine, _offer(give=BRICK), ActionType.OFFER_TRADE, "maximum negotiation rounds"
    )


def test_counter_menu_respects_per_player_offer_cap() -> None:
    engine = _trade_engine(max_offers_per_player=1)
    window = _window(max_offers_per_player=1)
    root = window.create_offer(_offer())
    second = window.create_offer(_offer(give=TWO_WOOD), allow_duplicate=True)
    window.create_offer(
        _offer(
            offered_by=Color.BLUE,
            audience=(Color.RED,),
            give=ORE,
            receive=BRICK,
            parent_offer_id=root.id,
        )
    )
    engine.state.trade_window = window
    responses = trade_response_actions(engine.state, Color.BLUE)
    assert Action(Color.BLUE, ActionType.ACCEPT_TRADE, second.id) in responses
    assert all(action.action_type != ActionType.COUNTER_OFFER for action in responses)
