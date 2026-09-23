"""Offer, counteroffer, and candidate contracts."""
from typing import Any

import pytest

from cle.game_engine.game import GameEngine
from cle.game_engine.models.player import Color
from cle.game_engine.trading import (
    TradeCandidate,
)

from .support import BRICK, COLORS, ORE, TWO_WOOD, WOOD, _offer, _window


def test_game_state_has_no_legacy_trade_projections() -> None:
    state = GameEngine(COLORS, seed=3, shuffle_players=False).state

    for name in (
        "active_trades",
        "counter_offers",
        "current_trade",
        "is_resolving_trade",
        "acceptees",
        "rejecters",
    ):
        assert not hasattr(state, name)


def test_trade_offer_is_one_semantic_who_and_what_contract() -> None:
    offer = _offer(give_any=1)

    assert offer.to_payload() == {
        "id": None,
        "offered_by": "RED",
        "audience": ["BLUE", "ORANGE", "WHITE"],
        "give": {"WOOD": 1},
        "receive": {"ORE": 1},
        "give_any": 1,
        "receive_any": 0,
        "parent_offer_id": None,
        "willing_by": [],
        "declined_by": [],
        "status": "active",
    }
    with pytest.raises(ValueError, match="same resource"):
        _offer(receive=WOOD)


def test_multiple_root_offers_acceptances_and_counters_coexist() -> None:
    window: Any = _window()
    first: Any = window.create_offer(_offer(give=TWO_WOOD))
    second: Any = window.create_offer(_offer(give=BRICK, receive=WOOD))
    counter = window.create_offer(
        _offer(
            offered_by=Color.BLUE,
            audience=(Color.RED,),
            give=ORE,
            receive=WOOD,
            parent_offer_id=first.id,
        )
    )
    window.signal_willingness(first.id, Color.ORANGE)
    window.signal_willingness(second.id, Color.WHITE)

    assert {offer.id for offer in window.active_offers} == {
        first.id,
        second.id,
        counter.id,
    }
    assert {
        (item.offer_id, item.counterparty)
        for item in window.executable_candidates()
    } == {
        (first.id, Color.ORANGE),
        (second.id, Color.WHITE),
        (counter.id, Color.BLUE),
    }


def test_only_turn_player_selects_one_willing_counterparty() -> None:
    window = _window()
    offer: Any = window.create_offer(_offer())
    window.signal_willingness(offer.id, Color.BLUE)
    window.signal_willingness(offer.id, Color.WHITE)
    candidates = window.executable_candidates()

    assert {candidate.counterparty for candidate in candidates} == {
        Color.BLUE,
        Color.WHITE,
    }
    assert window.selected_candidate is None
    with pytest.raises(ValueError, match="Only the turn player"):
        window.select(Color.BLUE, candidates[0])

    selected = next(
        candidate
        for candidate in candidates
        if candidate.counterparty == Color.WHITE
    )
    window.select(Color.RED, selected)

    assert window.selected_candidate == selected
    assert offer.active


def test_counteroffer_is_directed_only_to_turn_player() -> None:
    window = _window()
    root = window.create_offer(_offer())

    with pytest.raises(ValueError, match="only to the turn player"):
        window.create_offer(
            _offer(
                offered_by=Color.BLUE,
                audience=(Color.RED, Color.WHITE),
                give=ORE,
                receive=BRICK,
                parent_offer_id=root.id,
            )
        )


def test_counteroffer_is_candidate_only_for_turn_player() -> None:
    window = _window()
    root = window.create_offer(_offer())
    counter: Any = window.create_offer(
        _offer(
            offered_by=Color.BLUE,
            audience=(Color.RED,),
            give=ORE,
            receive=BRICK,
            parent_offer_id=root.id,
        )
    )

    with pytest.raises(ValueError, match="cannot respond"):
        window.signal_willingness(counter.id, Color.WHITE)
    assert window.executable_candidates() == (
        TradeCandidate(counter.id, Color.RED, Color.BLUE),
    )

    window.decline(counter.id, Color.RED)
    assert window.executable_candidates() == ()


def test_equivalent_directed_deals_are_rejected_in_opposite_orientation() -> None:
    window = _window()
    window.create_offer(_offer(audience=(Color.BLUE,)))

    with pytest.raises(ValueError, match="Equivalent offer"):
        window.create_offer(
            _offer(
                offered_by=Color.BLUE,
                audience=(Color.RED,),
                give=ORE,
                receive=WOOD,
            )
        )
