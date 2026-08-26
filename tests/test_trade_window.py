import pytest

from game_engine.game import GameEngine
from game_engine.models.actions import trade_response_actions
from game_engine.models.enums import ActionType
from game_engine.models.player import Color
from game_engine.trading import (
    TradeCandidate,
    TradeLimits,
    TradeOffer,
    TradeOfferStatus,
    TradeWindow,
    TradeWindowStatus,
)


ZERO = (0, 0, 0, 0, 0)
WOOD = (1, 0, 0, 0, 0)
TWO_WOOD = (2, 0, 0, 0, 0)
BRICK = (0, 1, 0, 0, 0)
ORE = (0, 0, 0, 0, 1)
COLORS = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)


def _window(**limits):
    return TradeWindow(
        id="trade-window-1",
        turn_player=Color.RED,
        participants=COLORS,
        limits=TradeLimits(**limits) if limits else TradeLimits(),
    )


def _offer(
    offered_by=Color.RED,
    audience=COLORS[1:],
    give=WOOD,
    receive=ORE,
    *,
    parent_offer_id=None,
    give_any=0,
    receive_any=0,
):
    return TradeOffer(
        offered_by=offered_by,
        audience=frozenset(audience),
        give=give,
        receive=receive,
        give_any=give_any,
        receive_any=receive_any,
        parent_offer_id=parent_offer_id,
    )


def test_game_state_has_no_legacy_trade_projections():
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


def test_trade_offer_is_one_semantic_who_and_what_contract():
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


def test_multiple_root_offers_acceptances_and_counters_coexist():
    window = _window()
    first = window.create_offer(_offer(give=TWO_WOOD))
    second = window.create_offer(_offer(give=BRICK, receive=WOOD))
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


def test_only_turn_player_selects_one_willing_counterparty():
    window = _window()
    offer = window.create_offer(_offer())
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


def test_counteroffer_is_directed_only_to_turn_player():
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


def test_counteroffer_is_candidate_only_for_turn_player():
    window = _window()
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

    with pytest.raises(ValueError, match="cannot respond"):
        window.signal_willingness(counter.id, Color.WHITE)
    assert window.executable_candidates() == (
        TradeCandidate(counter.id, Color.RED, Color.BLUE),
    )

    window.decline(counter.id, Color.RED)
    assert window.executable_candidates() == ()


def test_equivalent_directed_deals_are_rejected_in_opposite_orientation():
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


def test_counteroffer_cannot_be_countered():
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


def test_counter_action_is_not_generated_for_a_counteroffer():
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


def test_atomic_supersession_replaces_at_root_capacity():
    window = _window(
        max_active_root_offers=1,
        max_active_counteroffers=2,
        max_offers_per_player=1,
        max_negotiation_rounds=2,
        max_operations_per_player_round=1,
    )
    old = window.create_offer(_offer())
    new = window.create_offer(
        _offer(give=TWO_WOOD, receive=BRICK),
        supersedes_offer_id=old.id,
    )

    assert old.status == TradeOfferStatus.WITHDRAWN
    assert window.active_offers == (new,)


def test_selection_and_execution_close_all_remaining_offers():
    window = _window()
    selected = window.create_offer(_offer())
    other = window.create_offer(_offer(give=BRICK, receive=WOOD))
    window.signal_willingness(selected.id, Color.BLUE)

    window.select(Color.RED, window.executable_candidates()[0])
    window.mark_executed()

    assert window.status == TradeWindowStatus.CLOSED
    assert selected.status == TradeOfferStatus.EXECUTED
    assert other.status == TradeOfferStatus.EXPIRED


def test_negotiation_round_limit_blocks_new_offers_but_keeps_candidates():
    window = _window(
        max_active_root_offers=4,
        max_active_counteroffers=4,
        max_offers_per_player=3,
        max_negotiation_rounds=1,
        max_operations_per_player_round=2,
    )
    offer = window.create_offer(_offer())
    window.signal_willingness(offer.id, Color.BLUE)
    window.advance_round()

    with pytest.raises(ValueError, match="maximum negotiation rounds"):
        window.create_offer(_offer(give=BRICK, receive=WOOD))
    assert window.executable_candidates()


def test_invalid_or_empty_offer_fails_closed():
    with pytest.raises(ValueError, match="nonempty"):
        _offer(give=ZERO)
    with pytest.raises(ValueError, match="same resource"):
        _offer(receive=WOOD)
