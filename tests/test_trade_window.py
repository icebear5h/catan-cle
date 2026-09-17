from copy import deepcopy

import pytest

from cle.game_engine.game import GameEngine
from cle.game_engine.models.actions import generate_playable_actions, trade_response_actions
from cle.game_engine.models.enums import Action, ActionPrompt, ActionType
from cle.game_engine.models.player import Color
from cle.game_engine.state import apply_action, ensure_trade_window, new_trade_window
from cle.game_engine.state_functions import get_player_freqdeck
from cle.game_engine.trading import (
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


def _trade_engine(**limits):
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


def _assert_invalid_trade(engine, offer, action_type, message):
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
    counter, give, receive, give_any, receive_any
):
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
    offer = engine.step(action).resolved_action.value
    window = engine.state.trade_window
    if not counter:
        responses = trade_response_actions(engine.state, Color.BLUE)
        assert any(action.action_type == ActionType.COUNTER_OFFER for action in responses)
        accept = Action(Color.BLUE, ActionType.ACCEPT_TRADE, offer.id)
        assert accept in responses
        engine.step(accept)
        assert window.offers[offer.id].willing_by == {Color.BLUE}

    candidate = TradeCandidate(offer.id, Color.RED, Color.BLUE)
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

    exact = engine.step(
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


def test_exact_root_trade_still_executes():
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


@pytest.mark.parametrize(
    "kind, withdrawn",
    [("root", False), ("root", True), ("counter", False), ("counter", True),
     ("opposite-counter", False)],
)
def test_engine_rejects_duplicate_offers_before_mutation(kind, withdrawn):
    engine = _trade_engine()
    root = engine.step(
        Action(
            Color.RED,
            ActionType.OFFER_TRADE,
            _offer(audience=(Color.BLUE,) if kind == "opposite-counter" else COLORS[1:]),
        )
    ).resolved_action.value
    proposal = _offer()
    action_type = ActionType.OFFER_TRADE
    old_id = root.id
    if kind != "root":
        action_type = ActionType.COUNTER_OFFER
        proposal = _offer(
            offered_by=Color.BLUE,
            audience=(Color.RED,),
            give=ORE,
            receive=WOOD,
            parent_offer_id=root.id,
        )
        if kind == "counter":
            old_id = engine.step(
                Action(Color.BLUE, action_type, proposal)
            ).resolved_action.value.id
    if withdrawn:
        engine.state.trade_window.withdraw(old_id, proposal.offered_by)
        engine.state.playable_actions = generate_playable_actions(engine.state)
    _assert_invalid_trade(engine, proposal, action_type, "Equivalent offer")


@pytest.mark.parametrize(
    "counter, limits, message",
    [
        (False, {"max_active_root_offers": 1}, "maximum active root offers"),
        (False, {"max_offers_per_player": 1}, "maximum active offers for player"),
        (False, {"max_negotiation_rounds": 1}, "maximum negotiation rounds"),
        (True, {"max_active_counteroffers": 1}, "maximum active counteroffers"),
        (True, {"max_offers_per_player": 1}, "maximum active offers for player"),
        (True, {"max_negotiation_rounds": 1}, "maximum negotiation rounds"),
    ],
)
def test_engine_rejects_trade_limits_without_counting_preflight_cap_hits(counter, limits, message):
    engine = _trade_engine(**limits)
    root = engine.step(
        Action(Color.RED, ActionType.OFFER_TRADE, _offer())
    ).resolved_action.value
    proposal = _offer(give=BRICK)
    action_type = ActionType.OFFER_TRADE
    if counter:
        action_type = ActionType.COUNTER_OFFER
        proposal = _offer(
            offered_by=Color.BLUE,
            audience=(Color.RED,),
            give=ORE,
            receive=BRICK,
            parent_offer_id=root.id,
        )
        engine.step(Action(Color.BLUE, action_type, proposal))
        proposal = _offer(
            offered_by=Color.BLUE,
            audience=(Color.RED,),
            give=ORE,
            receive=WOOD,
            parent_offer_id=root.id,
        )
    window = engine.state.trade_window
    if "max_negotiation_rounds" in limits:
        window.advance_round()
    engine.state.playable_actions = generate_playable_actions(engine.state)
    _assert_invalid_trade(engine, proposal, action_type, message)
    assert window.cap_hits == 0

    before = deepcopy(window)
    with pytest.raises(ValueError, match=message):
        window.create_offer(proposal)
    before.cap_hits += 1
    assert window == before


@pytest.mark.parametrize(
    "offered_by, audience, counter, message",
    [
        (Color.RED, (Color.BLACK,), False, "non-participant"),
        (Color.BLUE, (Color.RED, Color.WHITE), True, "only to the turn player"),
        (Color.BLUE, (Color.WHITE,), True, "only to the turn player"),
        (Color.WHITE, (Color.RED,), True, "cannot respond"),
        (Color.BLACK, (Color.RED,), True, "offerer must be a window participant"),
    ],
)
def test_engine_rejects_invalid_trade_audiences_before_mutation(
    offered_by, audience, counter, message
):
    engine = _trade_engine()
    root = engine.step(
        Action(Color.RED, ActionType.OFFER_TRADE, _offer(audience=(Color.BLUE,)))
    ).resolved_action.value
    proposal = _offer(
        offered_by=offered_by,
        audience=audience,
        give=ORE,
        receive=BRICK,
        parent_offer_id=root.id if counter else None,
    )
    action_type = ActionType.COUNTER_OFFER if counter else ActionType.OFFER_TRADE
    _assert_invalid_trade(engine, proposal, action_type, message)


@pytest.mark.parametrize("closed", [False, True], ids=["missing-window", "closed-window"])
@pytest.mark.parametrize("valid", [False, True], ids=["invalid-audience", "valid"])
def test_root_preflight_does_not_install_or_replace_window(closed, valid):
    engine = _trade_engine()
    if closed:
        engine.step(Action(Color.RED, ActionType.OFFER_TRADE, _offer()))
        engine.state.trade_window.close()
    window = engine.state.trade_window
    before = deepcopy(window)
    action = Action(
        Color.RED,
        ActionType.OFFER_TRADE,
        _offer(audience=COLORS[1:] if valid else (Color.BLACK,)),
    )
    assert engine.is_action_valid(action) is valid
    assert engine.state.trade_window is window
    assert engine.state.trade_window == before
    if valid:
        engine.step(action)
        assert engine.state.trade_window is not window
        assert len(engine.state.trade_window.offers) == 1
    else:
        with pytest.raises(ValueError, match="not playable right now"):
            engine.step(action)
        assert engine.state.trade_window is window
        assert engine.state.trade_window == before


def test_offer_id_collision_is_rejected_before_supersession_mutates_window():
    window = _window()
    old = window.create_offer(_offer())
    proposal = _offer(give=TWO_WOOD)
    before = deepcopy(window)
    for operation in (window.validate_offer, window.create_offer):
        with pytest.raises(ValueError, match="already exists"):
            operation(proposal, supersedes_offer_id=old.id, offer_id=old.id)
        assert window == before


def test_engine_rejects_offer_id_collision_before_mutation():
    engine = _trade_engine()
    root = engine.step(
        Action(Color.RED, ActionType.OFFER_TRADE, _offer())
    ).resolved_action.value
    proposal = _offer(give=TWO_WOOD)
    proposal.id = root.id
    _assert_invalid_trade(engine, proposal, ActionType.OFFER_TRADE, "already exists")


def test_replay_duplicate_override_keeps_validation_nonmutating():
    window = _window(max_active_root_offers=1, max_offers_per_player=1)
    window.create_offer(_offer())
    window.round = window.limits.max_negotiation_rounds
    before = deepcopy(window)
    assert window.validate_offer(_offer(), offer_id="replay-offer", allow_duplicate=True) is None
    assert window == before
    offer = window.create_offer(_offer(), offer_id="replay-offer", allow_duplicate=True)
    assert offer.id == "replay-offer"
    assert len(window.active_offers) == 2
    assert window.cap_hits == 0


def test_empty_open_window_does_not_bypass_round_limit_in_menu():
    engine = _trade_engine(max_negotiation_rounds=1)
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


def test_counter_menu_respects_per_player_offer_cap():
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


def test_sequential_trade_windows_have_distinct_reproducible_ids_and_reject_stale_replies():
    engine = _trade_engine()
    first = engine.step(
        Action(Color.RED, ActionType.OFFER_TRADE, _offer(audience=(Color.BLUE,)))
    ).resolved_action.value
    first_window_id = engine.state.trade_window.id
    engine.step(Action(Color.BLUE, ActionType.ACCEPT_TRADE, first.id))
    engine.step(
        Action(Color.RED, ActionType.CONFIRM_TRADE, TradeCandidate(first.id, Color.RED, Color.BLUE))
    )
    snapshot = engine.snapshot()
    preview = new_trade_window(engine.state)
    assert preview.id != first_window_id
    assert preview.id == f"turn-{engine.state.num_turns}-trade-{len(engine.state.actions)}"
    assert engine.state.trade_window == snapshot.state.trade_window
    assert engine.state.trade_window.status == TradeWindowStatus.CLOSED
    second_action = Action(Color.RED, ActionType.OFFER_TRADE, _offer(give=BRICK))
    second = engine.step(second_action).resolved_action.value
    assert engine.state.trade_window.id == preview.id
    assert first.id != second.id
    second_window = deepcopy(engine.state.trade_window)
    stale = (
        Action(Color.BLUE, ActionType.ACCEPT_TRADE, first.id),
        Action(Color.BLUE, ActionType.REJECT_TRADE, first.id),
        Action(Color.RED, ActionType.CONFIRM_TRADE, TradeCandidate(first.id, Color.RED, Color.BLUE)),
        Action(Color.BLUE, ActionType.COUNTER_OFFER, _offer(
            offered_by=Color.BLUE, audience=(Color.RED,), give=ORE, receive=BRICK,
            parent_offer_id=first.id,
        )),
    )
    for action in stale:
        assert engine.is_action_valid(action) is False
        with pytest.raises(ValueError):
            engine.step(action)
    assert engine.state.trade_window == second_window
    engine.undo()
    assert engine.step(second_action).resolved_action.value.id == second.id
    engine.restore(snapshot)
    branch = engine.copy()
    assert engine.step(second_action).resolved_action.value.id == second.id
    assert branch.step(second_action).resolved_action.value.id == second.id


def test_new_trade_window_preview_is_pure_and_ignores_speech_revision():
    engine = _trade_engine()
    first = new_trade_window(engine.state)
    assert new_trade_window(engine.state) == first
    assert engine.state.trade_window is None
    engine.append_message(
        speaker=Color.RED, text="Considering a trade", audience=COLORS[1:],
        causation_id="test-trade-window",
    )
    assert new_trade_window(engine.state) == first
    assert engine.state.trade_window is None
    actual = ensure_trade_window(engine.state)
    assert actual == first
    assert actual is not first
    assert ensure_trade_window(engine.state) is actual


@pytest.mark.parametrize("counter", [False, True], ids=["root", "counter"])
@pytest.mark.parametrize("window_status", ["missing", "open", "closed"])
@pytest.mark.parametrize("offer_id", ["caller-assigned", ""])
def test_nonforced_trade_ids_are_rejected_before_any_state_mutation(
    counter, window_status, offer_id
):
    engine = _trade_engine()
    parent_id = "missing-root"
    if window_status != "missing":
        parent_id = engine.step(
            Action(Color.RED, ActionType.OFFER_TRADE, _offer())
        ).resolved_action.value.id
        if window_status == "closed":
            engine.state.trade_window.close()
    proposal = _offer(
        offered_by=Color.BLUE if counter else Color.RED,
        audience=(Color.RED,) if counter else COLORS[1:],
        give=ORE if counter else BRICK,
        receive=BRICK if counter else ORE,
        parent_offer_id=parent_id if counter else None,
    )
    proposal.id = offer_id
    action = Action(
        proposal.offered_by,
        ActionType.COUNTER_OFFER if counter else ActionType.OFFER_TRADE,
        proposal,
    )
    before = engine.snapshot()
    window = engine.state.trade_window
    proposed = deepcopy(proposal)
    with pytest.raises(ValueError, match="assigned by the engine"):
        apply_action(engine.state, action)
    assert proposal == proposed
    assert engine.state.trade_window is window
    assert engine.state.trade_window == before.state.trade_window
    assert engine.state.player_state == before.state.player_state
    assert engine.state.resource_freqdeck == before.state.resource_freqdeck
    assert engine.state.actions == before.state.actions
    assert engine.state.playable_actions == before.state.playable_actions
    assert engine.state.current_prompt == before.state.current_prompt
    assert engine.state.rng.getstate() == before.state.rng.getstate()
    assert tuple(engine.events) == before.events
    assert len(engine.history) == len(before.history)
    assert engine.is_action_valid(action) is False
    with pytest.raises(ValueError):
        engine.step(action)
    assert len(engine.history) == len(before.history)


@pytest.mark.parametrize("counter", [False, True], ids=["root", "counter"])
def test_forced_trade_creation_preserves_explicit_replay_ids(counter):
    engine = _trade_engine()
    proposal = _offer()
    proposal.id = "colonist-root"
    action_type = ActionType.OFFER_TRADE
    if counter:
        root = engine.step(Action(Color.RED, action_type, proposal), force=True).resolved_action.value
        proposal = _offer(
            offered_by=Color.BLUE, audience=(Color.RED,), give=ORE, receive=BRICK,
            parent_offer_id=root.id,
        )
        proposal.id = "colonist-counter"
        action_type = ActionType.COUNTER_OFFER
    transition = engine.step(Action(proposal.offered_by, action_type, proposal), force=True)
    assert transition.resolved_action.value.id == proposal.id
    assert engine.state.actions[-1].value.id == proposal.id
    assert engine.state.trade_window.offers[proposal.id].parent_offer_id == proposal.parent_offer_id
    assert transition.events[0].public_payload["id"] == proposal.id


def test_completed_offer_identity_cannot_be_reused_to_accept_different_terms():
    engine = _trade_engine()
    first = engine.step(
        Action(Color.RED, ActionType.OFFER_TRADE, _offer(audience=(Color.BLUE,)))
    ).resolved_action.value
    stale_accept = Action(Color.BLUE, ActionType.ACCEPT_TRADE, first.id)
    engine.step(stale_accept)
    engine.step(
        Action(Color.RED, ActionType.CONFIRM_TRADE, TradeCandidate(first.id, Color.RED, Color.BLUE))
    )
    before = engine.snapshot()
    closed_window = engine.state.trade_window
    replacement = _offer(give=BRICK, audience=(Color.BLUE,))
    replacement.id = first.id
    action = Action(Color.RED, ActionType.OFFER_TRADE, replacement)
    with pytest.raises(ValueError, match="assigned by the engine"):
        apply_action(engine.state, action)
    assert engine.state.trade_window is closed_window
    assert engine.state.trade_window == before.state.trade_window
    assert engine.state.actions == before.state.actions
    assert engine.is_action_valid(action) is False

    replacement.id = None
    second = engine.step(action).resolved_action.value
    assert second.id != first.id
    assert engine.is_action_valid(stale_accept) is False
    with pytest.raises(ValueError):
        engine.step(stale_accept)
    assert engine.state.trade_window.offers[second.id].willing_by == set()
    assert engine.is_action_valid(Action(Color.BLUE, ActionType.ACCEPT_TRADE, second.id)) is True
