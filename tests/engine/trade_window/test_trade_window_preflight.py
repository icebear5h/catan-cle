"""Engine preflight rejection before window mutation."""
from copy import deepcopy
from typing import Any

import pytest

from cle.game_engine.models.actions import generate_playable_actions
from cle.game_engine.models.enums import Action, ActionType
from cle.game_engine.models.player import Color

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


@pytest.mark.parametrize(
    "kind, withdrawn",
    [("root", False), ("root", True), ("counter", False), ("counter", True),
     ("opposite-counter", False)],
)
def test_engine_rejects_duplicate_offers_before_mutation(kind: str, withdrawn: bool) -> None:
    engine: Any = _trade_engine()
    root = engine.step(
        Action(
            Color.RED,
            ActionType.OFFER_TRADE,
            _offer(audience=(Color.BLUE,) if kind == "opposite-counter" else COLORS[1:]),
        )
    ).resolved_action.value
    proposal: Any = _offer()
    action_type = ActionType.OFFER_TRADE
    old_id: Any = root.id
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
def test_engine_rejects_trade_limits_without_counting_preflight_cap_hits(counter: bool, limits: dict[str, int], message: str) -> None:
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
    window: Any = engine.state.trade_window
    if "max_negotiation_rounds" in limits:
        window.advance_round()
    engine.state.playable_actions = generate_playable_actions(engine.state)
    _assert_invalid_trade(engine, proposal, action_type, message)
    assert window.cap_hits == 0

    before: Any = deepcopy(window)
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
    offered_by: Color, audience: tuple[Color, ...], counter: bool, message: str
) -> None:
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
def test_root_preflight_does_not_install_or_replace_window(closed: bool, valid: bool) -> None:
    engine: Any = _trade_engine()
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


def test_offer_id_collision_is_rejected_before_supersession_mutates_window() -> None:
    window = _window()
    old = window.create_offer(_offer())
    proposal = _offer(give=TWO_WOOD)
    before = deepcopy(window)
    for operation in (window.validate_offer, window.create_offer):
        with pytest.raises(ValueError, match="already exists"):
            operation(proposal, supersedes_offer_id=old.id, offer_id=old.id)
        assert window == before


def test_engine_rejects_offer_id_collision_before_mutation() -> None:
    engine = _trade_engine()
    root = engine.step(
        Action(Color.RED, ActionType.OFFER_TRADE, _offer())
    ).resolved_action.value
    proposal = _offer(give=TWO_WOOD)
    proposal.id = root.id
    _assert_invalid_trade(engine, proposal, ActionType.OFFER_TRADE, "already exists")


def test_replay_duplicate_override_keeps_validation_nonmutating() -> None:
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
