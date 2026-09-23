"""Window and offer identity, replay ids, and stale replies."""
from copy import deepcopy
from typing import Any

import pytest

from cle.game_engine.models.enums import Action, ActionType
from cle.game_engine.models.player import Color
from cle.game_engine.state import apply_action, ensure_trade_window, new_trade_window
from cle.game_engine.trading import (
    TradeCandidate,
    TradeWindowStatus,
)

from .support import BRICK, COLORS, ORE, _offer, _trade_engine


def test_sequential_trade_windows_have_distinct_reproducible_ids_and_reject_stale_replies() -> None:
    engine: Any = _trade_engine()
    first = engine.step(
        Action(Color.RED, ActionType.OFFER_TRADE, _offer(audience=(Color.BLUE,)))
    ).resolved_action.value
    first_window_id = engine.state.trade_window.id
    engine.step(Action(Color.BLUE, ActionType.ACCEPT_TRADE, first.id))
    engine.step(
        Action(Color.RED, ActionType.CONFIRM_TRADE, TradeCandidate(first.id, Color.RED, Color.BLUE))
    )
    snapshot = engine.snapshot()
    preview: Any = new_trade_window(engine.state)
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


def test_new_trade_window_preview_is_pure_and_ignores_speech_revision() -> None:
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
    counter: bool, window_status: str, offer_id: str
) -> None:
    engine: Any = _trade_engine()
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
def test_forced_trade_creation_preserves_explicit_replay_ids(counter: bool) -> None:
    engine: Any = _trade_engine()
    proposal: Any = _offer()
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
    transition: Any = engine.step(Action(proposal.offered_by, action_type, proposal), force=True)
    assert transition.resolved_action.value.id == proposal.id
    assert engine.state.actions[-1].value.id == proposal.id
    assert engine.state.trade_window.offers[proposal.id].parent_offer_id == proposal.parent_offer_id
    assert transition.events[0].public_payload["id"] == proposal.id


def test_completed_offer_identity_cannot_be_reused_to_accept_different_terms() -> None:
    engine: Any = _trade_engine()
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
    second: Any = engine.step(action).resolved_action.value
    assert second.id != first.id
    assert engine.is_action_valid(stale_accept) is False
    with pytest.raises(ValueError):
        engine.step(stale_accept)
    assert engine.state.trade_window.offers[second.id].willing_by == set()
    assert engine.is_action_valid(Action(Color.BLUE, ActionType.ACCEPT_TRADE, second.id)) is True
