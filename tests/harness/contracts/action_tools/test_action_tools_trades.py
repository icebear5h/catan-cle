"""Trade offers, audiences, counters, and cancellation."""
import json
import re
from copy import deepcopy
from dataclasses import replace
from itertools import permutations
from typing import Any

import pytest

from cle.game_engine.models.actions import (
    year_of_plenty_possibilities,
)
from cle.game_engine.models.enums import Action, ActionType
from cle.game_engine.models.player import Color
from cle.game_engine.trading import TradeCandidate
from cle.harness.action_tools import parse_tool_choice, render_action_tools
from cle.players.validation import action_from_choice

from .support import OPAQUE_ID, ORE, WOOD, _case, _context, _offer, _turn_engine, _window


@pytest.mark.parametrize(
    "tool", ["accept_offer", "reject_offer", "counter_offer", "confirm_trade", "cancel_trade"]
)
@pytest.mark.parametrize(
    "bad", [None, 1, [], "", "o01", "RED", OPAQUE_ID.lower(), f" {OPAQUE_ID}", "unknown:offer"]
)
def test_offer_ids_are_opaque_exact_and_never_color_or_latest_offer_fallbacks(tool: str, bad: object) -> None:
    context, arguments, _ = _case(tool)
    with pytest.raises(ValueError):
        parse_tool_choice(context, tool, {**arguments, "offer_id": bad})


def test_root_and_counter_offers_bind_actor_audience_parent_and_wildcards_without_resolution() -> None:
    context, arguments, _ = _case("offer_trade")
    choice: Any = parse_tool_choice(
        context, "offer_trade", {**arguments, "give_any": 2, "receive_any": 1}
    )
    assert choice.trade_offer == _offer(give_any=2, receive_any=1)
    assert choice.trade_offer.id is None
    choice = parse_tool_choice(
        context, "offer_trade", {"give": {}, "receive": {}, "give_any": 1, "receive_any": 2}
    )
    assert choice.trade_offer.give == choice.trade_offer.receive == (0, 0, 0, 0, 0)
    assert choice.trade_offer.give_any == 1
    assert choice.trade_offer.receive_any == 2
    counter_context, arguments, _ = _case("counter_offer")
    counter: Any = parse_tool_choice(counter_context, "counter_offer", arguments).trade_offer
    assert counter.parent_offer_id == OPAQUE_ID
    assert counter.offered_by == Color.BLUE
    assert counter.audience == frozenset({Color.RED})
    assert counter.give == (0, 0, 0, 0, 2)
    assert counter.receive == WOOD


@pytest.mark.parametrize("field", ["give_any", "receive_any"])
@pytest.mark.parametrize("bad", [True, False, -1, 1.0, "1", None, []])
def test_wildcard_counts_obey_trade_offer_invariants(field: str, bad: object) -> None:
    context, arguments, _ = _case("offer_trade")
    with pytest.raises(ValueError):
        parse_tool_choice(context, "offer_trade", {**arguments, field: bad})


def test_trade_invariants_funding_and_window_limits_are_checked_without_mutation() -> None:
    context, arguments, _ = _case("offer_trade")
    for bad in (
        {"give": {"WOOD": 1}, "receive": {"wood": 1}},
        {"give": {"WOOD": 5}, "receive": {"ORE": 1}},
        {**arguments, "give_any": 20},
    ):
        with pytest.raises(ValueError):
            parse_tool_choice(context, "offer_trade", bad)
    context.observation.trade_window = _window()
    before = deepcopy(context.observation.trade_window)
    with pytest.raises(ValueError, match="Equivalent offer"):
        parse_tool_choice(context, "offer_trade", arguments)
    assert context.observation.trade_window == before
    context.observation.trade_window.round = (
        context.observation.trade_window.limits.max_negotiation_rounds
    )
    with pytest.raises(ValueError, match="maximum negotiation"):
        parse_tool_choice(context, "offer_trade", {"give": {"BRICK": 1}, "receive": {"SHEEP": 1}})


def test_counter_requires_exact_active_root_and_does_not_counter_a_counter() -> None:
    context, arguments, _ = _case("counter_offer")
    window: Any = context.observation.trade_window
    window.withdraw(OPAQUE_ID, Color.RED)
    with pytest.raises(ValueError, match="not active"):
        parse_tool_choice(context, "counter_offer", arguments)
    context.observation.trade_window = _window()
    counter: Any = context.observation.trade_window.create_offer(
        _offer(
            actor=Color.WHITE,
            audience=(Color.RED,),
            give=(0, 1, 0, 0, 0),
            receive=ORE,
            parent=OPAQUE_ID,
        )
    )
    context = replace(
        context,
        legal_actions=(
            Action(
                Color.BLUE,
                ActionType.COUNTER_OFFER,
                f"COUNTER_OFFER:{counter.id}: supply a named trade_offer",
            ),
        ),
    )
    with pytest.raises(ValueError, match="counteroffer cannot be countered"):
        parse_tool_choice(context, "counter_offer", {**arguments, "offer_id": counter.id})


def test_confirm_uses_candidate_counterparty_not_turn_player_and_rejects_non_candidates() -> None:
    context, arguments, target = _case("confirm_trade")
    second: Any = Action(
        Color.RED, ActionType.CONFIRM_TRADE, TradeCandidate(OPAQUE_ID, Color.RED, Color.WHITE)
    )
    context: Any = replace(context, legal_actions=(second, target))
    for counterparty, expected in (("BLUE", target), ("white", second)):
        choice: Any = parse_tool_choice(
            context, "confirm_trade", {**arguments, "counterparty": counterparty}
        )
        assert action_from_choice(context, choice) == expected
    for counterparty in ("RED", "ORANGE", "UNKNOWN"):
        with pytest.raises(ValueError):
            parse_tool_choice(context, "confirm_trade", {**arguments, "counterparty": counterparty})


def test_concrete_counter_preserves_exact_parent_audience_and_resource_terms() -> None:
    context, arguments, _ = _case("counter_offer")
    concrete = _offer(
        actor=Color.BLUE,
        audience=(Color.RED,),
        give=(0, 0, 0, 0, 2),
        receive=WOOD,
        parent=OPAQUE_ID,
    )
    context = replace(
        context, legal_actions=(Action(Color.BLUE, ActionType.COUNTER_OFFER, concrete),)
    )
    choice = parse_tool_choice(context, "counter_offer", arguments)
    assert choice.trade_offer is None
    assert action_from_choice(context, choice).value == concrete
    for changed in (
        {**arguments, "offer_id": "different:parent"},
        {**arguments, "give": {"ORE": 1}},
        {**arguments, "receive_any": 1},
    ):
        with pytest.raises(ValueError, match="No legal"):
            parse_tool_choice(context, "counter_offer", changed)


def test_large_trade_request_stays_a_fixed_bundle_not_an_invented_bank_limit() -> None:
    context, arguments, _ = _case("offer_trade")
    choice: Any = parse_tool_choice(context, "offer_trade", {**arguments, "receive": {"ORE": 10**100}})
    assert choice.trade_offer.receive == (0, 0, 0, 0, 10**100)


@pytest.mark.parametrize("multiple", [False, True])
def test_cancel_requires_offer_id_even_when_only_one_offer_is_cancellable(multiple: bool) -> None:
    context, _, target = _case("cancel_trade")
    second = Action(Color.RED, ActionType.CANCEL_TRADE, "other:offer")
    context = replace(context, legal_actions=(target, second) if multiple else (target,))
    with pytest.raises(ValueError, match="Expected arguments offer_id"):
        parse_tool_choice(context, "cancel_trade", {})


def test_cancel_advertises_and_independently_withdraws_each_exact_live_offer_id() -> None:
    engine: Any = _turn_engine()
    for give, receive in (("WOOD", "ORE"), ("BRICK", "SHEEP")):
        context = _context(engine)
        choice = parse_tool_choice(
            context, "offer_trade", {"give": {give: 1}, "receive": {receive: 1}}
        )
        engine.step(action_from_choice(context, choice))
    context = _context(engine)
    cancels: Any = tuple(a for a in context.legal_actions if a.action_type == ActionType.CANCEL_TRADE)
    ids: Any = {action.value for action in cancels}
    assert len(ids) == 2
    assert all(isinstance(offer_id, str) for offer_id in ids)
    assert not engine.is_action_valid(Action(Color.RED, ActionType.CANCEL_TRADE, None))
    for menu in permutations(cancels):
        permuted: Any = replace(context, legal_actions=menu)
        assert render_action_tools(permuted).splitlines()[-1] == (
            f"cancel_trade(offer_id): {json.dumps(sorted(ids))}"
        )
        with pytest.raises(ValueError, match="No legal cancel_trade"):
            parse_tool_choice(permuted, "cancel_trade", {"offer_id": "unavailable:offer"})
        for target in menu:
            choice = parse_tool_choice(permuted, "cancel_trade", {"offer_id": target.value})
            action: Any = action_from_choice(permuted, choice)
            assert action == target
            branch: Any = deepcopy(engine)
            assert branch.is_action_valid(action)
            branch.step(action)
            assert {offer.id for offer in branch.state.trade_window.active_offers} == ids - {
                target.value
            }
            current: Any = _context(branch)
            with pytest.raises(ValueError, match="No legal cancel_trade"):
                parse_tool_choice(current, "cancel_trade", {"offer_id": target.value})


def test_guidance_is_compact_menu_scoped_literal_and_does_not_invent_hidden_stock() -> None:
    context = _context()
    rendered = render_action_tools(context)
    assert "<N00>" in rendered
    assert "&lt;" not in rendered
    assert "<action>" not in rendered
    assert "action_index" not in rendered
    assert re.search(r"(?m)^\s*\d+[.):]", rendered) is None
    assert len(rendered.splitlines()) == 3
    assert "play_knight(" not in rendered
    assert "maritime_trade(" not in rendered
    context.observation.my_dev_cards["KNIGHT"] = 99
    context.observation.opponent_dev_card_counts[Color.BLUE] = 888
    assert render_action_tools(context) == rendered
    yop = _context(actions=year_of_plenty_possibilities(Color.RED, [2] * 5))
    guidance = render_action_tools(yop)
    assert len(guidance.splitlines()) == 3
    assert guidance.count("play_year_of_plenty(") == 1
    assert "bank stock" not in guidance.lower()
