"""Semantic trade lifecycle, matching, and parsers."""
import json
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest

from cle.game_engine.models.enums import Action, ActionType
from cle.game_engine.models.player import Color
from cle.game_engine.trading import TradeCandidate, TradeOffer
from cle.harness.action_tools import (
    parse_tool_choice,
    render_action_tools,
)
from cle.harness.context import ContextAssembler, PlayerResponseParseError, PlayerResponseParser
from cle.harness.models import ModelResponse
from cle.harness.shared_suite import load_shared_prompt_suite
from cle.replay.runtime.step_executor import (
    _apply_trade_closures,
    _publish_replay_action,
    _publish_trade_overlay,
)

from .support import COLORS, WOOD, context, main_engine, offer, semantic


def test_semantic_trade_lifecycle_and_counter_direction() -> None:
    engine: Any = main_engine()
    root: Any = offer(engine)
    terms: Any = {"player": "RED", "give": {"ORE": 1}, "receive": {"WOOD": 1}}
    for tool, action_type, actor in (("reject_offer", ActionType.REJECT_TRADE, Color.WHITE), ("accept_offer", ActionType.ACCEPT_TRADE, Color.BLUE)):
        action: Any = semantic(engine, tool, actor, **terms)
        assert action == Action(actor, action_type, root.id)
        event = engine.step(action).events[0]
        assert event.public_payload["offer"]["give"] == {"WOOD": 1}
        assert event.public_payload["offer"]["receive"] == {"ORE": 1}
    engine.step(semantic(engine, "accept_offer", Color.ORANGE, **terms))
    confirm: Any = semantic(engine, "confirm_trade", player="BLUE", give={"WOOD": 1}, receive={"ORE": 1})
    assert confirm.value == TradeCandidate(root.id, Color.RED, Color.BLUE)
    event = engine.step(confirm).events[0]
    assert event.public_payload["offer"]["id"] == root.id
    assert engine.state.trade_window.status.value == "closed"

    engine = main_engine()
    root = offer(engine)
    counter = semantic(engine, "counter_offer", Color.BLUE, player="RED",
                       original={"give": {"ORE": 1}, "receive": {"WOOD": 1}},
                       proposed={"give": {"ORE": 2}, "receive": {"WOOD": 1}})
    counter_event: Any = engine.step(counter).events[0]
    assert counter_event.public_payload["original"]["give"] == {"WOOD": 1}
    confirm = semantic(engine, "confirm_trade", player="BLUE", give={"WOOD": 1}, receive={"ORE": 2})
    assert confirm.value.offer_id != root.id
    engine.step(confirm)


def test_semantic_matching_rejects_stale_wrong_direction_and_ambiguity() -> None:
    engine: Any = main_engine()
    root = offer(engine)
    with pytest.raises(ValueError, match="No active visible"):
        semantic(engine, "accept_offer", Color.BLUE, player="RED", give={"WOOD": 1}, receive={"ORE": 1})
    # Equivalent terms from distinct offers must not be guessed, even if only one
    # corresponding menu action is currently executable.
    duplicate: Any = deepcopy(root)
    duplicate.id = "another-active-offer"
    engine.state.trade_window.offers[duplicate.id] = duplicate
    with pytest.raises(ValueError, match="Ambiguous trade"):
        semantic(engine, "accept_offer", Color.BLUE, player="RED", give={"ORE": 1}, receive={"WOOD": 1})
    duplicate.status = type(duplicate.status).WITHDRAWN
    cancel = semantic(engine, "cancel_trade", player="BLUE", give={"WOOD": 1}, receive={"ORE": 1})
    event: Any = engine.step(cancel).events[0]
    assert event.public_payload["offer"]["give"] == {"WOOD": 1}
    with pytest.raises(ValueError, match="stale"):
        semantic(engine, "accept_offer", Color.BLUE, player="RED", give={"ORE": 1}, receive={"WOOD": 1})


def test_wildcard_events_visible_without_legal_menu_or_hidden_hands() -> None:
    engine: Any = main_engine()
    root: Any = offer(engine, receive=(0, 0, 0, 0, 0), receive_any=1)
    event_text: Any = ContextAssembler._format_events(tuple(engine.project_events(Color.BLUE))[-1:], shared=True)
    assert "RED gives 1 WOOD, receives 1 ANY" in event_text
    assert "not executable" in event_text
    assert root.id not in event_text
    ctx = context(engine, Color.BLUE)
    assert not engine.state.trade_window.executable_candidates()
    assert "accept_offer(" in render_action_tools(ctx, shared=True)
    action = semantic(engine, "reject_offer", Color.BLUE, player="RED", give={}, receive={"WOOD": 1}, give_any=1)
    engine.step(action)
    rendered = ContextAssembler._format_events(tuple(engine.project_events(Color.BLUE))[-1:], shared=True)
    assert "REJECT_TRADE RED gives 1 WOOD, receives 1 ANY" in rendered
    assert "IN_HAND" not in rendered


def test_shared_parser_requires_terms_while_legacy_id_parser_remains_valid() -> None:
    engine = main_engine()
    root = offer(engine)
    ctx = context(engine, Color.BLUE)
    assert parse_tool_choice(ctx, "accept_offer", {"offer_id": root.id}).action_index >= 0
    parser = PlayerResponseParser(load_shared_prompt_suite().decision_suite())
    with pytest.raises(PlayerResponseParseError, match="Expected arguments"):
        parser.parse(ctx, ModelResponse(json.dumps({"tool": "accept_offer", "arguments": {"offer_id": root.id}})))


def test_replay_source_only_proposals_keep_terms_in_responses_and_closure() -> None:
    engine = main_engine()
    source: Any = TradeOffer(Color.RED, frozenset(COLORS[1:]), WOOD, (0, 0, 0, 0, 0), receive_any=1, id="source-only")
    _publish_replay_action(engine, Action(Color.RED, ActionType.OFFER_TRADE, source))
    assert engine.state.trade_window is None
    runtime = SimpleNamespace(
        current_game=engine, replay_index=0,
        replay_data={"colonist_color_to_engine_idx": {"1": 0, "2": 1}},
    )
    _publish_replay_action(engine, Action(Color.BLUE, ActionType.REJECT_TRADE, source.id))
    _publish_trade_overlay(runtime, {"type": "CLEAR_TRADE_RESPONSE", "player": 2, "trade_id": source.id})
    _apply_trade_closures({"type": "CLOSE_TRADE", "creator": 1, "trade_id": source.id, "reason": "withdrawn"}, runtime)
    events: Any = tuple(engine.project_events(Color.BLUE))[-3:]
    for event in events:
        assert event.payload["offer"]["receive_any"] == 1
        rendered: Any = ContextAssembler._format_events((event,), shared=True)
        assert "RED gives 1 WOOD, receives 1 ANY" in rendered
        assert source.id not in rendered
    assert ContextAssembler._format_events((events[0],)).endswith("REJECT_TRADE source-only")
