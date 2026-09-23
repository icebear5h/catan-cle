"""Counteroffer, trade, and decision parsing."""
from dataclasses import replace
from typing import Any

import pytest

from cle.game_engine.models.actions import trade_response_actions
from cle.game_engine.models.enums import Action, ActionType
from cle.game_engine.models.player import Color
from cle.harness import (
    ModelResponse,
    PlayerResponseParseError,
    PlayerResponseParser,
    default_suite_path,
    load_context_suite,
)
from cle.sandbox import CatanSandbox

from .support import _sandbox_and_player, allows_deprecated_suite


@allows_deprecated_suite
def test_counteroffer_parser_preserves_engine_generated_parent_id(trade_sandbox: CatanSandbox) -> None:
    engine: Any = trade_sandbox.game_engine
    suite = load_context_suite(default_suite_path().with_name("catan_v10.yaml"))
    parser = PlayerResponseParser(suite)
    context: Any = trade_sandbox.decision_context()
    index = next(
        i
        for i, action in enumerate(context.legal_actions)
        if action.action_type == ActionType.OFFER_TRADE
    )
    root_choice: Any = parser.parse(
        context,
        ModelResponse(
            content=(
                f"<action>{index}</action>"
                '<trade_offer>{"give":{"WOOD":1},"receive":{"ORE":1}}</trade_offer>'
            )
        ),
    )
    selected: Any = context.action_at(root_choice.action_index)
    root: Any = engine.step(
        Action(selected.color, selected.action_type, root_choice.trade_offer)
    ).resolved_action.value
    context = replace(
        context,
        context_id=f"{engine.id}:{engine.revision}:BLUE",
        actor=Color.BLUE,
        observation=engine.observe(Color.BLUE),
        events=engine.project_events(Color.BLUE),
        legal_actions=tuple(trade_response_actions(engine.state, Color.BLUE)),
    )
    index = next(
        i
        for i, action in enumerate(context.legal_actions)
        if action.action_type == ActionType.COUNTER_OFFER
    )

    choice: Any = parser.parse(
        context,
        ModelResponse(
            content=(
                f"<action>{index}</action>"
                '<trade_offer>{"give":{"ORE":1},"receive":{"WOOD":2}}</trade_offer>'
            )
        ),
    )

    assert ":o1" in root.id
    assert choice.trade_offer.parent_offer_id == root.id
    assert choice.trade_offer.audience == frozenset({Color.RED})
    selected = context.action_at(choice.action_index)
    action = Action(selected.color, selected.action_type, choice.trade_offer)
    assert engine.is_action_valid(action)
    assert engine.step(action).resolved_action.value.parent_offer_id == root.id


@allows_deprecated_suite
@pytest.mark.parametrize(
    "payload",
    [
        '{"give":{"WOOD":2,"WOOD":1},"receive":{"ORE":1}}',
        '{"give":{"WOOD":1},"receive":{"ORE":2,"ORE":1}}',
        '{"give":{"WOOD":2},"give":{"WOOD":1},"receive":{"ORE":1}}',
        '{"give":{"WOOD":1},"receive":{"ORE":1},"receive":{"ORE":2}}',
        '{"give":{"WOOD":1},"receive":{"ORE":1},"give_any":0,"give_any":1}',
        '{"give":{"WOOD":1},"receive":{"ORE":1},"receive_any":0,"receive_any":1}',
        '{"give":{"WOOD":2,"wood":1},"receive":{"ORE":1}}',
    ],
)
def test_trade_parser_rejects_duplicate_json_keys(trade_sandbox: CatanSandbox, payload: str) -> None:
    context = trade_sandbox.decision_context()
    suite = load_context_suite(default_suite_path().with_name("catan_v10.yaml"))
    index = next(
        i
        for i, action in enumerate(context.legal_actions)
        if action.action_type == ActionType.OFFER_TRADE
    )

    with pytest.raises(PlayerResponseParseError, match="Duplicate"):
        PlayerResponseParser(suite).parse(
            context,
            ModelResponse(content=f"<action>{index}</action><trade_offer>{payload}</trade_offer>"),
        )


@allows_deprecated_suite
@pytest.mark.parametrize("text", [
    "<!-- <action>0</action> -->",
    "<!-- action_index: 0 -->",
    "<action index='0'>0</action>",
    "<action>0</action><action/>",
    "<action>0</action><action>1",
    "<action><action>0</action></action>",
    "<game_plan>one</game_plan><game_plan>two</game_plan><action>0</action>",
    '<action>0</action><discard>{"WOOD":4}</discard><discard>{"WOOD":4}</discard>',
])
def test_decision_parser_rejects_malformed_repeated_and_comment_only_controls(text: str) -> None:
    sandbox, _, _ = _sandbox_and_player([])
    suite = load_context_suite(default_suite_path().with_name("catan_v10.yaml"))
    with pytest.raises(PlayerResponseParseError):
        PlayerResponseParser(suite).parse(
            sandbox.decision_context(), ModelResponse(content=text)
        )


@allows_deprecated_suite
def test_decision_parser_keeps_xml_escaped_prose_and_raw_output_separate() -> None:
    sandbox, _, _ = _sandbox_and_player([])
    suite = load_context_suite(default_suite_path().with_name("catan_v10.yaml"))
    text: Any = (
        "<!-- <action>2</action> action_index: 3 -->"
        "<game_plan>ORE &gt; WOOD &amp; wheat &lt; sheep</game_plan><action>0</action>"
    )
    choice: Any = PlayerResponseParser(suite).parse(
        sandbox.decision_context(), ModelResponse(content=text)
    )
    assert choice.action_index == 0
    assert choice.game_plan == "ORE > WOOD & wheat < sheep"
    assert choice.raw_response == text
