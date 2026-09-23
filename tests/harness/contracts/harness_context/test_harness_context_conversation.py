"""Conversation retention, retries, and completion limits."""
import json
from typing import Any

import pytest

from cle.game_engine.board_tokens import node_token
from cle.game_engine.models.player import Color
from cle.harness import (
    ModelResponse,
    PlayerResponseParseError,
    PlayerResponseParser,
    default_suite_path,
    load_context_suite,
)

from .support import _response, _sandbox_and_player


@pytest.mark.parametrize("suite_file", ["catan_v10.yaml", "catan_v11.yaml"])
@pytest.mark.parametrize("content", ["", " \n\t"])
@pytest.mark.parametrize("reasoning_channel", [None, "text", "details"])
def test_missing_final_answer_is_not_parsed_from_reasoning(
    suite_file: str, content: str, reasoning_channel: str | None
) -> None:
    sandbox, _, _ = _sandbox_and_player([])
    context: Any = sandbox.decision_context()
    parser: Any = PlayerResponseParser(
        load_context_suite(default_suite_path().with_name(suite_file))
    )
    valid_action: Any = (
        "<action>0</action>"
        if parser.suite.response.format == "xml"
        else json.dumps({
            "tool": "build_settlement",
            "arguments": {"node": node_token(context.legal_actions[0].value)},
        })
    )
    assert parser.parse(context, ModelResponse(content=valid_action)).action_index == 0
    response = ModelResponse(
        content=content,
        native_reasoning=valid_action if reasoning_channel == "text" else "",
        native_reasoning_details=(
            ({"type": "reasoning.text", "text": valid_action},)
            if reasoning_channel == "details" else ()
        ),
        finish_reason="stop",
        provider_native_finish_reason="stop",
    )

    with pytest.raises(PlayerResponseParseError) as error:
        parser.parse(context, response)

    assert str(error.value) == (
        "The provider returned reasoning but no final answer."
        if reasoning_channel else "The provider returned no final answer."
    )


@pytest.mark.parametrize(
    ("finish_reason", "native_finish_reason"),
    [("length", None), (None, "max_tokens"), (None, "length")],
)
def test_missing_final_answer_reports_only_explicit_completion_limits(
    finish_reason: str | None, native_finish_reason: str | None
) -> None:
    sandbox, player, _ = _sandbox_and_player([])
    response = ModelResponse(
        content="", native_reasoning="unfinished analysis",
        finish_reason=finish_reason,
        provider_native_finish_reason=native_finish_reason,
    )

    with pytest.raises(PlayerResponseParseError) as error:
        PlayerResponseParser(player.suite).parse(sandbox.decision_context(), response)

    assert str(error.value) == (
        "The provider returned reasoning but no final answer. "
        "The provider reported a completion-token limit."
    )


@pytest.mark.asyncio
async def test_agent_player_keeps_full_conversation_and_full_visible_events() -> None:
    sandbox, player, transport = _sandbox_and_player(
        [_response(), _response(plan="connect the opening road"), _response()]
    )

    first: Any = await sandbox.step()
    second = await sandbox.step()

    assert first.transitions[0].resolved_action == first.contexts[0].legal_actions[0]
    assert second.transitions[0].resolved_action == second.contexts[0].legal_actions[0]
    assert first.attempts[0].choice.native_reasoning == "native analysis"
    assert first.attempts[0].choice.native_reasoning_details == (
        {"type": "reasoning.text"},
    )
    assert dict(first.attempts[0].choice.reasoning_request) == {
        "effort": "xhigh",
        "exclude": False,
    }
    assert first.attempts[0].choice.provider_response_id == "gen-context-test"
    assert first.attempts[0].choice.provider_request_id == "req-context-test"
    assert first.attempts[0].choice.provider_native_finish_reason == "stop"
    first_system = transport.requests[0].messages[0].content
    first_user = transport.requests[0].messages[-1].content
    assert first_system == "You are playing a game of Catan. You are playing as RED."
    assert "expert" not in first_system.lower()
    assert "prioritize" not in first_system.lower()
    assert len(first_system.split()) < 150
    first_request: Any = transport.requests[0]
    first_board: Any = first_request.board_presentation
    assert first_board.kind == "text"
    assert first_board.format == "indexed_tile_rows/v3"
    assert first_board.provenance.source_id == first_request.decision_id
    assert first_board.provenance.perspective == Color.RED
    assert first_board.provenance.identity_space == "canonical_engine_ids"
    assert "CATAN FULL PUBLIC GRAPH V1" in first_board.content
    assert first_board.byte_length > 10_000
    first_components = first_request.components
    assert [component.id for component in first_components] == [
        "system.identity",
        "environment.strategic_memory",
        "environment.visible_events",
        "environment.recent_table_talk",
        "environment.commitments",
        "environment.phase_info",
        "environment.board_state",
        "environment.resources",
        "environment.opponents",
        "environment.trade_window",
        "environment.phase_guidance",
        "environment.legal_actions",
        "environment.decision_request",
        "environment.response_schema",
    ]
    assert first_user == "\n\n".join(
        component.rendered for component in first_components[1:]
    )
    assert "DECISION FACTS:" in first_user
    assert "Round 1 (first settlement + road): RED -> BLUE -> WHITE -> ORANGE" in first_user
    assert "Round 2 (second settlement + road): ORANGE -> WHITE -> BLUE -> RED" in first_user
    assert "Your positions: round 1 = 1/4; round 2 = 4/4." in first_user
    assert "first settlement gives no starting cards" in first_user
    assert "second settlement is independent of the first road" in first_user
    assert "nominal diversity without buildable combinations can be weak" in first_user
    assert next(
        component.value
        for component in first_components
        if component.id == "environment.board_state"
    ).startswith("YOUR BUILDINGS:")
    assert next(
        component.value
        for component in first_components
        if component.id == "environment.resources"
    ).startswith("YOUR RESOURCES:")
    assert first_user.count("VALID ACTIONS:") == 1
    assert "<valid_actions>" not in first_user
    assert "dice=" in first_user
    assert "pip" in first_user.lower()
    assert player.session.strategic_memory == "connect the opening road"
    assert all(
        "CATAN FULL PUBLIC GRAPH" not in message.content
        for message in player.session.messages
    )
    assert [message.role for message in player.session.messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert [message.role for message in transport.requests[1].messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    second_user = transport.requests[1].messages[-1].content
    assert "0. RED: BUILD_SETTLEMENT" in second_user
    assert "more than one viable expansion location" in second_user
    assert "remains an independent choice" in second_user
    assert "does not need to connect to this road" in second_user

    while sandbox.current_actor() != Color.RED:
        await sandbox.step()
    await sandbox.step()

    third_user = transport.requests[2].messages[-1].content
    assert "0. RED: BUILD_SETTLEMENT" in third_user
    assert "1. RED: BUILD_ROAD" in third_user


@pytest.mark.asyncio
async def test_sandbox_snapshot_restores_private_player_conversation() -> None:
    sandbox, player, _ = _sandbox_and_player([_response(), _response()])
    await sandbox.step()
    snapshot = sandbox.snapshot()

    await sandbox.step()
    assert len(player.session.messages) == 4

    sandbox.restore(snapshot)

    assert len(player.session.messages) == 2
    assert player.session.strategic_memory == "expand toward wheat"
    assert len(player.session.receipts) == 1


@pytest.mark.asyncio
async def test_sandbox_retries_invalid_player_output_with_feedback() -> None:
    sandbox, _, transport = _sandbox_and_player(
        [ModelResponse(content="<action>999</action>"), _response()]
    )

    result = await sandbox.step()

    assert result.after_revision == 1
    assert len(transport.requests) == 2
    assert "CORRECTION FROM THE SANDBOX" in transport.requests[1].messages[-1].content
    assert len(sandbox.decision_trace) == 1
