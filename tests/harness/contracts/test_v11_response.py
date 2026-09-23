"""Strict JSON envelope and real AgentPlayer admission for v11 tools."""
import json
import pickle
import re
from collections.abc import Iterable
from dataclasses import replace
from typing import Any

import pytest

from cle.game_engine.board_tokens import node_token, tile_token
from cle.game_engine.events import GameEvent
from cle.game_engine.game import GameEngine
from cle.game_engine.models.actions import generate_playable_actions
from cle.game_engine.models.enums import ActionPrompt, ActionType
from cle.game_engine.models.player import Color
from cle.harness.communication import load_communication_suite
from cle.harness.context import ContextAssembler, PlayerResponseParseError, PlayerResponseParser
from cle.harness.models import ModelRequest, ModelResponse, PlayerSession
from cle.harness.suite import load_context_suite
from cle.players.agent import AgentPlayer
from cle.players.baseline import FirstLegalPlayer
from cle.players.contracts import PlayerContext
from cle.players.validation import action_from_choice
from cle.sandbox import CatanSandbox, RetryPolicy
from cle.sandbox.communication import CommunicationOpportunity

COLORS = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)


class NoCommunication:
    def pre_action(self, engine: GameEngine) -> tuple[CommunicationOpportunity, ...]:
        return ()

    def after_events(self, engine: GameEngine, events: tuple[GameEvent, ...], *, round_number: int) -> tuple[CommunicationOpportunity, ...]:
        return ()


class LocalTransport:
    def __init__(self, engine: GameEngine, responses: Iterable[str]) -> None:
        self.engine = engine
        self.responses = iter(responses)
        self.requests: list[ModelRequest] = []
        self.snapshots: list[bytes] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        self.snapshots.append(pickle.dumps(self.engine.snapshot()))
        return ModelResponse(content=next(self.responses), model="local/test")


def _sandbox(engine: GameEngine) -> CatanSandbox:
    return CatanSandbox(
        engine,
        {color: FirstLegalPlayer(color) for color in COLORS},
        communication_policy=NoCommunication(),
    )


@pytest.fixture
def context() -> PlayerContext:
    return _sandbox(GameEngine(COLORS, seed=7, shuffle_players=False)).decision_context()


def test_json_keeps_trained_token_and_provider_metadata(context: PlayerContext) -> None:
    target: Any = context.legal_actions[0]
    text: Any = json.dumps({
        "game_plan": 'Keep <N00> in mind; {"tool":"end_turn"} is just a note.',
        "tool": "build_settlement",
        "arguments": {"node": node_token(target.value)},
    })
    response = ModelResponse(
        content=text,
        model="local/test",
        usage=(("completion_tokens", 40),),
        latency_ms=12,
        native_reasoning="separate reasoning",
        native_reasoning_details=({"type": "reasoning.text"},),
        reasoning_request=(("effort", "high"),),
        provider_response_id="response-test",
        provider_request_id="request-test",
        provider_native_finish_reason="stop",
    )
    choice: Any = PlayerResponseParser(load_context_suite()).parse(context, response)
    assert action_from_choice(context, choice) == target
    assert choice.game_plan == json.loads(text)["game_plan"]
    assert choice.raw_response == text
    assert node_token(target.value) in choice.raw_response
    assert "&lt;" not in choice.raw_response
    assert choice.parse_warning is None
    for field in (
        "model", "usage", "latency_ms", "native_reasoning", "native_reasoning_details",
        "reasoning_request", "provider_response_id", "provider_request_id",
        "provider_native_finish_reason",
    ):
        assert getattr(choice, field) == getattr(response, field)


@pytest.mark.parametrize("text", [
    "", "null", "[]", "0", "{}",
    '<action>0</action>',
    '{"tool":"build_settlement"}',
    '{"arguments":{"node":"<N00>"}}',
    '{"tool":"build_settlement","arguments":null}',
    '{"tool":"build_settlement","arguments":[]}',
    '{"tool":"build_settlement","arguments":{"node":"<N00>"},"game_plan":false}',
    '{"tool":"build_settlement","arguments":{"node":"<N00>"},"action_index":0}',
    '{"tool":"build_settlement","arguments":{"node":"<N00>"},"tool":"end_turn"}',
    '{"tool":"build_settlement","arguments":{"node":"<N00>","node":"<N01>"}}',
    '{"tool":"build_settlement","arguments":{"node":"<N00>","extra":1}}',
    '{"tool":"build_settlement","arguments":{"node":0}}',
    '{"tool":"build_settlement","arguments":{"node":"N00"}}',
    '{"tool":"build_settlement","arguments":{"node":"&lt;N00&gt;"}}',
    r'{"tool":"build_settlement","arguments":{"node":"\u003cN00\u003e"}}',
    r'{"tool":"build_settlement","arguments":{"node":"<\u004e00>"}}',
    r'{"game_plan":"\"node\":\"<N00>\"", "tool":"build_settlement","arguments":{"node":"\u003cN00\u003e"}}',
    '{"tool":"build_settlement","arguments":{"node":"<n00>"}}',
    '{"tool":"build_settlement","arguments":{"node":"<N99>"}}',
    '{"tool":"build_settlement","arguments":{"node":"<T00>"}}',
    '{"tool":"build_settlement","arguments":{"node":NaN}}',
    '{"tool":"build_settlement","arguments":{"node":Infinity}}',
    '{"tool":false,"arguments":{}}',
    '{"tool":[],"arguments":{}}',
    '{"tool":"end_turn","arguments":{}}',
    '{"tool":"nonexistent","arguments":{}}',
    '{"tool":"build_settlement","arguments":{"node":"<N00>"}} extra',
    '{"tool":"build_settlement","arguments":{"node":"<N00>"}} {}',
    '```json\n{"tool":"build_settlement","arguments":{"node":"<N00>"}}\n```',
    '[' * 2000 + '0' + ']' * 2000,
    '{"tool":"build_settlement","arguments":{"node":' + '9' * 5000 + '}}',
    ' ' * (128 * 1024 + 1),
])
def test_invalid_envelope_never_falls_back_or_mutates_context(
    context: PlayerContext, text: str
) -> None:
    before = pickle.dumps(context)
    with pytest.raises(PlayerResponseParseError):
        PlayerResponseParser(load_context_suite()).parse(context, ModelResponse(content=text))
    assert pickle.dumps(context) == before


def test_default_packet_advertises_semantic_tools_and_preserves_token_identity(context: PlayerContext) -> None:
    suite = load_context_suite()
    session = PlayerSession(color=context.actor, session_id="tool-test")
    request = ContextAssembler(suite).assemble(context, session)
    menu = next(c.rendered for c in request.components if c.id == "environment.legal_actions")
    assert "build_settlement(node)" in menu
    assert node_token(context.legal_actions[0].value) in menu
    assert "0. " not in menu
    assert "play_knight" not in menu
    assert "zero-based" not in request.messages[-1].content
    # Reordering the engine's internal menu changes neither affordances nor semantics.
    reordered = replace(context, legal_actions=tuple(reversed(context.legal_actions)))
    other = ContextAssembler(suite).assemble(reordered, session)
    assert other.messages == request.messages


@pytest.mark.asyncio
async def test_json_knight_retry_preserves_state_then_commits_one_decision() -> None:
    # Reduced real engine state isolates Knight admission, not strategic gameplay.
    engine = GameEngine(COLORS, seed=7, shuffle_players=False, capture_history=True)
    state = engine.state
    state.is_initial_build_phase = False
    state.current_prompt = ActionPrompt.PLAY_TURN
    state.player_state["P0_KNIGHT_IN_HAND"] = 1
    state.player_state["P0_KNIGHT_OWNED_AT_START"] = True
    state.development_listdeck.remove("KNIGHT")
    state.playable_actions = generate_playable_actions(state)
    destination, tile = next(
        (coord, tile) for coord, tile in state.board.map.land_tiles.items()
        if coord != state.board.robber_coordinate
    )
    invalid = '{"tool":"play_knight","arguments":{"tile":"<T99>"}}'
    valid = json.dumps({
        "game_plan": "Move the robber with this Knight.",
        "tool": "play_knight",
        "arguments": {"tile": tile_token(tile.id)},
    })
    before = pickle.dumps(engine.snapshot())
    transport = LocalTransport(engine, [invalid, valid])
    agent: Any = AgentPlayer(
        Color.RED, transport, session_id="json-knight",
        suite=load_context_suite(),
        communication_suite=load_communication_suite(),
    )
    sandbox = _sandbox(engine)
    sandbox.players[Color.RED] = agent
    sandbox.retry_policy = RetryPolicy(max_decision_attempts=2)

    result: Any = await sandbox.step()

    assert transport.snapshots == [before, before]
    assert len(transport.requests) == 2
    assert "CORRECTION FROM THE SANDBOX" in transport.requests[1].messages[-1].content
    assert [item.requested_action.action_type for item in result.transitions] == [
        ActionType.PLAY_KNIGHT_CARD, ActionType.MOVE_ROBBER,
    ]
    assert len(result.attempts) == len(agent.session.receipts) == 1
    assert agent.session.messages[-1].content == valid
    assert len(agent.session.messages) == 2
    assert state.board.robber_coordinate == destination
    assert state.player_state["P0_KNIGHT_IN_HAND"] == 0
    receipt: Any = agent.session.receipts[result.context.context_id]
    assert receipt.after_revision == result.after_revision == 2
    assert receipt.choice.knight_destination == destination


def _main_phase_context() -> PlayerContext:
    """A post-roll turn where end_turn is legal (the fixture context is setup)."""
    engine = GameEngine(COLORS, seed=7, shuffle_players=False)
    while engine.state.is_initial_build_phase:
        engine.step(engine.state.playable_actions[0])
    engine.state.player_state["P0_HAS_ROLLED"] = True
    engine.state.playable_actions = generate_playable_actions(engine.state)
    return _sandbox(engine).decision_context()


@pytest.mark.parametrize("text", [
    '{"tool":"end_turn","game_plan":"pass"}',
    '{"tool":"end_turn"}',
])
def test_missing_arguments_on_a_no_parameter_tool_is_accepted(text: str) -> None:
    # The common live miss: a valid object that drops "arguments" on end_turn.
    # It means {} and is treated as such instead of costing a retry.
    context = _main_phase_context()
    end_turn: Any = next(i for i, a in enumerate(context.legal_actions) if a.action_type == ActionType.END_TURN)
    choice: Any = PlayerResponseParser(load_context_suite()).parse(context, ModelResponse(content=text))
    assert choice.action_index == end_turn


def test_missing_arguments_on_a_parameter_tool_still_fails_on_its_own_terms(context: PlayerContext) -> None:
    with pytest.raises(PlayerResponseParseError):
        PlayerResponseParser(load_context_suite()).parse(
            context, ModelResponse(content='{"tool":"build_settlement"}'),
        )


def test_stray_top_level_keys_are_named() -> None:
    with pytest.raises(PlayerResponseParseError, match=re.escape("Unexpected top-level keys: reason")):
        PlayerResponseParser(load_context_suite()).parse(
            _main_phase_context(), ModelResponse(content='{"tool":"end_turn","arguments":{},"reason":"tired"}'),
        )
