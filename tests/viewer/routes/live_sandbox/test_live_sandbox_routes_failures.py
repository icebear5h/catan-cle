"""Off-turn and post-action failures keep distinct, checkpointed diagnostics."""
import json
from dataclasses import replace
from typing import Any

import pytest
from flask import Flask

from cle.game_engine.events import PlayerEvent
from cle.game_engine.models.player import Color
from cle.harness import ModelResponse
from cle.players.contracts import (
    CommunicationChoice,
    CommunicationMode,
    PlayerAttempt,
    PlayerChoice,
    TalkContext,
)
from cle.sandbox import CatanSandbox
from cle.sandbox.catan import PlayerResponseError
from cle.sandbox.communication import (
    CommunicationAdmission,
    CommunicationOpportunity,
    ReactionReason,
)
from cle.sandbox.replay import ReplaySandbox
from playground.game_viewer.state import ServerState

from .conftest import (
    DummySocket,
    FixedTransport,
)


def live_sandbox(state: ServerState) -> CatanSandbox | ReplaySandbox:
    """The viewer's current sandbox, which every caller here has already created."""
    sandbox = state.current_sandbox
    assert sandbox is not None
    return sandbox



@pytest.mark.parametrize(
    ("final_output", "native_reasoning", "native_details"),
    [
        ("<game_plan>final output</game_plan>" * 200, "native analysis", ()),
        ("", "native analysis without a final answer", ()),
        ("<action>1</action>", "", ({"type": "reasoning.text", "text": "native detail"},)),
    ],
)
def test_off_turn_failure_preserves_distinct_diagnostics_over_http_and_ws(
    live_app: tuple[Flask, ServerState, DummySocket],
    monkeypatch: pytest.MonkeyPatch,
    final_output: str,
    native_reasoning: str,
    native_details: tuple[dict[str, str], ...],
) -> None:
    app, state, socket = live_app
    client: Any = app.test_client()
    started: Any = client.post(
        "/api/start-game",
        json={"mode": "random", "seed": 5, "shuffle_players": False, "palette": "canonical_four"},
    )
    assert started.status_code == 200
    sandbox: Any = state.current_sandbox
    assert sandbox.current_actor() == Color.RED
    before_revision: Any = sandbox.revision
    original_step = sandbox.step
    validation_error: Any = "The selected action parameters are no longer legal or affordable."
    attempt = PlayerAttempt(
        context_id="off-turn-blue",
        choice=PlayerChoice(action_index=1),
        validation_error=validation_error,
        model_response=ModelResponse(
            content=final_output,
            native_reasoning=native_reasoning,
            native_reasoning_details=native_details,
            reasoning_request=(("effort", "high"), ("exclude", False)),
            usage=(("completion_tokens", 123),),
            finish_reason="stop",
            provider_request_payload={"headers": {"Authorization": "secret-request-header"}},
            provider_response_payload={"internal": "unexposed-provider-payload"},
        ),
    )
    earlier_attempt = replace(attempt, context_id="off-turn-white")
    sandbox.decision_trace.append(replace(attempt, context_id="previous-failure"))
    opportunity = CommunicationOpportunity(
        player=Color.BLUE,
        cause=PlayerEvent(0, "pre-action", Color.BLUE, "PRE_ACTION", None),
        visible_through_sequence=0,
        reason=ReactionReason.PRE_ACTION,
        round=0,
    )
    sandbox.communication_trace.append(
        CommunicationAdmission(opportunity, CommunicationChoice(), accepted=True)
    )

    async def reject_off_turn() -> None:
        sandbox.decision_trace.extend((earlier_attempt, attempt))
        sandbox.communication_trace.append(
            CommunicationAdmission(replace(opportunity, round=1), CommunicationChoice(), accepted=True)
        )
        raise PlayerResponseError(Color.BLUE, (attempt,), validation_error)

    monkeypatch.setattr(sandbox, "step", reject_off_turn)
    failed = client.post("/api/step")

    assert failed.status_code == 422
    payload: Any = failed.get_json()
    assert payload["player"] == "BLUE"
    assert payload["trace_game_id"] == started.json["trace_game_id"]
    assert payload["attempt_count"] == 1
    assert payload["retryable"] is True
    diagnostic = payload["attempts"][0]
    assert diagnostic["context_id"] == "off-turn-blue"
    assert diagnostic["action_index"] == 1
    assert diagnostic["validation_error"] == validation_error
    assert diagnostic["final_response"] == final_output
    assert diagnostic["native_reasoning"] == native_reasoning
    assert diagnostic["native_reasoning_details"] == list(native_details)
    assert diagnostic["native_reasoning_chars"] == len(native_reasoning)
    assert diagnostic["reasoning_request"] == {"effort": "high", "exclude": False}
    assert diagnostic["usage"] == {"completion_tokens": 123}
    assert "secret-request-header" not in failed.get_data(as_text=True)
    assert "unexposed-provider-payload" not in failed.get_data(as_text=True)
    assert "provider_request_payload" not in diagnostic
    assert "provider_response_payload" not in diagnostic
    assert socket.emissions[-1][0] == "game_state"
    assert socket.emissions[-1][1]["last_live_step_error"] == payload
    assert state.last_live_step_error == payload
    assert sandbox.revision == before_revision
    assert state.step_processing is False
    trace: Any = client.get(f"/api/live-traces/{started.json['trace_game_id']}").json
    assert trace["step_count"] == 0
    assert trace["model_calls"] == []
    assert len(trace["failures"]) == 1
    failure = trace["failures"][0]
    assert failure["failure_id"] == payload["trace_failure_id"]
    assert failure["revision"] == before_revision
    assert failure["actor"] == "BLUE"
    assert failure["validation_error"] == validation_error
    assert [item["context_id"] for item in failure["attempts"]] == [
        "off-turn-white", "off-turn-blue"
    ]
    assert all(item["accepted"] is False for item in failure["attempts"])
    assert failure["attempts"][-1]["model_response"]["native_reasoning"] == native_reasoning
    assert len(failure["communication_attempts"]) == 1
    assert failure["communication_attempts"][0]["opportunity"]["round"] == 1
    assert state.live_trace_store.load_resume_point(started.json["trace_game_id"]).step_index is None

    monkeypatch.setattr(sandbox, "step", original_step)
    continued = client.post("/api/step")
    assert continued.status_code == 200
    assert continued.json["state"]["last_live_step_error"] is None
    assert state.last_live_step_error is None
    trace = client.get(f"/api/live-traces/{started.json['trace_game_id']}").json
    assert trace["step_count"] == 1
    assert len(trace["failures"]) == 1


@pytest.mark.parametrize("partial_message", [False, True])
def test_post_action_communication_failure_checkpoints_applied_result(
    live_app: tuple[Flask, ServerState, DummySocket], monkeypatch: pytest.MonkeyPatch, partial_message: bool
) -> None:
    app, state, socket = live_app
    response: Any = ModelResponse(
        content=json.dumps({
            "game_plan": "opening plan " * 400,
            "tool": "build_settlement",
            "arguments": {"node": "<N00>"},
        }),
        model="test/model",
        native_reasoning="native accepted analysis " * 250,
        native_reasoning_details=({"type": "reasoning.text", "text": "native detail"},),
        reasoning_request=(("effort", "high"), ("exclude", False)),
    )
    transport = FixedTransport(response=response)
    monkeypatch.setattr("cle.sandbox.factory.create_text_transport", lambda config: transport)
    client = app.test_client()
    started: Any = client.post(
        "/api/start-game",
        json={
            "mode": "llm_vs_random", "seed": 5,
            "shuffle_players": False, "palette": "canonical_four",
        },
    )
    assert started.status_code == 200
    game_id = started.json["trace_game_id"]
    sandbox: Any = state.current_sandbox
    before_revision = sandbox.revision
    red: Any = sandbox.players[Color.RED]

    async def blue_speech(context: TalkContext) -> CommunicationChoice:
        assert sandbox.revision == before_revision + 1
        assert len(red.session.receipts) == 1
        if partial_message:
            return CommunicationChoice(
                mode=CommunicationMode.SAY,
                text="One message was emitted before speech failed.",
                audience=tuple(
                    color for color in sandbox.game_engine.state.colors if color != Color.BLUE
                ),
            )
        raise RuntimeError("private speech transport failure details")

    async def invalid_white_speech(context: TalkContext) -> CommunicationChoice:
        return CommunicationChoice(
            mode=CommunicationMode.SAY, text="Invalid recipient", audience=(Color.BLACK,)
        )

    monkeypatch.setattr(sandbox.players[Color.BLUE], "communicate", blue_speech)
    if partial_message:
        monkeypatch.setattr(sandbox.players[Color.WHITE], "communicate", invalid_white_speech)
    stepped = client.post("/api/step")

    assert stepped.status_code == 200, stepped.get_json()
    payload = stepped.get_json()
    warning = payload["warning"]
    assert payload["status"] == "ok"
    assert payload["trace_step_index"] == 0
    assert payload["state"]["running"] is True
    assert payload["state"]["last_live_step_error"] == warning
    assert warning["action_applied"] is True
    assert warning["retryable"] is False
    assert warning["details"].startswith(
        "Game action was applied, but post-action communication failed."
    )
    assert "Do not retry the applied action" in warning["details"]
    assert "the next Step advances the game" in warning["details"]
    assert "private speech transport failure details" not in warning["details"]
    assert "attempts" not in warning
    assert "trace_failure_id" not in warning
    assert sandbox.revision == before_revision + 1 + int(partial_message)
    assert sandbox.game_engine.events[0].event_type == "BUILD_SETTLEMENT"
    assert len(red.session.receipts) == 1
    assert red.session.messages[-1].content == response.content
    assert len(transport.requests) == 1
    assert payload["reasoning_traces"][0]["native_reasoning"] == response.native_reasoning
    assert payload["reasoning_traces"][0]["native_reasoning_details"] == list(
        response.native_reasoning_details
    )
    assert any(entry["type"] == "building" for entry in payload["state"]["game_log"])
    # Speech rows are labelled with the game step, not the engine-event number.
    spoken: Any = [entry for entry in payload["state"]["game_log"] if entry["type"] == "message"]
    assert len(spoken) == int(partial_message)
    for entry in spoken:
        assert entry["step_index"] == payload["trace_step_index"] == 0
        assert entry["details"]["step_index"] == 0
        assert entry["details"]["sequence"] == 1
    assert socket.emissions == [("game_state", payload["state"])]
    assert state.last_live_step_error == warning
    assert state.step_processing is False

    stored: Any = client.get(f"/api/live-traces/{game_id}").json
    assert stored["step_count"] == 1
    assert stored["failures"] == []
    assert len(stored["model_calls"]) == 1
    call = stored["model_calls"][0]
    assert call["accepted"] is True
    assert call["response"]["content"] == response.content
    assert call["response"]["native_reasoning"] == response.native_reasoning
    assert stored["steps"][0]["after_revision"] == before_revision + 1
    assert len(stored["steps"][0]["result"]["message_events"]) == int(partial_message)
    assert [event["event_type"] for event in stored["events"]] == (
        ["BUILD_SETTLEMENT", "MESSAGE_SENT"]
        if partial_message else ["BUILD_SETTLEMENT"]
    )
    if partial_message:
        assert stored["events"][-1]["event"] == (
            stored["steps"][0]["result"]["message_events"][0]
        )
    checkpoint: Any = client.get(f"/api/live-traces/{game_id}/steps/0").json
    assert checkpoint["step"]["public_state"] == payload["state"]
    assert [
        entry["step_index"]
        for entry in checkpoint["step"]["public_state"]["game_log"]
        if entry["type"] == "message"
    ] == [0] * int(partial_message)
    assert checkpoint["model_calls"] == stored["model_calls"]

    loaded: Any = client.post(f"/api/live-traces/{game_id}/load")
    assert loaded.status_code == 200
    # Speech survives the round trip through the checkpoint.
    assert [
        entry for entry in loaded.json["state"]["game_log"]
        if entry["type"] == "message"
    ] == spoken
    assert live_sandbox(state).revision == sandbox.revision
    assert len(live_sandbox(state).players[Color.RED].session.receipts) == 1
    assert len(transport.requests) == 1
    transport.response = None
    continued: Any = client.post("/api/step")
    assert continued.status_code == 200
    assert continued.json["warning"] is None
    assert continued.json["trace_step_index"] == 1
    assert live_sandbox(state).game_engine.events[-1].event_type == "BUILD_ROAD"
    assert len(live_sandbox(state).players[Color.RED].session.receipts) == 2
