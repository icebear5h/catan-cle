"""A loaded live game surfaces its first invalid attempt without retries."""
from typing import Any

import pytest
from flask import Flask

from cle.sandbox import CatanSandbox
from cle.sandbox.factory import (
    DEFAULT_LIVE_MODEL,
)
from cle.sandbox.replay import ReplaySandbox
from playground.game_viewer.state import ServerState

from .conftest import (
    DummySocket,
    SuccessThenInvalidTransport,
)


def live_sandbox(state: ServerState) -> CatanSandbox | ReplaySandbox:
    """The viewer's current sandbox, which every caller here has already created."""
    sandbox = state.current_sandbox
    assert sandbox is not None
    return sandbox



def test_loaded_live_game_surfaces_first_invalid_model_attempt_without_retries(
    live_app: tuple[Flask, ServerState, DummySocket],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, state, socket = live_app
    transport = SuccessThenInvalidTransport()
    monkeypatch.setattr(
        "cle.sandbox.factory.create_text_transport",
        lambda config: transport,
    )
    client: Any = app.test_client()

    started: Any = client.post(
        "/api/start-game",
        json={
            "mode": "llm_vs_random",
            "seed": 23,
            "shuffle_players": False,
            "palette": "canonical_four",
            "reasoning": {"effort": "xhigh", "exclude": False},
            "max_tokens": 8_192,
            "max_decision_attempts": 3,
        },
    )
    first_step = client.post("/api/step")
    client.post(
        "/api/start-game",
        json={
            "mode": "random",
            "seed": 24,
            "palette": "canonical_four",
            "max_decision_attempts": 1,
        },
    )
    loaded: Any = client.post(
        f"/api/live-traces/{started.json['trace_game_id']}/load"
    )
    before_revision: Any = live_sandbox(state).revision

    failed_step: Any = client.post("/api/step")

    assert started.status_code == 200
    assert started.json["reasoning_request"] == {
        "effort": "xhigh",
        "exclude": False,
    }
    assert started.json["max_tokens"] == 8_192
    assert started.json["max_decision_attempts"] == 3
    assert first_step.status_code == 200
    assert loaded.status_code == 200, loaded.get_json()
    assert loaded.json["reasoning_request"] == {"enabled": False}
    assert loaded.json["max_tokens"] is None
    assert loaded.json["max_decision_attempts"] == 1
    assert len(socket.emissions) == 2
    emitted_event, emitted_state = socket.emissions[0]
    assert emitted_event == "game_state"
    assert emitted_state == loaded.json["state"]
    assert emitted_state["running"] is True
    assert emitted_state["live_trace_game_id"] == started.json["trace_game_id"]
    assert failed_step.status_code == 422
    assert failed_step.json["error"] == "Model returned no valid action"
    assert "Unknown action tool" in (
        failed_step.json["details"]
    )
    assert "No gameplay action was applied" in failed_step.json["details"]
    assert failed_step.json["attempt_count"] == 1
    assert len(failed_step.json["attempts"]) == 1
    attempt = failed_step.json["attempts"][0]
    expected_diagnostics = {
        "context_id": transport.requests[-1].decision_id,
        "action_index": None,
        "final_response": (
            '{"game_plan":"continue","tool":"invalid_tool","arguments":{}}'
        ),
        "finish_reason": "length",
        "latency_ms": 25,
        "model": "test/model",
        "native_reasoning": "private reasoning",
        "native_reasoning_details": [],
        "native_reasoning_chars": 17,
        "reasoning_request": {},
        "provider_native_finish_reason": "max_tokens",
        "provider_request_id": "req-invalid-test",
        "provider_response_id": "gen-invalid-test",
        "reasoning_tokens": 8_000,
        "usage": {
            "completion_tokens": 8_192,
            "completion_tokens_details": {"reasoning_tokens": 8_000},
        },
        # Retry feedback names the exact legal menu after a failed decision.
        "validation_error": (
            "Invalid tool call: Unknown action tool "
            "Your currently legal tools: build_road."
        ),
    }
    assert {key: attempt[key] for key in expected_diagnostics} == expected_diagnostics
    assert attempt["accepted"] is False
    assert attempt["notes_update"] is None
    assert attempt["context_policy"] is None
    assert attempt["memory_revision"] is None
    assert attempt["input_next_sequence"] is None
    assert attempt["channel"] is None
    model_request = attempt["request"]
    assert model_request["decision_id"] == transport.requests[-1].decision_id
    assert model_request["session_id"] == transport.requests[-1].session_id
    assert model_request["messages"] == [
        {"role": message.role, "content": message.content}
        for message in transport.requests[-1].messages
    ]
    assert model_request["context_policy"] is None
    assert model_request["memory_revision"] is None
    assert model_request["input_next_sequence"] is None
    assert model_request["channel"] is None
    assert failed_step.json["retryable"] is True
    assert failed_step.json["player"] == "RED"
    assert failed_step.json["trace_game_id"] == started.json["trace_game_id"]
    failure_event, failure_state = socket.emissions[1]
    assert failure_event == "game_state"
    assert failure_state["last_live_step_error"] == failed_step.json
    assert state.last_live_step_error == failed_step.json
    assert failure_state["live_inference"] == {
        "max_decision_attempts": 1,
        "max_tokens": None,
        "model": DEFAULT_LIVE_MODEL,
        "reasoning": {"enabled": False},
    }
    assert len(transport.requests) == 2
    assert live_sandbox(state).revision == before_revision
    assert state.step_processing is False
    stored: Any = client.get(f"/api/live-traces/{started.json['trace_game_id']}").json
    assert stored["step_count"] == 1
    assert stored["config"]["reasoning"] == {"effort": "xhigh", "exclude": False}
    assert stored["config"]["max_tokens"] == 8_192
    assert stored["config"]["max_decision_attempts"] == 3
    assert len(stored["failures"]) == 1
    failure = stored["failures"][0]
    assert failure["failure_id"] == failed_step.json["trace_failure_id"]
    assert failure["revision"] == before_revision
    assert failure["actor"] == "RED"
    assert failure["attempts"][0]["model_response"]["content"] == (
        failed_step.json["attempts"][0]["final_response"]
    )
