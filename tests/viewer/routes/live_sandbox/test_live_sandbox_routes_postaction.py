"""Post-action OpenRouter failures warn once and never offer a retry."""
import json
from typing import Any

import pytest
from flask import Flask

from cle.game_engine.models.player import Color
from cle.harness import ModelResponse
from cle.harness.models import ModelRequest
from cle.harness.providers.openrouter import OpenRouterHTTPFailure
from cle.players.contracts import AcceptanceResult, PlayerAttempt
from cle.sandbox import CatanSandbox
from cle.sandbox.replay import ReplaySandbox
from playground.game_viewer.routes.health import health_bp
from playground.game_viewer.state import ServerState

from .conftest import (
    DummySocket,
    FixedTransport,
    OpenRouterFailure,
    OpenRouterFailureFactory,
)


def live_sandbox(state: ServerState) -> CatanSandbox | ReplaySandbox:
    """The viewer's current sandbox, which every caller here has already created."""
    sandbox = state.current_sandbox
    assert sandbox is not None
    return sandbox



def test_post_action_openrouter_failure_is_warning_and_checkpoints_once(
    live_app: tuple[Flask, ServerState, DummySocket], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    openrouter_failure: OpenRouterFailureFactory,
) -> None:
    app, state, socket = live_app
    app.register_blueprint(health_bp)
    transport = FixedTransport()
    decision_complete = transport.complete
    speech_failures: Any = []
    fail_speech = True

    async def complete(request: ModelRequest) -> ModelResponse:
        if any(component.id == "environment.legal_actions" for component in request.components):
            return await decision_complete(request)
        transport.requests.append(request)
        blue_session = live_sandbox(state).players[Color.BLUE].status()["session_id"]
        if fail_speech and request.session_id == blue_session:
            failure = openrouter_failure(request)
            speech_failures.append(failure)
            raise failure from RuntimeError("private-post-action-provider-secret")
        return ModelResponse(content="<message>SILENCE</message>", model="test/model")

    monkeypatch.setattr(transport, "complete", complete)
    monkeypatch.setattr("cle.sandbox.factory.create_text_transport", lambda config: transport)
    client: Any = app.test_client()
    started = client.post(
        "/api/start-game",
        json={
            "mode": "llm", "seed": 5,
            "shuffle_players": False, "palette": "canonical_four",
        },
    )
    assert started.status_code == 200, started.get_json()
    game_id = started.json["trace_game_id"]
    sandbox: Any = state.current_sandbox
    stepped = client.post("/api/step")

    assert stepped.status_code == 200, stepped.get_json()
    payload: Any = stepped.get_json()
    warning: Any = payload["warning"]
    assert len(speech_failures) == 1
    failure = speech_failures[0]
    assert len([
        request for request in transport.requests if request.session_id == failure.session_id
    ]) == 1
    if isinstance(failure, OpenRouterHTTPFailure):
        assert warning["details"].count(str(failure)) == 1
        assert warning["provider_status_code"] == 403
        assert warning["provider_request_id"] == "req-rejected-test"
        assert "Provider policy rejected this model." in warning["details"]
        assert "Resolve the provider rejection before the next Step." in warning["details"]
        assert "Press Step to retry" not in warning["details"]
        assert "private-post-action-provider-secret" not in caplog.text
        assert "private-provider-secret" not in caplog.text
        assert all(record.exc_info is None for record in caplog.records)
    else:
        assert warning["details"] == (
            "Game action was applied, but post-action communication failed. "
            "Auto-play stopped. Do not retry the applied action; "
            "the next Step advances the game."
        )
    assert warning["action_applied"] is True
    assert warning["retryable"] is False
    assert warning["details"].startswith(
        "Game action was applied, but post-action communication failed."
    )
    assert "Do not retry the applied action" in warning["details"]
    assert "No gameplay action was applied" not in warning["details"]
    assert "attempts" not in warning
    assert "trace_failure_id" not in warning
    assert payload["trace_step_index"] == 0
    assert payload["state"]["last_live_step_error"] == warning
    assert state.last_live_step_error == warning
    assert client.get("/api/state").json["last_live_step_error"] == warning
    assert socket.emissions == [("game_state", payload["state"])]
    assert state.step_processing is False
    assert len(sandbox.game_engine.state.actions) == sandbox.revision == 1
    assert len(sandbox.players[Color.RED].session.receipts) == 1
    assert len(sandbox.players[Color.RED].session.messages) == 2
    stored: Any = client.get(f"/api/live-traces/{game_id}").json
    assert stored["step_count"] == 1
    assert stored["failures"] == []
    assert len(stored["steps"][0]["result"]["transitions"]) == 1
    assert [event["event_type"] for event in stored["events"]] == ["BUILD_SETTLEMENT"]
    decisions = [call for call in stored["model_calls"] if call["call_kind"] == "decision"]
    assert len(decisions) == 1
    assert decisions[0]["accepted"] is True
    assert decisions[0]["response"]["provider_response_id"] == "gen-live-test"
    assert all(call["context_id"] != speech_failures[0].context_id for call in stored["model_calls"])
    checkpoint: Any = client.get(f"/api/live-traces/{game_id}/steps/0").json
    assert checkpoint["step"]["public_state"] == payload["state"]
    assert "private-post-action-provider-secret" not in json.dumps([payload, stored])
    assert "private-provider-secret" not in json.dumps([payload, stored])

    calls_before_load = len(transport.requests)
    loaded = client.post(f"/api/live-traces/{game_id}/load")
    assert loaded.status_code == 200, loaded.get_json()
    assert len(transport.requests) == calls_before_load
    assert live_sandbox(state).revision == 1
    assert len(live_sandbox(state).game_engine.state.actions) == 1
    assert len(live_sandbox(state).players[Color.RED].session.receipts) == 1
    fail_speech = False
    continued: Any = client.post("/api/step")
    assert continued.status_code == 200, continued.get_json()
    assert continued.json["warning"] is None
    assert continued.json["state"]["last_live_step_error"] is None
    assert continued.json["trace_step_index"] == 1
    assert len(live_sandbox(state).game_engine.state.actions) == 2
    assert len(live_sandbox(state).players[Color.RED].session.receipts) == 2
    assert live_sandbox(state).game_engine.events[-1].event_type == "BUILD_ROAD"
    stored = client.get(f"/api/live-traces/{game_id}").json
    assert stored["step_count"] == 2
    assert len([call for call in stored["model_calls"] if call["call_kind"] == "decision"]) == 2


def test_openrouter_error_from_accept_callback_does_not_offer_action_retry(
    live_app: tuple[Flask, ServerState, DummySocket], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    openrouter_failure: OpenRouterFailureFactory,
) -> None:
    app, state, socket = live_app
    transport = FixedTransport()
    monkeypatch.setattr("cle.sandbox.factory.create_text_transport", lambda config: transport)
    client = app.test_client()
    started: Any = client.post(
        "/api/start-game",
        json={
            "mode": "llm_vs_random", "seed": 5,
            "shuffle_players": False, "palette": "canonical_four",
        },
    )
    assert started.status_code == 200, started.get_json()
    sandbox: Any = state.current_sandbox
    red: Any = sandbox.players[Color.RED]
    accept = red.accept
    failures: list[OpenRouterFailure] = []

    def fail_after_accept(attempt: PlayerAttempt, result: AcceptanceResult) -> None:
        accept(attempt, result)
        failure = openrouter_failure(attempt.model_request)
        failures.append(failure)
        raise failure from RuntimeError("private-accept-provider-secret")

    monkeypatch.setattr(red, "accept", fail_after_accept)
    failed = client.post("/api/step")

    assert failed.status_code == 502, failed.get_json()
    payload: Any = failed.get_json()
    assert len(failures) == 1
    failure = failures[0]
    if isinstance(failure, OpenRouterHTTPFailure):
        assert payload["error"] == "OpenRouter rejected the request"
        assert payload["provider_status_code"] == 403
        assert payload["provider_request_id"] == "req-rejected-test"
        assert "Resolve the provider rejection before continuing." in payload["details"]
    else:
        assert payload["error"] == "OpenRouter connection failed"
    assert str(failure) in payload["details"]
    assert "A gameplay action was applied before this error. Do not repeat it" in payload["details"]
    assert payload["transport_attempt_count"] == failure.attempts
    assert payload["action_applied"] is True
    assert payload["retryable"] is False
    assert "No gameplay action was applied" not in payload["details"]
    assert "Press Step to retry" not in payload["details"]
    assert payload["player"] == "RED"
    assert payload["context_id"] == transport.requests[0].decision_id
    assert len(sandbox.game_engine.state.actions) == sandbox.revision == 1
    assert len(red.session.receipts) == 1
    assert len(transport.requests) == 1
    assert state.last_live_step_error == payload
    assert socket.emissions[-1][1]["last_live_step_error"] == payload
    assert state.step_processing is False
    stored: Any = client.get(f"/api/live-traces/{started.json['trace_game_id']}").json
    assert len(stored["failures"]) == 1
    assert stored["failures"][0]["revision"] == sandbox.revision
    assert stored["failures"][0]["validation_error"] == payload["details"]
    assert stored["step_count"] == 0
    assert stored["model_calls"] == []
    assert "attempts" not in payload
    assert stored["failures"][0]["attempts"] == []
    diagnostics = json.dumps([payload, socket.emissions, stored]) + caplog.text
    assert "private-accept-provider-secret" not in diagnostics
    assert "private-provider-secret" not in diagnostics
    assert all(record.exc_info is None for record in caplog.records)
