"""Generic transport errors and HTTP 403s checkpoint safe diagnostics."""
import json
import pickle
from typing import Any

import httpx
import pytest
from flask import Flask

from cle.game_engine.models.player import Color
from cle.harness.models import ModelRequest
from cle.harness.providers import (
    OpenRouterConfig,
    OpenRouterTransport,
)
from cle.sandbox import CatanSandbox
from cle.sandbox.replay import ReplaySandbox
from playground.game_viewer.async_runtime import sandbox_async_runtime
from playground.game_viewer.routes.health import health_bp
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
    "error",
    [
        RuntimeError("Callback failed with SSLV3_ALERT_BAD_RECORD_MAC in its message"),
        httpx.ReadError("An unrelated provider read failed"),
        httpx.HTTPStatusError(
            "An unrelated provider HTTP 401 failed",
            request=httpx.Request("POST", "https://openrouter.test/api/v1/chat/completions"),
            response=httpx.Response(401),
        ),
    ],
)
def test_generic_live_transport_errors_checkpoint_safe_diagnostics(
    live_app: tuple[Flask, ServerState, DummySocket],
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    app, state, socket = live_app
    transport = FixedTransport()

    async def complete(request: ModelRequest) -> None:
        transport.requests.append(request)
        raise error

    monkeypatch.setattr(transport, "complete", complete)
    monkeypatch.setattr("cle.sandbox.factory.create_text_transport", lambda config: transport)
    client: Any = app.test_client()
    started: Any = client.post(
        "/api/start-game",
        json={
            "mode": "llm_vs_random", "seed": 5,
            "shuffle_players": False, "palette": "canonical_four",
        },
    )
    assert started.status_code == 200, started.get_json()

    failed = client.post("/api/step")

    assert failed.status_code == 500
    payload: Any = failed.get_json()
    assert payload["error"] == "Sandbox step failed"
    assert payload["details"] == (
        f"{type(error).__name__}. No gameplay action was applied. "
        "Inspect the failure before retrying."
    )
    assert payload["player"] == "RED"
    assert payload["action_applied"] is False
    assert payload["retryable"] is False
    assert payload["checkpoint_saved"] is True
    assert payload["trace_game_id"] == started.json["trace_game_id"]
    assert state.last_live_step_error == payload
    assert len(socket.emissions) == 1
    assert socket.emissions[0][0] == "game_state"
    assert socket.emissions[0][1]["last_live_step_error"] == payload
    assert state.step_processing is False
    assert live_sandbox(state).revision == 0
    assert live_sandbox(state).game_engine.state.actions == []
    assert len(transport.requests) == 1
    stored: Any = client.get(f"/api/live-traces/{started.json['trace_game_id']}").json
    assert len(stored["failures"]) == 1
    failure = stored["failures"][0]
    assert failure["failure_id"] == payload["trace_failure_id"]
    assert failure["actor"] == "RED"
    assert failure["revision"] == 0
    assert failure["validation_error"] == payload["details"]
    assert failure["attempts"] == []
    assert failure["communication_attempts"] == []
    assert str(error) not in json.dumps([payload, socket.emissions, stored])
    assert stored["step_count"] == 0
    assert stored["model_calls"] == []
    resume: Any = state.live_trace_store.load_resume_point(started.json["trace_game_id"])
    assert resume.snapshot.player_states == live_sandbox(state).snapshot().player_states
    assert resume.snapshot.pending_decision_revision is None
    assert resume.public_state["last_live_step_error"]["details"] == payload["details"]


@pytest.mark.parametrize("include_details", [True, False])
def test_live_openrouter_http403_retains_safe_rejection_without_retries(
    live_app: tuple[Flask, ServerState, DummySocket], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, include_details: bool
) -> None:
    app, state, socket = live_app
    app.register_blueprint(health_bp)
    http_requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        http_requests.append(request)
        return httpx.Response(
            403,
            json={"error": {
                "code": 403,
                **({"message": "Provider policy rejected this model. local-test-key"}
                   if include_details else {}),
                "metadata": {"raw": "private-upstream-response"},
            }},
            headers={"x-request-id": "req-http403-route"} if include_details else {},
            request=request,
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport = OpenRouterTransport(
        OpenRouterConfig(
            model="test/model", max_retries=3,
            endpoint="https://openrouter.test/api/v1/chat/completions",
        ),
        api_key="local-test-key",
        client=http_client,
    )
    monkeypatch.setattr("cle.sandbox.factory.create_text_transport", lambda config: transport)
    client: Any = app.test_client()
    try:
        started: Any = client.post(
            "/api/start-game",
            json={
                "mode": "llm_vs_random", "seed": 5,
                "shuffle_players": False, "palette": "canonical_four",
                "max_decision_attempts": 3,
            },
        )
        assert started.status_code == 200, started.get_json()
        game_id: Any = started.json["trace_game_id"]
        sandbox: Any = state.current_sandbox
        red: Any = sandbox.players[Color.RED]
        context_id = sandbox.decision_context().context_id
        before_engine: Any = pickle.dumps(sandbox.game_engine.snapshot())
        before_session: Any = red.session.snapshot()

        failed = client.post("/api/step")

        assert failed.status_code == 502, failed.get_json()
        payload: Any = failed.get_json()
        assert payload["error"] == "OpenRouter rejected the request"
        assert payload["provider_status_code"] == 403
        assert "403" in payload["details"]
        assert payload["player"] == "RED"
        assert payload["context_id"] == context_id
        assert payload["transport_attempt_count"] == len(http_requests) == 1
        assert payload["retryable"] is False
        assert payload["action_applied"] is False
        assert "No gameplay action was applied." in payload["details"]
        assert "Resolve the provider rejection before retrying." in payload["details"]
        assert "Press Step to retry" not in payload["details"]
        assert "attempts" not in payload
        assert "final_response" not in payload
        assert "native_reasoning" not in payload
        assert pickle.dumps(sandbox.game_engine.snapshot()) == before_engine
        assert red.session.snapshot() == before_session
        assert sandbox.revision == 0
        assert sandbox.game_engine.state.actions == []
        assert sandbox._step_state is None
        assert state.step_processing is False
        assert state.last_live_step_error == payload
        assert client.get("/api/state").json["last_live_step_error"] == payload
        assert len(socket.emissions) == 1
        event, emitted = socket.emissions[0]
        assert event == "game_state"
        assert emitted["last_live_step_error"] == payload
        assert emitted["running"] is True
        assert emitted["events"] == []
        stored: Any = client.get(f"/api/live-traces/{game_id}").json
        assert stored["step_count"] == 0
        assert stored["model_calls"] == []
        assert len(stored["failures"]) == 1
        failure = stored["failures"][0]
        assert failure["failure_id"] == payload["trace_failure_id"]
        assert failure["actor"] == "RED"
        assert failure["revision"] == 0
        assert failure["attempts"] == []
        assert failure["communication_attempts"] == []
        assert failure["validation_error"] in payload["details"]
        assert "403" in failure["validation_error"]
        if include_details:
            assert payload["provider_request_id"] == "req-http403-route"
            assert "req-http403-route" in failure["validation_error"]
            assert "Provider policy rejected this model." in failure["validation_error"]
        else:
            assert payload.get("provider_request_id") is None
        assert state.live_trace_store.load_resume_point(game_id).step_index is None
        assert len(http_requests) == 1
        diagnostics = json.dumps([payload, emitted, stored]) + caplog.text
        assert "local-test-key" not in diagnostics
        assert "private-upstream-response" not in diagnostics
        assert all(record.exc_info is None for record in caplog.records)
    finally:
        sandbox_async_runtime.run(http_client.aclose())
