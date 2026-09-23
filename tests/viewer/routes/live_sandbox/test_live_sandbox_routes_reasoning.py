"""Reasoning-only replies are rejected as illegal until a manual step."""
import json
import pickle
from typing import Any

import httpx
import pytest
from flask import Flask

from cle.game_engine.models.player import Color
from cle.harness.providers import (
    OpenRouterConfig,
    OpenRouterTransport,
)
from playground.game_viewer.async_runtime import sandbox_async_runtime
from playground.game_viewer.routes.health import health_bp
from playground.game_viewer.state import ServerState

from .conftest import (
    DummySocket,
    _opening_call,
)


def test_live_openrouter_reasoning_only_rejects_legal_json_until_manual_step(live_app: tuple[Flask, ServerState, DummySocket], monkeypatch: pytest.MonkeyPatch) -> None:
    app, state, socket = live_app
    app.register_blueprint(health_bp)
    http_requests = []
    final_answer = False

    def handler(request: httpx.Request) -> httpx.Response:
        http_requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "gen-reka-final-answer" if final_answer else "gen-reka-reasoning-only",
                "model": "reka/test-model",
                "choices": [{
                    "message": {
                        "role": "assistant", "content": output if final_answer else None,
                        "reasoning": output,
                    },
                    "finish_reason": "stop", "native_finish_reason": "stop",
                }],
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport = OpenRouterTransport(
        OpenRouterConfig(
            model="reka/test-model", max_tokens=None,
            reasoning={"effort": "high", "exclude": False},
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
                "mode": "llm_vs_random", "model": "reka/test-model", "seed": 5,
                "shuffle_players": False, "palette": "canonical_four",
                "max_decision_attempts": 1, "max_tokens": None,
            },
        )
        assert started.status_code == 200, started.get_json()
        assert started.json["max_decision_attempts"] == 1
        assert started.json["max_tokens"] is None
        game_id: Any = started.json["trace_game_id"]
        sandbox: Any = state.current_sandbox
        red: Any = sandbox.players[Color.RED]
        assert red.suite.response.format == "json"
        context: Any = sandbox.decision_context()
        output: Any = _opening_call(
            red._assembler.assemble(context, red.session), "build a legal opening"
        )
        before_revision: Any = sandbox.revision
        before_engine: Any = pickle.dumps(sandbox.game_engine.snapshot())
        before_session: Any = red.session.snapshot()

        failed = client.post("/api/step")

        assert failed.status_code == 422, failed.get_json()
        payload: Any = failed.get_json()
        # Retry feedback names the exact legal menu after a failed decision.
        diagnostic = (
            "The provider returned reasoning but no final answer. "
            "Your currently legal tools: build_settlement."
        )
        assert payload["error"] == "Model returned no valid action"
        assert payload["details"] == (
            f"{diagnostic} No final model output was returned. "
            "No gameplay action was applied. Press Step to ask the model again."
        )
        assert "JSONDecodeError" not in failed.get_data(as_text=True)
        assert payload["attempt_count"] == len(payload["attempts"]) == len(http_requests) == 1
        assert "max_tokens" not in json.loads(http_requests[0].content)
        assert payload["player"] == "RED"
        assert payload["retryable"] is True
        assert payload["trace_game_id"] == game_id
        attempt: Any = payload["attempts"][0]
        assert attempt["context_id"] == context.context_id
        assert attempt["action_index"] is None
        assert attempt["validation_error"] == diagnostic
        assert attempt["final_response"] == ""
        assert attempt["native_reasoning"] == output
        assert attempt["finish_reason"] == attempt["provider_native_finish_reason"] == "stop"
        assert attempt["provider_response_id"] == "gen-reka-reasoning-only"
        assert sandbox.revision == before_revision
        assert pickle.dumps(sandbox.game_engine.snapshot()) == before_engine
        assert red.session.snapshot() == before_session
        assert red.session.messages == []
        assert red.session.receipts == {}
        assert state.step_processing is False
        assert sandbox._step_state is None
        assert state.last_live_step_error == payload
        retained: Any = client.get("/api/state")
        assert retained.status_code == 200
        assert retained.json["last_live_step_error"] == payload
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
        failure: Any = stored["failures"][0]
        assert failure["failure_id"] == payload["trace_failure_id"]
        assert failure["revision"] == before_revision
        assert failure["actor"] == "RED"
        assert failure["validation_error"] == diagnostic
        assert len(failure["attempts"]) == 1
        saved_attempt = failure["attempts"][0]
        assert saved_attempt["accepted"] is False
        assert saved_attempt["choice"] is None
        response = saved_attempt["model_response"]
        assert response["content"] == ""
        assert response["native_reasoning"] == output
        assert response["finish_reason"] == response["provider_native_finish_reason"] == "stop"
        assert response["provider_response_id"] == "gen-reka-reasoning-only"
        provider_payload = response["provider_response_payload"]
        assert provider_payload["id"] == response["provider_response_id"]
        provider_choice = provider_payload["choices"][0]
        assert provider_choice["finish_reason"] == provider_choice["native_finish_reason"] == "stop"
        assert provider_choice["message"]["content"] is None
        assert provider_choice["message"]["reasoning"] == output
        assert "tool_calls" not in provider_choice["message"]
        assert "max_tokens" not in response["provider_request_payload"]
        assert state.live_trace_store.load_resume_point(game_id).step_index is None

        # Only a new manual Step may admit the same legal JSON as a final answer.
        final_answer = True
        continued: Any = client.post("/api/step")
        assert continued.status_code == 200, continued.get_json()
        assert len(http_requests) == 2
        assert continued.json["warning"] is None
        assert continued.json["trace_step_index"] == 0
        assert continued.json["state"]["last_live_step_error"] is None
        assert client.get("/api/state").json["last_live_step_error"] is None
        assert state.last_live_step_error is None
        assert sandbox.revision == before_revision + 1
        assert len(sandbox.game_engine.state.actions) == 1
        assert len(sandbox.game_engine.events) == 1
        assert sandbox.game_engine.events[0].event_type == "BUILD_SETTLEMENT"
        assert len(red.session.receipts) == 1
        assert len(red.session.messages) == 2
        assert red.session.messages[-1].content == output
        # Receipts keep the decision only; the raw output is on the stored call below.
        assert next(iter(red.session.receipts.values())).choice.raw_response == ""
        stored = client.get(f"/api/live-traces/{game_id}").json
        assert stored["step_count"] == len(stored["model_calls"]) == 1
        assert stored["model_calls"][0]["accepted"] is True
        assert stored["model_calls"][0]["response"]["content"] == output
        assert stored["failures"] == [failure]
    finally:
        sandbox_async_runtime.run(http_client.aclose())
