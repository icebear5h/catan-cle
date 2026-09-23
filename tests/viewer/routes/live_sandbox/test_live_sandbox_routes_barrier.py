"""Barrier failures use the registered session, not the turn actor."""
import asyncio
import json
import pickle
from typing import Any

import pytest
from flask import Flask

from cle.game_engine.models.player import Color
from cle.game_engine.state import ensure_trade_window
from cle.game_engine.trading import TradeOffer
from cle.harness import ModelResponse
from cle.harness.models import ModelRequest
from cle.harness.providers.openrouter import OpenRouterHTTPFailure
from playground.game_viewer.routes.health import health_bp
from playground.game_viewer.state import ServerState

from .conftest import (
    DummySocket,
    FixedTransport,
    OpenRouterFailureFactory,
)


def test_openrouter_barrier_failure_uses_registered_session_not_turn_actor(
    live_app: tuple[Flask, ServerState, DummySocket], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    openrouter_failure: OpenRouterFailureFactory,
) -> None:
    app, state, socket = live_app
    app.register_blueprint(health_bp)
    transport = FixedTransport()
    monkeypatch.setattr("cle.sandbox.factory.create_text_transport", lambda config: transport)
    client: Any = app.test_client()
    started: Any = client.post(
        "/api/start-game",
        json={
            "mode": "llm", "seed": 5,
            "shuffle_players": False, "palette": "canonical_four",
        },
    )
    assert started.status_code == 200, started.get_json()
    game_id = started.json["trace_game_id"]
    sandbox: Any = state.current_sandbox
    sandbox.players[Color.BLUE].session.session_id = "opaque-session-without-a-color"
    offer = ensure_trade_window(sandbox.game_engine.state).create_offer(
        TradeOffer(
            offered_by=Color.RED,
            audience=frozenset({Color.BLUE, Color.WHITE, Color.ORANGE}),
            give=(1, 0, 0, 0, 0), receive=(0, 1, 0, 0, 0),
        )
    )
    assert sandbox.current_actor() == Color.RED
    before_revision: Any = sandbox.revision
    before_engine: Any = pickle.dumps(sandbox.game_engine.snapshot())
    before_sessions: Any = {color: player.session.snapshot() for color, player in sandbox.players.items()}
    siblings_ready = asyncio.Event()
    responses = {}
    failures = []

    async def complete(request: ModelRequest) -> ModelResponse:
        transport.requests.append(request)
        color: Any = next(
            color for color, player in sandbox.players.items()
            if player.status()["session_id"] == request.session_id
        )
        if color == Color.BLUE:
            await asyncio.wait_for(siblings_ready.wait(), timeout=2)
            failure = openrouter_failure(request)
            failures.append(failure)
            raise failure from RuntimeError("private-barrier-provider-secret")
        assert color in {Color.WHITE, Color.ORANGE}
        response = ModelResponse(
            content=json.dumps({
                "game_plan": "decline this offer", "tool": "reject_offer",
                "arguments": {"offer_id": offer.id},
            }),
            model="test/model", provider_response_id=f"completed-{color.value}",
        )
        responses[color] = response
        if len(responses) == 2:
            siblings_ready.set()
        return response

    monkeypatch.setattr(transport, "complete", complete)
    failed = client.post("/api/step")

    assert len(failures) == 1
    assert len(sandbox.decision_trace) == 2
    assert all(attempt.choice is not None for attempt in sandbox.decision_trace)
    assert failed.status_code == 502, failed.get_json()
    payload: Any = failed.get_json()
    failure = failures[0]
    http_rejection = isinstance(failure, OpenRouterHTTPFailure)
    if http_rejection:
        assert payload["error"] == "OpenRouter rejected the request"
        assert payload["provider_status_code"] == 403
        assert payload["provider_request_id"] == "req-rejected-test"
        assert "Resolve the provider rejection before retrying." in payload["details"]
        assert "Press Step to retry" not in payload["details"]
    else:
        assert payload["error"] == "OpenRouter connection failed"
        assert "No gameplay action was applied. Press Step to retry." in payload["details"]
    assert str(failure) in payload["details"]
    assert payload["player"] == "BLUE"
    assert payload["context_id"] == failure.context_id
    assert payload["transport_attempt_count"] == failure.attempts
    assert payload["retryable"] is (not http_rejection)
    assert payload["action_applied"] is False
    assert "No gameplay action was applied." in payload["details"]
    assert state.last_live_step_error == payload
    assert client.get("/api/state").json["last_live_step_error"] == payload
    assert len(socket.emissions) == 1
    assert socket.emissions[-1][1]["last_live_step_error"] == payload
    assert len(transport.requests) == 3
    assert sandbox.revision == before_revision
    assert sandbox.game_engine.state.actions == []
    assert offer.declined_by == set()
    assert pickle.dumps(sandbox.game_engine.snapshot()) == before_engine
    assert {color: player.session.snapshot() for color, player in sandbox.players.items()} == (
        before_sessions
    )
    assert all(player.session.receipts == {} for player in sandbox.players.values())
    assert all(player.session.messages == [] for player in sandbox.players.values())
    assert state.step_processing is False
    assert sandbox._step_state is None
    stored: Any = client.get(f"/api/live-traces/{game_id}").json
    assert stored["step_count"] == 0
    assert stored["model_calls"] == []
    assert len(stored["failures"]) == 1
    record = stored["failures"][0]
    assert record["failure_id"] == payload["trace_failure_id"]
    assert record["actor"] == "BLUE"
    assert record["revision"] == before_revision
    assert record["validation_error"] == str(failures[0])
    assert record["communication_attempts"] == []
    assert len(record["attempts"]) == 2
    assert [attempt["model_response"]["provider_response_id"] for attempt in record["attempts"]] == [
        "completed-WHITE", "completed-ORANGE",
    ]
    assert all(attempt["accepted"] is False for attempt in record["attempts"])
    assert all("withheld" in attempt["validation_error"].lower() for attempt in record["attempts"])
    assert all(attempt["context_id"] != payload["context_id"] for attempt in record["attempts"])
    assert len(payload["attempts"]) == 2
    assert [attempt["provider_response_id"] for attempt in payload["attempts"]] == [
        "completed-WHITE", "completed-ORANGE",
    ]
    assert all("withheld" in attempt["validation_error"].lower() for attempt in payload["attempts"])
    assert all(str(failure) in attempt["validation_error"] for attempt in record["attempts"])
    diagnostics = json.dumps([payload, socket.emissions, stored]) + caplog.text
    assert "private-barrier-provider-secret" not in diagnostics
    assert "private-provider-secret" not in diagnostics
    assert all(record.exc_info is None for record in caplog.records)
