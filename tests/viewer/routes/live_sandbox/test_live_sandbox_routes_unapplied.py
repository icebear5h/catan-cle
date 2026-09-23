"""OpenRouter failures preserve unapplied-step diagnostics."""
import json
import re
from dataclasses import replace
from typing import Any

import pytest
from flask import Flask

from cle.game_engine.events import PlayerEvent
from cle.game_engine.models.player import Color
from cle.harness import ModelResponse
from cle.harness.models import ModelRequest
from cle.harness.providers.openrouter import OpenRouterHTTPFailure
from cle.players.contracts import (
    CommunicationChoice,
    PlayerAttempt,
)
from cle.sandbox.communication import (
    CommunicationAdmission,
    CommunicationOpportunity,
    ReactionReason,
)
from playground.game_viewer.routes.health import health_bp
from playground.game_viewer.state import ServerState

from .conftest import (
    DummySocket,
    FixedTransport,
    OpenRouterFailureFactory,
)


@pytest.mark.parametrize(
    ("failure_phase", "storage_fails"),
    [
        ("decision", False),
        ("preaction", False),
        ("after_speech", False),
        ("after_rejection", False),
        ("decision", True),
    ],
)
def test_openrouter_failure_preserves_unapplied_step_diagnostics(
    live_app: tuple[Flask, ServerState, DummySocket], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    openrouter_failure: OpenRouterFailureFactory,
    failure_phase: str,
    storage_fails: bool,
) -> None:
    app, state, socket = live_app
    app.register_blueprint(health_bp)
    transport = FixedTransport()
    recovered_complete = transport.complete
    failures = []
    invalid_output = '{"game_plan":"retry","tool":"invalid_tool","arguments":{}}'
    speech_output = (
        "<message>Does anyone want to trade wood?</message>"
        "<intent>QUESTION</intent><audience>PUBLIC</audience>"
    )

    async def complete(request: ModelRequest) -> ModelResponse:
        transport.requests.append(request)
        is_decision = any(
            component.id == "environment.legal_actions" for component in request.components
        )
        if failure_phase == "after_speech" and not is_decision:
            return ModelResponse(content=speech_output, model="test/model")
        if failure_phase == "after_rejection" and len(transport.requests) == 1:
            return ModelResponse(content=invalid_output, model="test/model")
        failure = openrouter_failure(request)
        failures.append(failure)
        raise failure from RuntimeError("Authorization: Bearer private-provider-secret")

    monkeypatch.setattr(transport, "complete", complete)
    monkeypatch.setattr("cle.sandbox.factory.create_text_transport", lambda config: transport)
    client: Any = app.test_client()
    started: Any = client.post(
        "/api/start-game",
        json={
            "mode": "llm_vs_random", "seed": 5,
            "shuffle_players": False, "palette": "canonical_four",
            "max_decision_attempts": 2,
        },
    )
    assert started.status_code == 200, started.get_json()
    game_id: Any = started.json["trace_game_id"]
    sandbox: Any = state.current_sandbox
    red: Any = sandbox.players[Color.RED]
    before_revision: Any = sandbox.revision
    before_actions: Any = list(sandbox.game_engine.state.actions)
    opportunity: Any = CommunicationOpportunity(
        player=Color.RED,
        cause=PlayerEvent(before_revision, "pre-action-provider-test", Color.RED, "PRE_ACTION", None),
        visible_through_sequence=before_revision - 1,
        reason=ReactionReason.PRE_ACTION,
        round=0,
    )
    if failure_phase in {"preaction", "after_speech"}:
        monkeypatch.setattr(sandbox.communication_policy, "pre_action", lambda engine: (opportunity,))
    sandbox.decision_trace.append(PlayerAttempt("earlier-failure", None, "earlier rejection"))
    sandbox.communication_trace.append(
        CommunicationAdmission(replace(opportunity, round=99), CommunicationChoice(), accepted=True)
    )
    persistence_calls = []

    def fail_persistence(*args: object, **kwargs: object) -> None:
        persistence_calls.append((args, kwargs))
        raise RuntimeError("private-storage-path/live.sqlite3: private-storage-secret")

    if storage_fails:
        monkeypatch.setattr(state.live_trace_store, "record_failure", fail_persistence)

    failed = client.post("/api/step")

    assert len(failures) == 1
    assert len(transport.requests) == 1 + int(failure_phase in {"after_speech", "after_rejection"})
    assert sandbox.game_engine.state.actions == before_actions
    assert sandbox.revision == before_revision + int(failure_phase == "after_speech")
    assert failed.status_code == 502, failed.get_json()
    payload: Any = failed.get_json()
    failure = failures[0]
    http_rejection = isinstance(failure, OpenRouterHTTPFailure)
    if http_rejection:
        assert payload["error"] == "OpenRouter rejected the request"
        assert payload["provider_status_code"] == 403
        assert payload["provider_request_id"] == "req-rejected-test"
        assert "Provider policy rejected this model." in payload["details"]
        assert "Resolve the provider rejection before retrying." in payload["details"]
        assert "Press Step to retry" not in payload["details"]
    else:
        assert payload["error"] == "OpenRouter connection failed"
        assert "SSLV3_ALERT_BAD_RECORD_MAC" in payload["details"]
        if storage_fails:
            assert "Press Step to retry" not in payload["details"]
            assert "Do not retry until storage is repaired" in payload["details"]
        else:
            assert "No gameplay action was applied. Press Step to retry." in payload["details"]
        assert "provider_status_code" not in payload
    assert str(failure) in payload["details"]
    assert "No gameplay action was applied." in payload["details"]
    assert payload["player"] == "RED"
    assert payload["context_id"] == transport.requests[-1].decision_id == failure.context_id
    assert failure.session_id == red.status()["session_id"]
    assert payload["transport_attempt_count"] == failure.attempts == (1 if http_rejection else 3)
    assert payload["action_applied"] is False
    assert payload["retryable"] is (not http_rejection and not storage_fails)
    assert payload["checkpoint_saved"] is (not storage_fails)
    assert payload["trace_game_id"] == game_id
    if failure_phase == "after_rejection":
        assert len(payload["attempts"]) == 1
        assert payload["attempts"][0]["final_response"] == invalid_output
    else:
        assert "attempts" not in payload
    assert "final_response" not in payload
    assert "native_reasoning" not in payload
    assert state.last_live_step_error == payload
    assert client.get("/api/state").json["last_live_step_error"] == payload
    assert len(socket.emissions) == 1
    event, emitted = socket.emissions[0]
    assert event == "game_state"
    assert emitted["last_live_step_error"] == payload
    assert emitted["running"] is True
    assert emitted["live_trace_game_id"] == game_id
    assert [event.event_type for event in sandbox.game_engine.events] == (
        ["MESSAGE_SENT"] if failure_phase == "after_speech" else []
    )
    assert len(emitted["events"]) == int(failure_phase == "after_speech")
    assert red.session.receipts == {}
    assert red.session.messages == []
    assert red.session.strategic_memory == ""
    assert state.step_processing is False
    assert sandbox._step_state is None

    stored: Any = client.get(f"/api/live-traces/{game_id}").json
    assert stored["step_count"] == 0
    assert stored["model_calls"] == []
    assert state.live_trace_store.load_resume_point(game_id).step_index is None
    if storage_fails:
        assert payload["trace_failure_id"] is None
        assert stored["failures"] == []
        assert len(persistence_calls) == 1
        args, kwargs = persistence_calls[0]
        assert args == (game_id,)
        assert kwargs["revision"] == sandbox.revision
        assert kwargs["player"] == Color.RED
        assert kwargs["validation_error"] == str(failure)
        assert list(kwargs["attempts"]) == []
        assert list(kwargs["communication_attempts"]) == []
        persistence_notices = [
            sentence for sentence in payload["details"].split(". ")
            if re.search(r"persist|sav|stor", sentence, re.IGNORECASE)
        ]
        assert any(
            re.search(r"fail|could not|unable|not saved", sentence, re.IGNORECASE)
            for sentence in persistence_notices
        )
    else:
        assert payload["trace_failure_id"]
        assert len(stored["failures"]) == 1
        record: Any = stored["failures"][0]
        assert record["failure_id"] == payload["trace_failure_id"]
        assert record["actor"] == "RED"
        assert record["revision"] == sandbox.revision
        assert record["validation_error"] == str(failure)
        if http_rejection:
            assert "Provider policy rejected this model." in record["validation_error"]
            assert "req-rejected-test" in record["validation_error"]
        assert len(record["attempts"]) == int(failure_phase == "after_rejection")
        if failure_phase == "after_rejection":
            attempt = record["attempts"][0]
            assert attempt["context_id"] == transport.requests[0].decision_id
            assert attempt["accepted"] is False
            assert attempt["model_response"]["content"] == invalid_output
        assert len(record["communication_attempts"]) == int(failure_phase == "after_speech")
        if failure_phase == "after_speech":
            speech = record["communication_attempts"][0]
            assert speech["accepted"] is True
            assert speech["opportunity"]["round"] == 0
            assert speech["model_response"]["content"] == speech_output
    diagnostics = json.dumps([payload, emitted, stored]) + caplog.text
    assert "private-provider-secret" not in diagnostics
    assert "private-storage-secret" not in diagnostics
    assert "private-storage-path" not in diagnostics
    assert all(record.exc_info is None for record in caplog.records)

    if failure_phase == "decision" and not storage_fails:
        monkeypatch.setattr(transport, "complete", recovered_complete)
        continued: Any = client.post("/api/step")
        assert continued.status_code == 200, continued.get_json()
        assert continued.json["warning"] is None
        assert continued.json["trace_step_index"] == 0
        assert continued.json["state"]["last_live_step_error"] is None
        assert state.last_live_step_error is None
        assert len(sandbox.game_engine.state.actions) == 1
        assert len(red.session.receipts) == 1
        assert len(transport.requests) == 2
        stored = client.get(f"/api/live-traces/{game_id}").json
        assert stored["step_count"] == len(stored["model_calls"]) == 1
        assert len(stored["failures"]) == 1
