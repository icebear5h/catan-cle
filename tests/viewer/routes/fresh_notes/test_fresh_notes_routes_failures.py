"""Failed actions and speech checkpoint without leaking provider bodies."""
import asyncio
import json
from typing import Any

import pytest

from cle.game_engine.models.player import Color
from cle.game_engine.state import ensure_trade_window
from cle.game_engine.trading import TradeOffer
from cle.harness.communication import default_communication_suite_path
from cle.harness.models import ModelRequest, ModelResponse
from cle.harness.prompt_store import (
    resolve_prompt_suites,
    save_shared_prompt_override,
)
from cle.harness.shared_suite import parse_shared_prompt_suite
from cle.harness.suite import default_suite_path
from cle.sandbox import CatanSandbox
from cle.sandbox.replay import ReplaySandbox
from cle.traces import SQLiteLiveTraceStore
from playground.game_viewer.state import ServerState

from .conftest import LiveApp
from .support import (
    _action_content,
    _reach_main_action,
    _start,
)


def live_sandbox(state: ServerState) -> CatanSandbox | ReplaySandbox:
    """The viewer's current sandbox, which every caller here has already created."""
    sandbox = state.current_sandbox
    assert sandbox is not None
    return sandbox



@pytest.mark.parametrize("kind,status", [("parse", 422), ("tls", 502), ("http", 502), ("generic", 500)])
def test_pre_speech_notes_survive_failed_action_checkpoint_and_resume(live_app: LiveApp, kind: str, status: int) -> None:
    client, state, websocket, transport = live_app
    game_id = _start(client)
    _reach_main_action(client, state)
    sandbox: Any = state.current_sandbox
    actor: Any = sandbox.current_actor()
    before = sandbox.players[actor].snapshot().session
    revision: Any = sandbox.revision
    action_count: Any = len(sandbox.game_engine.state.actions)
    transport.talk_notes = "accepted pre-speech notes at failed boundary"
    transport.speech_once = True
    transport.action_failure = kind
    request_cursor = len(transport.requests)
    failed: Any = client.post("/api/step")
    assert failed.status_code == status, failed.json
    assert failed.json["checkpoint_saved"] is True
    assert failed.json["action_applied"] is False
    assert state.step_processing is False
    assert websocket.emissions[-1][1]["last_live_step_error"] == failed.json
    assert sandbox.revision == revision + 1
    assert len(sandbox.game_engine.state.actions) == action_count
    assert [request.channel for request in transport.requests[request_cursor:]] == ["action", "action"]
    session = sandbox.players[actor].snapshot().session
    assert session.strategic_memory == transport.talk_notes
    assert session.memory_revision == before.memory_revision + 1
    assert session.action_next_sequence == revision
    assert session.talk_next_sequence == before.talk_next_sequence
    expected: Any = sandbox.snapshot()
    store = SQLiteLiveTraceStore(state.live_trace_store.path)
    point: Any = store.load_resume_point(game_id)
    assert point.snapshot.player_states == expected.player_states
    assert point.snapshot.engine.events == expected.engine.events
    assert point.public_state.get("step_processing", False) is False
    assert point.public_state["last_live_step_error"]["action_applied"] is False
    trace: Any = store.get_game(game_id)
    assert trace["failures"][-1]["communication_attempts"][0]["accepted"] is True
    assert "PRIVATE_PROVIDER_BODY" not in json.dumps(trace)
    assert "never-expose" not in json.dumps(websocket.emissions)
    if kind == "parse":
        assert failed.json["attempts"][0]["accepted"] is False
        assert failed.json["attempts"][0]["memory_revision"] == session.memory_revision
    loaded = client.post(f"/api/live-traces/{game_id}/load")
    assert loaded.status_code == 200, loaded.json
    restored: Any = state.current_sandbox
    assert restored.snapshot().player_states == expected.player_states
    assert restored.game_engine.state.rng.getstate() == expected.engine.state.rng.getstate()
    transport.action_failure = None
    retry_cursor = len(transport.requests)
    continued = client.post("/api/step")
    assert continued.status_code == 200, continued.json
    assert len(restored.game_engine.state.actions) == action_count + 1
    # Restoring the exact continuation must not run accepted pre-action speech twice.
    assert transport.requests[retry_cursor].channel == "action"
    assert transport.requests[retry_cursor].memory_revision == session.memory_revision


@pytest.mark.parametrize("kind", ["parse", "tls", "generic"])
def test_failure_checkpoint_write_error_is_explicit_and_not_retryable(live_app: LiveApp, monkeypatch: pytest.MonkeyPatch, kind: str) -> None:
    client, state, websocket, transport = live_app
    game_id = _start(client)
    transport.action_failure = kind

    def fail_save(*args: object, **kwargs: object) -> None:
        assert state.step_processing is False
        assert kwargs["snapshot"].player_states == live_sandbox(state).snapshot().player_states
        raise OSError("PRIVATE_STORAGE_SECRET")

    monkeypatch.setattr(state.live_trace_store, "record_failure", fail_save)
    failed: Any = client.post("/api/step")
    assert failed.status_code == {"parse": 422, "tls": 502, "generic": 500}[kind], failed.json
    assert len(transport.requests) == 1
    assert failed.json["retryable"] is False
    assert failed.json["checkpoint_saved"] is False
    assert failed.json["trace_failure_id"] is None
    assert "could not be saved" in failed.json["details"]
    assert "Press Step" not in failed.json["details"]
    assert "PRIVATE_STORAGE_SECRET" not in json.dumps(failed.json)
    assert state.last_live_step_error == websocket.emissions[-1][1]["last_live_step_error"]
    assert state.step_processing is False
    assert state.live_trace_store.get_game(game_id)["step_count"] == 0


def test_applied_step_storage_failure_does_not_rerun_or_claim_old_state(live_app: LiveApp, monkeypatch: pytest.MonkeyPatch) -> None:
    client, state, websocket, transport = live_app
    game_id = _start(client)
    writes = []

    def fail_save(*args: object, **kwargs: object) -> None:
        writes.append(kwargs)
        raise OSError("PRIVATE_STORAGE_SECRET")

    monkeypatch.setattr(state.live_trace_store, "record_step", fail_save)
    failed: Any = client.post("/api/step")
    assert failed.status_code == 500, failed.json
    assert failed.json["action_applied"] is True
    assert failed.json["retryable"] is False
    assert failed.json["checkpoint_saved"] is False
    assert len(writes) == 1
    assert len(live_sandbox(state).game_engine.state.actions) == 1
    assert sum(request.channel == "action" for request in transport.requests) == 1
    assert live_sandbox(state).players[Color.RED].session.strategic_memory == "accepted action notes"
    assert state.live_trace_store.get_game(game_id)["step_count"] == 0
    assert failed.json["state"] == websocket.emissions[-1][1]
    assert "could not be saved" in failed.json["details"]
    assert "PRIVATE_STORAGE_SECRET" not in json.dumps(failed.json)


@pytest.mark.parametrize("failure", [True, "cancelled"], ids=["exception", "cancellation"])
def test_post_action_communication_failure_records_one_applied_step(live_app: LiveApp, failure: bool | str) -> None:
    client, state, websocket, transport = live_app
    # Explicit historical fresh contract retains its after-build polling behavior.
    source: Any = resolve_prompt_suites().shared
    bundle: Any = parse_shared_prompt_suite(source.source).model_copy(update={"reactive_speech": False})
    save_shared_prompt_override(source=bundle.model_dump_json(), expected_sha256=source.sha256)
    game_id = _start(client)
    transport.talk_failure = failure
    response: Any = client.post("/api/step")
    assert response.status_code == 200, response.json
    assert response.json["warning"]["action_applied"] is True
    assert response.json["warning"]["retryable"] is False
    trace: Any = state.live_trace_store.get_game(game_id)
    assert trace["step_count"] == 1
    assert trace["failures"] == []
    assert any(
        call["accepted"] and call["call_kind"] == "decision"
        and call["request"]["context_policy"] == "fresh_notes"
        for call in trace["model_calls"]
    )
    assert state.live_trace_store.load_snapshot(game_id).player_states == live_sandbox(state).snapshot().player_states
    assert state.last_live_step_error == websocket.emissions[-1][1]["last_live_step_error"]
    assert "PRIVATE_PROVIDER_BODY" not in json.dumps(response.json)


def test_reactive_cancellation_checkpoints_selected_speech_without_a_fake_game_step(live_app: LiveApp) -> None:
    client, state, _, transport = live_app
    game_id = _start(client)
    transport.speech_once = True
    transport.speech_respondents = ["BLUE"]
    transport.talk_failure = "cancelled"
    failed = client.post("/api/step")
    assert failed.status_code == 500
    assert failed.json["checkpoint_saved"] and not failed.json["action_applied"]
    trace: Any = state.live_trace_store.get_game(game_id)
    assert trace["step_count"] == 0
    speech: Any = trace["failures"][-1]["communication_attempts"][0]
    assert speech["accepted"] and speech["choice"]["mode"] == "say"
    assert speech["choice"]["trigger_reason"] == "standalone_speech"
    point = state.live_trace_store.load_resume_point(game_id)
    assert point.snapshot.speech_used
    assert not point.snapshot.engine.state.actions
    assert len(point.snapshot.engine.events) == 1
    transport.talk_failure = False
    assert client.post(f"/api/live-traces/{game_id}/load").status_code == 200
    continued = client.post("/api/step")
    assert continued.status_code == 200
    assert len(live_sandbox(state).game_engine.state.actions) == 1
    assert len([e for e in live_sandbox(state).game_engine.events if e.event_type == "MESSAGE_SENT"]) == 1


def test_generic_accept_callback_failure_checkpoints_applied_action(live_app: LiveApp, monkeypatch: pytest.MonkeyPatch) -> None:
    client, state, websocket, _ = live_app
    game_id = _start(client)
    sandbox: Any = state.current_sandbox

    def fail_accept(*args: object) -> None:
        raise RuntimeError("PRIVATE_PROVIDER_BODY api_key=never-expose")

    monkeypatch.setattr(sandbox.players[Color.RED], "accept", fail_accept)
    response: Any = client.post("/api/step")
    assert response.status_code == 500, response.json
    assert response.json["action_applied"] is True
    assert response.json["retryable"] is False
    assert response.json["checkpoint_saved"] is True
    assert "Do not repeat" in response.json["details"]
    point: Any = state.live_trace_store.load_resume_point(game_id)
    assert point.snapshot.player_states == sandbox.snapshot().player_states
    assert len(point.snapshot.engine.state.actions) == 1
    assert state.last_live_step_error == websocket.emissions[-1][1]["last_live_step_error"]
    assert "PRIVATE_PROVIDER_BODY" not in json.dumps(state.live_trace_store.get_game(game_id))


@pytest.mark.parametrize("channel", ["decision", "communication"])
def test_generic_sibling_failure_does_not_persist_exception_body(live_app: LiveApp, monkeypatch: pytest.MonkeyPatch, channel: str) -> None:
    client, state, websocket, transport = live_app
    monkeypatch.setenv("CATAN_CONTEXT_SUITE", str(default_suite_path()))
    monkeypatch.setenv("CATAN_COMMUNICATION_SUITE", str(default_communication_suite_path()))
    game_id: Any = _start(client)
    sandbox: Any = state.current_sandbox
    offer = None
    if channel == "decision":
        offer = ensure_trade_window(sandbox.game_engine.state).create_offer(TradeOffer(
            offered_by=Color.RED, audience=frozenset({Color.BLUE, Color.WHITE, Color.ORANGE}),
            give=(1, 0, 0, 0, 0), receive=(0, 1, 0, 0, 0),
        ))
    ready = asyncio.Event()
    completed = []

    async def complete(request: ModelRequest) -> ModelResponse:
        actor = next(
            color for color, player in sandbox.players.items()
            if player.session.session_id == request.session_id
        )
        if actor == Color.RED:
            return ModelResponse(_action_content(request, legacy=True))
        if actor == Color.BLUE:
            await asyncio.wait_for(ready.wait(), timeout=2)
            raise RuntimeError("PRIVATE_PROVIDER_BODY api_key=never-expose")
        completed.append(actor)
        if len(completed) == 2:
            ready.set()
        return ModelResponse(
            json.dumps({
                "tool": "reject_offer", "arguments": {"offer_id": offer.id},
                "game_plan": "withheld sibling plan",
            }) if offer is not None else "<message>SILENCE</message>"
        )

    monkeypatch.setattr(transport, "complete", complete)
    response = client.post("/api/step")
    assert response.status_code == (500 if channel == "decision" else 200), response.json
    assert len(completed) == 2
    trace: Any = state.live_trace_store.get_game(game_id)
    if channel == "decision":
        attempts: Any = trace["failures"][0]["attempts"]
        assert len(attempts) == 2
        assert all(not attempt["accepted"] for attempt in attempts)
        assert all(attempt["choice"]["game_plan"] == "withheld sibling plan" for attempt in attempts)
    else:
        assert trace["step_count"] == 1 and trace["failures"] == []
        attempts = trace["steps"][0]["result"]["communication_attempts"]
        assert len(attempts) == 2 and all(not attempt["accepted"] for attempt in attempts)
    assert "PRIVATE_PROVIDER_BODY" not in json.dumps([response.json, trace, websocket.emissions])
    assert "never-expose" not in json.dumps([response.json, trace, websocket.emissions])
    assert state.live_trace_store.load_snapshot(game_id).player_states == sandbox.snapshot().player_states
