"""Offline live/replay adapter coverage for exact fresh-notes boundaries."""

import asyncio
import json
import pickle
import re
import socket
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from typing import Callable
from threading import Event

import httpx
import pytest
from flask import Flask

from cle.game_engine.models.enums import ActionType
from cle.game_engine.board_tokens import edge_token, node_token, tile_token
from cle.game_engine.models.player import Color
from cle.game_engine.state import ensure_trade_window
from cle.game_engine.trading import TradeOffer
from cle.harness.communication import default_communication_suite_path
from cle.harness.catan_board_surface import ImageBoardPresenter
from cle.harness.decision import request_player_attempt
from cle.harness.models import ModelRequest, ModelResponse
from cle.harness.prompt_store import (
    load_active_prompt_suites,
    resolve_prompt_suites,
    save_prompt_suite_overrides,
    save_shared_prompt_override,
)
from cle.harness.providers.openrouter import OpenRouterHTTPFailure, OpenRouterTLSFailure
from cle.harness.shared_suite import default_shared_suite_path, parse_shared_prompt_suite
from cle.harness.suite import default_suite_path
from cle.players.contracts import PlayerAttempt, PlayerChoice, PlayerContext
from cle.sandbox.decision import build_decision_context
from cle.sandbox.factory import ActivePromptConfigurationError, LiveSandboxConfig, create_live_sandbox, materialize_live_prompt_suites
from cle.sandbox.replay import ReplaySandbox
from cle.traces import SQLiteLiveTraceStore
from playground.game_viewer.replay import decision_preview
from playground.game_viewer.routes import live_game
from playground.game_viewer.routes import prompt_suite as prompt_routes
from playground.game_viewer.routes.health import health_bp
from playground.game_viewer.routes.replay import replay_bp
from playground.game_viewer.state import ServerState


def _no_network(*args, **kwargs):
    raise AssertionError("Network is forbidden in fresh-notes route tests")


def _action_content(request, *, legacy=False, context=None):
    if context is not None and not legacy:
        action = next((a for a in context.legal_actions if a.action_type == ActionType.END_TURN), context.legal_actions[0])
        arguments = {}
        if action.action_type == ActionType.BUILD_SETTLEMENT:
            tool, arguments = "build_settlement", {"node": node_token(action.value)}
        elif action.action_type == ActionType.BUILD_ROAD:
            tool, arguments = "build_road", {"edge": edge_token(action.value)}
        elif action.action_type == ActionType.MOVE_ROBBER:
            tile = context.observation.board_map.land_tiles[action.value]
            tool, arguments = "move_robber", {"tile": tile_token(tile.id)}
        elif action.action_type == ActionType.STEAL:
            tool, arguments = "steal_from", {"player": action.value[0].value}
        elif action.action_type == ActionType.END_TURN:
            tool = "end_turn"
        elif action.action_type == ActionType.ROLL:
            tool = "roll_dice"
        else:
            raise AssertionError(f"Unexpected engine fixture action: {action}")
        return json.dumps({"tool": tool, "arguments": arguments, "notes": "accepted action notes"})
    menu = next(
        component.value for component in request.components
        if component.id == "environment.legal_actions"
    )
    arguments = {}
    if "end_turn()" in menu:
        tool = "end_turn"
    elif "build_settlement(node):" in menu:
        tool = "build_settlement"
        arguments = {"node": re.search(r"<N\d{2}>", menu).group()}
    elif "build_road(edge):" in menu:
        tool = "build_road"
        arguments = {"edge": re.search(r"<E\d{2}_\d{2}>", menu).group()}
    elif "move_robber(tile):" in menu:
        tool = "move_robber"
        arguments = {"tile": re.search(r"<T\d{2}>", menu).group()}
    elif "steal_from(player):" in menu:
        tool = "steal_from"
        arguments = {"player": re.search(r"steal_from\(player\): (\w+)", menu)[1]}
    else:
        raise AssertionError(f"Unexpected test decision menu: {menu}")
    return json.dumps({
        "tool": tool, "arguments": arguments,
        "game_plan" if legacy else "notes": "accepted action notes",
    })


@dataclass
class LocalTransport:
    requests: list = field(default_factory=list)
    action_failure: str | None = None
    talk_failure: bool | str = False
    talk_notes: str = "accepted speech notes"
    speech_once: bool = False
    speech_respondents: list[str] = field(default_factory=list)
    legacy: bool = False
    context_factory: Callable[[ModelRequest], PlayerContext] | None = None

    async def complete(self, request):
        self.requests.append(request)
        if self.legacy and not any(
            component.id == "environment.legal_actions" for component in request.components
        ):
            return ModelResponse("<mode>SILENCE</mode>")
        if request.channel == "talk":
            if self.talk_failure == "cancelled":
                raise asyncio.CancelledError("post-action speech cancelled")
            if self.talk_failure:
                raise RuntimeError("PRIVATE_PROVIDER_BODY api_key=never-expose")
            return ModelResponse(json.dumps({"mode": "silence", "notes": self.talk_notes}))
        if self.speech_once:
            self.speech_once = False
            return ModelResponse(json.dumps({
                "tool": "say", "arguments": {"text": "Leave this spot open?",
                "respondents": self.speech_respondents}, "notes": self.talk_notes,
            }))
        if self.action_failure == "parse":
            return ModelResponse('{"tool":"invalid","arguments":{},"notes":"REJECTED_NOTES"}')
        if self.action_failure == "tls":
            raise OpenRouterTLSFailure(request, model="test/local", attempts=3)
        if self.action_failure == "http":
            raise OpenRouterHTTPFailure(
                request, model="test/local", attempts=1,
                response=httpx.Response(
                    403, json={"error": {"message": "Provider policy rejected this model."}},
                    headers={"x-request-id": "req-local"},
                ),
            )
        if self.action_failure == "generic":
            raise RuntimeError("PRIVATE_PROVIDER_BODY api_key=never-expose")
        context = self.context_factory(request) if self.context_factory is not None else None
        return ModelResponse(_action_content(request, legacy=self.legacy, context=context), model="test/local")


@dataclass
class RecordingSocket:
    emissions: list = field(default_factory=list)

    def emit(self, event, payload):
        self.emissions.append((event, payload))


@pytest.fixture(autouse=True)
def isolated_environment(tmp_path, monkeypatch):
    assert tmp_path.is_dir() and tmp_path.parent.is_dir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.setenv("CATAN_LIVE_TRACE_DB", str(tmp_path / "live.sqlite3"))
    monkeypatch.setenv("CATAN_PROMPT_SUITE_DIR", str(tmp_path / "prompts"))
    for name in (
        "CATAN_SHARED_SUITE", "CATAN_CONTEXT_SUITE", "CATAN_COMMUNICATION_SUITE",
        "VLLM_BASE_URL", "OPENROUTER_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(socket, "create_connection", _no_network)
    monkeypatch.setattr(httpx.Client, "send", _no_network)
    monkeypatch.setattr(httpx.AsyncClient, "send", _no_network)


@pytest.fixture
def live_app(tmp_path, monkeypatch):
    transport = LocalTransport()
    monkeypatch.setattr("cle.sandbox.factory.create_text_transport", lambda config: transport)
    app = Flask(__name__)
    app.config["TESTING"] = True
    state = ServerState()
    transport.context_factory = lambda request: build_decision_context(
        state.current_sandbox.game_engine, Color(request.decision_id.rsplit(":", 1)[-1]),
    )
    state.live_trace_store = SQLiteLiveTraceStore(tmp_path / "live.sqlite3")
    websocket = RecordingSocket()
    app.config.update(SERVER_STATE=state, SOCKETIO=websocket)
    app.config["REPLAY_COMPLETION_TRANSPORT_FACTORY"] = lambda **config: transport
    app.register_blueprint(live_game.live_game_bp)
    app.register_blueprint(health_bp)
    app.register_blueprint(replay_bp)
    return app.test_client(), state, websocket, transport


def _start(client):
    response = client.post("/api/start-game", json={
        "mode": "llm", "model": "test/local", "seed": 9,
        "palette": "canonical_four", "shuffle_players": False,
        "reasoning": {"enabled": False}, "max_decision_attempts": 1,
    })
    assert response.status_code == 200, response.json
    return response.json["trace_game_id"]


def _reach_main_action(client, state):
    for _ in range(24):
        engine = state.current_sandbox.game_engine
        context = state.current_sandbox.decision_context()
        types = {action.action_type for action in context.legal_actions}
        if context.phase == "main_game" and ActionType.END_TURN in types:
            return
        response = client.post("/api/step")
        assert response.status_code == 200, response.json
        assert engine is state.current_sandbox.game_engine
    raise AssertionError("Opening did not reach an ordinary main-game decision")


def test_new_game_keeps_saved_trace_and_starts_with_empty_memory(live_app):
    client, state, _, _ = live_app
    old_id = _start(client)
    assert client.post("/api/step").status_code == 200
    old_trace = state.live_trace_store.get_game(old_id)
    assert state.current_sandbox.players[Color.RED].session.strategic_memory
    assert client.post("/api/reset").status_code == 200
    assert state.current_sandbox is None
    new_id = _start(client)
    assert new_id != old_id
    assert state.current_sandbox.game_engine.state.actions == []
    for player in state.current_sandbox.players.values():
        assert player.session.strategic_memory == ""
        assert player.session.memory_revision == 0
        assert player.session.receipts == {}
    archived = state.live_trace_store.get_game(old_id)
    assert archived["status"] == "reset"
    assert archived["model_calls"] == old_trace["model_calls"]
    assert archived["steps"] == old_trace["steps"]


def test_stored_shared_source_and_path_are_historical_only():
    config = materialize_live_prompt_suites(LiveSandboxConfig(
        mode="llm", model="test/local", shared_suite_path=str(default_shared_suite_path()),
    ))
    restored = live_game._config_from_stored_payload(asdict(config))
    assert restored.shared_suite is None and restored.shared_suite_path is None
    public = live_game._live_inference_payload(materialize_live_prompt_suites(restored))
    assert public["context_policy"] == "fresh_notes"
    assert public["shared_suite"] == {
        "id": config.shared_suite.id, "version": config.shared_suite.version,
        "sha256": config.shared_suite.sha256,
    }
    assert "source" not in public["shared_suite"]


def test_new_live_game_restores_notes_with_active_shared_source(live_app):
    client, state, _, transport = live_app
    game_id = _start(client)
    source = resolve_prompt_suites().shared
    trace = client.get(f"/api/live-traces/{game_id}").json
    assert trace["config"]["shared_suite"]["source"] == source.source
    assert trace["config"]["shared_suite"]["sha256"] == source.sha256
    assert trace["config"]["decision_suite"] is None
    stepped = client.post("/api/step")
    assert stepped.status_code == 200, stepped.json
    sandbox = state.current_sandbox
    red = sandbox.players[Color.RED]
    assert red.session.strategic_memory == "accepted action notes"
    assert red.session.memory_revision == 1
    assert red.session.messages == []
    traces = stepped.json["reasoning_traces"]
    assert traces[0]["accepted"] is True
    assert traces[0]["notes_update"] == red.session.strategic_memory
    assert traces[0]["context_policy"] == "fresh_notes"
    assert sum(item["call_kind"] == "communication" for item in traces) == 0
    assert all(item["accepted"] for item in traces)
    assert all(item.context_policy == "fresh_notes" for item in transport.requests)
    expected = sandbox.snapshot()
    changed = save_shared_prompt_override(
        source=source.source.replace("{{ resources }}", "EDITED RESOURCE HEADING: {{ resources }}"),
        expected_sha256=source.sha256,
    )
    loaded = client.post(f"/api/live-traces/{game_id}/load")
    assert loaded.status_code == 200, loaded.json
    assert loaded.json["state"]["live_inference"]["shared_suite"]["sha256"] == changed.shared.sha256
    assert state.current_sandbox.snapshot().player_states == expected.player_states
    assert resolve_prompt_suites() == changed
    restored = state.current_sandbox.players[Color.RED]
    assert restored.suite.context.memory_mode == restored.communication_suite.memory_mode == "fresh_notes"
    assert restored.suite.components["resources"].template == parse_shared_prompt_suite(changed.shared.source).components["resources"].template
    assert changed.shared.sha256 != source.sha256
    continued = client.post("/api/step")
    assert continued.status_code == 200, continued.json
    assert continued.json["trace_step_index"] == 1
    assert len(continued.json["reasoning_traces"]) == 1
    assert restored.session.messages == []


def test_live_editor_changes_next_action_and_speech_without_rewriting_history(live_app):
    client, state, _, transport = live_app
    client.application.register_blueprint(prompt_routes.prompt_suite_bp)
    game_id = _start(client)
    assert client.post("/api/step").status_code == 200
    history = deepcopy(state.live_trace_store.get_game(game_id))
    old_requests = deepcopy(transport.requests)
    sandbox = state.current_sandbox
    snapshot = sandbox.snapshot()
    editor = client.get("/api/prompt-suite").json
    assert not editor["saving_locked"]
    document = editor["shared"]["document"]
    document["components"]["board_state"]["template"] = "NEXT REQUEST BOARD\n{{ board_state }}"
    saved = client.put("/api/prompt-suite", json={
        "shared": document, "expected": {"shared": editor["shared"]["sha256"]},
    })
    assert saved.status_code == 200, saved.json
    assert sandbox.snapshot().player_states == snapshot.player_states
    transport.speech_once = True
    transport.speech_respondents = ["BLUE"]
    assert client.post("/api/step").status_code == 200
    assert state.current_sandbox is sandbox
    calls = transport.requests[len(old_requests):]
    assert {call.channel for call in calls} == {"action", "talk"}
    for call in calls:
        assert "NEXT REQUEST BOARD" in "\n".join(message.content for message in call.messages)
        assert call.prompt_sources[0].sha256 == saved.json["shared"]["sha256"]
    assert transport.requests[:len(old_requests)] == old_requests
    after = state.live_trace_store.get_game(game_id)
    assert after["model_calls"][:len(history["model_calls"])] == history["model_calls"]
    assert after["steps"][0] == history["steps"][0]
    assert after["config"] == history["config"]
    durable = deepcopy(after["model_calls"])
    before_load = sandbox.snapshot()
    loaded = client.post(f"/api/live-traces/{game_id}/load")
    assert loaded.status_code == 200
    assert state.current_sandbox.snapshot().player_states == before_load.player_states
    assert loaded.json["state"]["game"] == after["steps"][-1]["public_state"]["game"]
    restored = state.current_sandbox.snapshot().engine
    assert restored.events == before_load.engine.events
    assert restored.state.player_state == before_load.engine.state.player_state
    assert restored.state.resource_freqdeck == before_load.engine.state.resource_freqdeck
    assert restored.state.development_listdeck == before_load.engine.state.development_listdeck
    assert restored.state.rng.getstate() == before_load.engine.state.rng.getstate()
    assert state.live_trace_store.get_game(game_id)["model_calls"] == durable


@pytest.mark.parametrize("notes_limit", [500, 3])
def test_inflight_edit_keeps_old_admission_and_updates_next_actual_boundary(live_app, monkeypatch, notes_limit):
    client, state, _, transport = live_app
    client.application.register_blueprint(prompt_routes.prompt_suite_bp)
    game_id = _start(client)
    editor = client.get("/api/prompt-suite").json
    entered, release, published = Event(), Event(), Event()
    original_complete = transport.complete
    original_save = prompt_routes.save_shared_prompt_override

    async def paused(request):
        if request.channel == "action":
            entered.set()
            assert release.wait(5)
        return await original_complete(request)

    def save_and_signal(**kwargs):
        result = original_save(**kwargs)
        published.set()
        return result

    monkeypatch.setattr(transport, "complete", paused)
    monkeypatch.setattr(prompt_routes, "save_shared_prompt_override", save_and_signal)
    document = editor["shared"]["document"]
    document["components"]["board_state"]["template"] = "EDIT DURING INFERENCE\n{{ board_state }}"
    document["max_notes_chars"] = notes_limit
    with ThreadPoolExecutor(max_workers=2) as pool:
        step = pool.submit(client.application.test_client().post, "/api/step")
        try:
            assert entered.wait(5)
            save = pool.submit(client.application.test_client().put, "/api/prompt-suite", json={
                "shared": document, "expected": {"shared": editor["shared"]["sha256"]},
            })
            assert published.wait(5), "Prompt save waited for the in-flight model"
        finally:
            release.set()
        result, saved = step.result(timeout=5), save.result(timeout=5)
    assert result.status_code == saved.status_code == 200
    action, *speech = transport.requests
    assert action.channel == "action"
    assert action.prompt_sources[0].sha256 == editor["shared"]["sha256"]
    assert "EDIT DURING INFERENCE" not in str(action.messages)
    assert speech == []
    assert result.json["warning"] is None
    assert len(state.current_sandbox.game_engine.state.actions) == 1
    following = client.post("/api/step")
    if notes_limit == 3:
        # The old request's notes exceed the NEW limit. They must still commit
        # under the original contract before the next boundary rejects rebinding.
        assert following.status_code == 409
        assert following.json["action_applied"] is False
        assert "notes must not exceed 3" in following.json["details"]
        assert state.current_sandbox.players[Color.RED].session.strategic_memory == "accepted action notes"
    else:
        assert following.status_code == 200
        assert transport.requests[-1].prompt_sources[0].sha256 == saved.json["shared"]["sha256"]
    stored = state.live_trace_store.get_game(game_id)["model_calls"]
    assert stored[0]["request"]["prompt_sources"][0]["sha256"] == editor["shared"]["sha256"]


def test_context_mode_migration_preserves_notes_history_and_channel_events(live_app, monkeypatch):
    client, state, _, transport = live_app
    _start(client)
    assert client.post("/api/step").status_code == 200
    sandbox = state.current_sandbox
    before = sandbox.snapshot()
    monkeypatch.setenv("CATAN_CONTEXT_SUITE", str(default_suite_path()))
    monkeypatch.setenv("CATAN_COMMUNICATION_SUITE", str(default_communication_suite_path()))
    sandbox._refresh_inference_policy()
    for color, prior in before.player_states:
        session = sandbox.players[color].session
        assert session.context_policy == "legacy"
        assert session.strategic_memory == prior.session.strategic_memory
        assert session.action_next_sequence == prior.session.action_next_sequence
        assert session.talk_next_sequence == prior.session.talk_next_sequence
    # An event arrives during legacy mode. Its mixed acknowledgment must not
    # make either fresh channel skip that event when switching back.
    sandbox.game_engine.append_message(
        speaker=Color.BLUE, text="New trade information",
        audience=tuple(color for color in sandbox.players if color != Color.BLUE),
        causation_id="migration-message",
    )
    for player in sandbox.players.values():
        player.acknowledge_events(sandbox.revision)
    legacy = sandbox.snapshot()
    monkeypatch.delenv("CATAN_CONTEXT_SUITE")
    monkeypatch.delenv("CATAN_COMMUNICATION_SUITE")
    sandbox._refresh_inference_policy()
    assert pickle.dumps(sandbox.snapshot().engine) == pickle.dumps(legacy.engine)
    for color, prior in legacy.player_states:
        session = sandbox.players[color].session
        assert session.context_policy == "fresh_notes"
        assert session.strategic_memory == prior.session.strategic_memory
        assert tuple(session.messages) == prior.session.messages
        assert session.action_next_sequence == session.talk_next_sequence == 0
        assert session.event_cursor == prior.session.event_cursor
    cursor = len(transport.requests)
    transport.speech_once = True
    transport.speech_respondents = ["BLUE"]
    assert client.post("/api/step").status_code == 200
    assert "New trade information" in str(transport.requests[cursor].messages)
    assert any("New trade information" in str(call.messages) for call in transport.requests[cursor + 1:])


def test_incompatible_notes_limit_rebind_is_atomic_and_recoverable(live_app):
    client, state, _, transport = live_app
    _start(client)
    assert client.post("/api/step").status_code == 200
    sandbox = state.current_sandbox
    last = list(sandbox.players.values())[-1]
    last.session.strategic_memory = "Retain these private notes exactly"
    before = pickle.dumps(sandbox.snapshot())
    players = dict(sandbox.players)
    active = resolve_prompt_suites().shared
    bundle = parse_shared_prompt_suite(active.source)
    changed = save_shared_prompt_override(
        source=bundle.model_copy(update={"max_notes_chars": 3}).model_dump_json(),
        expected_sha256=active.sha256,
    )
    with pytest.raises(ActivePromptConfigurationError, match="notes must not exceed"):
        sandbox._refresh_inference_policy()
    assert sandbox.players == players
    assert pickle.dumps(sandbox.snapshot()) == before
    count = len(transport.requests)
    failed = client.post("/api/step")
    assert failed.status_code == 409
    assert not failed.json["action_applied"]
    assert len(transport.requests) == count
    assert pickle.dumps(sandbox.snapshot()) == before
    save_shared_prompt_override(source=active.source, expected_sha256=changed.shared.sha256)
    assert client.post("/api/step").status_code == 200


def test_legacy_saved_game_load_uses_active_shared_config_and_keeps_notes(live_app):
    client, state, _, transport = live_app
    transport.legacy = True
    started = client.post("/api/start-game", json={
        "mode": "llm", "model": "historical/model", "seed": 7,
        "palette": "canonical_four", "shuffle_players": False,
        "context_suite_path": str(default_suite_path()),
        "communication_suite_path": str(default_communication_suite_path()),
    })
    assert started.status_code == 200
    game_id = started.json["trace_game_id"]
    assert client.post("/api/step").status_code == 200
    before = state.current_sandbox.snapshot()
    history = deepcopy(state.live_trace_store.get_game(game_id))
    # Simulate a process with a different explicit active selection, not a game
    # reset: loading must use this model and shared defaults, not saved paths.
    state.active_live_config = LiveSandboxConfig(model="current/model", reasoning={"enabled": False})
    transport.legacy = False
    loaded = client.post(f"/api/live-traces/{game_id}/load")
    assert loaded.status_code == 200, loaded.json
    assert loaded.json["model"] == "current/model"
    assert loaded.json["state"]["live_inference"]["context_policy"] == "fresh_notes"
    for color, old in before.player_states:
        player = state.current_sandbox.players[color]
        assert player.session.strategic_memory == old.session.strategic_memory
        assert tuple(player.session.messages) == old.session.messages
        assert player.session.action_next_sequence == player.session.talk_next_sequence == 0
        assert player.session.memory_revision == old.session.memory_revision + 1
    assert tuple(state.current_sandbox.game_engine.events) == before.engine.events
    assert client.post("/api/step").status_code == 200
    after = state.live_trace_store.get_game(game_id)
    assert after["config"] == history["config"]
    assert after["model_calls"][:len(history["model_calls"])] == history["model_calls"]
    assert all(call["request"]["prompt_sources"][0]["kind"] == "shared"
               for call in after["model_calls"][len(history["model_calls"]):])


@pytest.mark.parametrize("kind,status", [("parse", 422), ("tls", 502), ("http", 502), ("generic", 500)])
def test_pre_speech_notes_survive_failed_action_checkpoint_and_resume(live_app, kind, status):
    client, state, websocket, transport = live_app
    game_id = _start(client)
    _reach_main_action(client, state)
    sandbox = state.current_sandbox
    actor = sandbox.current_actor()
    before = sandbox.players[actor].snapshot().session
    revision = sandbox.revision
    action_count = len(sandbox.game_engine.state.actions)
    transport.talk_notes = "accepted pre-speech notes at failed boundary"
    transport.speech_once = True
    transport.action_failure = kind
    request_cursor = len(transport.requests)
    failed = client.post("/api/step")
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
    expected = sandbox.snapshot()
    store = SQLiteLiveTraceStore(state.live_trace_store.path)
    point = store.load_resume_point(game_id)
    assert point.snapshot.player_states == expected.player_states
    assert point.snapshot.engine.events == expected.engine.events
    assert point.public_state.get("step_processing", False) is False
    assert point.public_state["last_live_step_error"]["action_applied"] is False
    trace = store.get_game(game_id)
    assert trace["failures"][-1]["communication_attempts"][0]["accepted"] is True
    assert "PRIVATE_PROVIDER_BODY" not in json.dumps(trace)
    assert "never-expose" not in json.dumps(websocket.emissions)
    if kind == "parse":
        assert failed.json["attempts"][0]["accepted"] is False
        assert failed.json["attempts"][0]["memory_revision"] == session.memory_revision
    loaded = client.post(f"/api/live-traces/{game_id}/load")
    assert loaded.status_code == 200, loaded.json
    restored = state.current_sandbox
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
def test_failure_checkpoint_write_error_is_explicit_and_not_retryable(live_app, monkeypatch, kind):
    client, state, websocket, transport = live_app
    game_id = _start(client)
    transport.action_failure = kind

    def fail_save(*args, **kwargs):
        assert state.step_processing is False
        assert kwargs["snapshot"].player_states == state.current_sandbox.snapshot().player_states
        raise OSError("PRIVATE_STORAGE_SECRET")

    monkeypatch.setattr(state.live_trace_store, "record_failure", fail_save)
    failed = client.post("/api/step")
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


def test_applied_step_storage_failure_does_not_rerun_or_claim_old_state(live_app, monkeypatch):
    client, state, websocket, transport = live_app
    game_id = _start(client)
    writes = []

    def fail_save(*args, **kwargs):
        writes.append(kwargs)
        raise OSError("PRIVATE_STORAGE_SECRET")

    monkeypatch.setattr(state.live_trace_store, "record_step", fail_save)
    failed = client.post("/api/step")
    assert failed.status_code == 500, failed.json
    assert failed.json["action_applied"] is True
    assert failed.json["retryable"] is False
    assert failed.json["checkpoint_saved"] is False
    assert len(writes) == 1
    assert len(state.current_sandbox.game_engine.state.actions) == 1
    assert sum(request.channel == "action" for request in transport.requests) == 1
    assert state.current_sandbox.players[Color.RED].session.strategic_memory == "accepted action notes"
    assert state.live_trace_store.get_game(game_id)["step_count"] == 0
    assert failed.json["state"] == websocket.emissions[-1][1]
    assert "could not be saved" in failed.json["details"]
    assert "PRIVATE_STORAGE_SECRET" not in json.dumps(failed.json)


@pytest.mark.parametrize("failure", [True, "cancelled"], ids=["exception", "cancellation"])
def test_post_action_communication_failure_records_one_applied_step(live_app, failure):
    client, state, websocket, transport = live_app
    # Explicit historical fresh contract retains its after-build polling behavior.
    source = resolve_prompt_suites().shared
    bundle = parse_shared_prompt_suite(source.source).model_copy(update={"reactive_speech": False})
    save_shared_prompt_override(source=bundle.model_dump_json(), expected_sha256=source.sha256)
    game_id = _start(client)
    transport.talk_failure = failure
    response = client.post("/api/step")
    assert response.status_code == 200, response.json
    assert response.json["warning"]["action_applied"] is True
    assert response.json["warning"]["retryable"] is False
    trace = state.live_trace_store.get_game(game_id)
    assert trace["step_count"] == 1
    assert trace["failures"] == []
    assert any(
        call["accepted"] and call["call_kind"] == "decision"
        and call["request"]["context_policy"] == "fresh_notes"
        for call in trace["model_calls"]
    )
    assert state.live_trace_store.load_snapshot(game_id).player_states == state.current_sandbox.snapshot().player_states
    assert state.last_live_step_error == websocket.emissions[-1][1]["last_live_step_error"]
    assert "PRIVATE_PROVIDER_BODY" not in json.dumps(response.json)


def test_reactive_cancellation_checkpoints_selected_speech_without_a_fake_game_step(live_app):
    client, state, _, transport = live_app
    game_id = _start(client)
    transport.speech_once = True
    transport.speech_respondents = ["BLUE"]
    transport.talk_failure = "cancelled"
    failed = client.post("/api/step")
    assert failed.status_code == 500
    assert failed.json["checkpoint_saved"] and not failed.json["action_applied"]
    trace = state.live_trace_store.get_game(game_id)
    assert trace["step_count"] == 0
    speech = trace["failures"][-1]["communication_attempts"][0]
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
    assert len(state.current_sandbox.game_engine.state.actions) == 1
    assert len([e for e in state.current_sandbox.game_engine.events if e.event_type == "MESSAGE_SENT"]) == 1


def test_generic_accept_callback_failure_checkpoints_applied_action(live_app, monkeypatch):
    client, state, websocket, _ = live_app
    game_id = _start(client)
    sandbox = state.current_sandbox

    def fail_accept(*args):
        raise RuntimeError("PRIVATE_PROVIDER_BODY api_key=never-expose")

    monkeypatch.setattr(sandbox.players[Color.RED], "accept", fail_accept)
    response = client.post("/api/step")
    assert response.status_code == 500, response.json
    assert response.json["action_applied"] is True
    assert response.json["retryable"] is False
    assert response.json["checkpoint_saved"] is True
    assert "Do not repeat" in response.json["details"]
    point = state.live_trace_store.load_resume_point(game_id)
    assert point.snapshot.player_states == sandbox.snapshot().player_states
    assert len(point.snapshot.engine.state.actions) == 1
    assert state.last_live_step_error == websocket.emissions[-1][1]["last_live_step_error"]
    assert "PRIVATE_PROVIDER_BODY" not in json.dumps(state.live_trace_store.get_game(game_id))


@pytest.mark.parametrize("channel", ["decision", "communication"])
def test_generic_sibling_failure_does_not_persist_exception_body(live_app, monkeypatch, channel):
    client, state, websocket, transport = live_app
    monkeypatch.setenv("CATAN_CONTEXT_SUITE", str(default_suite_path()))
    monkeypatch.setenv("CATAN_COMMUNICATION_SUITE", str(default_communication_suite_path()))
    game_id = _start(client)
    sandbox = state.current_sandbox
    offer = None
    if channel == "decision":
        offer = ensure_trade_window(sandbox.game_engine.state).create_offer(TradeOffer(
            offered_by=Color.RED, audience=frozenset({Color.BLUE, Color.WHITE, Color.ORANGE}),
            give=(1, 0, 0, 0, 0), receive=(0, 1, 0, 0, 0),
        ))
    ready = asyncio.Event()
    completed = []

    async def complete(request):
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
    trace = state.live_trace_store.get_game(game_id)
    if channel == "decision":
        attempts = trace["failures"][0]["attempts"]
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


@pytest.mark.parametrize("image_board", [False, True])
def test_failed_attempt_exposes_typed_notes_and_provenance_not_provider_bodies(image_board):
    board = None
    if image_board:
        sandbox = create_live_sandbox(LiveSandboxConfig(seed=9, shuffle_players=False))
        board = ImageBoardPresenter().present(sandbox.decision_context())
    request = ModelRequest(
        "decision", "session", (), memory_revision=4, input_next_sequence=12,
        context_policy="fresh_notes", channel="action", board_presentation=board,
    )
    payload = live_game._failed_attempt_payload(PlayerAttempt(
        "decision", PlayerChoice(0, notes_update="unaccepted parsed notes"), "withheld",
        model_request=request,
        model_response=ModelResponse(
            "visible completion", provider_request_payload={"secret": "RAW_REQUEST"},
            provider_response_payload={"body": "RAW_RESPONSE"},
        ),
    ))
    assert payload["accepted"] is False
    assert payload["notes_update"] == "unaccepted parsed notes"
    assert payload["context_policy"] == "fresh_notes"
    assert payload["request"]["memory_revision"] == payload["memory_revision"] == 4
    assert payload["input_next_sequence"] == 12
    assert payload["channel"] == "action"
    assert "RAW_REQUEST" not in json.dumps(payload)
    assert "RAW_RESPONSE" not in json.dumps(payload)
    if image_board:
        assert payload["request"]["board_presentation"]["kind"] == "image"
        assert payload["request"]["board_presentation"]["data"] is None


def test_request_attempt_accepts_matched_pair_and_only_explicit_fresh_seed():
    source = resolve_prompt_suites().shared
    bundle = parse_shared_prompt_suite(source.source)
    transport = LocalTransport()
    sandbox = create_live_sandbox(LiveSandboxConfig(seed=9, shuffle_players=False))
    context = replace(sandbox.decision_context(), visible_through_sequence=-1)
    transport.context_factory = lambda request: context
    for seed in (None, "own explicit notes"):
        attempt = asyncio.run(request_player_attempt(
            context, transport, suite=bundle.decision_suite(),
            communication_suite=bundle.communication_suite(),
            game_plan="DO_NOT_COPY_LEGACY_MEMORY", notes_seed=seed,
        ))
        assert attempt.validation_error is None
        assert attempt.choice.notes_update == "accepted action notes"
        notes = next(item for item in attempt.model_request.components if item.id == "environment.notes")
        assert dict(notes.variables)["notes"] == (seed or "")
        assert all(
            "DO_NOT_COPY_LEGACY_MEMORY" not in message.content
            for message in attempt.model_request.messages
        )


@pytest.mark.parametrize("legacy", [False, True])
def test_replay_preview_resolves_once_and_never_promotes_notes_or_resets_sources(live_app, monkeypatch, legacy):
    client, state, _, transport = live_app
    _start(client)
    live = state.current_sandbox
    engine = live.game_engine
    engine.step(engine.state.playable_actions[0])
    for index in range(20):
        engine.append_message(
            speaker=Color.BLUE, text=f"available message {index:02d}",
            audience=tuple(color for color in engine.state.colors if color != Color.BLUE),
            causation_id=f"message:{index}",
        )
    engine.append_message(
        speaker=Color.BLUE, text="HIDDEN_OTHER_SEAT_MESSAGE", audience=(Color.WHITE,),
        causation_id="private",
    )
    live.players[Color.RED].session.strategic_memory = "DO_NOT_COPY_RESIDENT_MEMORY"
    if legacy:
        pair = load_active_prompt_suites()
        save_prompt_suite_overrides(
            decision_source=pair.decision.source, communication_source=pair.communication.source,
            expected_decision_sha256=pair.decision.sha256,
            expected_communication_sha256=pair.communication.sha256,
        )
        transport.legacy = True
    sources = resolve_prompt_suites()
    state.replay_mode = True
    state.replay_data = {"game_id": str(engine.id), "parsed_actions": [{}] * 4}
    state.replay_index = 1
    state.current_sandbox = ReplaySandbox(state, engine, players=live.players)
    before = pickle.dumps(live.snapshot())
    resolutions = []

    def resolve_once():
        resolutions.append(True)
        return resolve_prompt_suites()

    monkeypatch.setattr(decision_preview, "resolve_prompt_suites", resolve_once)
    for _ in range(2):
        response = client.post("/api/replay-llm-response", json={
            "model": "test/local", "reasoning": {"enabled": False},
            "game_plan": "EXPLICIT_HISTORICAL_PLAN",
        })
        assert response.status_code == 200, response.json
        result = response.json
        assert result["parse_error"] is None
        assert result["accepted"] is False
        assert result["stale"] is False
        assert pickle.dumps(live.snapshot()) == before
        assert state.replay_index == 1
        assert resolve_prompt_suites() == sources
        messages = "\n".join(message["content"] for message in result["model_messages"])
        assert "DO_NOT_COPY_RESIDENT_MEMORY" not in messages
        assert "HIDDEN_OTHER_SEAT_MESSAGE" not in messages
        if not legacy:
            assert "EXPLICIT_HISTORICAL_PLAN" not in messages
            assert "available message 00" in messages and "available message 19" in messages
            assert "BUILD_SETTLEMENT" in messages
            assert result["notes_update"] == "accepted action notes"
            assert result["context_bootstrap"] == "cold"
            assert result["input_start_sequence"] == result["memory_revision"] == 0
            assert result["input_next_sequence"] == engine.revision
            assert result["notes_seeded"] is False
            assert result["shared_suite"] == {
                "id": sources.shared.id, "version": sources.shared.version,
                "sha256": sources.shared.sha256,
            }
        else:
            assert "EXPLICIT_HISTORICAL_PLAN" in messages
            assert "shared_suite" not in result
            assert result["game_plan"] == "accepted action notes"
    assert len(resolutions) == 2


@pytest.mark.parametrize("owner", [None, Color.BLUE, Color.RED])
def test_replay_explicit_notes_require_matching_own_seat(live_app, owner):
    client, state, _, transport = live_app
    _start(client)
    live = state.current_sandbox
    state.replay_data = {"game_id": live.game_engine.id, "parsed_actions": [{}]}
    replay = ReplaySandbox(state, live.game_engine, players=live.players)
    arguments = dict(
        model="test/local", game_plan="legacy", reasoning_request={"enabled": False},
        transport_factory=lambda **config: transport, notes_seed="EXPLICIT_OWN_NOTES",
        notes_player=owner,
    )
    before = live.snapshot().player_states
    if owner != Color.RED:
        with pytest.raises(ValueError, match="current decision player"):
            asyncio.run(decision_preview.generate_decision_preview(replay, **arguments))
        assert transport.requests == []
    else:
        result = asyncio.run(decision_preview.generate_decision_preview(replay, **arguments))
        assert result["notes_seeded"] is True
        assert result["accepted"] is False
        assert "EXPLICIT_OWN_NOTES" in json.dumps(result["model_messages"])
    assert live.snapshot().player_states == before


def test_explicit_historical_factory_pair_remains_legacy(live_app, monkeypatch):
    client, state, _, transport = live_app
    monkeypatch.setenv("CATAN_CONTEXT_SUITE", str(default_suite_path()))
    monkeypatch.setenv("CATAN_COMMUNICATION_SUITE", str(default_communication_suite_path()))
    transport.legacy = True
    game_id = _start(client)
    response = client.post("/api/step")
    assert response.status_code == 200, response.json
    config = state.live_trace_store.get_game(game_id)["config"]
    assert config["shared_suite"] is None
    assert config["decision_suite"]["version"] == "11.0.0"
    assert state.current_sandbox.players[Color.RED].session.context_policy == "legacy"
