"""New live games keep saved traces, notes, and active shared sources."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict
from threading import Event
from typing import Any

import pytest

from cle.game_engine.models.player import Color
from cle.harness.models import ModelRequest, ModelResponse
from cle.harness.prompt_store import (
    ActivePromptSuites,
    resolve_prompt_suites,
    save_shared_prompt_override,
)
from cle.harness.shared_suite import default_shared_suite_path, parse_shared_prompt_suite
from cle.sandbox import CatanSandbox
from cle.sandbox.factory import (
    LiveSandboxConfig,
    materialize_live_prompt_suites,
)
from cle.sandbox.replay import ReplaySandbox
from playground.game_viewer.routes import live_game
from playground.game_viewer.routes import prompt_suite as prompt_routes
from playground.game_viewer.state import ServerState

from .conftest import LiveApp
from .support import (
    _start,
)


def live_sandbox(state: ServerState) -> CatanSandbox | ReplaySandbox:
    """The viewer's current sandbox, which every caller here has already created."""
    sandbox = state.current_sandbox
    assert sandbox is not None
    return sandbox



def test_new_game_keeps_saved_trace_and_starts_with_empty_memory(live_app: LiveApp) -> None:
    client, state, _, _ = live_app
    old_id: Any = _start(client)
    assert client.post("/api/step").status_code == 200
    old_trace: Any = state.live_trace_store.get_game(old_id)
    assert live_sandbox(state).players[Color.RED].session.strategic_memory
    assert client.post("/api/reset").status_code == 200
    assert state.current_sandbox is None
    new_id = _start(client)
    assert new_id != old_id
    assert live_sandbox(state).game_engine.state.actions == []
    for player in live_sandbox(state).players.values():
        assert player.session.strategic_memory == ""
        assert player.session.memory_revision == 0
        assert player.session.receipts == {}
    archived: Any = state.live_trace_store.get_game(old_id)
    assert archived["status"] == "reset"
    assert archived["model_calls"] == old_trace["model_calls"]
    assert archived["steps"] == old_trace["steps"]


def test_stored_shared_source_and_path_are_historical_only() -> None:
    config: Any = materialize_live_prompt_suites(LiveSandboxConfig(
        mode="llm", model="test/local", shared_suite_path=str(default_shared_suite_path()),
    ))
    restored = live_game._config_from_stored_payload(asdict(config))
    assert restored.shared_suite is None and restored.shared_suite_path is None
    public: Any = live_game._live_inference_payload(materialize_live_prompt_suites(restored))
    assert public["context_policy"] == "fresh_notes"
    assert public["shared_suite"] == {
        "id": config.shared_suite.id, "version": config.shared_suite.version,
        "sha256": config.shared_suite.sha256,
    }
    assert "source" not in public["shared_suite"]


def test_new_live_game_restores_notes_with_active_shared_source(live_app: LiveApp) -> None:
    client, state, _, transport = live_app
    game_id = _start(client)
    source: Any = resolve_prompt_suites().shared
    trace: Any = client.get(f"/api/live-traces/{game_id}").json
    assert trace["config"]["shared_suite"]["source"] == source.source
    assert trace["config"]["shared_suite"]["sha256"] == source.sha256
    assert trace["config"]["decision_suite"] is None
    stepped: Any = client.post("/api/step")
    assert stepped.status_code == 200, stepped.json
    sandbox: Any = state.current_sandbox
    red: Any = sandbox.players[Color.RED]
    assert red.session.strategic_memory == "accepted action notes"
    assert red.session.memory_revision == 1
    assert red.session.messages == []
    traces: Any = stepped.json["reasoning_traces"]
    assert traces[0]["accepted"] is True
    assert traces[0]["notes_update"] == red.session.strategic_memory
    assert traces[0]["context_policy"] == "fresh_notes"
    assert sum(item["call_kind"] == "communication" for item in traces) == 0
    assert all(item["accepted"] for item in traces)
    assert all(item.context_policy == "fresh_notes" for item in transport.requests)
    expected: Any = sandbox.snapshot()
    changed: Any = save_shared_prompt_override(
        source=source.source.replace("{{ resources }}", "EDITED RESOURCE HEADING: {{ resources }}"),
        expected_sha256=source.sha256,
    )
    loaded: Any = client.post(f"/api/live-traces/{game_id}/load")
    assert loaded.status_code == 200, loaded.json
    assert loaded.json["state"]["live_inference"]["shared_suite"]["sha256"] == changed.shared.sha256
    assert live_sandbox(state).snapshot().player_states == expected.player_states
    assert resolve_prompt_suites() == changed
    restored: Any = live_sandbox(state).players[Color.RED]
    assert restored.suite.context.memory_mode == restored.communication_suite.memory_mode == "fresh_notes"
    assert restored.suite.components["resources"].template == parse_shared_prompt_suite(changed.shared.source).components["resources"].template
    assert changed.shared.sha256 != source.sha256
    continued: Any = client.post("/api/step")
    assert continued.status_code == 200, continued.json
    assert continued.json["trace_step_index"] == 1
    assert len(continued.json["reasoning_traces"]) == 1
    assert restored.session.messages == []


def test_live_editor_changes_next_action_and_speech_without_rewriting_history(live_app: LiveApp) -> None:
    client, state, _, transport = live_app
    client.application.register_blueprint(prompt_routes.prompt_suite_bp)
    game_id: Any = _start(client)
    assert client.post("/api/step").status_code == 200
    history: Any = deepcopy(state.live_trace_store.get_game(game_id))
    old_requests = deepcopy(transport.requests)
    sandbox: Any = state.current_sandbox
    snapshot: Any = sandbox.snapshot()
    editor: Any = client.get("/api/prompt-suite").json
    assert not editor["saving_locked"]
    document: Any = editor["shared"]["document"]
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
    calls: Any = transport.requests[len(old_requests):]
    assert {call.channel for call in calls} == {"action", "talk"}
    for call in calls:
        assert "NEXT REQUEST BOARD" in "\n".join(message.content for message in call.messages)
        assert call.prompt_sources[0].sha256 == saved.json["shared"]["sha256"]
    assert transport.requests[:len(old_requests)] == old_requests
    after: Any = state.live_trace_store.get_game(game_id)
    assert after["model_calls"][:len(history["model_calls"])] == history["model_calls"]
    assert after["steps"][0] == history["steps"][0]
    assert after["config"] == history["config"]
    durable: Any = deepcopy(after["model_calls"])
    before_load: Any = sandbox.snapshot()
    loaded: Any = client.post(f"/api/live-traces/{game_id}/load")
    assert loaded.status_code == 200
    assert live_sandbox(state).snapshot().player_states == before_load.player_states
    assert loaded.json["state"]["game"] == after["steps"][-1]["public_state"]["game"]
    restored = live_sandbox(state).snapshot().engine
    assert restored.events == before_load.engine.events
    assert restored.state.player_state == before_load.engine.state.player_state
    assert restored.state.resource_freqdeck == before_load.engine.state.resource_freqdeck
    assert restored.state.development_listdeck == before_load.engine.state.development_listdeck
    assert restored.state.rng.getstate() == before_load.engine.state.rng.getstate()
    assert state.live_trace_store.get_game(game_id)["model_calls"] == durable


@pytest.mark.parametrize("notes_limit", [500, 3])
def test_inflight_edit_keeps_old_admission_and_updates_next_actual_boundary(live_app: LiveApp, monkeypatch: pytest.MonkeyPatch, notes_limit: int) -> None:
    client, state, _, transport = live_app
    client.application.register_blueprint(prompt_routes.prompt_suite_bp)
    game_id = _start(client)
    editor: Any = client.get("/api/prompt-suite").json
    entered, release, published = Event(), Event(), Event()
    original_complete = transport.complete
    original_save = prompt_routes.save_shared_prompt_override

    async def paused(request: ModelRequest) -> ModelResponse:
        if request.channel == "action":
            entered.set()
            assert release.wait(5)
        return await original_complete(request)

    def save_and_signal(**kwargs: object) -> ActivePromptSuites:
        result: Any = original_save(**kwargs)
        published.set()
        return result

    monkeypatch.setattr(transport, "complete", paused)
    monkeypatch.setattr(prompt_routes, "save_shared_prompt_override", save_and_signal)
    document: Any = editor["shared"]["document"]
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
    assert len(live_sandbox(state).game_engine.state.actions) == 1
    following: Any = client.post("/api/step")
    if notes_limit == 3:
        # The old request's notes exceed the NEW limit. They must still commit
        # under the original contract before the next boundary rejects rebinding.
        assert following.status_code == 409
        assert following.json["action_applied"] is False
        assert "notes must not exceed 3" in following.json["details"]
        assert live_sandbox(state).players[Color.RED].session.strategic_memory == "accepted action notes"
    else:
        assert following.status_code == 200
        assert transport.requests[-1].prompt_sources[0].sha256 == saved.json["shared"]["sha256"]
    stored: Any = state.live_trace_store.get_game(game_id)["model_calls"]
    assert stored[0]["request"]["prompt_sources"][0]["sha256"] == editor["shared"]["sha256"]
