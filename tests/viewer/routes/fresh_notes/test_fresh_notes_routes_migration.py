"""Context-mode migration and notes-limit rebinds stay atomic."""
import pickle
from copy import deepcopy
from typing import Any

import pytest

from cle.game_engine.models.player import Color
from cle.harness.communication import default_communication_suite_path
from cle.harness.prompt_store import (
    resolve_prompt_suites,
    save_shared_prompt_override,
)
from cle.harness.shared_suite import parse_shared_prompt_suite
from cle.harness.suite import default_suite_path
from cle.sandbox import CatanSandbox
from cle.sandbox.factory import (
    ActivePromptConfigurationError,
    LiveSandboxConfig,
)
from cle.sandbox.replay import ReplaySandbox
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



def test_context_mode_migration_preserves_notes_history_and_channel_events(live_app: LiveApp, monkeypatch: pytest.MonkeyPatch) -> None:
    client, state, _, transport = live_app
    _start(client)
    assert client.post("/api/step").status_code == 200
    sandbox: Any = state.current_sandbox
    before = sandbox.snapshot()
    monkeypatch.setenv("CATAN_CONTEXT_SUITE", str(default_suite_path()))
    monkeypatch.setenv("CATAN_COMMUNICATION_SUITE", str(default_communication_suite_path()))
    sandbox._refresh_inference_policy()
    for color, prior in before.player_states:
        session: Any = sandbox.players[color].session
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
    legacy: Any = sandbox.snapshot()
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


def test_incompatible_notes_limit_rebind_is_atomic_and_recoverable(live_app: LiveApp) -> None:
    client, state, _, transport = live_app
    _start(client)
    assert client.post("/api/step").status_code == 200
    sandbox: Any = state.current_sandbox
    last = list(sandbox.players.values())[-1]
    last.session.strategic_memory = "Retain these private notes exactly"
    before = pickle.dumps(sandbox.snapshot())
    players: Any = dict(sandbox.players)
    active: Any = resolve_prompt_suites().shared
    bundle: Any = parse_shared_prompt_suite(active.source)
    changed: Any = save_shared_prompt_override(
        source=bundle.model_copy(update={"max_notes_chars": 3}).model_dump_json(),
        expected_sha256=active.sha256,
    )
    with pytest.raises(ActivePromptConfigurationError, match="notes must not exceed"):
        sandbox._refresh_inference_policy()
    assert sandbox.players == players
    assert pickle.dumps(sandbox.snapshot()) == before
    count = len(transport.requests)
    failed: Any = client.post("/api/step")
    assert failed.status_code == 409
    assert not failed.json["action_applied"]
    assert len(transport.requests) == count
    assert pickle.dumps(sandbox.snapshot()) == before
    save_shared_prompt_override(source=active.source, expected_sha256=changed.shared.sha256)
    assert client.post("/api/step").status_code == 200


def test_legacy_saved_game_load_uses_active_shared_config_and_keeps_notes(live_app: LiveApp) -> None:
    client, state, _, transport = live_app
    transport.legacy = True
    started: Any = client.post("/api/start-game", json={
        "mode": "llm", "model": "historical/model", "seed": 7,
        "palette": "canonical_four", "shuffle_players": False,
        "context_suite_path": str(default_suite_path()),
        "communication_suite_path": str(default_communication_suite_path()),
    })
    assert started.status_code == 200
    game_id: Any = started.json["trace_game_id"]
    assert client.post("/api/step").status_code == 200
    before: Any = live_sandbox(state).snapshot()
    history: Any = deepcopy(state.live_trace_store.get_game(game_id))
    # Simulate a process with a different explicit active selection, not a game
    # reset: loading must use this model and shared defaults, not saved paths.
    state.active_live_config = LiveSandboxConfig(model="current/model", reasoning={"enabled": False})
    transport.legacy = False
    loaded: Any = client.post(f"/api/live-traces/{game_id}/load")
    assert loaded.status_code == 200, loaded.json
    assert loaded.json["model"] == "current/model"
    assert loaded.json["state"]["live_inference"]["context_policy"] == "fresh_notes"
    for color, old in before.player_states:
        player: Any = live_sandbox(state).players[color]
        assert player.session.strategic_memory == old.session.strategic_memory
        assert tuple(player.session.messages) == old.session.messages
        assert player.session.action_next_sequence == player.session.talk_next_sequence == 0
        assert player.session.memory_revision == old.session.memory_revision + 1
    assert tuple(live_sandbox(state).game_engine.events) == before.engine.events
    assert client.post("/api/step").status_code == 200
    after: Any = state.live_trace_store.get_game(game_id)
    assert after["config"] == history["config"]
    assert after["model_calls"][:len(history["model_calls"])] == history["model_calls"]
    assert all(call["request"]["prompt_sources"][0]["kind"] == "shared"
               for call in after["model_calls"][len(history["model_calls"]):])
