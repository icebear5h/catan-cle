"""Replay previews resolve once and never promote notes or reset sources."""
import asyncio
import json
import pickle
from typing import Any

import pytest

from cle.game_engine.models.player import Color
from cle.harness.communication import default_communication_suite_path
from cle.harness.prompt_store import (
    ActivePromptSuites,
    load_active_prompt_suites,
    resolve_prompt_suites,
    save_prompt_suite_overrides,
)
from cle.harness.suite import default_suite_path
from cle.sandbox import CatanSandbox
from cle.sandbox.replay import ReplaySandbox
from playground.game_viewer.replay import decision_preview
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



@pytest.mark.parametrize("legacy", [False, True])
def test_replay_preview_resolves_once_and_never_promotes_notes_or_resets_sources(live_app: LiveApp, monkeypatch: pytest.MonkeyPatch, legacy: bool) -> None:
    client, state, _, transport = live_app
    _start(client)
    live: Any = state.current_sandbox
    engine: Any = live.game_engine
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
        pair: Any = load_active_prompt_suites()
        save_prompt_suite_overrides(
            decision_source=pair.decision.source, communication_source=pair.communication.source,
            expected_decision_sha256=pair.decision.sha256,
            expected_communication_sha256=pair.communication.sha256,
        )
        transport.legacy = True
    sources: Any = resolve_prompt_suites()
    state.replay_mode = True
    state.replay_data = {"game_id": str(engine.id), "parsed_actions": [{}] * 4}
    state.replay_index = 1
    state.current_sandbox = ReplaySandbox(state, engine, players=live.players)
    before: Any = pickle.dumps(live.snapshot())
    resolutions: list[bool] = []

    def resolve_once() -> ActivePromptSuites:
        resolutions.append(True)
        return resolve_prompt_suites()

    monkeypatch.setattr(decision_preview, "resolve_prompt_suites", resolve_once)
    for _ in range(2):
        response: Any = client.post("/api/replay-llm-response", json={
            "model": "test/local", "reasoning": {"enabled": False},
            "game_plan": "EXPLICIT_HISTORICAL_PLAN",
        })
        assert response.status_code == 200, response.json
        result: Any = response.json
        assert result["parse_error"] is None
        assert result["accepted"] is False
        assert result["stale"] is False
        assert pickle.dumps(live.snapshot()) == before
        assert state.replay_index == 1
        assert resolve_prompt_suites() == sources
        messages: Any = "\n".join(message["content"] for message in result["model_messages"])
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
def test_replay_explicit_notes_require_matching_own_seat(live_app: LiveApp, owner: Color | None) -> None:
    client, state, _, transport = live_app
    _start(client)
    live: Any = state.current_sandbox
    state.replay_data = {"game_id": live.game_engine.id, "parsed_actions": [{}]}
    replay: Any = ReplaySandbox(state, live.game_engine, players=live.players)
    arguments: Any = dict(
        model="test/local", game_plan="legacy", reasoning_request={"enabled": False},
        transport_factory=lambda **config: transport, notes_seed="EXPLICIT_OWN_NOTES",
        notes_player=owner,
    )
    before: Any = live.snapshot().player_states
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


def test_explicit_historical_factory_pair_remains_legacy(live_app: LiveApp, monkeypatch: pytest.MonkeyPatch) -> None:
    client, state, _, transport = live_app
    monkeypatch.setenv("CATAN_CONTEXT_SUITE", str(default_suite_path()))
    monkeypatch.setenv("CATAN_COMMUNICATION_SUITE", str(default_communication_suite_path()))
    transport.legacy = True
    game_id: Any = _start(client)
    response = client.post("/api/step")
    assert response.status_code == 200, response.json
    config: Any = state.live_trace_store.get_game(game_id)["config"]
    assert config["shared_suite"] is None
    assert config["decision_suite"]["version"] == "11.0.0"
    assert live_sandbox(state).players[Color.RED].session.context_policy == "legacy"
