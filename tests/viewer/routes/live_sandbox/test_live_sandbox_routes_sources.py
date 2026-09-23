"""Saved games use the active source while retaining the recorded one."""
from pathlib import Path
from typing import Any

import pytest
from flask import Flask

from cle.game_engine.models.player import Color
from cle.harness import CompletionTransport, ModelResponse, default_suite_path
from cle.sandbox import CatanSandbox
from cle.sandbox.factory import LiveSandboxConfig
from cle.sandbox.replay import ReplaySandbox
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



@pytest.mark.parametrize("version", ["8.0.0", "10.0.0"])
def test_saved_game_uses_active_source_but_retains_recorded_source(
    live_app: tuple[Flask, ServerState, DummySocket],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    version: str,
) -> None:
    app, state, _ = live_app
    source_path = tmp_path / "active-decision.yaml"
    original_source: Any = (
        default_suite_path()
        .with_name(f"catan_v{version.split('.')[0]}.yaml")
        .read_text(encoding="utf-8")
        .replace(
            "Resolve the required discard.",
            "ORIGINAL RECORDED DISCARD GUIDANCE.",
        )
    )
    source_path.write_text(original_source, encoding="utf-8")
    monkeypatch.setenv("CATAN_CONTEXT_SUITE", str(source_path))
    monkeypatch.setattr(
        "cle.sandbox.factory.create_text_transport",
        lambda config: FixedTransport(response=ModelResponse(content="<action>0</action>")),
    )
    client = app.test_client()

    started: Any = client.post(
        "/api/start-game",
        json={
            "mode": "llm_vs_random",
            "seed": 13,
            "shuffle_players": False,
            "palette": "canonical_four",
        },
    )
    game_id = started.json["trace_game_id"]
    source_path.write_text(
        original_source.replace(
            "ORIGINAL RECORDED DISCARD GUIDANCE.",
            "NEW GLOBAL DISCARD GUIDANCE.",
        ),
        encoding="utf-8",
    )
    replacement = client.post(
        "/api/start-game",
        json={
            "mode": "random",
            "seed": 14,
            "palette": "canonical_four",
        },
    )
    loaded = client.post(f"/api/live-traces/{game_id}/load")

    assert started.status_code == 200
    assert replacement.status_code == 200
    assert loaded.status_code == 200, loaded.get_json()
    restored: Any = live_sandbox(state).players[Color.RED]
    assert restored.suite.version == version
    assert restored.suite.response.format == "xml"
    assert restored.suite.phase_guidance["discarding"].startswith(
        "NEW GLOBAL DISCARD GUIDANCE."
    )
    assert ("Maximizing raw pip count is not the objective" in (
        restored.suite.phase_guidance["initial_settlement_1"]
    )) == (version == "10.0.0")
    trace: Any = client.get(f"/api/live-traces/{game_id}").json
    assert trace["config"]["decision_suite"]["version"] == version
    assert trace["config"]["decision_suite"]["source"] == original_source
    stepped = client.post("/api/step")
    assert stepped.status_code == 200, stepped.get_json()
    assert next(iter(restored.session.receipts.values())).choice.action_index == 0


def test_llm_live_route_preserves_explicit_reasoning_off(live_app: tuple[Flask, ServerState, DummySocket], monkeypatch: pytest.MonkeyPatch) -> None:
    app, state, socket = live_app
    transport = FixedTransport()
    configs: list[LiveSandboxConfig] = []

    def create_transport(config: LiveSandboxConfig) -> CompletionTransport:
        configs.append(config)
        return transport

    monkeypatch.setattr(
        "cle.sandbox.factory.create_text_transport",
        create_transport,
    )

    client = app.test_client()
    response: Any = client.post(
        "/api/start-game",
        json={
            "mode": "llm_vs_random",
            "reasoning": {"enabled": False},
            "shuffle_players": False,
        },
    )

    assert response.status_code == 200
    assert response.get_json()["reasoning_request"] == {"enabled": False}
    assert dict(configs[0].reasoning) == {"enabled": False}
    game_id: Any = response.json["trace_game_id"]
    stored_config: Any = state.live_trace_store.get_game(game_id)["config"]

    loaded: Any = client.post(f"/api/live-traces/{game_id}/load")

    assert loaded.status_code == 200, loaded.get_json()
    assert loaded.json["reasoning_request"] == {"enabled": False}
    assert loaded.json["max_tokens"] is None
    assert loaded.json["max_decision_attempts"] == 3
    assert loaded.json["state"]["live_inference"]["reasoning"] == {"enabled": False}
    assert socket.emissions[-1][1] == loaded.json["state"]
    assert dict(configs[-1].reasoning) == {"enabled": False}
    assert transport.requests == []
    assert state.live_trace_store.get_game(game_id)["config"] == stored_config
