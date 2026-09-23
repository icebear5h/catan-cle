"""The LLM live route uses the pinned agent without the legacy player."""
import json
import pickle
from hashlib import sha256
from typing import Any

import pytest
from flask import Flask

from cle.game_engine.models.player import Color
from cle.harness import CompletionTransport, default_suite_path
from cle.harness.communication import default_communication_suite_path
from cle.sandbox import CatanSandbox
from cle.sandbox.factory import (
    DEFAULT_LIVE_MODEL,
    LiveSandboxConfig,
)
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



def test_llm_live_route_uses_pinned_v11_agent_without_legacy_llm_player(
    live_app: tuple[Flask, ServerState, DummySocket],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, state, _ = live_app
    transport = FixedTransport()
    configs: list[LiveSandboxConfig] = []
    monkeypatch.delenv("CATAN_LLM_MODEL", raising=False)

    def create_transport(config: LiveSandboxConfig) -> CompletionTransport:
        configs.append(config)
        return transport

    monkeypatch.setattr(
        "cle.sandbox.factory.create_text_transport",
        create_transport,
    )
    client: Any = app.test_client()

    started: Any = client.post(
        "/api/start-game",
        json={
            "mode": "llm_vs_random",
            "seed": 9,
            "shuffle_players": False,
            "palette": "canonical_four",
        },
    )
    stepped: Any = client.post("/api/step")

    assert started.status_code == 200
    assert started.json["model"] == DEFAULT_LIVE_MODEL
    assert started.json["board_surface"] == "indexed_tile_rows"
    assert configs[0].model == DEFAULT_LIVE_MODEL
    assert configs[0].board_surface == "indexed_tile_rows"
    assert stepped.status_code == 200
    red_player: Any = live_sandbox(state).players[live_sandbox(state).game_engine.state.colors[0]]
    assert red_player.suite.response.format == "json"
    assert red_player.status()["kind"] == "agent"
    assert red_player.status()["messages"] == 2
    assert [message.role for message in transport.requests[0].messages] == [
        "system",
        "user",
    ]
    user_message = transport.requests[0].messages[-1].content
    assert "BOARD STATE:" in user_message
    menu = next(
        component.value for component in transport.requests[0].components
        if component.id == "environment.legal_actions"
    )
    assert "build_settlement" in menu
    assert "<N00>" in menu
    assert "action_index" not in user_message
    assert "<action>" not in user_message
    assert "<game_plan>" not in user_message
    assert "<valid_actions>" not in user_message
    assert "rationale" not in user_message.lower()
    assert "STRATEGIC OBJECTIVE FOR THIS PROBE:" in user_message
    assert "Maximizing raw pip count is not the objective" in user_message
    assert "opening archetype" in (
        user_message
    )
    assert "Round 1 (first settlement + road): RED -> BLUE -> WHITE -> ORANGE" in (
        user_message
    )
    assert dict(configs[0].reasoning) == {"effort": "high", "exclude": False}
    assert configs[0].max_tokens is None
    assert started.json["max_tokens"] is None
    assert started.json["state"]["live_inference"] == {
        "max_decision_attempts": 3,
        "max_tokens": None,
        "model": DEFAULT_LIVE_MODEL,
        "reasoning": {"effort": "high", "exclude": False},
    }
    # The receipt keeps the decision; the raw call and its reasoning are
    # recorded once, on the accepted model call in the trace store.
    receipt = next(iter(red_player.session.receipts.values()))
    assert receipt.choice.action_index == 0
    assert receipt.choice.rationale == ""
    assert receipt.choice.raw_response == ""
    assert receipt.choice.native_reasoning == ""
    accepted_call = next(
        call for call in client.get(
            f"/api/live-traces/{started.json['trace_game_id']}/steps/0"
        ).json["model_calls"]
        if call["accepted"]
    )
    assert json.loads(accepted_call["response"]["content"]) == {
        "game_plan": "build a balanced opening",
        "tool": "build_settlement",
        "arguments": {"node": "<N00>"},
    }
    assert accepted_call["response"]["native_reasoning"] == "native live analysis"
    assert accepted_call["response"]["native_reasoning_details"] == [{"type": "reasoning.text"}]
    assert accepted_call["response"]["reasoning_request"] == {
        "effort": "high",
        "exclude": False,
    }
    trace = stepped.json["reasoning_traces"][0]
    assert trace["schema"] == "live-reasoning-trace-v2"
    assert "rationale" not in trace
    assert "rationale_source" not in trace
    assert trace["native_reasoning"] == "native live analysis"
    assert trace["native_reasoning_source"] == "provider_response"
    assert trace["native_reasoning_returned"] is True
    assert trace["finish_reason"] == "stop"
    assert trace["provider_native_finish_reason"] == "stop"
    assert trace["provider_response_id"] == "gen-live-test"
    assert trace["provider_request_id"] == "req-live-test"
    stored: Any = client.get(f"/api/live-traces/{started.json['trace_game_id']}").json
    assert stored["model_calls"][0]["call_kind"] == "decision"
    assert stored["model_calls"][0]["response"]["provider_response_id"] == ("gen-live-test")
    assert stored["model_calls"][0]["request"]["messages"][-1]["role"] == ("user")
    assert stored["model_calls"][0]["request"]["components"][0]["id"] == (
        "system.identity"
    )
    assert stored["config"]["reasoning"] == {
        "effort": "high",
        "exclude": False,
    }
    assert stored["config"]["max_tokens"] is None
    assert stored["config"]["decision_suite"]["id"] == "catan-agent"
    assert stored["config"]["decision_suite"]["version"] == "11.0.0"
    assert stored["config"]["shared_suite"] is None
    for name, path in (
        ("decision_suite", default_suite_path()),
        ("communication_suite", default_communication_suite_path()),
    ):
        source: Any = path.read_text(encoding="utf-8")
        assert stored["config"][name]["source"] == source
        assert stored["config"][name]["sha256"] == sha256(source.encode("utf-8")).hexdigest()
    assert "environment.board_state" in {
        component["id"]
        for component in stored["model_calls"][0]["request"]["components"]
    }
    checkpoint: Any = client.get(f"/api/live-traces/{started.json['trace_game_id']}/steps/0").json
    assert checkpoint["model_calls"][0]["accepted"] is True
    assert "rationale" not in checkpoint["model_calls"][0]["choice"]
    assert checkpoint["model_calls"][0]["response"]["native_reasoning"] == ("native live analysis")
    assert checkpoint["model_calls"][0]["request"]["messages"][-1]["role"] == ("user")
    receipts_before: Any = pickle.dumps(red_player.session.receipts)
    loaded = client.post(f"/api/live-traces/{started.json['trace_game_id']}/load")
    assert loaded.status_code == 200, loaded.get_json()
    restored_red: Any = live_sandbox(state).players[Color.RED]
    assert restored_red.session.context_policy == "legacy"
    assert pickle.dumps(restored_red.session.receipts) == receipts_before
    assert restored_red.status()["messages"] == 2
    continued: Any = None
    for _ in range(8):
        continued = client.post("/api/step")
        assert continued.status_code == 200
        if restored_red.status()["messages"] == 4:
            break
    assert continued is not None
    assert continued.json["trace_step_index"] >= 1
    assert restored_red.status()["messages"] == 4
    assert "observation" not in trace
    assert "available_actions" not in trace
    assert not hasattr(state, "llm_thinking")
    assert live_sandbox(state).game_engine.state.colors[0] == Color.RED
