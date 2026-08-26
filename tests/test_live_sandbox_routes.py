from dataclasses import dataclass, field

import pytest
from flask import Flask

from cle.harness import ModelResponse
from cle.sandbox.factory import LiveSandboxConfig, create_text_transport
from cle.traces import SQLiteLiveTraceStore
from game_engine.models.player import Color
from playground.game_viewer.routes.live_game import live_game_bp
from playground.game_viewer.state import ServerState


@dataclass
class DummySocket:
    emissions: list = field(default_factory=list)

    def emit(self, event, payload):
        self.emissions.append((event, payload))


@dataclass
class FixedTransport:
    requests: list = field(default_factory=list)

    async def complete(self, request):
        self.requests.append(request)
        return ModelResponse(
            content=(
                "<game_plan>build a balanced opening</game_plan>"
                "<rationale>take the first exact legal option</rationale>"
                "<action>0</action>"
            ),
            model="test/model",
            finish_reason="stop",
            native_reasoning="native live analysis",
            native_reasoning_details=({"type": "reasoning.text"},),
            reasoning_request=(("effort", "xhigh"), ("exclude", False)),
            provider_response_id="gen-live-test",
            provider_request_id="req-live-test",
            provider_native_finish_reason="stop",
        )


@pytest.fixture
def live_app(tmp_path):
    app = Flask(__name__)
    state = ServerState()
    state.live_trace_store = SQLiteLiveTraceStore(tmp_path / "live.sqlite3")
    socket = DummySocket()
    app.config["SERVER_STATE"] = state
    app.config["SOCKETIO"] = socket
    app.register_blueprint(live_game_bp)
    return app, state, socket


def test_random_live_routes_delegate_ticks_to_owned_sandbox(live_app):
    app, state, socket = live_app
    client = app.test_client()

    started = client.post(
        "/api/start-game",
        json={
            "mode": "random",
            "name": "  First opening  ",
            "seed": 5,
            "shuffle_players": False,
        },
    )

    assert started.status_code == 200
    assert started.json["state"]["running"] is True
    assert started.json["state"]["game"] is not None
    trace_game_id = started.json["trace_game_id"]
    assert trace_game_id
    assert started.json["trace_display_name"] == "First opening"
    assert started.json["trace_database"].endswith("live.sqlite3")
    assert state.current_sandbox is not None
    assert set(state.current_sandbox.players) == set(
        state.current_sandbox.game_engine.state.colors
    )
    assert all(
        player.status()["kind"] == "first_legal"
        for player in state.current_sandbox.players.values()
    )

    stepped = client.post("/api/step")

    assert stepped.status_code == 200
    assert stepped.json["state"]["running"] is True
    assert stepped.json["reasoning_traces"] == []
    assert stepped.json["trace_game_id"] == trace_game_id
    assert stepped.json["trace_step_index"] == 0
    assert len(stepped.json["state"]["events"]) == 1
    assert state.current_sandbox.revision == 1
    assert len(state.current_sandbox.game_engine.state.actions) == 1
    assert socket.emissions == []
    payload = stepped.json["state"]
    assert payload["sandbox_players"]["RED"]["kind"] == "first_legal"
    assert "llm_thinking" not in payload
    assert payload["all_player_resources"]["RED"] == {"TOTAL": 0}
    assert all(
        not key.endswith("_IN_HAND")
        for key in payload["game"]["player_state"]
    )
    renamed = client.patch(
        f"/api/live-traces/{trace_game_id}",
        json={"name": "Opening experiment"},
    )
    assert renamed.status_code == 200
    assert renamed.json["display_name"] == "Opening experiment"
    trace_list = client.get("/api/live-traces").json
    assert trace_list["games"][0]["game_id"] == trace_game_id
    assert trace_list["games"][0]["display_name"] == "Opening experiment"
    trace = client.get(f"/api/live-traces/{trace_game_id}").json
    assert trace["step_count"] == 1
    assert trace["steps"][0]["after_revision"] == 1
    assert len(trace["events"]) == 1

    replacement = client.post(
        "/api/start-game",
        json={"mode": "random", "seed": 6, "shuffle_players": False},
    )
    assert replacement.status_code == 200
    assert replacement.json["trace_game_id"] != trace_game_id
    loaded = client.post(f"/api/live-traces/{trace_game_id}/load")
    assert loaded.status_code == 200
    assert loaded.json["trace_game_id"] == trace_game_id
    assert loaded.json["trace_display_name"] == "Opening experiment"
    assert loaded.json["loaded_step_index"] == 0
    assert loaded.json["state"]["running"] is True
    assert state.current_sandbox.revision == 1

    continued = client.post("/api/step")
    assert continued.status_code == 200
    assert continued.json["trace_game_id"] == trace_game_id
    assert continued.json["trace_step_index"] == 1
    assert state.current_sandbox.revision == 2
    active_sandbox = state.current_sandbox
    checkpoint = client.get(
        f"/api/live-traces/{trace_game_id}/steps/0"
    )
    assert checkpoint.status_code == 200
    assert checkpoint.json["latest_step_index"] == 1
    assert checkpoint.json["step"]["after_revision"] == 1
    assert len(checkpoint.json["step"]["public_state"]["events"]) == 1
    assert checkpoint.json["model_calls"] == []
    assert state.current_sandbox is active_sandbox
    assert state.current_sandbox.revision == 2
    assert client.get(
        f"/api/live-traces/{trace_game_id}/steps/99"
    ).status_code == 404
    assert client.post("/api/live-traces/missing/load").status_code == 404
    assert client.patch(
        f"/api/live-traces/{trace_game_id}",
        json={"name": "x" * 81},
    ).status_code == 400
    assert client.post("/api/auto-play").status_code == 404


def test_live_step_serializes_color_payload_without_websocket(live_app):
    app, state, socket = live_app
    client = app.test_client()
    started = client.post(
        "/api/start-game",
        json={"mode": "random", "seed": 19, "shuffle_players": False},
    )

    assert started.status_code == 200
    for _ in range(31):
        stepped = client.post("/api/step")
        assert stepped.status_code == 200, stepped.get_json()
        assert stepped.content_type == "application/json"

    assert stepped.json["state"]["events"][-1]["payload"][0] in {
        "RED",
        "BLUE",
        "WHITE",
        "ORANGE",
    }
    assert state.current_sandbox.revision == 31
    assert socket.emissions == []


def test_unsupported_transport_cannot_fake_native_reasoning(monkeypatch):
    monkeypatch.setenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")

    with pytest.raises(ValueError, match="does not implement"):
        create_text_transport(
            LiveSandboxConfig(
                mode="llm",
                reasoning={"effort": "high", "exclude": False},
            )
        )


def test_live_factory_accepts_snapshotted_trade_and_communication_limits(live_app):
    app, state, _ = live_app
    response = app.test_client().post(
        "/api/start-game",
        json={
            "mode": "random",
            "trade_limits": {"max_active_root_offers": 2},
            "communication_limits": {"recent_message_window": 4},
        },
    )

    assert response.status_code == 200
    assert response.json["trade_limits"]["max_active_root_offers"] == 2
    assert response.json["communication_limits"]["recent_message_window"] == 4
    engine = state.current_sandbox.game_engine
    assert engine.state.trade_limits.max_active_root_offers == 2
    assert engine.communication_limits.recent_message_window == 4


def test_llm_live_route_uses_yaml_agent_player_without_legacy_llm_player(
    live_app,
    monkeypatch,
):
    app, state, _ = live_app
    transport = FixedTransport()
    configs = []

    def create_transport(config):
        configs.append(config)
        return transport

    monkeypatch.setattr(
        "cle.sandbox.factory.create_text_transport",
        create_transport,
    )
    client = app.test_client()

    started = client.post(
        "/api/start-game",
        json={"mode": "llm_vs_random", "seed": 9, "shuffle_players": False},
    )
    stepped = client.post("/api/step")

    assert started.status_code == 200
    assert stepped.status_code == 200
    red_player = state.current_sandbox.players[
        state.current_sandbox.game_engine.state.colors[0]
    ]
    assert red_player.status()["kind"] == "agent"
    assert red_player.status()["messages"] == 2
    assert [message.role for message in transport.requests[0].messages] == [
        "system",
        "user",
    ]
    user_message = transport.requests[0].messages[-1].content
    assert "CURRENT GAME STATE:" in user_message
    assert user_message.count("VALID ACTIONS:") == 1
    assert "<valid_actions>" not in user_message
    assert dict(configs[0].reasoning) == {"effort": "xhigh", "exclude": False}
    receipt = next(iter(red_player.session.receipts.values()))
    assert receipt.choice.rationale == "take the first exact legal option"
    assert receipt.choice.native_reasoning == "native live analysis"
    assert receipt.choice.native_reasoning_details == ({"type": "reasoning.text"},)
    assert dict(receipt.choice.reasoning_request) == {
        "effort": "xhigh",
        "exclude": False,
    }
    trace = stepped.json["reasoning_traces"][0]
    assert trace["schema"] == "live-reasoning-trace-v1"
    assert trace["rationale"] == "take the first exact legal option"
    assert trace["rationale_source"] == "model_response_xml"
    assert trace["native_reasoning"] == "native live analysis"
    assert trace["native_reasoning_source"] == "provider_response"
    assert trace["native_reasoning_returned"] is True
    assert trace["finish_reason"] == "stop"
    assert trace["provider_native_finish_reason"] == "stop"
    assert trace["provider_response_id"] == "gen-live-test"
    assert trace["provider_request_id"] == "req-live-test"
    stored = client.get(
        f"/api/live-traces/{started.json['trace_game_id']}"
    ).json
    assert stored["model_calls"][0]["call_kind"] == "decision"
    assert stored["model_calls"][0]["response"]["provider_response_id"] == (
        "gen-live-test"
    )
    assert stored["model_calls"][0]["request"]["messages"][-1]["role"] == (
        "user"
    )
    checkpoint = client.get(
        f"/api/live-traces/{started.json['trace_game_id']}/steps/0"
    ).json
    assert checkpoint["model_calls"][0]["accepted"] is True
    assert checkpoint["model_calls"][0]["choice"]["rationale"] == (
        "take the first exact legal option"
    )
    assert checkpoint["model_calls"][0]["response"]["native_reasoning"] == (
        "native live analysis"
    )
    assert checkpoint["model_calls"][0]["request"]["messages"][-1]["role"] == (
        "user"
    )
    loaded = client.post(
        f"/api/live-traces/{started.json['trace_game_id']}/load"
    )
    assert loaded.status_code == 200, loaded.get_json()
    restored_red = state.current_sandbox.players[Color.RED]
    assert restored_red.status()["messages"] == 2
    continued = None
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
    assert state.current_sandbox.game_engine.state.colors[0] == Color.RED


def test_llm_live_route_preserves_explicit_reasoning_off(live_app, monkeypatch):
    app, _, _ = live_app
    transport = FixedTransport()
    configs = []

    def create_transport(config):
        configs.append(config)
        return transport

    monkeypatch.setattr(
        "cle.sandbox.factory.create_text_transport",
        create_transport,
    )

    response = app.test_client().post(
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
