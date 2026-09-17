import asyncio
from dataclasses import dataclass, field, replace
from hashlib import sha256
import json
import pickle
import re
import ssl

import httpx
import pytest
from flask import Flask

from cle.harness import ModelResponse, default_suite_path
from cle.harness.communication import default_communication_suite_path
from cle.harness.providers import OpenRouterConfig, OpenRouterTransport
from cle.harness.providers.openrouter import OpenRouterHTTPFailure, OpenRouterTLSFailure
from cle.players.contracts import CommunicationChoice, CommunicationMode, PlayerAttempt, PlayerChoice
from cle.sandbox.catan import PlayerResponseError
from cle.sandbox.communication import CommunicationAdmission, CommunicationOpportunity, ReactionReason
from cle.sandbox.factory import (
    DEFAULT_LIVE_MAX_DECISION_ATTEMPTS,
    DEFAULT_LIVE_MODEL,
    DEFAULT_LIVE_REASONING_EFFORT,
    LiveSandboxConfig,
    create_live_sandbox,
    create_text_transport,
    resolve_live_model,
)
from cle.traces import SQLiteLiveTraceStore
from cle.game_engine.events import PlayerEvent
from cle.game_engine.models.player import Color
from cle.game_engine.state import ensure_trade_window
from cle.game_engine.trading import TradeOffer
from playground.game_viewer.async_runtime import sandbox_async_runtime
from playground.game_viewer.live.game_logging import log_game_event
from playground.game_viewer.routes.health import health_bp
from playground.game_viewer.routes.live_game import (
    _config_from_stored_payload,
    live_game_bp,
)
from playground.game_viewer.state import ServerState


@dataclass
class DummySocket:
    emissions: list = field(default_factory=list)

    def emit(self, event, payload):
        self.emissions.append((event, payload))


def _opening_call(request, plan):
    menu = next(
        component.value for component in request.components
        if component.id == "environment.legal_actions"
    )
    if "build_settlement" in menu:
        tool, parameter, pattern = "build_settlement", "node", r"<N\d{2}>"
    else:
        assert "build_road" in menu
        tool, parameter, pattern = "build_road", "edge", r"<E\d{2}_\d{2}>"
    token = re.search(pattern, menu)
    assert token is not None, menu
    return json.dumps({
        "game_plan": plan, "tool": tool, "arguments": {parameter: token.group()},
    })


@dataclass
class FixedTransport:
    requests: list = field(default_factory=list)
    response: ModelResponse | None = None

    async def complete(self, request):
        self.requests.append(request)
        if self.response is not None:
            return self.response
        return ModelResponse(
            content=_opening_call(request, "build a balanced opening"),
            model="test/model",
            finish_reason="stop",
            native_reasoning="native live analysis",
            native_reasoning_details=({"type": "reasoning.text"},),
            reasoning_request=(("effort", "high"), ("exclude", False)),
            provider_response_id="gen-live-test",
            provider_request_id="req-live-test",
            provider_native_finish_reason="stop",
        )


@dataclass
class SuccessThenInvalidTransport:
    requests: list = field(default_factory=list)

    async def complete(self, request):
        self.requests.append(request)
        if len(self.requests) == 1:
            content = _opening_call(request, "start")
        else:
            content = '{"game_plan":"continue","tool":"invalid_tool","arguments":{}}'
        return ModelResponse(
            content=content,
            model="test/model",
            usage=(
                ("completion_tokens", 8_192),
                (
                    "completion_tokens_details",
                    {"reasoning_tokens": 8_000},
                ),
            ),
            latency_ms=25,
            finish_reason="length",
            native_reasoning="private reasoning",
            provider_response_id="gen-invalid-test",
            provider_request_id="req-invalid-test",
            provider_native_finish_reason="max_tokens",
        )


@pytest.fixture
def live_app(tmp_path, monkeypatch):
    monkeypatch.setenv("CATAN_PROMPT_SUITE_DIR", str(tmp_path / "prompt-suites"))
    # These historical scripts use game_plan/XML, not the shared fresh-notes policy.
    monkeypatch.delenv("CATAN_SHARED_SUITE", raising=False)
    monkeypatch.setenv("CATAN_CONTEXT_SUITE", str(default_suite_path()))
    monkeypatch.setenv("CATAN_COMMUNICATION_SUITE", str(default_communication_suite_path()))
    app = Flask(__name__)
    state = ServerState()
    state.live_trace_store = SQLiteLiveTraceStore(tmp_path / "live.sqlite3")
    socket = DummySocket()
    app.config["SERVER_STATE"] = state
    app.config["SOCKETIO"] = socket
    app.register_blueprint(live_game_bp)
    return app, state, socket


def test_broadcast_game_log_is_whole_history_not_a_tail(live_app):
    app, state, socket = live_app
    app.register_blueprint(health_bp)
    client = app.test_client()

    started = client.post(
        "/api/start-game",
        json={"mode": "random", "seed": 5, "palette": "canonical_four"},
    )
    assert started.status_code == 200
    opening_rows = len(state.game_log)

    spoken = log_game_event(
        state, "message", "[QUESTION] Anyone want wood?", color="RED",
        details={"event_type": "MESSAGE_SENT", "payload": {}, "sequence": 469},
    )
    spoken["step_index"] = 12
    for roll in range(80):
        log_game_event(state, "dice", f"Rolled {roll}", color="BLUE")

    snapshot = client.get("/api/state").json
    # Speech logged 80 rows back stays reachable; a 50-row tail would drop it.
    assert len(snapshot["game_log"]) == opening_rows + 81
    assert [
        entry for entry in snapshot["game_log"] if entry["type"] == "message"
    ] == [spoken]
    assert snapshot["game_log"][-1]["message"] == "Rolled 79"

    # A new game clears the history.
    client.post("/api/start-game", json={"mode": "random", "seed": 6, "palette": "canonical_four"})
    assert len(client.get("/api/state").json["game_log"]) == opening_rows


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
            "palette": "canonical_four",
        },
    )

    assert started.status_code == 200
    assert started.json["state"]["running"] is True
    assert started.json["state"]["game"] is not None
    assert started.json["palette"] == "canonical_four"
    assert started.json["board_surface"] == "indexed_tile_rows"
    assert started.json["realized_colors"] == ["RED", "BLUE", "WHITE", "ORANGE"]
    trace_game_id = started.json["trace_game_id"]
    assert trace_game_id
    assert started.json["state"]["live_trace_game_id"] == trace_game_id
    assert started.json["trace_display_name"] == "First opening"
    assert started.json["trace_database"].endswith("live.sqlite3")
    assert state.current_sandbox is not None
    assert set(state.current_sandbox.players) == set(state.current_sandbox.game_engine.state.colors)
    assert all(
        player.status()["kind"] == "first_legal"
        for player in state.current_sandbox.players.values()
    )

    stepped = client.post("/api/step")

    assert stepped.status_code == 200
    assert stepped.json["state"]["running"] is True
    assert stepped.json["reasoning_traces"] == []
    assert stepped.json["trace_game_id"] == trace_game_id
    assert stepped.json["state"]["live_trace_game_id"] == trace_game_id
    assert stepped.json["trace_step_index"] == 0
    assert len(stepped.json["state"]["events"]) == 1
    assert state.current_sandbox.revision == 1
    assert len(state.current_sandbox.game_engine.state.actions) == 1
    assert socket.emissions == []
    payload = stepped.json["state"]
    assert payload["sandbox_players"]["RED"]["kind"] == "first_legal"
    assert "llm_thinking" not in payload
    assert payload["all_player_resources"]["RED"] == {"TOTAL": 0}
    assert all(not key.endswith("_IN_HAND") for key in payload["game"]["player_state"])
    # The public projection stays redacted; hand contents ride the separate
    # spectator field the viewer reveals behind its Hands toggle.
    red_hand = payload["player_hands"]["RED"]
    assert red_hand["resources"] == {
        "WOOD": 0,
        "BRICK": 0,
        "SHEEP": 0,
        "WHEAT": 0,
        "ORE": 0,
    }
    assert red_hand["dev_cards"]["total_in_hand"] == 0
    assert red_hand["dev_cards"]["in_hand"] == {
        "KNIGHT": 0,
        "YEAR_OF_PLENTY": 0,
        "MONOPOLY": 0,
        "ROAD_BUILDING": 0,
        "VICTORY_POINT": 0,
    }
    assert set(payload["player_hands"]) == set(payload["all_player_resources"])
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
        json={
            "mode": "random",
            "seed": 6,
            "shuffle_players": False,
            "palette": "canonical_four",
        },
    )
    assert replacement.status_code == 200
    assert replacement.json["trace_game_id"] != trace_game_id
    loaded = client.post(f"/api/live-traces/{trace_game_id}/load")
    assert loaded.status_code == 200
    assert loaded.json["trace_game_id"] == trace_game_id
    assert loaded.json["state"]["live_trace_game_id"] == trace_game_id
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
    inference_before = state.live_inference
    usage = client.get(f"/api/live-traces/{trace_game_id}?view=usage")
    assert usage.status_code == 200
    assert usage.json == {"game_id": trace_game_id, "step_count": 2, "calls": [], "failure_calls": []}
    assert state.live_inference is inference_before
    assert client.get("/api/live-traces/missing?view=usage").status_code == 404
    checkpoint = client.get(f"/api/live-traces/{trace_game_id}/steps/0")
    assert checkpoint.status_code == 200
    assert checkpoint.json["latest_step_index"] == 1
    assert checkpoint.json["step"]["after_revision"] == 1
    assert len(checkpoint.json["step"]["public_state"]["events"]) == 1
    assert checkpoint.json["model_calls"] == []
    assert state.current_sandbox is active_sandbox
    assert state.current_sandbox.revision == 2
    assert client.get(f"/api/live-traces/{trace_game_id}/steps/99").status_code == 404
    assert client.post("/api/live-traces/missing/load").status_code == 404
    assert (
        client.patch(
            f"/api/live-traces/{trace_game_id}",
            json={"name": "x" * 81},
        ).status_code
        == 400
    )
    assert client.post("/api/auto-play").status_code == 404


def test_live_route_defaults_to_random_all_and_persists_realized_colors(live_app):
    app, state, _ = live_app
    client = app.test_client()

    started = client.post(
        "/api/start-game",
        json={"mode": "random", "seed": 2026, "shuffle_players": False},
    )

    assert started.status_code == 200
    assert started.json["palette"] == "random_all"
    assert len(started.json["realized_colors"]) == 4
    assert len(set(started.json["realized_colors"])) == 4
    assert started.json["realized_colors"] == [
        color.value for color in state.current_sandbox.game_engine.state.colors
    ]
    trace = client.get(f"/api/live-traces/{started.json['trace_game_id']}").json
    assert trace["config"]["palette"] == "random_all"
    assert trace["config"]["board_surface"] == "indexed_tile_rows"
    assert trace["config"]["realized_colors"] == started.json["realized_colors"]


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

    assert stepped.json["state"]["events"][-1]["payload"][0] in {color.value for color in Color}
    assert state.current_sandbox.revision == 31
    assert socket.emissions == []


def test_stored_games_do_not_select_an_inference_board_contract():
    config = _config_from_stored_payload(
        {"mode": "random", "seed": 7, "palette": "canonical_four"}
    )

    assert config.board_surface == "indexed_tile_rows"


def test_saved_unpinned_llm_game_uses_current_prompt_defaults(live_app, monkeypatch):
    app, state, socket = live_app
    transport = FixedTransport()
    monkeypatch.setattr("cle.sandbox.factory.create_text_transport", lambda config: transport)
    sandbox = create_live_sandbox(
        LiveSandboxConfig(
            mode="llm_vs_random", seed=7, shuffle_players=False,
            palette="canonical_four", context_suite_path=str(default_suite_path()),
            communication_suite_path=str(default_communication_suite_path()),
        ),
        transport=transport,
    )
    game_id = sandbox.game_engine.id
    snapshot = sandbox.snapshot()
    state.live_trace_store.start_game(
        game_id, config={"mode": "llm_vs_random", "seed": 7}, snapshot=snapshot,
    )
    stored_before = state.live_trace_store.get_game(game_id)
    client = app.test_client()
    started = client.post("/api/start-game", json={"mode": "random", "seed": 8})
    assert started.status_code == 200
    active = state.current_sandbox
    active_before = pickle.dumps(active.snapshot())

    loaded = client.post(f"/api/live-traces/{game_id}/load")

    assert loaded.status_code == 200, loaded.json
    assert state.current_sandbox is not active
    assert state.current_sandbox.snapshot().player_states == snapshot.player_states
    assert pickle.dumps(active.snapshot()) == active_before
    assert state.live_trace_game_id == game_id
    assert state.live_trace_store.get_game(game_id)["config"] == stored_before["config"]
    assert pickle.dumps(state.live_trace_store.load_snapshot(game_id)) == pickle.dumps(snapshot)
    assert transport.requests == []
    assert socket.emissions[-1][0] == "game_state"


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({}, {"effort": "high", "exclude": False}),
        ({"reasoning": None}, {"effort": "high", "exclude": False}),
        ({"reasoning": {"enabled": False}}, {"effort": "high", "exclude": False}),
        (
            {"reasoning": {"effort": "xhigh", "exclude": False}},
            {"effort": "high", "exclude": False},
        ),
        ({"reasoning": {}}, {"effort": "high", "exclude": False}),
    ],
)
def test_stored_reasoning_never_overrides_current_defaults(payload, expected):
    config = _config_from_stored_payload({"mode": "llm_vs_random", **payload})

    assert config.reasoning == expected


def test_live_model_defaults_to_qwen_and_preserves_explicit_overrides(monkeypatch):
    monkeypatch.delenv("CATAN_LLM_MODEL", raising=False)

    assert DEFAULT_LIVE_MODEL == "qwen/qwen3.8-27b"
    assert DEFAULT_LIVE_MAX_DECISION_ATTEMPTS == 3
    assert DEFAULT_LIVE_REASONING_EFFORT == "high"
    assert LiveSandboxConfig().max_tokens is None
    assert dict(LiveSandboxConfig().reasoning) == {
        "effort": "high",
        "exclude": False,
    }
    assert resolve_live_model(None) == DEFAULT_LIVE_MODEL
    assert resolve_live_model("  custom/model  ") == "custom/model"


def test_live_route_can_select_typed_image_board_surface(live_app, monkeypatch):
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
    client = app.test_client()

    started = client.post(
        "/api/start-game",
        json={
            "mode": "llm_vs_random",
            "seed": 91,
            "shuffle_players": False,
            "palette": "canonical_four",
            "board_surface": "image",
        },
    )
    stepped = client.post("/api/step")

    assert started.status_code == 200, started.get_json()
    assert started.json["board_surface"] == "image"
    assert configs[0].board_surface == "image"
    assert stepped.status_code == 200, stepped.get_json()
    board = transport.requests[0].board_presentation
    assert board.kind == "image"
    assert board.media_type == "image/png"
    assert board.contains_entity_labels is False


def test_unsupported_transport_cannot_fake_native_reasoning(monkeypatch):
    monkeypatch.setenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")

    with pytest.raises(ValueError, match="does not implement"):
        create_text_transport(
            LiveSandboxConfig(
                mode="llm",
                reasoning={"effort": "high", "exclude": False},
            )
        )


@pytest.mark.asyncio
async def test_live_openrouter_defaults_to_high_reasoning_without_token_cap(
    monkeypatch,
):
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    transport = create_text_transport(LiveSandboxConfig(mode="llm"))

    assert isinstance(transport, OpenRouterTransport)
    assert transport.config.max_tokens is None
    assert dict(transport.reasoning_request) == {
        "effort": "high",
        "exclude": False,
    }

    await transport.aclose()


def test_live_factory_does_not_fall_back_to_groq(monkeypatch):
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "obsolete")

    with pytest.raises(ValueError, match="VLLM_BASE_URL or OPENROUTER_API_KEY"):
        create_text_transport(
            LiveSandboxConfig(
                mode="llm",
                reasoning={"enabled": False},
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


def test_llm_live_route_uses_pinned_v11_agent_without_legacy_llm_player(
    live_app,
    monkeypatch,
):
    app, state, _ = live_app
    transport = FixedTransport()
    configs = []
    monkeypatch.delenv("CATAN_LLM_MODEL", raising=False)

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
        json={
            "mode": "llm_vs_random",
            "seed": 9,
            "shuffle_players": False,
            "palette": "canonical_four",
        },
    )
    stepped = client.post("/api/step")

    assert started.status_code == 200
    assert started.json["model"] == DEFAULT_LIVE_MODEL
    assert started.json["board_surface"] == "indexed_tile_rows"
    assert configs[0].model == DEFAULT_LIVE_MODEL
    assert configs[0].board_surface == "indexed_tile_rows"
    assert stepped.status_code == 200
    red_player = state.current_sandbox.players[state.current_sandbox.game_engine.state.colors[0]]
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
    stored = client.get(f"/api/live-traces/{started.json['trace_game_id']}").json
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
        source = path.read_text(encoding="utf-8")
        assert stored["config"][name]["source"] == source
        assert stored["config"][name]["sha256"] == sha256(source.encode("utf-8")).hexdigest()
    assert "environment.board_state" in {
        component["id"]
        for component in stored["model_calls"][0]["request"]["components"]
    }
    checkpoint = client.get(f"/api/live-traces/{started.json['trace_game_id']}/steps/0").json
    assert checkpoint["model_calls"][0]["accepted"] is True
    assert "rationale" not in checkpoint["model_calls"][0]["choice"]
    assert checkpoint["model_calls"][0]["response"]["native_reasoning"] == ("native live analysis")
    assert checkpoint["model_calls"][0]["request"]["messages"][-1]["role"] == ("user")
    receipts_before = pickle.dumps(red_player.session.receipts)
    loaded = client.post(f"/api/live-traces/{started.json['trace_game_id']}/load")
    assert loaded.status_code == 200, loaded.get_json()
    restored_red = state.current_sandbox.players[Color.RED]
    assert restored_red.session.context_policy == "legacy"
    assert pickle.dumps(restored_red.session.receipts) == receipts_before
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


def test_loaded_live_game_surfaces_first_invalid_model_attempt_without_retries(
    live_app,
    monkeypatch,
):
    app, state, socket = live_app
    transport = SuccessThenInvalidTransport()
    monkeypatch.setattr(
        "cle.sandbox.factory.create_text_transport",
        lambda config: transport,
    )
    client = app.test_client()

    started = client.post(
        "/api/start-game",
        json={
            "mode": "llm_vs_random",
            "seed": 23,
            "shuffle_players": False,
            "palette": "canonical_four",
            "reasoning": {"effort": "xhigh", "exclude": False},
            "max_tokens": 8_192,
            "max_decision_attempts": 3,
        },
    )
    first_step = client.post("/api/step")
    client.post(
        "/api/start-game",
        json={
            "mode": "random",
            "seed": 24,
            "palette": "canonical_four",
            "max_decision_attempts": 1,
        },
    )
    loaded = client.post(
        f"/api/live-traces/{started.json['trace_game_id']}/load"
    )
    before_revision = state.current_sandbox.revision

    failed_step = client.post("/api/step")

    assert started.status_code == 200
    assert started.json["reasoning_request"] == {
        "effort": "xhigh",
        "exclude": False,
    }
    assert started.json["max_tokens"] == 8_192
    assert started.json["max_decision_attempts"] == 3
    assert first_step.status_code == 200
    assert loaded.status_code == 200, loaded.get_json()
    assert loaded.json["reasoning_request"] == {"enabled": False}
    assert loaded.json["max_tokens"] is None
    assert loaded.json["max_decision_attempts"] == 1
    assert len(socket.emissions) == 2
    emitted_event, emitted_state = socket.emissions[0]
    assert emitted_event == "game_state"
    assert emitted_state == loaded.json["state"]
    assert emitted_state["running"] is True
    assert emitted_state["live_trace_game_id"] == started.json["trace_game_id"]
    assert failed_step.status_code == 422
    assert failed_step.json["error"] == "Model returned no valid action"
    assert "Unknown action tool" in (
        failed_step.json["details"]
    )
    assert "No gameplay action was applied" in failed_step.json["details"]
    assert failed_step.json["attempt_count"] == 1
    assert len(failed_step.json["attempts"]) == 1
    attempt = failed_step.json["attempts"][0]
    expected_diagnostics = {
        "context_id": transport.requests[-1].decision_id,
        "action_index": None,
        "final_response": (
            '{"game_plan":"continue","tool":"invalid_tool","arguments":{}}'
        ),
        "finish_reason": "length",
        "latency_ms": 25,
        "model": "test/model",
        "native_reasoning": "private reasoning",
        "native_reasoning_details": [],
        "native_reasoning_chars": 17,
        "reasoning_request": {},
        "provider_native_finish_reason": "max_tokens",
        "provider_request_id": "req-invalid-test",
        "provider_response_id": "gen-invalid-test",
        "reasoning_tokens": 8_000,
        "usage": {
            "completion_tokens": 8_192,
            "completion_tokens_details": {"reasoning_tokens": 8_000},
        },
        "validation_error": "Invalid tool call: Unknown action tool",
    }
    assert {key: attempt[key] for key in expected_diagnostics} == expected_diagnostics
    assert attempt["accepted"] is False
    assert attempt["notes_update"] is None
    assert attempt["context_policy"] is None
    assert attempt["memory_revision"] is None
    assert attempt["input_next_sequence"] is None
    assert attempt["channel"] is None
    model_request = attempt["request"]
    assert model_request["decision_id"] == transport.requests[-1].decision_id
    assert model_request["session_id"] == transport.requests[-1].session_id
    assert model_request["messages"] == [
        {"role": message.role, "content": message.content}
        for message in transport.requests[-1].messages
    ]
    assert model_request["context_policy"] is None
    assert model_request["memory_revision"] is None
    assert model_request["input_next_sequence"] is None
    assert model_request["channel"] is None
    assert failed_step.json["retryable"] is True
    assert failed_step.json["player"] == "RED"
    assert failed_step.json["trace_game_id"] == started.json["trace_game_id"]
    failure_event, failure_state = socket.emissions[1]
    assert failure_event == "game_state"
    assert failure_state["last_live_step_error"] == failed_step.json
    assert state.last_live_step_error == failed_step.json
    assert failure_state["live_inference"] == {
        "max_decision_attempts": 1,
        "max_tokens": None,
        "model": DEFAULT_LIVE_MODEL,
        "reasoning": {"enabled": False},
    }
    assert len(transport.requests) == 2
    assert state.current_sandbox.revision == before_revision
    assert state.step_processing is False
    stored = client.get(f"/api/live-traces/{started.json['trace_game_id']}").json
    assert stored["step_count"] == 1
    assert stored["config"]["reasoning"] == {"effort": "xhigh", "exclude": False}
    assert stored["config"]["max_tokens"] == 8_192
    assert stored["config"]["max_decision_attempts"] == 3
    assert len(stored["failures"]) == 1
    failure = stored["failures"][0]
    assert failure["failure_id"] == failed_step.json["trace_failure_id"]
    assert failure["revision"] == before_revision
    assert failure["actor"] == "RED"
    assert failure["attempts"][0]["model_response"]["content"] == (
        failed_step.json["attempts"][0]["final_response"]
    )


@pytest.mark.parametrize(
    ("final_output", "native_reasoning", "native_details"),
    [
        ("<game_plan>final output</game_plan>" * 200, "native analysis", ()),
        ("", "native analysis without a final answer", ()),
        ("<action>1</action>", "", ({"type": "reasoning.text", "text": "native detail"},)),
    ],
)
def test_off_turn_failure_preserves_distinct_diagnostics_over_http_and_ws(
    live_app,
    monkeypatch,
    final_output,
    native_reasoning,
    native_details,
):
    app, state, socket = live_app
    client = app.test_client()
    started = client.post(
        "/api/start-game",
        json={"mode": "random", "seed": 5, "shuffle_players": False, "palette": "canonical_four"},
    )
    assert started.status_code == 200
    sandbox = state.current_sandbox
    assert sandbox.current_actor() == Color.RED
    before_revision = sandbox.revision
    original_step = sandbox.step
    validation_error = "The selected action parameters are no longer legal or affordable."
    attempt = PlayerAttempt(
        context_id="off-turn-blue",
        choice=PlayerChoice(action_index=1),
        validation_error=validation_error,
        model_response=ModelResponse(
            content=final_output,
            native_reasoning=native_reasoning,
            native_reasoning_details=native_details,
            reasoning_request=(("effort", "high"), ("exclude", False)),
            usage=(("completion_tokens", 123),),
            finish_reason="stop",
            provider_request_payload={"headers": {"Authorization": "secret-request-header"}},
            provider_response_payload={"internal": "unexposed-provider-payload"},
        ),
    )
    earlier_attempt = replace(attempt, context_id="off-turn-white")
    sandbox.decision_trace.append(replace(attempt, context_id="previous-failure"))
    opportunity = CommunicationOpportunity(
        player=Color.BLUE,
        cause=PlayerEvent(0, "pre-action", Color.BLUE, "PRE_ACTION", None),
        visible_through_sequence=0,
        reason=ReactionReason.PRE_ACTION,
        round=0,
    )
    sandbox.communication_trace.append(
        CommunicationAdmission(opportunity, CommunicationChoice(), accepted=True)
    )

    async def reject_off_turn():
        sandbox.decision_trace.extend((earlier_attempt, attempt))
        sandbox.communication_trace.append(
            CommunicationAdmission(replace(opportunity, round=1), CommunicationChoice(), accepted=True)
        )
        raise PlayerResponseError(Color.BLUE, (attempt,), validation_error)

    monkeypatch.setattr(sandbox, "step", reject_off_turn)
    failed = client.post("/api/step")

    assert failed.status_code == 422
    payload = failed.get_json()
    assert payload["player"] == "BLUE"
    assert payload["trace_game_id"] == started.json["trace_game_id"]
    assert payload["attempt_count"] == 1
    assert payload["retryable"] is True
    diagnostic = payload["attempts"][0]
    assert diagnostic["context_id"] == "off-turn-blue"
    assert diagnostic["action_index"] == 1
    assert diagnostic["validation_error"] == validation_error
    assert diagnostic["final_response"] == final_output
    assert diagnostic["native_reasoning"] == native_reasoning
    assert diagnostic["native_reasoning_details"] == list(native_details)
    assert diagnostic["native_reasoning_chars"] == len(native_reasoning)
    assert diagnostic["reasoning_request"] == {"effort": "high", "exclude": False}
    assert diagnostic["usage"] == {"completion_tokens": 123}
    assert "secret-request-header" not in failed.get_data(as_text=True)
    assert "unexposed-provider-payload" not in failed.get_data(as_text=True)
    assert "provider_request_payload" not in diagnostic
    assert "provider_response_payload" not in diagnostic
    assert socket.emissions[-1][0] == "game_state"
    assert socket.emissions[-1][1]["last_live_step_error"] == payload
    assert state.last_live_step_error == payload
    assert sandbox.revision == before_revision
    assert state.step_processing is False
    trace = client.get(f"/api/live-traces/{started.json['trace_game_id']}").json
    assert trace["step_count"] == 0
    assert trace["model_calls"] == []
    assert len(trace["failures"]) == 1
    failure = trace["failures"][0]
    assert failure["failure_id"] == payload["trace_failure_id"]
    assert failure["revision"] == before_revision
    assert failure["actor"] == "BLUE"
    assert failure["validation_error"] == validation_error
    assert [item["context_id"] for item in failure["attempts"]] == [
        "off-turn-white", "off-turn-blue"
    ]
    assert all(item["accepted"] is False for item in failure["attempts"])
    assert failure["attempts"][-1]["model_response"]["native_reasoning"] == native_reasoning
    assert len(failure["communication_attempts"]) == 1
    assert failure["communication_attempts"][0]["opportunity"]["round"] == 1
    assert state.live_trace_store.load_resume_point(started.json["trace_game_id"]).step_index is None

    monkeypatch.setattr(sandbox, "step", original_step)
    continued = client.post("/api/step")
    assert continued.status_code == 200
    assert continued.json["state"]["last_live_step_error"] is None
    assert state.last_live_step_error is None
    trace = client.get(f"/api/live-traces/{started.json['trace_game_id']}").json
    assert trace["step_count"] == 1
    assert len(trace["failures"]) == 1


@pytest.mark.parametrize("partial_message", [False, True])
def test_post_action_communication_failure_checkpoints_applied_result(
    live_app, monkeypatch, partial_message
):
    app, state, socket = live_app
    response = ModelResponse(
        content=json.dumps({
            "game_plan": "opening plan " * 400,
            "tool": "build_settlement",
            "arguments": {"node": "<N00>"},
        }),
        model="test/model",
        native_reasoning="native accepted analysis " * 250,
        native_reasoning_details=({"type": "reasoning.text", "text": "native detail"},),
        reasoning_request=(("effort", "high"), ("exclude", False)),
    )
    transport = FixedTransport(response=response)
    monkeypatch.setattr("cle.sandbox.factory.create_text_transport", lambda config: transport)
    client = app.test_client()
    started = client.post(
        "/api/start-game",
        json={
            "mode": "llm_vs_random", "seed": 5,
            "shuffle_players": False, "palette": "canonical_four",
        },
    )
    assert started.status_code == 200
    game_id = started.json["trace_game_id"]
    sandbox = state.current_sandbox
    before_revision = sandbox.revision
    red = sandbox.players[Color.RED]

    async def blue_speech(context):
        assert sandbox.revision == before_revision + 1
        assert len(red.session.receipts) == 1
        if partial_message:
            return CommunicationChoice(
                mode=CommunicationMode.SAY,
                text="One message was emitted before speech failed.",
                audience=tuple(
                    color for color in sandbox.game_engine.state.colors if color != Color.BLUE
                ),
            )
        raise RuntimeError("private speech transport failure details")

    async def invalid_white_speech(context):
        return CommunicationChoice(
            mode=CommunicationMode.SAY, text="Invalid recipient", audience=(Color.BLACK,)
        )

    monkeypatch.setattr(sandbox.players[Color.BLUE], "communicate", blue_speech)
    if partial_message:
        monkeypatch.setattr(sandbox.players[Color.WHITE], "communicate", invalid_white_speech)
    stepped = client.post("/api/step")

    assert stepped.status_code == 200, stepped.get_json()
    payload = stepped.get_json()
    warning = payload["warning"]
    assert payload["status"] == "ok"
    assert payload["trace_step_index"] == 0
    assert payload["state"]["running"] is True
    assert payload["state"]["last_live_step_error"] == warning
    assert warning["action_applied"] is True
    assert warning["retryable"] is False
    assert warning["details"].startswith(
        "Game action was applied, but post-action communication failed."
    )
    assert "Do not retry the applied action" in warning["details"]
    assert "the next Step advances the game" in warning["details"]
    assert "private speech transport failure details" not in warning["details"]
    assert "attempts" not in warning
    assert "trace_failure_id" not in warning
    assert sandbox.revision == before_revision + 1 + int(partial_message)
    assert sandbox.game_engine.events[0].event_type == "BUILD_SETTLEMENT"
    assert len(red.session.receipts) == 1
    assert red.session.messages[-1].content == response.content
    assert len(transport.requests) == 1
    assert payload["reasoning_traces"][0]["native_reasoning"] == response.native_reasoning
    assert payload["reasoning_traces"][0]["native_reasoning_details"] == list(
        response.native_reasoning_details
    )
    assert any(entry["type"] == "building" for entry in payload["state"]["game_log"])
    # Speech rows are labelled with the game step, not the engine-event number.
    spoken = [entry for entry in payload["state"]["game_log"] if entry["type"] == "message"]
    assert len(spoken) == int(partial_message)
    for entry in spoken:
        assert entry["step_index"] == payload["trace_step_index"] == 0
        assert entry["details"]["step_index"] == 0
        assert entry["details"]["sequence"] == 1
    assert socket.emissions == [("game_state", payload["state"])]
    assert state.last_live_step_error == warning
    assert state.step_processing is False

    stored = client.get(f"/api/live-traces/{game_id}").json
    assert stored["step_count"] == 1
    assert stored["failures"] == []
    assert len(stored["model_calls"]) == 1
    call = stored["model_calls"][0]
    assert call["accepted"] is True
    assert call["response"]["content"] == response.content
    assert call["response"]["native_reasoning"] == response.native_reasoning
    assert stored["steps"][0]["after_revision"] == before_revision + 1
    assert len(stored["steps"][0]["result"]["message_events"]) == int(partial_message)
    assert [event["event_type"] for event in stored["events"]] == (
        ["BUILD_SETTLEMENT", "MESSAGE_SENT"]
        if partial_message else ["BUILD_SETTLEMENT"]
    )
    if partial_message:
        assert stored["events"][-1]["event"] == (
            stored["steps"][0]["result"]["message_events"][0]
        )
    checkpoint = client.get(f"/api/live-traces/{game_id}/steps/0").json
    assert checkpoint["step"]["public_state"] == payload["state"]
    assert [
        entry["step_index"]
        for entry in checkpoint["step"]["public_state"]["game_log"]
        if entry["type"] == "message"
    ] == [0] * int(partial_message)
    assert checkpoint["model_calls"] == stored["model_calls"]

    loaded = client.post(f"/api/live-traces/{game_id}/load")
    assert loaded.status_code == 200
    # Speech survives the round trip through the checkpoint.
    assert [
        entry for entry in loaded.json["state"]["game_log"]
        if entry["type"] == "message"
    ] == spoken
    assert state.current_sandbox.revision == sandbox.revision
    assert len(state.current_sandbox.players[Color.RED].session.receipts) == 1
    assert len(transport.requests) == 1
    transport.response = None
    continued = client.post("/api/step")
    assert continued.status_code == 200
    assert continued.json["warning"] is None
    assert continued.json["trace_step_index"] == 1
    assert state.current_sandbox.game_engine.events[-1].event_type == "BUILD_ROAD"
    assert len(state.current_sandbox.players[Color.RED].session.receipts) == 2


@pytest.mark.parametrize("version", ["8.0.0", "10.0.0"])
def test_saved_game_uses_active_source_but_retains_recorded_source(
    live_app,
    monkeypatch,
    tmp_path,
    version,
):
    app, state, _ = live_app
    source_path = tmp_path / "active-decision.yaml"
    original_source = (
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

    started = client.post(
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
    restored = state.current_sandbox.players[Color.RED]
    assert restored.suite.version == version
    assert restored.suite.response.format == "xml"
    assert restored.suite.phase_guidance["discarding"].startswith(
        "NEW GLOBAL DISCARD GUIDANCE."
    )
    assert ("Maximizing raw pip count is not the objective" in (
        restored.suite.phase_guidance["initial_settlement_1"]
    )) == (version == "10.0.0")
    trace = client.get(f"/api/live-traces/{game_id}").json
    assert trace["config"]["decision_suite"]["version"] == version
    assert trace["config"]["decision_suite"]["source"] == original_source
    stepped = client.post("/api/step")
    assert stepped.status_code == 200, stepped.get_json()
    assert next(iter(restored.session.receipts.values())).choice.action_index == 0


def test_llm_live_route_preserves_explicit_reasoning_off(live_app, monkeypatch):
    app, state, socket = live_app
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
    response = client.post(
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
    game_id = response.json["trace_game_id"]
    stored_config = state.live_trace_store.get_game(game_id)["config"]

    loaded = client.post(f"/api/live-traces/{game_id}/load")

    assert loaded.status_code == 200, loaded.get_json()
    assert loaded.json["reasoning_request"] == {"enabled": False}
    assert loaded.json["max_tokens"] is None
    assert loaded.json["max_decision_attempts"] == 3
    assert loaded.json["state"]["live_inference"]["reasoning"] == {"enabled": False}
    assert socket.emissions[-1][1] == loaded.json["state"]
    assert dict(configs[-1].reasoning) == {"enabled": False}
    assert transport.requests == []
    assert state.live_trace_store.get_game(game_id)["config"] == stored_config


@pytest.fixture(params=["tls", "http403"])
def openrouter_failure(request):
    def make_failure(model_request):
        if request.param == "tls":
            return OpenRouterTLSFailure(model_request, model="test/model", attempts=3)
        return OpenRouterHTTPFailure(
            model_request, model="test/model", attempts=1,
            response=httpx.Response(
                403,
                json={"error": {"message": (
                    "Provider policy rejected this model. private-provider-secret"
                )}},
                headers={"x-request-id": "req-rejected-test"},
                request=httpx.Request("POST", "https://openrouter.test/api/v1/chat/completions"),
            ),
            redactions=("private-provider-secret",),
        )

    return make_failure


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
    live_app, monkeypatch, caplog, openrouter_failure, failure_phase, storage_fails
):
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

    async def complete(request):
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
    client = app.test_client()
    started = client.post(
        "/api/start-game",
        json={
            "mode": "llm_vs_random", "seed": 5,
            "shuffle_players": False, "palette": "canonical_four",
            "max_decision_attempts": 2,
        },
    )
    assert started.status_code == 200, started.get_json()
    game_id = started.json["trace_game_id"]
    sandbox = state.current_sandbox
    red = sandbox.players[Color.RED]
    before_revision = sandbox.revision
    before_actions = list(sandbox.game_engine.state.actions)
    opportunity = CommunicationOpportunity(
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

    def fail_persistence(*args, **kwargs):
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
    payload = failed.get_json()
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

    stored = client.get(f"/api/live-traces/{game_id}").json
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
        record = stored["failures"][0]
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
        continued = client.post("/api/step")
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


def test_openrouter_barrier_failure_uses_registered_session_not_turn_actor(
    live_app, monkeypatch, caplog, openrouter_failure
):
    app, state, socket = live_app
    app.register_blueprint(health_bp)
    transport = FixedTransport()
    monkeypatch.setattr("cle.sandbox.factory.create_text_transport", lambda config: transport)
    client = app.test_client()
    started = client.post(
        "/api/start-game",
        json={
            "mode": "llm", "seed": 5,
            "shuffle_players": False, "palette": "canonical_four",
        },
    )
    assert started.status_code == 200, started.get_json()
    game_id = started.json["trace_game_id"]
    sandbox = state.current_sandbox
    sandbox.players[Color.BLUE].session.session_id = "opaque-session-without-a-color"
    offer = ensure_trade_window(sandbox.game_engine.state).create_offer(
        TradeOffer(
            offered_by=Color.RED,
            audience=frozenset({Color.BLUE, Color.WHITE, Color.ORANGE}),
            give=(1, 0, 0, 0, 0), receive=(0, 1, 0, 0, 0),
        )
    )
    assert sandbox.current_actor() == Color.RED
    before_revision = sandbox.revision
    before_engine = pickle.dumps(sandbox.game_engine.snapshot())
    before_sessions = {color: player.session.snapshot() for color, player in sandbox.players.items()}
    siblings_ready = asyncio.Event()
    responses = {}
    failures = []

    async def complete(request):
        transport.requests.append(request)
        color = next(
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
    payload = failed.get_json()
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
    stored = client.get(f"/api/live-traces/{game_id}").json
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


def test_post_action_openrouter_failure_is_warning_and_checkpoints_once(
    live_app, monkeypatch, caplog, openrouter_failure
):
    app, state, socket = live_app
    app.register_blueprint(health_bp)
    transport = FixedTransport()
    decision_complete = transport.complete
    speech_failures = []
    fail_speech = True

    async def complete(request):
        if any(component.id == "environment.legal_actions" for component in request.components):
            return await decision_complete(request)
        transport.requests.append(request)
        blue_session = state.current_sandbox.players[Color.BLUE].status()["session_id"]
        if fail_speech and request.session_id == blue_session:
            failure = openrouter_failure(request)
            speech_failures.append(failure)
            raise failure from RuntimeError("private-post-action-provider-secret")
        return ModelResponse(content="<message>SILENCE</message>", model="test/model")

    monkeypatch.setattr(transport, "complete", complete)
    monkeypatch.setattr("cle.sandbox.factory.create_text_transport", lambda config: transport)
    client = app.test_client()
    started = client.post(
        "/api/start-game",
        json={
            "mode": "llm", "seed": 5,
            "shuffle_players": False, "palette": "canonical_four",
        },
    )
    assert started.status_code == 200, started.get_json()
    game_id = started.json["trace_game_id"]
    sandbox = state.current_sandbox
    stepped = client.post("/api/step")

    assert stepped.status_code == 200, stepped.get_json()
    payload = stepped.get_json()
    warning = payload["warning"]
    assert len(speech_failures) == 1
    failure = speech_failures[0]
    assert len([
        request for request in transport.requests if request.session_id == failure.session_id
    ]) == 1
    if isinstance(failure, OpenRouterHTTPFailure):
        assert warning["details"].count(str(failure)) == 1
        assert warning["provider_status_code"] == 403
        assert warning["provider_request_id"] == "req-rejected-test"
        assert "Provider policy rejected this model." in warning["details"]
        assert "Resolve the provider rejection before the next Step." in warning["details"]
        assert "Press Step to retry" not in warning["details"]
        assert "private-post-action-provider-secret" not in caplog.text
        assert "private-provider-secret" not in caplog.text
        assert all(record.exc_info is None for record in caplog.records)
    else:
        assert warning["details"] == (
            "Game action was applied, but post-action communication failed. "
            "Auto-play stopped. Do not retry the applied action; "
            "the next Step advances the game."
        )
    assert warning["action_applied"] is True
    assert warning["retryable"] is False
    assert warning["details"].startswith(
        "Game action was applied, but post-action communication failed."
    )
    assert "Do not retry the applied action" in warning["details"]
    assert "No gameplay action was applied" not in warning["details"]
    assert "attempts" not in warning
    assert "trace_failure_id" not in warning
    assert payload["trace_step_index"] == 0
    assert payload["state"]["last_live_step_error"] == warning
    assert state.last_live_step_error == warning
    assert client.get("/api/state").json["last_live_step_error"] == warning
    assert socket.emissions == [("game_state", payload["state"])]
    assert state.step_processing is False
    assert len(sandbox.game_engine.state.actions) == sandbox.revision == 1
    assert len(sandbox.players[Color.RED].session.receipts) == 1
    assert len(sandbox.players[Color.RED].session.messages) == 2
    stored = client.get(f"/api/live-traces/{game_id}").json
    assert stored["step_count"] == 1
    assert stored["failures"] == []
    assert len(stored["steps"][0]["result"]["transitions"]) == 1
    assert [event["event_type"] for event in stored["events"]] == ["BUILD_SETTLEMENT"]
    decisions = [call for call in stored["model_calls"] if call["call_kind"] == "decision"]
    assert len(decisions) == 1
    assert decisions[0]["accepted"] is True
    assert decisions[0]["response"]["provider_response_id"] == "gen-live-test"
    assert all(call["context_id"] != speech_failures[0].context_id for call in stored["model_calls"])
    checkpoint = client.get(f"/api/live-traces/{game_id}/steps/0").json
    assert checkpoint["step"]["public_state"] == payload["state"]
    assert "private-post-action-provider-secret" not in json.dumps([payload, stored])
    assert "private-provider-secret" not in json.dumps([payload, stored])

    calls_before_load = len(transport.requests)
    loaded = client.post(f"/api/live-traces/{game_id}/load")
    assert loaded.status_code == 200, loaded.get_json()
    assert len(transport.requests) == calls_before_load
    assert state.current_sandbox.revision == 1
    assert len(state.current_sandbox.game_engine.state.actions) == 1
    assert len(state.current_sandbox.players[Color.RED].session.receipts) == 1
    fail_speech = False
    continued = client.post("/api/step")
    assert continued.status_code == 200, continued.get_json()
    assert continued.json["warning"] is None
    assert continued.json["state"]["last_live_step_error"] is None
    assert continued.json["trace_step_index"] == 1
    assert len(state.current_sandbox.game_engine.state.actions) == 2
    assert len(state.current_sandbox.players[Color.RED].session.receipts) == 2
    assert state.current_sandbox.game_engine.events[-1].event_type == "BUILD_ROAD"
    stored = client.get(f"/api/live-traces/{game_id}").json
    assert stored["step_count"] == 2
    assert len([call for call in stored["model_calls"] if call["call_kind"] == "decision"]) == 2


def test_openrouter_error_from_accept_callback_does_not_offer_action_retry(
    live_app, monkeypatch, caplog, openrouter_failure
):
    app, state, socket = live_app
    transport = FixedTransport()
    monkeypatch.setattr("cle.sandbox.factory.create_text_transport", lambda config: transport)
    client = app.test_client()
    started = client.post(
        "/api/start-game",
        json={
            "mode": "llm_vs_random", "seed": 5,
            "shuffle_players": False, "palette": "canonical_four",
        },
    )
    assert started.status_code == 200, started.get_json()
    sandbox = state.current_sandbox
    red = sandbox.players[Color.RED]
    accept = red.accept
    failures = []

    def fail_after_accept(attempt, result):
        accept(attempt, result)
        failure = openrouter_failure(attempt.model_request)
        failures.append(failure)
        raise failure from RuntimeError("private-accept-provider-secret")

    monkeypatch.setattr(red, "accept", fail_after_accept)
    failed = client.post("/api/step")

    assert failed.status_code == 502, failed.get_json()
    payload = failed.get_json()
    assert len(failures) == 1
    failure = failures[0]
    if isinstance(failure, OpenRouterHTTPFailure):
        assert payload["error"] == "OpenRouter rejected the request"
        assert payload["provider_status_code"] == 403
        assert payload["provider_request_id"] == "req-rejected-test"
        assert "Resolve the provider rejection before continuing." in payload["details"]
    else:
        assert payload["error"] == "OpenRouter connection failed"
    assert str(failure) in payload["details"]
    assert "A gameplay action was applied before this error. Do not repeat it" in payload["details"]
    assert payload["transport_attempt_count"] == failure.attempts
    assert payload["action_applied"] is True
    assert payload["retryable"] is False
    assert "No gameplay action was applied" not in payload["details"]
    assert "Press Step to retry" not in payload["details"]
    assert payload["player"] == "RED"
    assert payload["context_id"] == transport.requests[0].decision_id
    assert len(sandbox.game_engine.state.actions) == sandbox.revision == 1
    assert len(red.session.receipts) == 1
    assert len(transport.requests) == 1
    assert state.last_live_step_error == payload
    assert socket.emissions[-1][1]["last_live_step_error"] == payload
    assert state.step_processing is False
    stored = client.get(f"/api/live-traces/{started.json['trace_game_id']}").json
    assert len(stored["failures"]) == 1
    assert stored["failures"][0]["revision"] == sandbox.revision
    assert stored["failures"][0]["validation_error"] == payload["details"]
    assert stored["step_count"] == 0
    assert stored["model_calls"] == []
    assert "attempts" not in payload
    assert stored["failures"][0]["attempts"] == []
    diagnostics = json.dumps([payload, socket.emissions, stored]) + caplog.text
    assert "private-accept-provider-secret" not in diagnostics
    assert "private-provider-secret" not in diagnostics
    assert all(record.exc_info is None for record in caplog.records)


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("Callback failed with SSLV3_ALERT_BAD_RECORD_MAC in its message"),
        httpx.ReadError("An unrelated provider read failed"),
        httpx.HTTPStatusError(
            "An unrelated provider HTTP 401 failed",
            request=httpx.Request("POST", "https://openrouter.test/api/v1/chat/completions"),
            response=httpx.Response(401),
        ),
    ],
)
def test_generic_live_transport_errors_checkpoint_safe_diagnostics(live_app, monkeypatch, error):
    app, state, socket = live_app
    transport = FixedTransport()

    async def complete(request):
        transport.requests.append(request)
        raise error

    monkeypatch.setattr(transport, "complete", complete)
    monkeypatch.setattr("cle.sandbox.factory.create_text_transport", lambda config: transport)
    client = app.test_client()
    started = client.post(
        "/api/start-game",
        json={
            "mode": "llm_vs_random", "seed": 5,
            "shuffle_players": False, "palette": "canonical_four",
        },
    )
    assert started.status_code == 200, started.get_json()

    failed = client.post("/api/step")

    assert failed.status_code == 500
    payload = failed.get_json()
    assert payload["error"] == "Sandbox step failed"
    assert payload["details"] == (
        f"{type(error).__name__}. No gameplay action was applied. "
        "Inspect the failure before retrying."
    )
    assert payload["player"] == "RED"
    assert payload["action_applied"] is False
    assert payload["retryable"] is False
    assert payload["checkpoint_saved"] is True
    assert payload["trace_game_id"] == started.json["trace_game_id"]
    assert state.last_live_step_error == payload
    assert len(socket.emissions) == 1
    assert socket.emissions[0][0] == "game_state"
    assert socket.emissions[0][1]["last_live_step_error"] == payload
    assert state.step_processing is False
    assert state.current_sandbox.revision == 0
    assert state.current_sandbox.game_engine.state.actions == []
    assert len(transport.requests) == 1
    stored = client.get(f"/api/live-traces/{started.json['trace_game_id']}").json
    assert len(stored["failures"]) == 1
    failure = stored["failures"][0]
    assert failure["failure_id"] == payload["trace_failure_id"]
    assert failure["actor"] == "RED"
    assert failure["revision"] == 0
    assert failure["validation_error"] == payload["details"]
    assert failure["attempts"] == []
    assert failure["communication_attempts"] == []
    assert str(error) not in json.dumps([payload, socket.emissions, stored])
    assert stored["step_count"] == 0
    assert stored["model_calls"] == []
    resume = state.live_trace_store.load_resume_point(started.json["trace_game_id"])
    assert resume.snapshot.player_states == state.current_sandbox.snapshot().player_states
    assert resume.snapshot.pending_decision_revision is None
    assert resume.public_state["last_live_step_error"]["details"] == payload["details"]


@pytest.mark.parametrize("include_details", [True, False])
def test_live_openrouter_http403_retains_safe_rejection_without_retries(
    live_app, monkeypatch, caplog, include_details
):
    app, state, socket = live_app
    app.register_blueprint(health_bp)
    http_requests = []

    def handler(request):
        http_requests.append(request)
        return httpx.Response(
            403,
            json={"error": {
                "code": 403,
                **({"message": "Provider policy rejected this model. local-test-key"}
                   if include_details else {}),
                "metadata": {"raw": "private-upstream-response"},
            }},
            headers={"x-request-id": "req-http403-route"} if include_details else {},
            request=request,
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport = OpenRouterTransport(
        OpenRouterConfig(
            model="test/model", max_retries=3,
            endpoint="https://openrouter.test/api/v1/chat/completions",
        ),
        api_key="local-test-key",
        client=http_client,
    )
    monkeypatch.setattr("cle.sandbox.factory.create_text_transport", lambda config: transport)
    client = app.test_client()
    try:
        started = client.post(
            "/api/start-game",
            json={
                "mode": "llm_vs_random", "seed": 5,
                "shuffle_players": False, "palette": "canonical_four",
                "max_decision_attempts": 3,
            },
        )
        assert started.status_code == 200, started.get_json()
        game_id = started.json["trace_game_id"]
        sandbox = state.current_sandbox
        red = sandbox.players[Color.RED]
        context_id = sandbox.decision_context().context_id
        before_engine = pickle.dumps(sandbox.game_engine.snapshot())
        before_session = red.session.snapshot()

        failed = client.post("/api/step")

        assert failed.status_code == 502, failed.get_json()
        payload = failed.get_json()
        assert payload["error"] == "OpenRouter rejected the request"
        assert payload["provider_status_code"] == 403
        assert "403" in payload["details"]
        assert payload["player"] == "RED"
        assert payload["context_id"] == context_id
        assert payload["transport_attempt_count"] == len(http_requests) == 1
        assert payload["retryable"] is False
        assert payload["action_applied"] is False
        assert "No gameplay action was applied." in payload["details"]
        assert "Resolve the provider rejection before retrying." in payload["details"]
        assert "Press Step to retry" not in payload["details"]
        assert "attempts" not in payload
        assert "final_response" not in payload
        assert "native_reasoning" not in payload
        assert pickle.dumps(sandbox.game_engine.snapshot()) == before_engine
        assert red.session.snapshot() == before_session
        assert sandbox.revision == 0
        assert sandbox.game_engine.state.actions == []
        assert sandbox._step_state is None
        assert state.step_processing is False
        assert state.last_live_step_error == payload
        assert client.get("/api/state").json["last_live_step_error"] == payload
        assert len(socket.emissions) == 1
        event, emitted = socket.emissions[0]
        assert event == "game_state"
        assert emitted["last_live_step_error"] == payload
        assert emitted["running"] is True
        assert emitted["events"] == []
        stored = client.get(f"/api/live-traces/{game_id}").json
        assert stored["step_count"] == 0
        assert stored["model_calls"] == []
        assert len(stored["failures"]) == 1
        failure = stored["failures"][0]
        assert failure["failure_id"] == payload["trace_failure_id"]
        assert failure["actor"] == "RED"
        assert failure["revision"] == 0
        assert failure["attempts"] == []
        assert failure["communication_attempts"] == []
        assert failure["validation_error"] in payload["details"]
        assert "403" in failure["validation_error"]
        if include_details:
            assert payload["provider_request_id"] == "req-http403-route"
            assert "req-http403-route" in failure["validation_error"]
            assert "Provider policy rejected this model." in failure["validation_error"]
        else:
            assert payload.get("provider_request_id") is None
        assert state.live_trace_store.load_resume_point(game_id).step_index is None
        assert len(http_requests) == 1
        diagnostics = json.dumps([payload, emitted, stored]) + caplog.text
        assert "local-test-key" not in diagnostics
        assert "private-upstream-response" not in diagnostics
        assert all(record.exc_info is None for record in caplog.records)
    finally:
        sandbox_async_runtime.run(http_client.aclose())


def test_live_openrouter_reasoning_only_rejects_legal_json_until_manual_step(live_app, monkeypatch):
    app, state, socket = live_app
    app.register_blueprint(health_bp)
    http_requests = []
    final_answer = False

    def handler(request):
        http_requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "gen-reka-final-answer" if final_answer else "gen-reka-reasoning-only",
                "model": "reka/test-model",
                "choices": [{
                    "message": {
                        "role": "assistant", "content": output if final_answer else None,
                        "reasoning": output,
                    },
                    "finish_reason": "stop", "native_finish_reason": "stop",
                }],
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport = OpenRouterTransport(
        OpenRouterConfig(
            model="reka/test-model", max_tokens=None,
            reasoning={"effort": "high", "exclude": False},
            endpoint="https://openrouter.test/api/v1/chat/completions",
        ),
        api_key="local-test-key",
        client=http_client,
    )
    monkeypatch.setattr("cle.sandbox.factory.create_text_transport", lambda config: transport)
    client = app.test_client()
    try:
        started = client.post(
            "/api/start-game",
            json={
                "mode": "llm_vs_random", "model": "reka/test-model", "seed": 5,
                "shuffle_players": False, "palette": "canonical_four",
                "max_decision_attempts": 1, "max_tokens": None,
            },
        )
        assert started.status_code == 200, started.get_json()
        assert started.json["max_decision_attempts"] == 1
        assert started.json["max_tokens"] is None
        game_id = started.json["trace_game_id"]
        sandbox = state.current_sandbox
        red = sandbox.players[Color.RED]
        assert red.suite.response.format == "json"
        context = sandbox.decision_context()
        output = _opening_call(
            red._assembler.assemble(context, red.session), "build a legal opening"
        )
        before_revision = sandbox.revision
        before_engine = pickle.dumps(sandbox.game_engine.snapshot())
        before_session = red.session.snapshot()

        failed = client.post("/api/step")

        assert failed.status_code == 422, failed.get_json()
        payload = failed.get_json()
        diagnostic = "The provider returned reasoning but no final answer."
        assert payload["error"] == "Model returned no valid action"
        assert payload["details"] == (
            f"{diagnostic} No final model output was returned. "
            "No gameplay action was applied. Press Step to ask the model again."
        )
        assert "JSONDecodeError" not in failed.get_data(as_text=True)
        assert payload["attempt_count"] == len(payload["attempts"]) == len(http_requests) == 1
        assert "max_tokens" not in json.loads(http_requests[0].content)
        assert payload["player"] == "RED"
        assert payload["retryable"] is True
        assert payload["trace_game_id"] == game_id
        attempt = payload["attempts"][0]
        assert attempt["context_id"] == context.context_id
        assert attempt["action_index"] is None
        assert attempt["validation_error"] == diagnostic
        assert attempt["final_response"] == ""
        assert attempt["native_reasoning"] == output
        assert attempt["finish_reason"] == attempt["provider_native_finish_reason"] == "stop"
        assert attempt["provider_response_id"] == "gen-reka-reasoning-only"
        assert sandbox.revision == before_revision
        assert pickle.dumps(sandbox.game_engine.snapshot()) == before_engine
        assert red.session.snapshot() == before_session
        assert red.session.messages == []
        assert red.session.receipts == {}
        assert state.step_processing is False
        assert sandbox._step_state is None
        assert state.last_live_step_error == payload
        retained = client.get("/api/state")
        assert retained.status_code == 200
        assert retained.json["last_live_step_error"] == payload
        assert len(socket.emissions) == 1
        event, emitted = socket.emissions[0]
        assert event == "game_state"
        assert emitted["last_live_step_error"] == payload
        assert emitted["running"] is True
        assert emitted["events"] == []

        stored = client.get(f"/api/live-traces/{game_id}").json
        assert stored["step_count"] == 0
        assert stored["model_calls"] == []
        assert len(stored["failures"]) == 1
        failure = stored["failures"][0]
        assert failure["failure_id"] == payload["trace_failure_id"]
        assert failure["revision"] == before_revision
        assert failure["actor"] == "RED"
        assert failure["validation_error"] == diagnostic
        assert len(failure["attempts"]) == 1
        saved_attempt = failure["attempts"][0]
        assert saved_attempt["accepted"] is False
        assert saved_attempt["choice"] is None
        response = saved_attempt["model_response"]
        assert response["content"] == ""
        assert response["native_reasoning"] == output
        assert response["finish_reason"] == response["provider_native_finish_reason"] == "stop"
        assert response["provider_response_id"] == "gen-reka-reasoning-only"
        provider_payload = response["provider_response_payload"]
        assert provider_payload["id"] == response["provider_response_id"]
        provider_choice = provider_payload["choices"][0]
        assert provider_choice["finish_reason"] == provider_choice["native_finish_reason"] == "stop"
        assert provider_choice["message"]["content"] is None
        assert provider_choice["message"]["reasoning"] == output
        assert "tool_calls" not in provider_choice["message"]
        assert "max_tokens" not in response["provider_request_payload"]
        assert state.live_trace_store.load_resume_point(game_id).step_index is None

        # Only a new manual Step may admit the same legal JSON as a final answer.
        final_answer = True
        continued = client.post("/api/step")
        assert continued.status_code == 200, continued.get_json()
        assert len(http_requests) == 2
        assert continued.json["warning"] is None
        assert continued.json["trace_step_index"] == 0
        assert continued.json["state"]["last_live_step_error"] is None
        assert client.get("/api/state").json["last_live_step_error"] is None
        assert state.last_live_step_error is None
        assert sandbox.revision == before_revision + 1
        assert len(sandbox.game_engine.state.actions) == 1
        assert len(sandbox.game_engine.events) == 1
        assert sandbox.game_engine.events[0].event_type == "BUILD_SETTLEMENT"
        assert len(red.session.receipts) == 1
        assert len(red.session.messages) == 2
        assert red.session.messages[-1].content == output
        # Receipts keep the decision only; the raw output is on the stored call below.
        assert next(iter(red.session.receipts.values())).choice.raw_response == ""
        stored = client.get(f"/api/live-traces/{game_id}").json
        assert stored["step_count"] == len(stored["model_calls"]) == 1
        assert stored["model_calls"][0]["accepted"] is True
        assert stored["model_calls"][0]["response"]["content"] == output
        assert stored["failures"] == [failure]
    finally:
        sandbox_async_runtime.run(http_client.aclose())


def test_live_openrouter_tls_recovery_with_local_http_client_applies_once(live_app, monkeypatch):
    app, state, socket = live_app
    http_requests = []
    clients = []
    async_client = httpx.AsyncClient
    output = json.dumps({
        "game_plan": "recover the opening", "tool": "build_settlement",
        "arguments": {"node": "<N00>"},
    })

    def handler(request):
        http_requests.append(request)
        if len(http_requests) == 1:
            raise httpx.ReadError("private-tls-record-failure", request=request) from ssl.SSLError(
                1, "[SSL: SSLV3_ALERT_BAD_RECORD_MAC] sslv3 alert bad record mac"
            )
        return httpx.Response(
            200,
            json={
                "id": "gen-recovered-once", "model": "test/model",
                "choices": [{"message": {"content": output}, "finish_reason": "stop"}],
            },
        )

    def local_client(**kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        client = async_client(**kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr("cle.harness.providers.openrouter.httpx.AsyncClient", local_client)
    transport = OpenRouterTransport(
        OpenRouterConfig(
            model="test/model", max_retries=1,
            endpoint="https://openrouter.test/api/v1/chat/completions",
        ),
        api_key="local-test-key",
    )
    monkeypatch.setattr("cle.sandbox.factory.create_text_transport", lambda config: transport)
    client = app.test_client()
    try:
        started = client.post(
            "/api/start-game",
            json={
                "mode": "llm_vs_random", "seed": 5,
                "shuffle_players": False, "palette": "canonical_four",
            },
        )
        assert started.status_code == 200, started.get_json()
        sandbox = state.current_sandbox
        assert sandbox.game_engine.state.playable_actions[0].value == 0
        stepped = client.post("/api/step")

        assert stepped.status_code == 200, stepped.get_json()
        assert stepped.json["warning"] is None
        assert stepped.json["trace_step_index"] == 0
        assert stepped.json["state"]["last_live_step_error"] is None
        assert len(http_requests) == 2
        assert len(clients) == 2
        assert http_requests[0].content == http_requests[1].content
        assert http_requests[0].headers["x-session-id"] == http_requests[1].headers["x-session-id"]
        assert len(sandbox.game_engine.state.actions) == sandbox.revision == 1
        assert sandbox.game_engine.events[0].event_type == "BUILD_SETTLEMENT"
        red = sandbox.players[Color.RED]
        assert len(red.session.receipts) == 1
        assert len(red.session.messages) == 2
        assert socket.emissions == []
        assert state.step_processing is False
        stored = client.get(f"/api/live-traces/{started.json['trace_game_id']}").json
        assert stored["step_count"] == 1
        # The recovered output is recorded once, in the trace store; receipts
        # keep only the accepted decision.
        assert [call["response"]["content"] for call in stored["model_calls"]] == [output]
        assert next(iter(red.session.receipts.values())).choice.raw_response == ""
        assert stored["failures"] == []
        assert len(stored["steps"][0]["result"]["transitions"]) == 1
        assert len(stored["events"]) == 1
        assert len(stored["model_calls"]) == 1
        assert stored["model_calls"][0]["accepted"] is True
        assert stored["model_calls"][0]["response"]["provider_response_id"] == "gen-recovered-once"
    finally:
        sandbox_async_runtime.run(transport.aclose())
        for http_client in clients:
            sandbox_async_runtime.run(http_client.aclose())


def test_snapshots_carry_the_whole_game_log_not_a_window(live_app):
    """Speech at step 99 must still be visible at step 400: no trailing-N slice."""
    app, state, _ = live_app
    client = app.test_client()
    assert client.post("/api/start-game", json={"mode": "random", "seed": 5}).status_code == 200
    state.game_log.extend(
        {"type": "message", "message": f"table talk {index}", "color": "RED",
         "details": {"sequence": index}}
        for index in range(120)
    )
    state.game_log.append({"type": "dice", "message": "Rolled 3 + 4 = 7", "color": "BLUE"})

    stepped = client.post("/api/step")
    assert stepped.status_code == 200
    log = stepped.json["state"]["game_log"]
    assert len(log) >= 121
    assert [entry["message"] for entry in log if entry["type"] == "message"][:2] == [
        "table talk 0", "table talk 1",
    ]
    stored = client.get(f"/api/live-traces/{stepped.json['trace_game_id']}/steps/0").json
    assert len(stored["step"]["public_state"]["game_log"]) == len(log)
