from dataclasses import dataclass, field, replace

import pytest
from flask import Flask

from cle.harness import ModelResponse, default_suite_path
from cle.harness.providers import OpenRouterTransport
from cle.players.contracts import CommunicationChoice, CommunicationMode, PlayerAttempt, PlayerChoice
from cle.sandbox.catan import PlayerResponseError
from cle.sandbox.communication import CommunicationAdmission, CommunicationOpportunity, ReactionReason
from cle.sandbox.factory import (
    DEFAULT_LIVE_MAX_DECISION_ATTEMPTS,
    DEFAULT_LIVE_MODEL,
    DEFAULT_LIVE_REASONING_EFFORT,
    LiveSandboxConfig,
    create_text_transport,
    resolve_live_model,
)
from cle.traces import SQLiteLiveTraceStore
from cle.game_engine.events import PlayerEvent
from cle.game_engine.models.player import Color
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


@dataclass
class FixedTransport:
    requests: list = field(default_factory=list)
    response: ModelResponse | None = None

    async def complete(self, request):
        self.requests.append(request)
        if self.response is not None:
            return self.response
        return ModelResponse(
            content=(
                "<game_plan>build a balanced opening</game_plan>"
                "<action>0</action>"
            ),
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
            content = "<game_plan>start</game_plan><action>0</action>"
        else:
            content = "<game_plan>continue</game_plan><action>999</action>"
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


def test_stored_games_without_board_contract_restore_legacy_semantic_mode():
    config = _config_from_stored_payload(
        {"mode": "random", "seed": 7, "palette": "canonical_four"}
    )

    assert config.board_surface == "legacy_semantic"


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({}, {"effort": "high", "exclude": False}),
        ({"reasoning": None}, {"effort": "high", "exclude": False}),
        ({"reasoning": {"enabled": False}}, {"enabled": False}),
        (
            {"reasoning": {"effort": "xhigh", "exclude": False}},
            {"effort": "xhigh", "exclude": False},
        ),
        ({"reasoning": {}}, {}),
    ],
)
def test_stored_reasoning_defaults_only_when_missing_or_none(payload, expected):
    config = _config_from_stored_payload({"mode": "llm_vs_random", **payload})

    assert config.reasoning == expected


def test_live_model_defaults_to_qwen_and_preserves_explicit_overrides(monkeypatch):
    monkeypatch.delenv("CATAN_LLM_MODEL", raising=False)

    assert DEFAULT_LIVE_MODEL == "qwen/qwen3.8-27b"
    assert DEFAULT_LIVE_MAX_DECISION_ATTEMPTS == 1
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


def test_llm_live_route_uses_yaml_agent_player_without_legacy_llm_player(
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
    assert red_player.status()["kind"] == "agent"
    assert red_player.status()["messages"] == 2
    assert [message.role for message in transport.requests[0].messages] == [
        "system",
        "user",
    ]
    user_message = transport.requests[0].messages[-1].content
    assert "BOARD STATE:" in user_message
    assert user_message.count("VALID ACTIONS:") == 1
    assert "<valid_actions>" not in user_message
    assert "rationale" not in user_message.lower()
    assert "STRATEGIC OBJECTIVE FOR THIS PROBE:" in user_message
    assert "Maximizing raw pip count is not the objective" in user_message
    assert "nominal diversity without buildable combinations can be weak" in (
        user_message
    )
    assert "Round 1 (first settlement + road): RED -> BLUE -> WHITE -> ORANGE" in (
        user_message
    )
    assert dict(configs[0].reasoning) == {"effort": "high", "exclude": False}
    assert configs[0].max_tokens is None
    assert started.json["max_tokens"] is None
    assert started.json["state"]["live_inference"] == {
        "max_decision_attempts": 1,
        "max_tokens": None,
        "model": DEFAULT_LIVE_MODEL,
        "reasoning": {"effort": "high", "exclude": False},
    }
    receipt = next(iter(red_player.session.receipts.values()))
    assert receipt.choice.rationale == ""
    assert receipt.choice.native_reasoning == "native live analysis"
    assert receipt.choice.native_reasoning_details == ({"type": "reasoning.text"},)
    assert dict(receipt.choice.reasoning_request) == {
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
    assert stored["config"]["decision_suite"]["version"] == "10.0.0"
    assert len(stored["config"]["decision_suite"]["sha256"]) == 64
    assert "environment.board_state" in {
        component["id"]
        for component in stored["model_calls"][0]["request"]["components"]
    }
    checkpoint = client.get(f"/api/live-traces/{started.json['trace_game_id']}/steps/0").json
    assert checkpoint["model_calls"][0]["accepted"] is True
    assert "rationale" not in checkpoint["model_calls"][0]["choice"]
    assert checkpoint["model_calls"][0]["response"]["native_reasoning"] == ("native live analysis")
    assert checkpoint["model_calls"][0]["request"]["messages"][-1]["role"] == ("user")
    loaded = client.post(f"/api/live-traces/{started.json['trace_game_id']}/load")
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
    assert loaded.json["reasoning_request"] == {
        "effort": "xhigh",
        "exclude": False,
    }
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
    assert "Choose an action index from 0 to 2; received 999" in (
        failed_step.json["details"]
    )
    assert "No gameplay action was applied" in failed_step.json["details"]
    assert failed_step.json["attempt_count"] == 1
    assert failed_step.json["attempts"] == [
        {
            "context_id": transport.requests[-1].decision_id,
            "action_index": None,
            "final_response": (
                "<game_plan>continue</game_plan><action>999</action>"
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
            "validation_error": (
                "Choose an action index from 0 to 2; received 999."
            ),
        }
    ]
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
        "reasoning": {"effort": "xhigh", "exclude": False},
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
        content=f"<game_plan>{'opening plan ' * 400}</game_plan><action>0</action>",
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
    assert checkpoint["model_calls"] == stored["model_calls"]

    loaded = client.post(f"/api/live-traces/{game_id}/load")
    assert loaded.status_code == 200
    assert state.current_sandbox.revision == sandbox.revision
    assert len(state.current_sandbox.players[Color.RED].session.receipts) == 1
    assert len(transport.requests) == 1
    continued = client.post("/api/step")
    assert continued.status_code == 200
    assert continued.json["warning"] is None
    assert continued.json["trace_step_index"] == 1
    assert state.current_sandbox.game_engine.events[-1].event_type == "BUILD_ROAD"
    assert len(state.current_sandbox.players[Color.RED].session.receipts) == 2


def test_saved_game_restores_recorded_suite_after_source_file_changes(
    live_app,
    monkeypatch,
    tmp_path,
):
    app, state, _ = live_app
    source_path = tmp_path / "active-decision.yaml"
    original_source = (
        default_suite_path()
        .with_name("catan_v8.yaml")
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
        lambda config: FixedTransport(),
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
    assert restored.suite.version == "8.0.0"
    assert restored.suite.phase_guidance["discarding"].startswith(
        "ORIGINAL RECORDED DISCARD GUIDANCE."
    )
    assert "Maximizing raw pip count is not the objective" not in (
        restored.suite.phase_guidance["initial_settlement_1"]
    )
    trace = client.get(f"/api/live-traces/{game_id}").json
    assert trace["config"]["decision_suite"]["version"] == "8.0.0"
    assert trace["config"]["decision_suite"]["source"] == original_source


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
    assert loaded.json["max_decision_attempts"] == 1
    assert loaded.json["state"]["live_inference"]["reasoning"] == {"enabled": False}
    assert socket.emissions[-1][1] == loaded.json["state"]
    assert dict(configs[-1].reasoning) == {"enabled": False}
    assert transport.requests == []
    assert state.live_trace_store.get_game(game_id)["config"] == stored_config
