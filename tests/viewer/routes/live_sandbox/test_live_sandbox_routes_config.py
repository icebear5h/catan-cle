"""Live game configuration, board surfaces, and transports stay explicit."""
import pickle
from typing import Any

import pytest
from flask import Flask

from cle.harness import CompletionTransport, default_suite_path
from cle.harness.communication import default_communication_suite_path
from cle.harness.providers import (
    CerebrasTransport,
    OpenRouterTransport,
)
from cle.sandbox import CatanSandbox
from cle.sandbox.factory import (
    DEFAULT_LIVE_MAX_DECISION_ATTEMPTS,
    DEFAULT_LIVE_MODEL,
    DEFAULT_LIVE_REASONING_EFFORT,
    LiveSandboxConfig,
    create_live_sandbox,
    create_text_transport,
    resolve_live_model,
)
from cle.sandbox.replay import ReplaySandbox
from playground.game_viewer.routes.live_game import (
    _config_from_stored_payload,
)
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



def test_stored_games_do_not_select_an_inference_board_contract() -> None:
    config = _config_from_stored_payload(
        {"mode": "random", "seed": 7, "palette": "canonical_four"}
    )

    assert config.board_surface == "indexed_tile_rows"


def test_saved_unpinned_llm_game_uses_current_prompt_defaults(live_app: tuple[Flask, ServerState, DummySocket], monkeypatch: pytest.MonkeyPatch) -> None:
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
    game_id: Any = sandbox.game_engine.id
    snapshot: Any = sandbox.snapshot()
    state.live_trace_store.start_game(
        game_id, config={"mode": "llm_vs_random", "seed": 7}, snapshot=snapshot,
    )
    stored_before: Any = state.live_trace_store.get_game(game_id)
    client = app.test_client()
    started = client.post("/api/start-game", json={"mode": "random", "seed": 8})
    assert started.status_code == 200
    active: Any = state.current_sandbox
    active_before: Any = pickle.dumps(active.snapshot())

    loaded = client.post(f"/api/live-traces/{game_id}/load")

    assert loaded.status_code == 200, loaded.json
    assert state.current_sandbox is not active
    assert live_sandbox(state).snapshot().player_states == snapshot.player_states
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
def test_stored_reasoning_never_overrides_current_defaults(
    payload: dict[str, Any], expected: dict[str, str | bool]
) -> None:
    config = _config_from_stored_payload({"mode": "llm_vs_random", **payload})

    assert config.reasoning == expected


def test_live_model_defaults_to_qwen_and_preserves_explicit_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_live_route_can_select_typed_image_board_surface(live_app: tuple[Flask, ServerState, DummySocket], monkeypatch: pytest.MonkeyPatch) -> None:
    app, _, _ = live_app
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

    started: Any = client.post(
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
    board: Any = transport.requests[0].board_presentation
    assert board.kind == "image"
    assert board.media_type == "image/png"
    assert board.contains_entity_labels is False


def test_unsupported_transport_cannot_fake_native_reasoning(monkeypatch: pytest.MonkeyPatch) -> None:
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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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


@pytest.mark.asyncio
async def test_live_cerebras_model_prefix_selects_cerebras_over_env_providers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    monkeypatch.setenv("CEREBRAS_API_KEY", "cerebras-key")
    transport = create_text_transport(
        LiveSandboxConfig(mode="llm", model="cerebras/qwen-3.8-27b")
    )

    assert isinstance(transport, CerebrasTransport)
    assert transport.config.model == "qwen-3.8-27b"
    assert transport.config.max_tokens is None
    assert transport.reasoning_effort == "high"
    assert dict(transport.reasoning_request) == {
        "effort": "high",
        "exclude": False,
    }

    await transport.aclose()


def test_live_cerebras_rejects_image_board_surface(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CEREBRAS_API_KEY", "cerebras-key")

    with pytest.raises(ValueError, match="text-only"):
        create_text_transport(
            LiveSandboxConfig(
                mode="llm",
                model="cerebras/qwen-3.8-27b",
                board_surface="image",
            )
        )


def test_live_factory_does_not_fall_back_to_groq(monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_live_factory_accepts_snapshotted_trade_and_communication_limits(live_app: tuple[Flask, ServerState, DummySocket]) -> None:
    app, state, _ = live_app
    response: Any = app.test_client().post(
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
    engine = live_sandbox(state).game_engine
    assert engine.state.trade_limits.max_active_root_offers == 2
    assert engine.communication_limits.recent_message_window == 4
