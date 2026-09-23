"""The replay LLM route answers without advancing or racing the cursor."""
from threading import Lock
from types import SimpleNamespace
from typing import Any

from .support import (
    _fake_general_provider_response,
    _make_game,
    _make_route_app,
    _replay_data,
)


def test_replay_llm_route_returns_response_without_advancing_cursor(legacy_prompt_pair: None) -> None:
    game = _make_game()
    state = SimpleNamespace(
        replay_mode=True,
        replay_data=_replay_data(),
        current_game=game,
        replay_index=0,
        replay_llm_lock=Lock(),
    )
    app = _make_route_app(state)
    before_actions = list(game.state.actions)
    before_history_length = len(game.history)

    response = app.test_client().post(
        "/api/replay-llm-response",
        json={
            "model": "anthropic/claude-sonnet-4",
            "game_plan": "Build efficiently.",
            "reasoning": {"effort": "xhigh", "exclude": False},
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["schema"] == "agent-decision-preview-v2"
    assert payload["game_id"] == "test-game"
    assert payload["replay_index"] == 0
    assert payload["stale"] is False
    assert payload["game_plan"] == "Prioritize production and expansion."
    assert payload["parse_error"] is None
    assert payload["action_index"] == 0
    assert '"tool":"build_settlement"' in payload["raw_response"]
    assert "rationale" not in payload
    assert payload["native_reasoning"] == "private native analysis"
    assert payload["native_reasoning_returned"] is True
    assert payload["native_reasoning_missing"] is False
    assert payload["reasoning_tokens"] == 18
    assert payload["reasoning_request"] == {
        "effort": "xhigh",
        "exclude": False,
    }
    assert payload["provider_response_id"] == "gen-replay-test"
    assert payload["provider_request_id"] == "req-replay-test"
    assert payload["provider_native_finish_reason"] == "stop"
    assert "YOUR CURRENT GAME PLAN" in payload["model_messages"][1]["content"]
    assert "Build efficiently." in payload["model_messages"][1]["content"]
    assert "rationale" not in payload["model_messages"][1]["content"].lower()
    assert state.replay_index == 0
    assert game.state.actions == before_actions
    assert len(game.history) == before_history_length


def test_replay_llm_route_marks_response_stale_if_cursor_moves_during_query(legacy_prompt_pair: None) -> None:
    game = _make_game()
    state = SimpleNamespace(
        replay_mode=True,
        replay_data=_replay_data(),
        current_game=game,
        replay_index=0,
        replay_llm_lock=Lock(),
    )

    def move_cursor_during_query(**kwargs: object) -> dict[str, Any]:
        state.replay_index = 1
        return _fake_general_provider_response(**kwargs)

    app = _make_route_app(state, query_fn=move_cursor_during_query)
    response = app.test_client().post(
        "/api/replay-llm-response",
        json={"model": "openrouter/auto"},
    )

    assert response.status_code == 200
    assert response.get_json()["stale"] is True
    assert response.get_json()["reasoning_request"] == {
        "effort": "xhigh",
        "exclude": False,
    }


def test_replay_llm_route_preserves_explicit_reasoning_off(legacy_prompt_pair: None) -> None:
    game = _make_game()
    state = SimpleNamespace(
        replay_mode=True,
        replay_data=_replay_data(),
        current_game=game,
        replay_index=0,
        replay_llm_lock=Lock(),
    )
    app = _make_route_app(state)

    response = app.test_client().post(
        "/api/replay-llm-response",
        json={
            "model": "openrouter/auto",
            "reasoning": {"enabled": False},
        },
    )

    assert response.status_code == 200
    assert response.get_json()["reasoning_request"] == {"enabled": False}


def test_replay_llm_route_rejects_invalid_model_and_concurrent_request(legacy_prompt_pair: None) -> None:
    game = _make_game()
    state = SimpleNamespace(
        replay_mode=True,
        replay_data=_replay_data(),
        current_game=game,
        replay_index=0,
        replay_llm_lock=Lock(),
    )
    app = _make_route_app(state)
    client = app.test_client()

    invalid = client.post(
        "/api/replay-llm-response",
        json={"model": "bad model with spaces"},
    )
    assert invalid.status_code == 400
    assert "unsupported characters" in invalid.get_json()["error"]

    hidden_reasoning = client.post(
        "/api/replay-llm-response",
        json={
            "model": "openrouter/auto",
            "reasoning": {"effort": "high", "exclude": True},
        },
    )
    assert hidden_reasoning.status_code == 400
    assert "exclude must be false" in hidden_reasoning.get_json()["error"]

    state.replay_llm_lock.acquire()
    try:
        concurrent = client.post(
            "/api/replay-llm-response",
            json={"model": "openrouter/auto"},
        )
    finally:
        state.replay_llm_lock.release()

    assert concurrent.status_code == 429
