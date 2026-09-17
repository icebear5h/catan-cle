from threading import Lock
from types import SimpleNamespace

import pytest
from flask import Flask

from cle.harness.models import ModelResponse
from cle.harness.prompt_store import resolve_prompt_suites
from cle.replay.activity import (
    format_visible_replay_activity,
    select_recent_activity_rows,
)
from cle.replay.colonist.event_parser import parse_colonist_events_to_actions
from cle.sandbox.replay import ReplaySandbox
from cle.game_engine.game import GameEngine
from cle.game_engine.models.player import Color
from playground.game_viewer.routes.replay import replay_bp


@pytest.fixture
def legacy_prompt_pair(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "playground.game_viewer.replay.decision_preview.resolve_prompt_suites",
        lambda: resolve_prompt_suites(
            directory=tmp_path, legacy=True, use_environment=False,
        ),
    )


def _make_game():
    return GameEngine(
        [
            Color.RED,
            Color.BLUE,
            Color.WHITE,
            Color.ORANGE,
        ],
        shuffle_players=False,
    )


def _replay_data(parsed_actions=None):
    return {
        "game_id": "test-game",
        "parsed_actions": parsed_actions or [],
        "colonist_color_to_engine_idx": {"1": 0, "2": 1, "9": 2, "3": 3},
    }


def _fake_general_provider_response(**kwargs):
    return {
        "content": (
            '{"game_plan":"Prioritize production and expansion.",'
            '"tool":"build_settlement","arguments":{"node":"<N00>"}}'
        ),
        "model": kwargs["model"],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 30,
            "completion_tokens_details": {"reasoning_tokens": 18},
        },
        "latency_ms": 25,
        "native_reasoning": "private native analysis",
        "native_reasoning_details": [{"type": "reasoning.text"}],
        "provider_response_id": "gen-replay-test",
        "provider_request_id": "req-replay-test",
        "provider_native_finish_reason": "stop",
    }


class _QueryTransport:
    def __init__(self, query_fn, config):
        self.query_fn = query_fn
        self.config = config

    async def complete(self, request):
        result = self.query_fn(
            **self.config,
            system_prompt=request.messages[0].content,
            prompt=request.messages[-1].content,
        )
        return ModelResponse(
            content=result.get("content") or "",
            model=result.get("model"),
            usage=tuple((result.get("usage") or {}).items()),
            latency_ms=result.get("latency_ms"),
            finish_reason=result.get("finish_reason"),
            native_reasoning=result.get("native_reasoning") or "",
            native_reasoning_details=tuple(
                result.get("native_reasoning_details") or ()
            ),
            reasoning_request=tuple(self.config["reasoning"].items()),
            provider_response_id=result.get("provider_response_id"),
            provider_request_id=result.get("provider_request_id"),
            provider_native_finish_reason=result.get(
                "provider_native_finish_reason"
            ),
        )


def _make_route_app(state, query_fn=_fake_general_provider_response):
    if not hasattr(state, "current_sandbox"):
        state.current_sandbox = ReplaySandbox(state, state.current_game)
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.config["SERVER_STATE"] = state
    app.config["REPLAY_COMPLETION_TRANSPORT_FACTORY"] = (
        lambda **config: _QueryTransport(query_fn, config)
    )
    app.register_blueprint(replay_bp)
    return app


def test_roll_parser_records_only_trustworthy_public_payout_deltas():
    events = [
        {
            "stateChange": {
                "currentState": {
                    "actionState": 24,
                    "currentTurnPlayerColor": 1,
                },
                "playerStates": {
                    "1": {"resourceCards": {"cards": [1]}},
                    "2": {"resourceCards": {"cards": [3]}},
                    "3": {"resourceCards": {"cards": [1, 2]}},
                },
            }
        },
        {
            "stateChange": {
                "diceState": {
                    "dice1": 4,
                    "dice2": 5,
                    "diceThrown": True,
                },
                "currentState": {"currentTurnPlayerColor": 1},
                "playerStates": {
                    "1": {"resourceCards": {"cards": [1, 4, 4]}},
                    "2": {"resourceCards": {"cards": [3, 5]}},
                    "3": {"resourceCards": {"cards": [1, 2]}},
                    "9": {"resourceCards": {"cards": [2]}},
                },
            }
        },
    ]

    actions = parse_colonist_events_to_actions(
        events,
        player_ids=[1, 2, 3, 9],
    )
    roll = next(action for action in actions if action["type"] == "ROLL")

    assert roll["resource_payouts"] == {
        1: (0, 0, 0, 2, 0),
        2: (0, 0, 0, 0, 1),
    }
    assert roll["resource_payouts_complete"] is False
    assert 3 not in roll["resource_payouts"]
    assert 9 not in roll["resource_payouts"]


def test_roll_parser_rejects_mixed_negative_resource_deltas():
    actions = parse_colonist_events_to_actions(
        [
            {
                "stateChange": {
                    "currentState": {
                        "actionState": 24,
                        "currentTurnPlayerColor": 1,
                    },
                    "playerStates": {
                        "1": {"resourceCards": {"cards": [1]}},
                        "2": {"resourceCards": {"cards": [2]}},
                    },
                }
            },
            {
                "stateChange": {
                    "diceState": {
                        "dice1": 4,
                        "dice2": 5,
                        "diceThrown": True,
                    },
                    "currentState": {"currentTurnPlayerColor": 1},
                    "playerStates": {
                        "1": {"resourceCards": {"cards": [1, 4]}},
                        "2": {"resourceCards": {"cards": []}},
                    },
                }
            },
        ],
        player_ids=[1, 2],
    )
    roll = next(action for action in actions if action["type"] == "ROLL")

    assert roll["resource_payouts"] == {}
    assert roll["resource_payouts_complete"] is False


def test_roll_parser_records_complete_public_payouts_for_known_roster():
    actions = parse_colonist_events_to_actions(
        [
            {
                "stateChange": {
                    "currentState": {
                        "actionState": 24,
                        "currentTurnPlayerColor": 1,
                    },
                    "playerStates": {
                        "1": {"resourceCards": {"cards": [1]}},
                        "2": {"resourceCards": {"cards": [3]}},
                    },
                }
            },
            {
                "stateChange": {
                    "diceState": {
                        "dice1": 4,
                        "dice2": 5,
                        "diceThrown": True,
                    },
                    "currentState": {"currentTurnPlayerColor": 1},
                    "playerStates": {
                        "1": {"resourceCards": {"cards": [1, 4, 4]}},
                        "2": {"resourceCards": {"cards": [3, 5]}},
                    },
                }
            },
        ],
        player_ids=[1, 2],
    )
    roll = next(action for action in actions if action["type"] == "ROLL")

    assert roll["resource_payouts"] == {
        1: (0, 0, 0, 2, 0),
        2: (0, 0, 0, 0, 1),
    }
    assert roll["resource_payouts_complete"] is True


def test_roll_parser_marks_trustworthy_no_production_roll():
    actions = parse_colonist_events_to_actions(
        [
            {
                "stateChange": {
                    "currentState": {
                        "actionState": 24,
                        "currentTurnPlayerColor": 1,
                    },
                    "playerStates": {
                        "1": {"resourceCards": {"cards": [1]}},
                        "2": {"resourceCards": {"cards": [3]}},
                    },
                }
            },
            {
                "stateChange": {
                    "diceState": {
                        "dice1": 3,
                        "dice2": 4,
                        "diceThrown": True,
                    },
                    "currentState": {"currentTurnPlayerColor": 1},
                }
            },
        ],
        player_ids=[1, 2],
    )
    roll = next(action for action in actions if action["type"] == "ROLL")

    assert roll["resource_payouts"] == {}
    assert roll["resource_payouts_complete"] is True


def test_recent_activity_selects_previous_completed_and_current_partial_turn():
    actions = [
        {"type": "ROLL", "player": 1},
        {"type": "END_TURN", "player": 1},
        {"type": "ROLL", "player": 2},
        {"type": "BUILD_ROAD", "player": 2},
        {"type": "END_TURN", "player": 2},
        {"type": "ROLL", "player": 9},
        {"type": "BUY_DEVELOPMENT_CARD", "player": 9},
    ]

    rows, window = select_recent_activity_rows(actions, replay_index=len(actions))

    assert rows == actions[2:]
    assert window == {
        "start_replay_index": 2,
        "end_replay_index": 7,
        "row_count": 5,
        "truncated": False,
    }


def test_activity_formatter_exposes_public_roll_payouts_without_full_hands():
    game = _make_game()
    replay_data = _replay_data()
    action = {
        "type": "ROLL",
        "player": 1,
        "dice": [6, 4],
        "resource_payouts": {
            1: (0, 0, 0, 1, 0),
            2: (0, 0, 1, 0, 0),
        },
        "resource_payouts_complete": True,
        "expected_resources": {
            1: [1, 1, 4],
            2: [3, 5, 5, 5],
        },
    }

    activity = format_visible_replay_activity(
        action,
        replay_data,
        game.state.colors,
        Color.WHITE,
    )
    partial = format_visible_replay_activity(
        {
            "type": "ROLL",
            "player": 1,
            "dice": [4, 5],
            "resource_payouts": {2: (1, 0, 0, 0, 0)},
            "resource_payouts_complete": False,
        },
        replay_data,
        game.state.colors,
        Color.RED,
    )
    no_payout = format_visible_replay_activity(
        {
            "type": "ROLL",
            "player": 2,
            "dice": [3, 4],
            "resource_payouts": {},
            "resource_payouts_complete": True,
        },
        replay_data,
        game.state.colors,
        Color.RED,
    )

    assert activity == (
        "RED: rolled 6 + 4 = 10\n"
        "  - RED: +1 WHEAT\n"
        "  - BLUE: +1 SHEEP"
    )
    assert "ORE" not in activity
    assert partial == (
        "RED: rolled 4 + 5 = 9\n"
        "  - BLUE: +1 WOOD\n"
        "  - Additional payouts may be unavailable"
    )
    assert no_payout == (
        "BLUE: rolled 3 + 4 = 7\n"
        "  - No resource payouts"
    )


def test_activity_formatter_redacts_hidden_cards_and_steals():
    game = _make_game()
    replay_data = _replay_data()
    colors = game.state.colors

    opponent_card = format_visible_replay_activity(
        {"type": "BUY_DEVELOPMENT_CARD", "player": 2, "card_type": "MONOPOLY"},
        replay_data,
        colors,
        Color.RED,
    )
    own_card = format_visible_replay_activity(
        {"type": "BUY_DEVELOPMENT_CARD", "player": 1, "card_type": "KNIGHT"},
        replay_data,
        colors,
        Color.RED,
    )
    hidden_steal = format_visible_replay_activity(
        {"type": "STEAL", "player": 2, "victim": 9, "stolen_resource": "ORE"},
        replay_data,
        colors,
        Color.RED,
    )
    visible_steal = format_visible_replay_activity(
        {"type": "STEAL", "player": 2, "victim": 1, "stolen_resource": "WHEAT"},
        replay_data,
        colors,
        Color.RED,
    )

    assert opponent_card == "BLUE: bought an unknown development card"
    assert "MONOPOLY" not in opponent_card
    assert own_card == "RED: bought development card KNIGHT"
    assert hidden_steal == "BLUE: stole an unknown resource from WHITE"
    assert "ORE" not in hidden_steal
    assert visible_steal == "BLUE: stole WHEAT from RED"


def test_replay_llm_route_returns_response_without_advancing_cursor(legacy_prompt_pair):
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


def test_replay_llm_route_marks_response_stale_if_cursor_moves_during_query(legacy_prompt_pair):
    game = _make_game()
    state = SimpleNamespace(
        replay_mode=True,
        replay_data=_replay_data(),
        current_game=game,
        replay_index=0,
        replay_llm_lock=Lock(),
    )

    def move_cursor_during_query(**kwargs):
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


def test_replay_llm_route_preserves_explicit_reasoning_off(legacy_prompt_pair):
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


def test_replay_llm_route_rejects_invalid_model_and_concurrent_request(legacy_prompt_pair):
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
