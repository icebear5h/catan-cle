from copy import deepcopy
from threading import Lock
from types import SimpleNamespace

from flask import Flask

from engine.game import Game
from engine.models.player import Color, SimplePlayer
from playground.game_viewer.colonist.event_parser import (
    parse_colonist_events_to_actions,
)
from playground.game_viewer.replay.llm_response import (
    CONTEXT_VERSION,
    _build_prompts,
    build_replay_decision_context,
    format_visible_replay_activity,
    generate_replay_llm_response,
    parse_replay_llm_output,
    select_recent_activity_rows,
    validate_model_id,
)
from playground.game_viewer.routes.replay import replay_bp


def _make_game():
    return Game(
        [
            SimplePlayer(Color.RED),
            SimplePlayer(Color.BLUE),
            SimplePlayer(Color.WHITE),
            SimplePlayer(Color.ORANGE),
        ],
        shuffle_players=False,
    )


def _replay_data(parsed_actions=None):
    return {
        "game_id": "test-game",
        "parsed_actions": parsed_actions or [],
        "colonist_color_to_engine_idx": {"1": 0, "2": 1, "9": 2, "3": 3},
    }


def _fake_provider_response(**kwargs):
    return {
        "content": (
            "<goals>Prioritize production and expansion.</goals>"
            "<reasoning>This legal placement has the strongest long-term value.</reasoning>"
            "<action>0</action>"
            "<message>BLUE, leave my ore alone and I will not robber your wheat.</message>"
        ),
        "model": kwargs["model"],
        "usage": {"prompt_tokens": 100, "completion_tokens": 30},
        "latency_ms": 25,
    }


def _make_route_app(state, query_fn=_fake_provider_response):
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.config["SERVER_STATE"] = state
    app.config["REPLAY_LLM_QUERY_FN"] = query_fn
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


def test_setup_phase_rules_only_appear_during_initial_placement():
    game = _make_game()
    replay_data = _replay_data([{"type": "END_TURN", "player": 2}])

    setup_context = build_replay_decision_context(
        game.state.copy(), replay_data, 1, ""
    )
    assert setup_context["phase"] == "initial_placement"
    system_prompt, _ = _build_prompts(setup_context)
    assert "Setup phase rules" in system_prompt
    assert "collect one resource" in system_prompt

    main_state = game.state.copy()
    main_state.is_initial_build_phase = False
    main_context = build_replay_decision_context(main_state, replay_data, 1, "")
    assert main_context["phase"] == "main"
    main_system_prompt, _ = _build_prompts(main_context)
    assert "Setup phase rules" not in main_system_prompt


def test_observation_excludes_embedded_action_menu():
    game = _make_game()
    context = build_replay_decision_context(game.state.copy(), _replay_data(), 0, "")

    assert "<valid_actions>" not in context["observation"]
    assert "</valid_actions>" not in context["observation"]
    # The board state itself must survive the strip.
    assert "<board_state>" in context["observation"]
    assert "</game_state>" in context["observation"]

    _, user_prompt = _build_prompts(context)
    # The indexed legal_actions list is the single action menu in the prompt.
    assert user_prompt.count("<legal_actions>") == 1
    assert "<valid_actions>" not in user_prompt
    assert context["available_actions"][0]["description"] in user_prompt


def test_parse_replay_llm_output_extracts_optional_table_talk_message():
    parsed = parse_replay_llm_output(
        "<goals>Win.</goals><reasoning>Best value.</reasoning><action>0</action>"
        "<message>WHITE, want to trade sheep for brick?</message>",
        action_count=2,
    )
    assert parsed["message"] == "WHITE, want to trade sheep for brick?"
    assert parsed["parse_error"] is None

    no_message = parse_replay_llm_output(
        "<goals>Win.</goals><reasoning>Best value.</reasoning><action>0</action>",
        action_count=2,
    )
    assert no_message["message"] == ""
    assert no_message["parse_error"] is None


def test_parse_replay_llm_output_never_substitutes_an_invalid_action():
    parsed = parse_replay_llm_output(
        "<goals>Build cities.</goals><reasoning>Strong production.</reasoning><action>99</action>",
        action_count=3,
    )

    assert parsed["goals"] == "Build cities."
    assert parsed["reasoning"] == "Strong production."
    assert parsed["action_index"] is None
    assert "outside the valid range" in parsed["parse_error"]


def test_generate_replay_response_uses_custom_model_and_does_not_mutate_game():
    game = _make_game()
    replay_data = _replay_data(
        [
            {
                "type": "ROLL",
                "player": 2,
                "dice": [3, 4],
                "resource_payouts": {2: (0, 0, 0, 1, 0)},
                "resource_payouts_complete": True,
            },
            {"type": "END_TURN", "player": 2},
        ]
    )
    provider_call = {}

    def capture_provider_call(**kwargs):
        provider_call.update(kwargs)
        return _fake_provider_response(**kwargs)

    before = {
        "actions": list(game.state.actions),
        "resources": list(game.state.resource_freqdeck),
        "player_state": deepcopy(game.state.player_state),
        "history_length": len(game.history),
        "replay_data": deepcopy(replay_data),
    }

    result = generate_replay_llm_response(
        game=game,
        replay_data=replay_data,
        replay_index=2,
        model="google/gemini-2.5-flash",
        prior_goals="Expand toward ore.",
        query_fn=capture_provider_call,
    )

    assert result["context_version"] == CONTEXT_VERSION
    assert result["player_color"] == "RED"
    assert result["requested_model"] == "google/gemini-2.5-flash"
    assert result["action_index"] == 0
    assert result["action"] == result["available_actions"][0]["action"]
    assert provider_call["provider"] == "openrouter"
    assert provider_call["model"] == "google/gemini-2.5-flash"
    assert "Expand toward ore." in provider_call["prompt"]
    assert "prior_completed_turn_plus_current_partial_turn" in provider_call["prompt"]
    assert "\n  - BLUE: +1 WHEAT" in provider_call["prompt"]

    assert game.state.actions == before["actions"]
    assert game.state.resource_freqdeck == before["resources"]
    assert game.state.player_state == before["player_state"]
    assert len(game.history) == before["history_length"]
    assert replay_data == before["replay_data"]


def test_generate_replay_response_surfaces_provider_truncation():
    game = _make_game()
    replay_data = _replay_data([{"type": "END_TURN", "player": 2}])

    def truncated_provider(**kwargs):
        return {
            "content": (
                "<goals>Secure five more victory points.</goals>"
                "<reasoning>I possess only 1 WOOD and 2 SHEEP, which falls short of"
            ),
            "model": kwargs["model"],
            "usage": {"prompt_tokens": 900, "completion_tokens": 8192},
            "latency_ms": 90,
            "finish_reason": "length",
        }

    result = generate_replay_llm_response(
        game=game,
        replay_data=replay_data,
        replay_index=1,
        model="qwen/qwen3.7-flash",
        prior_goals="",
        query_fn=truncated_provider,
    )

    assert result["finish_reason"] == "length"
    assert result["response_truncated"] is True
    assert result["action_index"] is None
    assert result["parse_error"]


def test_generate_replay_response_requests_expanded_completion_budget():
    game = _make_game()
    replay_data = _replay_data([{"type": "END_TURN", "player": 2}])
    provider_call = {}

    def capture_provider_call(**kwargs):
        provider_call.update(kwargs)
        return _fake_provider_response(**kwargs)

    result = generate_replay_llm_response(
        game=game,
        replay_data=replay_data,
        replay_index=1,
        model="google/gemini-2.5-flash",
        prior_goals="",
        query_fn=capture_provider_call,
    )

    assert provider_call["max_tokens"] == 8_192
    assert result["response_truncated"] is False
    assert result["finish_reason"] is None


def test_replay_llm_route_returns_response_without_advancing_cursor():
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
        json={"model": "anthropic/claude-sonnet-4", "goals": "Build efficiently."},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["game_id"] == "test-game"
    assert payload["replay_index"] == 0
    assert payload["stale"] is False
    assert state.replay_index == 0
    assert game.state.actions == before_actions
    assert len(game.history) == before_history_length


def test_replay_llm_route_marks_response_stale_if_cursor_moves_during_query():
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
        return _fake_provider_response(**kwargs)

    app = _make_route_app(state, query_fn=move_cursor_during_query)
    response = app.test_client().post(
        "/api/replay-llm-response",
        json={"model": "openrouter/auto"},
    )

    assert response.status_code == 200
    assert response.get_json()["stale"] is True


def test_replay_llm_route_rejects_invalid_model_and_concurrent_request():
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

    state.replay_llm_lock.acquire()
    try:
        concurrent = client.post(
            "/api/replay-llm-response",
            json={"model": validate_model_id("openrouter/auto")},
        )
    finally:
        state.replay_llm_lock.release()

    assert concurrent.status_code == 429
