from types import SimpleNamespace

import pytest
import yaml

from evals.decision_buckets import (
    STATE_FEATURE_SCHEMA,
    classify_decision_records,
    decision_state_features,
    default_bucket_suite_path,
    load_decision_bucket_suite,
)
from game_engine.game import GameEngine
from game_engine.models.player import Color


def _features(
    *,
    turns=0,
    actor_public=2,
    actor_actual=2,
    opponent_max=2,
    max_public=None,
    actions=(),
    road_length=0,
    played_knights=0,
    turn_owner=True,
):
    return {
        "schema": STATE_FEATURE_SCHEMA,
        "completed_turns": turns,
        "table_round": turns // 4 + 1,
        "player_count": 4,
        "actor_color": "BLUE",
        "turn_owner_color": "BLUE" if turn_owner else "RED",
        "actor_is_turn_owner": turn_owner,
        "actor_public_vp": actor_public,
        "actor_actual_vp": actor_actual,
        "opponent_max_public_vp": opponent_max,
        "max_public_vp": max_public if max_public is not None else max(actor_public, opponent_max),
        "public_vp": {
            "BLUE": actor_public,
            "RED": opponent_max,
            "ORANGE": 2,
            "BLACK": 2,
        },
        "actor_longest_road_length": road_length,
        "actor_played_knights": played_knights,
        "actor_has_longest_road": False,
        "actor_has_largest_army": False,
        "available_action_types": list(actions),
    }


def _record(index, action_type, *, phase="main_game", features=None, forced=False):
    return {
        "decision_id": f"game-1:{index}",
        "game_id": "game-1",
        "replay_index": index,
        "phase": phase,
        "effective_action_type": action_type,
        "forced": forced,
        "actor": {"engine_color": "BLUE"},
        "state_features": features,
    }


def test_bucket_suite_is_strict_versioned_and_complete():
    suite = load_decision_bucket_suite()

    assert suite.schema_id == "decision-bucket-suite-v1"
    assert suite.version == "1.0.0"
    assert tuple(rule.id for rule in suite.stage_rules) == (
        "setup",
        "early",
        "mid",
        "late",
    )
    assert len(suite.buckets) == 24
    assert len(suite.review_labels) == 12
    assert {verdict.id for verdict in suite.verdicts} == {
        "strong",
        "reasonable",
        "questionable",
        "blunder",
    }


def test_bucket_suite_rejects_unknown_fields(tmp_path):
    raw = yaml.safe_load(default_bucket_suite_path().read_text())
    raw["surprise"] = True
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw))

    with pytest.raises(ValueError, match="surprise"):
        load_decision_bucket_suite(path)


def test_setup_ordinals_produce_first_second_and_shared_road_buckets():
    records = [
        _record(0, "BUILD_SETTLEMENT", phase="initial_placement"),
        _record(1, "BUILD_ROAD", phase="initial_placement"),
        _record(12, "BUILD_SETTLEMENT", phase="initial_placement"),
        _record(13, "BUILD_ROAD", phase="initial_placement"),
    ]

    assignments = classify_decision_records(records)

    assert assignments[0]["bucket_ids"] == ["first_settlement"]
    assert assignments[1]["bucket_ids"] == ["initial_road"]
    assert assignments[2]["bucket_ids"] == ["second_settlement"]
    assert assignments[3]["bucket_ids"] == ["initial_road"]
    assert {assignment["stage"] for assignment in assignments} == {"setup"}
    assert assignments[0]["episode_ids"]["first_settlement"] == "game-1:setup:BLUE"


def test_early_robber_and_trade_decisions_are_multi_labelled():
    records = [
        _record(
            20,
            "MOVE_ROBBER",
            features=_features(turns=5, actions=("MOVE_ROBBER", "END_TURN")),
        ),
        _record(
            21,
            "REJECT_TRADE",
            features=_features(
                turns=5,
                actions=("ACCEPT_TRADE", "REJECT_TRADE", "COUNTER_OFFER"),
                turn_owner=False,
            ),
        ),
    ]

    assignments = classify_decision_records(records)

    assert assignments[0]["stage"] == "early"
    assert "early_robber" in assignments[0]["bucket_ids"]
    assert assignments[0]["episode_ids"]["early_robber"] == "game-1:turn:5:robber"
    assert assignments[1]["stage"] == "early"
    assert "early_trade_response" in assignments[1]["bucket_ids"]
    assert "multi_action_turn" not in assignments[1]["bucket_ids"]


def test_late_critical_turn_combines_win_denial_and_end_turn_buckets():
    record = _record(
        90,
        "END_TURN",
        features=_features(
            turns=40,
            actor_public=9,
            actor_actual=9,
            opponent_max=9,
            actions=("BUILD_CITY", "BUY_DEVELOPMENT_CARD", "END_TURN"),
        ),
    )

    assignment = classify_decision_records([record])[0]

    assert assignment["stage"] == "late"
    assert assignment["critical"] is True
    assert set(assignment["bucket_ids"]) == {
        "late_turn",
        "late_win_conversion",
        "late_opponent_denial",
        "end_turn_discipline",
    }
    assert assignment["evidence"]["actor_actual_vp"] == 9
    assert assignment["evidence"]["opponent_max_public_vp"] == 9


def test_state_feature_capture_uses_public_opponent_vp_and_actor_private_vp():
    game = GameEngine(
        [Color.BLUE, Color.RED, Color.ORANGE, Color.BLACK],
        shuffle_players=False,
        seed=2,
    )
    game.state.player_state["P0_VICTORY_POINTS"] = 7
    game.state.player_state["P0_ACTUAL_VICTORY_POINTS"] = 8
    game.state.player_state["P1_VICTORY_POINTS"] = 9
    action = SimpleNamespace(action_type=SimpleNamespace(value="BUILD_CITY"))

    features = decision_state_features(game.state, Color.BLUE, [action])

    assert features["schema"] == STATE_FEATURE_SCHEMA
    assert features["actor_public_vp"] == 7
    assert features["actor_actual_vp"] == 8
    assert features["opponent_max_public_vp"] == 9
    assert features["max_public_vp"] == 9
    assert features["available_action_types"] == ["BUILD_CITY"]
    assert features["actor_is_turn_owner"] is True
