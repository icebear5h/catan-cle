from types import SimpleNamespace

from cle.game_engine.game import GameEngine
from cle.game_engine.models.player import Color
from cle.replay.runtime.audit import (
    colonist_victory_points,
    expected_final_state_by_engine_index,
    record_replay_issue,
    replay_issues_since,
    sync_final_replay_state,
    validate_final_replay_state,
)


def test_colonist_victory_points_weights_city_awards_and_hidden_vp():
    result = colonist_victory_points({"0": 3, "1": 2, "2": 1, "3": 1, "4": 1})

    assert result["public"] == 11
    assert result["actual"] == 12


def test_replay_issue_filtering_by_severity():
    state = SimpleNamespace(replay_index=7)

    record_replay_issue(
        state,
        kind="forced_overlay",
        action_hint={"type": "OFFER_TRADE", "index": 12},
        message="forced",
        severity="warning",
    )
    record_replay_issue(
        state,
        kind="unmatched",
        action_hint={"type": "END_TURN", "index": 13},
        message="bad",
        severity="error",
    )

    assert [issue["kind"] for issue in replay_issues_since(state, 0)] == ["unmatched"]
    assert [issue["kind"] for issue in replay_issues_since(state, 0, "warning")] == [
        "forced_overlay",
        "unmatched",
    ]


def test_final_state_sync_forces_colonist_scoreboard_fields():
    game = GameEngine(
        [
            Color.RED,
            Color.BLUE,
            Color.WHITE,
            Color.ORANGE,
        ],
        shuffle_players=False,
    )
    state = SimpleNamespace(
        replay_index=1,
        current_game=game,
        replay_semantic_issues=[],
        replay_final_state_synced=False,
        replay_pending_dev_card=None,
        replay_data={
            "colonist_color_to_engine_idx": {"1": 0},
            "initial_state": {
                "mechanicLargestArmyState": {"1": {}},
                "mechanicLongestRoadState": {"1": {"longestRoad": 0}},
            },
            "events": [
                {
                    "stateChange": {
                        "mechanicLargestArmyState": {"1": {"hasLargestArmy": True}},
                        "mechanicLongestRoadState": {
                            "1": {"longestRoad": 5, "hasLongestRoad": True}
                        },
                    }
                }
            ],
            "end_game_state": {
                "players": {
                    "1": {
                        "victoryPoints": {"0": 3, "1": 2, "2": 1, "3": 1, "4": 1},
                        "winningPlayer": True,
                    }
                }
            },
        },
    )

    expected = expected_final_state_by_engine_index(state.replay_data)
    assert expected[0]["actual_vp"] == 12
    assert validate_final_replay_state(state)

    sync_final_replay_state(state)

    assert validate_final_replay_state(state) == []
    assert state.current_game.state.player_state["P0_ACTUAL_VICTORY_POINTS"] == 12
    assert state.current_game.state.player_state["P0_HAS_ARMY"] is True
    assert state.current_game.state.player_state["P0_HAS_ROAD"] is True
    assert state.current_game.state.player_state["P0_LONGEST_ROAD_LENGTH"] == 5
    assert any(issue["kind"] == "forced_final_state_sync" for issue in state.replay_semantic_issues)
