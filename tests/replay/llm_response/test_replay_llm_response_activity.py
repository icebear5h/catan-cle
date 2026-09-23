"""Recent-activity selection and formatting hide private hands and steals."""
from typing import Any

from cle.game_engine.models.player import Color
from cle.replay.activity import (
    format_visible_replay_activity,
    select_recent_activity_rows,
)

from .support import _make_game, _replay_data


def test_recent_activity_selects_previous_completed_and_current_partial_turn() -> None:
    actions: Any = [
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


def test_activity_formatter_exposes_public_roll_payouts_without_full_hands() -> None:
    game: Any = _make_game()
    replay_data: Any = _replay_data()
    action: Any = {
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


def test_activity_formatter_redacts_hidden_cards_and_steals() -> None:
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
