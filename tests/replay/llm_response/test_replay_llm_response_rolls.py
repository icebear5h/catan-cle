"""Colonist roll parsing keeps only trustworthy public payout deltas."""

from cle.replay.colonist.event_parser import parse_colonist_events_to_actions


def test_roll_parser_records_only_trustworthy_public_payout_deltas() -> None:
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


def test_roll_parser_rejects_mixed_negative_resource_deltas() -> None:
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


def test_roll_parser_records_complete_public_payouts_for_known_roster() -> None:
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


def test_roll_parser_marks_trustworthy_no_production_roll() -> None:
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
