"""Rolls, legacy action text, and table talk are each logged once."""
from types import SimpleNamespace
from typing import Any

from cle.game_engine.models.enums import Action, ActionType
from cle.game_engine.models.player import Color
from playground.game_viewer.live.game_logging import (
    analyze_transitions,
    format_action_for_display,
    format_trade_event,
    normalize_public_state_game_log,
    stamp_message_step_indexes,
)


def test_single_roll_transition_logs_payout_once() -> None:
    before: Any = SimpleNamespace(colors=(Color.RED,), player_state={})
    after: Any = SimpleNamespace(
        colors=(Color.RED,),
        player_state={"P0_WOOD_IN_HAND": 1},
        last_dice_roll=(3, 3),
    )
    transition = SimpleNamespace(
        requested_action=Action(Color.RED, ActionType.ROLL, None),
        events=(),
    )
    state = SimpleNamespace(game_log=[])

    analyze_transitions(state, (transition,), before, after)

    assert [entry["message"] for entry in state.game_log] == [
        "Rolled 3 + 3 = 6",
        "Gained 1🪵",
    ]


def test_non_trade_action_display_remains_legacy_action_text() -> None:
    action = Action(Color.RED, ActionType.BUILD_SETTLEMENT, 12)

    assert format_action_for_display(action) == str(action)
    assert format_trade_event("BUILD_SETTLEMENT", 12) is None


def test_table_talk_messages_are_logged_once_across_messages_and_transitions() -> None:
    spoken: Any = SimpleNamespace(
        event_type="MESSAGE_SENT",
        public_payload={
            "speaker": Color.GREEN,
            "text": "Leave T09 alone?",
            "audience": [Color.BLUE],
            "respondents": [Color.BLUE],
        },
        sequence=135,
    )
    transition: Any = SimpleNamespace(
        requested_action=Action(Color.RED, ActionType.END_TURN, None),
        events=(spoken,),
    )
    state: Any = SimpleNamespace(game_log=[])

    analyze_transitions(state, (transition,), None, None, messages=(spoken,))

    assert [entry["message"] for entry in state.game_log] == [
        "Leave T09 alone?",
    ]
    assert stamp_message_step_indexes(state.game_log, 0, 8) == 1
    assert state.game_log[0]["step_index"] == 8
    assert [entry["type"] for entry in state.game_log] == ["message"]
    assert [entry["color"] for entry in state.game_log] == ["GREEN"]


def test_stamp_message_step_indexes_labels_only_this_step_speech() -> None:
    game_log: Any = [
        {"type": "message", "message": "earlier turn", "details": {"sequence": 100}},
        {"type": "building", "message": "Built a settlement", "details": None},
        {"type": "message", "message": "this turn", "details": {"sequence": 201}},
        {"type": "message", "message": "reply", "details": {"sequence": 202}},
    ]

    assert stamp_message_step_indexes(game_log, 1, 12) == 2

    assert "step_index" not in game_log[0]
    assert "step_index" not in game_log[0]["details"]
    assert "step_index" not in game_log[1]
    assert [entry["step_index"] for entry in game_log[2:]] == [12, 12]
    assert [entry["details"] for entry in game_log[2:]] == [
        {"sequence": 201, "step_index": 12},
        {"sequence": 202, "step_index": 12},
    ]


def test_stamp_message_step_indexes_tolerates_steps_without_speech() -> None:
    game_log: Any = [{"type": "dice", "message": "Rolled 3 + 4 = 7", "details": None}]

    assert stamp_message_step_indexes(game_log, 0, 3) == 0
    assert game_log == [{"type": "dice", "message": "Rolled 3 + 4 = 7", "details": None}]
    assert stamp_message_step_indexes(game_log, len(game_log), 4) == 0


def test_sliced_checkpoint_log_regains_speech_rows_from_events() -> None:
    """A checkpoint written under the old 50-row window kept its events but not its
    older speech rows; loading it must rebuild them in engine-sequence order."""
    def said(sequence: int, speaker: str, text: str) -> dict[str, Any]:
        return {
            "sequence": sequence, "actor": speaker, "event_type": "MESSAGE_SENT",
            "payload": {
                "speaker": speaker, "text": text,
                "audience": [], "respondents": [],
            },
        }
    events: Any = [
        {"sequence": 134, "actor": "BRONZE", "event_type": "ROLL", "payload": {}},
        said(135, "GREEN", "put the robber on T06"),
        said(136, "ORANGE", "T16 only touches BLUE"),
        {"sequence": 300, "actor": "BLUE", "event_type": "BUILD_ROAD", "payload": {}},
        said(344, "BLUE", "T01 hits GREEN's city"),
        said(470, "GREEN", "echoing BRONZE"),
    ]
    sliced_log: Any = [
        {"type": "dice", "timestamp": 10.0, "message": "Rolled 3 + 4 = 7", "color": "BLUE", "details": None},
        {"type": "message", "timestamp": 11.0, "message": "T01 hits GREEN's city", "color": "BLUE",
         "step_index": 235, "details": {"event_type": "MESSAGE_SENT", "sequence": 344, "step_index": 235,
                                        "payload": events[4]["payload"]}},
        {"type": "building", "timestamp": 12.0, "message": "Built a road", "color": "BLUE",
         "details": {"sequence": 400}},
    ]

    normalized: Any = normalize_public_state_game_log({"game_log": sliced_log, "events": events})
    log: Any = normalized["game_log"]

    assert [(entry["type"], entry.get("details", {}) and entry["details"].get("sequence")) for entry in log] == [
        ("dice", None),
        ("message", 135),
        ("message", 136),
        ("message", 344),
        ("building", 400),
        ("message", 470),
    ]
    rebuilt = log[1]
    assert rebuilt["message"] == "put the robber on T06"
    assert rebuilt["color"] == "GREEN"
    assert rebuilt["details"]["payload"]["text"] == "put the robber on T06"
    assert rebuilt["timestamp"] == 11.0  # borrows its neighbour's clock
    assert log[2]["message"] == "T16 only touches BLUE"
    # The row that survived the slice is untouched, including its step label.
    assert log[3] == sliced_log[1]
    assert log[5]["timestamp"] == 12.0
    # Idempotent: a second pass adds nothing.
    again = normalize_public_state_game_log({"game_log": log, "events": events})["game_log"]
    assert again == log
    # No events: nothing is invented. No log at all: speech is still rebuilt.
    assert normalize_public_state_game_log({"game_log": sliced_log})["game_log"] == sliced_log
    from_events_only = normalize_public_state_game_log({"game_log": None, "events": events})["game_log"]
    assert [entry["details"]["sequence"] for entry in from_events_only] == [135, 136, 344, 470]
    assert {entry["type"] for entry in from_events_only} == {"message"}
