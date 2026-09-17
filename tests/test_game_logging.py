from types import SimpleNamespace

from cle.game_engine.models.enums import Action, ActionType
from cle.game_engine.models.player import Color
from cle.game_engine.trading import TradeOffer
from playground.game_viewer.live.game_logging import (
    analyze_action,
    analyze_transitions,
    format_action_for_display,
    format_trade_event,
    normalize_public_state_game_log,
    stamp_message_step_indexes,
)


def _offer(*, parent_offer_id=None):
    return TradeOffer(
        offered_by=Color.RED,
        audience=frozenset({Color.BLUE, Color.WHITE, Color.ORANGE}),
        give=(0, 1, 0, 0, 0),
        receive=(1, 0, 0, 0, 0),
        parent_offer_id=parent_offer_id,
    )


def test_trade_action_display_uses_named_resource_contract():
    action = Action(Color.RED, ActionType.OFFER_TRADE, _offer())
    state = SimpleNamespace(game_log=[])

    analyze_action(state, action, game_state=None)

    assert format_action_for_display(action) == (
        "Offered 1 BRICK for 1 WOOD to BLUE, ORANGE, WHITE"
    )
    assert format_action_for_display(action, include_actor=True) == (
        "RED: Offered 1 BRICK for 1 WOOD to BLUE, ORANGE, WHITE"
    )
    assert state.game_log == [
        {
            "type": "trade",
            "timestamp": state.game_log[0]["timestamp"],
            "message": "Offered 1 BRICK for 1 WOOD to BLUE, ORANGE, WHITE",
            "color": "RED",
            "details": {
                "action_type": "OFFER_TRADE",
                "payload": _offer().to_payload(),
            },
        }
    ]
    assert "TradeOffer(" not in state.game_log[0]["message"]


def test_counteroffer_display_retains_parent_and_audience():
    action = Action(
        Color.RED,
        ActionType.COUNTER_OFFER,
        _offer(parent_offer_id="turn-6-trade:o1"),
    )

    assert format_action_for_display(action) == (
        "Counter-offered 1 BRICK for 1 WOOD to BLUE, ORANGE, WHITE "
        "(counter to turn-6-trade:o1)"
    )


def test_legacy_trade_log_is_projected_from_structured_public_event():
    public_state = {
        "game_log": [
            {
                "type": "trade",
                "timestamp": 123.0,
                "message": "Action(color=C.RED, value=TradeOffer(...))",
                "color": "RED",
                "details": None,
            }
        ],
        "events": [
            {
                "sequence": 17,
                "actor": "RED",
                "event_type": "OFFER_TRADE",
                "payload": _offer().to_payload(),
            }
        ],
    }

    normalized = normalize_public_state_game_log(public_state)

    assert normalized["game_log"][0]["message"] == (
        "Offered 1 BRICK for 1 WOOD to BLUE, ORANGE, WHITE"
    )
    assert normalized["game_log"][0]["details"] == {
        "action_type": "OFFER_TRADE",
        "payload": _offer().to_payload(),
        "sequence": 17,
    }
    assert public_state["game_log"][0]["message"].startswith("Action(")


def test_missing_batched_trade_responses_are_restored_in_engine_order():
    offer_payload = _offer().to_payload()
    public_state = {
        "game_log": [
            {
                "type": "trade",
                "timestamp": 100.0,
                "message": "Offered 1 BRICK for 1 WOOD",
                "color": "RED",
                "details": {
                    "action_type": "OFFER_TRADE",
                    "payload": offer_payload,
                    "sequence": 17,
                },
            },
            {
                "type": "trade",
                "timestamp": 101.0,
                "message": "Signaled willingness for offer trade:o1",
                "color": "ORANGE",
                "details": {
                    "action_type": "ACCEPT_TRADE",
                    "payload": "trade:o1",
                },
            },
        ],
        "events": [
            {
                "sequence": 17,
                "actor": "RED",
                "event_type": "OFFER_TRADE",
                "payload": offer_payload,
            },
            {
                "sequence": 18,
                "actor": "BLUE",
                "event_type": "REJECT_TRADE",
                "payload": "trade:o1",
            },
            {
                "sequence": 19,
                "actor": "WHITE",
                "event_type": "ACCEPT_TRADE",
                "payload": "trade:o1",
            },
            {
                "sequence": 20,
                "actor": "ORANGE",
                "event_type": "ACCEPT_TRADE",
                "payload": "trade:o1",
            },
        ],
    }

    normalized = normalize_public_state_game_log(public_state)

    assert [entry["message"] for entry in normalized["game_log"]] == [
        "Offered 1 BRICK for 1 WOOD to BLUE, ORANGE, WHITE",
        "Declined offer trade:o1",
        "Signaled willingness for offer trade:o1",
        "Signaled willingness for offer trade:o1",
    ]
    assert [entry["color"] for entry in normalized["game_log"]] == [
        "RED",
        "BLUE",
        "WHITE",
        "ORANGE",
    ]
    assert [entry["details"]["sequence"] for entry in normalized["game_log"]] == [
        17,
        18,
        19,
        20,
    ]


def test_transition_batch_logs_every_trade_response():
    transitions = tuple(
        SimpleNamespace(
            requested_action=Action(color, action_type, "trade:o1"),
            events=(
                SimpleNamespace(
                    event_type=action_type.value,
                    public_payload="trade:o1",
                    sequence=sequence,
                ),
            ),
        )
        for sequence, color, action_type in (
            (18, Color.BLUE, ActionType.REJECT_TRADE),
            (19, Color.WHITE, ActionType.ACCEPT_TRADE),
            (20, Color.ORANGE, ActionType.ACCEPT_TRADE),
        )
    )
    state = SimpleNamespace(game_log=[])

    analyze_transitions(state, transitions, None, None)

    assert [entry["message"] for entry in state.game_log] == [
        "Declined offer trade:o1",
        "Signaled willingness for offer trade:o1",
        "Signaled willingness for offer trade:o1",
    ]
    assert [entry["color"] for entry in state.game_log] == [
        "BLUE",
        "WHITE",
        "ORANGE",
    ]
    assert [entry["details"]["sequence"] for entry in state.game_log] == [
        18,
        19,
        20,
    ]


def test_single_roll_transition_logs_payout_once():
    before = SimpleNamespace(colors=(Color.RED,), player_state={})
    after = SimpleNamespace(
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


def test_non_trade_action_display_remains_legacy_action_text():
    action = Action(Color.RED, ActionType.BUILD_SETTLEMENT, 12)

    assert format_action_for_display(action) == str(action)
    assert format_trade_event("BUILD_SETTLEMENT", 12) is None


def test_table_talk_messages_are_logged_once_across_messages_and_transitions():
    spoken = SimpleNamespace(
        event_type="MESSAGE_SENT",
        public_payload={
            "speaker": Color.GREEN,
            "text": "Leave T09 alone?",
            "audience": [Color.BLUE],
            "respondents": [Color.BLUE],
        },
        sequence=135,
    )
    transition = SimpleNamespace(
        requested_action=Action(Color.RED, ActionType.END_TURN, None),
        events=(spoken,),
    )
    state = SimpleNamespace(game_log=[])

    analyze_transitions(state, (transition,), None, None, messages=(spoken,))

    assert [entry["message"] for entry in state.game_log] == [
        "Leave T09 alone?",
    ]
    assert stamp_message_step_indexes(state.game_log, 0, 8) == 1
    assert state.game_log[0]["step_index"] == 8
    assert [entry["type"] for entry in state.game_log] == ["message"]
    assert [entry["color"] for entry in state.game_log] == ["GREEN"]


def test_stamp_message_step_indexes_labels_only_this_step_speech():
    game_log = [
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


def test_stamp_message_step_indexes_tolerates_steps_without_speech():
    game_log = [{"type": "dice", "message": "Rolled 3 + 4 = 7", "details": None}]

    assert stamp_message_step_indexes(game_log, 0, 3) == 0
    assert game_log == [{"type": "dice", "message": "Rolled 3 + 4 = 7", "details": None}]
    assert stamp_message_step_indexes(game_log, len(game_log), 4) == 0


def test_sliced_checkpoint_log_regains_speech_rows_from_events():
    """A checkpoint written under the old 50-row window kept its events but not its
    older speech rows; loading it must rebuild them in engine-sequence order."""
    said = lambda sequence, speaker, text: {
        "sequence": sequence, "actor": speaker, "event_type": "MESSAGE_SENT",
        "payload": {"speaker": speaker, "text": text, "audience": [], "respondents": []},
    }
    events = [
        {"sequence": 134, "actor": "BRONZE", "event_type": "ROLL", "payload": {}},
        said(135, "GREEN", "put the robber on T06"),
        said(136, "ORANGE", "T16 only touches BLUE"),
        {"sequence": 300, "actor": "BLUE", "event_type": "BUILD_ROAD", "payload": {}},
        said(344, "BLUE", "T01 hits GREEN's city"),
        said(470, "GREEN", "echoing BRONZE"),
    ]
    sliced_log = [
        {"type": "dice", "timestamp": 10.0, "message": "Rolled 3 + 4 = 7", "color": "BLUE", "details": None},
        {"type": "message", "timestamp": 11.0, "message": "T01 hits GREEN's city", "color": "BLUE",
         "step_index": 235, "details": {"event_type": "MESSAGE_SENT", "sequence": 344, "step_index": 235,
                                        "payload": events[4]["payload"]}},
        {"type": "building", "timestamp": 12.0, "message": "Built a road", "color": "BLUE",
         "details": {"sequence": 400}},
    ]

    normalized = normalize_public_state_game_log({"game_log": sliced_log, "events": events})
    log = normalized["game_log"]

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
