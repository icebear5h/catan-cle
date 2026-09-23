"""Revalidation and backward navigation restore replay trade state."""
from typing import Any

import pytest

from cle.game_engine.game import GameEngine
from cle.game_engine.models.actions import generate_playable_actions
from cle.game_engine.models.enums import Action, ActionType
from cle.game_engine.models.player import Color
from cle.replay.colonist.event_parser import parse_colonist_events_to_actions
from cle.replay.runtime.navigation import (
    replay_goto_sequential_logic,
)
from cle.replay.runtime.step_executor import replay_step_logic
from cle.sandbox.replay import ReplaySandbox
from playground.game_viewer.routes.websocket import broadcast_game_state
from playground.game_viewer.state import ServerState

from .support import (
    make_multi_offer_engine_game,
    trade_offer_event,
)


@pytest.mark.parametrize(
    ("resource_key", "error_text"),
    [
        ("P0_WOOD_IN_HAND", "can no longer afford"),
        ("P1_BRICK_IN_HAND", "can no longer afford"),
    ],
)
def test_confirm_trade_revalidates_both_hands_before_transfer(
    resource_key: str, error_text: str
) -> None:
    game: Any = make_multi_offer_engine_game()
    game.state.player_state[resource_key] = 0
    game.state.playable_actions = generate_playable_actions(game.state)
    confirmation = Action(
        Color.RED,
        ActionType.CONFIRM_TRADE,
        next(
            candidate
            for candidate in game.state.trade_window.executable_candidates()
            if candidate.counterparty == Color.BLUE
        ),
    )
    hands_before = {
        key: value
        for key, value in game.state.player_state.items()
        if key.endswith("_IN_HAND")
    }

    assert confirmation not in game.state.playable_actions
    with pytest.raises(ValueError, match=error_text):
        game.step(confirmation, force=True)

    assert {
        key: value
        for key, value in game.state.player_state.items()
        if key.endswith("_IN_HAND")
    } == hands_before


def test_backward_navigation_restarts_replay_running_state() -> None:
    events: Any = [
        trade_offer_event(),
        {
            "stateChange": {
                "tradeState": {"activeOffers": {"trade-1": None}}
            }
        },
    ]
    actions = parse_colonist_events_to_actions(events)
    state: Any = ServerState()
    players = [
        Color.RED,
        Color.BLUE,
        Color.WHITE,
        Color.ORANGE,
    ]
    state.current_game = GameEngine(players, shuffle_players=False)
    state.current_players = players
    state.replay_mode = True
    state.game_running = True
    state.replay_data = {
        "events": events,
        "parsed_actions": actions,
        "colonist_color_to_engine_idx": {"1": 0},
        "end_game_state": {},
    }
    for _ in actions:
        replay_step_logic(state, lambda: None)
    assert state.game_running is False

    replay_goto_sequential_logic(
        state,
        1,
        lambda: replay_step_logic(state, lambda: None),
        lambda: None,
    )
    assert state.replay_index == 1
    assert state.game_running is True

    replay_goto_sequential_logic(
        state,
        0,
        lambda: replay_step_logic(state, lambda: None),
        lambda: None,
    )
    assert state.replay_index == 0
    assert state.game_running is True


def test_counter_only_replay_state_does_not_broadcast_fake_legacy_offer() -> None:
    events: Any = [
        {
            "stateChange": {
                "tradeState": {
                    "activeOffers": {
                        "counter-1": {
                            "creator": 2,
                            "offeredResources": [2],
                            "wantedResources": [1],
                            "counterOfferInResponseToTradeId": "trade-1",
                        }
                    }
                }
            }
        }
    ]
    actions = parse_colonist_events_to_actions(events)
    state: Any = ServerState()
    players = [
        Color.RED,
        Color.BLUE,
        Color.WHITE,
        Color.ORANGE,
    ]
    state.current_game = GameEngine(players, shuffle_players=False)
    state.current_players = players
    state.replay_mode = True
    state.game_running = True
    state.replay_data = {
        "game_id": "test",
        "events": events,
        "parsed_actions": actions,
        "total_events": len(actions),
        "colonist_color_to_engine_idx": {"1": 0, "2": 1},
        "end_game_state": {},
    }
    replay_step_logic(state, lambda: None)
    state.current_sandbox = ReplaySandbox(state, state.current_game)

    class SocketRecorder:
        def __init__(self) -> None:
            self.payload: dict[str, Any] | None = None

        def emit(self, event: str, payload: dict[str, Any]) -> None:
            assert event == "game_state"
            self.payload = payload

    socket: Any = SocketRecorder()
    broadcast_game_state(socket, state)

    window = state.current_game.state.trade_window
    assert window is not None
    assert window.active_offers == ()
    assert socket.payload["trade_state"] is None
    source_offer = socket.payload["replay"]["trade_ledger"][0]
    assert source_offer["trade_id"] == "counter-1"
    assert source_offer["creator"] == 2
    assert source_offer["offered"] == [0, 1, 0, 0, 0]
    assert source_offer["wanted"] == [1, 0, 0, 0, 0]
    events = state.current_game.project_events(Color.WHITE)
    assert len(events) == 1
    assert events[0].event_type == "COUNTER_OFFER"
    assert events[0].actor == Color.BLUE
    assert events[0].payload["parent_offer_id"] == "trade-1"
    assert events[0].payload["give"] == {"BRICK": 1}
    assert events[0].payload["receive"] == {"WOOD": 1}
    assert "audience" not in events[0].payload
