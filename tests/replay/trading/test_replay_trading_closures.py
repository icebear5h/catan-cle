"""Confirmed and standalone trade closures stay undoable."""
from copy import deepcopy
from typing import Any

from cle.game_engine.game import GameEngine
from cle.game_engine.models.enums import ActionType
from cle.game_engine.models.player import Color
from cle.replay.colonist.event_parser import parse_colonist_events_to_actions
from cle.replay.runtime.navigation import (
    replay_undo_logic,
)
from cle.replay.runtime.step_executor import replay_step_logic
from playground.game_viewer.state import ServerState

from .support import (
    engine_snapshot,
    make_confirmation_state,
    trade_offer_event,
)


def test_confirm_trade_undo_restores_the_complete_pre_step_transaction() -> None:
    state = make_confirmation_state()
    before_engine = engine_snapshot(state)
    before_log = deepcopy(state.game_log)
    before_issues = deepcopy(state.replay_semantic_issues)
    before_divergence = deepcopy(state.first_divergence_step)
    before_pending_dev = deepcopy(state.replay_pending_dev_card)

    first_result: Any = replay_step_logic(state, lambda: None)

    assert first_result["status"] == "trade_applied"
    assert state.replay_index == 1
    assert state.replay_actions_per_step == [1]
    assert len(state.replay_step_checkpoints) == 1
    assert state.current_game.state.actions[-1].action_type == ActionType.CONFIRM_TRADE
    assert state.current_game.state.player_state["P0_WOOD_IN_HAND"] == 1
    assert state.current_game.state.player_state["P0_BRICK_IN_HAND"] == 1
    assert state.current_game.state.player_state["P1_WOOD_IN_HAND"] == 1
    assert state.current_game.state.player_state["P1_BRICK_IN_HAND"] == 0
    for color in state.current_game.state.colors:
        event: Any = state.current_game.project_events(color)[-1]
        assert event.event_type == "CONFIRM_TRADE"
        assert event.payload == {
            "offer_id": None, "turn_player": "RED", "counterparty": "BLUE",
            "give": {"WOOD": 1}, "receive": {"BRICK": 1},
        }
    after_first_confirmation = engine_snapshot(state)

    undo_result: Any = replay_undo_logic(state, lambda: None)

    assert undo_result["status"] == "ok"
    assert undo_result["event_index"] == 0
    assert undo_result["actions_undone"] == 1
    assert engine_snapshot(state) == before_engine
    assert state.game_log == before_log
    assert state.replay_semantic_issues == before_issues
    assert state.first_divergence_step == before_divergence
    assert state.replay_pending_dev_card == before_pending_dev
    assert state.replay_actions_per_step == []
    assert state.replay_step_checkpoints == []
    assert state.game_running is True

    second_result: Any = replay_step_logic(state, lambda: None)

    assert second_result["status"] == "trade_applied"
    assert engine_snapshot(state) == after_first_confirmation


def test_standalone_trade_closure_is_parsed_executed_and_undoable() -> None:
    actions = parse_colonist_events_to_actions([
        trade_offer_event(),
        {
            "stateChange": {
                "tradeState": {"activeOffers": {"trade-1": None}}
            }
        },
    ])

    assert [action["type"] for action in actions] == [
        "OFFER_TRADE",
        "CLOSE_TRADE",
    ]
    assert actions[1]["trade_id"] == "trade-1"
    assert actions[1]["creator"] == 1
    assert actions[1]["reason"] == "cancelled"
    assert actions[1]["is_counter_offer"] is False

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
        "events": [{}, {}],
        "parsed_actions": actions,
        "colonist_color_to_engine_idx": {"1": 0},
        "end_game_state": {},
    }
    resources_before = engine_snapshot(state)["hands"]

    offer_result: Any = replay_step_logic(state, lambda: None)

    assert offer_result["status"] in {"ok", "overlay_applied"}
    assert state.current_game.state.trade_window.offers["trade-1"].active

    close_result: Any = replay_step_logic(state, lambda: None)

    assert close_result["status"] == "closed"
    assert not state.current_game.state.trade_window.offers["trade-1"].active
    assert engine_snapshot(state)["hands"] == resources_before
    assert [event.event_type for event in state.current_game.events] == [
        "OFFER_TRADE", "CLOSE_TRADE",
    ]
    assert state.current_game.events[-1].public_payload == {
        "offer_id": "trade-1", "parent_offer_id": None, "reason": "cancelled",
    }
    assert [action.action_type for action in state.current_game.state.actions] == [
        ActionType.OFFER_TRADE,
    ]
    assert state.replay_actions_per_step == [1, 0]
    assert any(
        issue["kind"] == "replayed_trade_closure"
        for issue in state.replay_semantic_issues
    )

    replay_undo_logic(state, lambda: None)

    assert state.replay_index == 1
    assert state.current_game.state.trade_window.offers["trade-1"].active
    assert not any(
        issue["kind"] == "replayed_trade_closure"
        for issue in state.replay_semantic_issues
    )
    assert [event.event_type for event in state.current_game.events] == ["OFFER_TRADE"]
    assert replay_step_logic(state, lambda: None)["status"] == "closed"
    assert [event.sequence for event in state.current_game.events] == [0, 1]
