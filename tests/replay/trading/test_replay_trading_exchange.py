"""Resource trades publish one exact closure before the exchange."""
from copy import deepcopy
from typing import Any

import pytest

from cle.game_engine.game import GameEngine
from cle.game_engine.models.enums import Action, ActionType
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


@pytest.mark.parametrize(
    ("log_text", "expected_action_type"),
    [
        (
            {
                "type": 115,
                "playerColor": 1,
                "acceptingPlayerColor": 2,
                "givenCardEnums": [1],
                "receivedCardEnums": [2],
            },
            "CONFIRM_TRADE",
        ),
        (
            {
                "type": 116,
                "playerColor": 1,
                "givenCardEnums": [1, 1, 1, 1],
                "receivedCardEnums": [2],
            },
            "MARITIME_TRADE",
        ),
    ],
)
def test_resource_trade_actions_follow_one_deterministic_closure(
    log_text: dict[str, int | list[int]], expected_action_type: str
) -> None:
    actions = parse_colonist_events_to_actions([
        trade_offer_event(),
        {
            "stateChange": {
                "tradeState": {"activeOffers": {"trade-1": None}},
                "gameLogState": {"log-1": {"text": log_text}},
            }
        },
    ])

    assert [action["type"] for action in actions] == [
        "OFFER_TRADE",
        "CLOSE_TRADE",
        expected_action_type,
    ]
    assert actions[1]["trade_id"] == "trade-1"
    assert actions[1]["reason"] == "transaction_closed"
    assert actions[1]["expected_resources"] == {}
    assert actions[2]["trade_closures_preceded"] is True


@pytest.mark.parametrize("action_type", ["CONFIRM_TRADE", "MARITIME_TRADE"])
@pytest.mark.parametrize("attached", [False, True])
def test_resource_trade_publishes_one_exact_closure_before_exchange(action_type: str, attached: bool) -> None:
    state: Any = make_confirmation_state()
    game: Any = state.current_game
    game.state.player_state["P0_WOOD_IN_HAND"] = 5
    closure = {
        "index": 0, "type": "CLOSE_TRADE", "player": 1, "creator": 1,
        "trade_id": "trade-red", "reason": "transaction_closed",
    }
    exchange = {
        "index": 0, "type": action_type, "player": 1, "acceptor": 2,
        "offered": (1, 0, 0, 0, 0), "given": (4, 0, 0, 0, 0),
        "received": (0, 1, 0, 0, 0),
        "expected_resources": {"private-oracle": "MUST_NOT_PUBLISH"},
    }
    if attached:
        exchange["closed_trades"] = [closure]
        actions = [exchange]
    else:
        exchange["trade_closures_preceded"] = True
        # Older projections may retain this metadata after emitting a closure row.
        exchange["closed_trades"] = [closure]
        actions = [closure, exchange]
    state.replay_data["parsed_actions"] = actions
    for _ in actions:
        result = replay_step_logic(state, lambda: None, allow_lookahead=False)
        assert isinstance(result, dict), result
    assert [event.event_type for event in game.events] == ["CLOSE_TRADE", action_type]
    assert [event.sequence for event in game.events] == [0, 1]
    assert game.events[0].public_payload == {
        "offer_id": "trade-red", "parent_offer_id": None, "reason": "transaction_closed",
    }
    assert "MUST_NOT_PUBLISH" not in repr(game.events)
    assert "expected_resources" not in repr(game.events)
    assert game.state.trade_window.offers["trade-orange"].active
    assert game.state.trade_window.offers["counter-white"].active
    assert len(game.state.actions) == 1
    assert game.state.actions[0].action_type == ActionType[action_type]
    expected_wood: Any = 4 if action_type == "CONFIRM_TRADE" else 1
    assert game.state.player_state["P0_WOOD_IN_HAND"] == expected_wood
    assert game.state.player_state["P0_BRICK_IN_HAND"] == 1
    expected_events: Any = deepcopy(game.events)
    assert replay_undo_logic(state, lambda: None)["status"] == "ok"
    assert len(game.events) == (0 if attached else 1)
    replay_step_logic(state, lambda: None, allow_lookahead=False)
    assert game.events == expected_events


def test_forced_road_repairs_caches_and_publishes_an_undoable_action() -> None:
    state: Any = ServerState()
    game: Any = GameEngine(
        (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE),
        seed=4, shuffle_players=False, capture_history=True,
    )
    state.current_game = game
    state.replay_mode = state.game_running = True
    edge = tuple(sorted(next(iter(game.state.board.buildable_subgraph.edges))))
    state.edge_to_edge_map = {"_forced": list(edge)}
    state.replay_data = {
        "parsed_actions": [{
            "index": 0, "type": "BUILD_ROAD", "player": 1,
            "colonist_edge": "forced",
        }],
        "events": [{}], "colonist_color_to_engine_idx": {"1": 0},
        "end_game_state": {},
    }
    before = engine_snapshot(state)
    assert replay_step_logic(state, lambda: None)["status"] == "ok"
    assert game.state.board.roads[edge] == Color.RED
    assert game.state.board.roads[edge[::-1]] == Color.RED
    assert game.state.board.road_lengths[Color.RED] == 1
    assert game.state.player_state["P0_LONGEST_ROAD_LENGTH"] == 1
    assert game.state.board.connected_components[Color.RED] == [set(edge)]
    assert game.state.player_state["P0_ROADS_AVAILABLE"] == 14
    assert game.state.board.road_color is None
    assert game.state.actions == [Action(Color.RED, ActionType.BUILD_ROAD, edge)]
    assert game.events[0].event_type == "BUILD_ROAD"
    assert game.project_events(Color.BLUE)[0].payload == edge
    assert state.replay_actions_per_step == [1]
    assert any(issue["kind"] == "forced_replay_overlay" for issue in state.replay_semantic_issues)
    assert replay_undo_logic(state, lambda: None)["actions_undone"] == 1
    assert engine_snapshot(state) == before
    assert game.state.board.roads == {}
    assert game.state.board.connected_components[Color.RED] == []
    assert game.rng is game.state.rng
