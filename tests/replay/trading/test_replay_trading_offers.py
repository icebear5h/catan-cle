"""Road mismatches and offer snapshots do not cancel pending offers."""
from copy import deepcopy
from typing import Any

import pytest

from cle.game_engine.events import GameEvent
from cle.game_engine.game import GameEngine
from cle.game_engine.models.actions import generate_playable_actions
from cle.game_engine.models.enums import Action, ActionType
from cle.game_engine.models.player import Color
from cle.game_engine.trading import TradeLimits, TradeOffer
from cle.replay.colonist.event_parser import parse_colonist_events_to_actions
from cle.replay.runtime.navigation import (
    replay_undo_logic,
)
from cle.replay.runtime.step_executor import replay_step_logic
from playground.game_viewer.state import ServerState

from .support import (
    RESOURCES,
    engine_snapshot,
    trade_offer_event,
)


def test_main_game_road_mismatch_does_not_cancel_offers_or_advance_turn() -> None:
    state: Any = ServerState()
    game = GameEngine(
        (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE),
        seed=4, shuffle_players=False, capture_history=True,
    )
    for _ in range(16):
        game.step(game.state.playable_actions[0])
    game.step(Action(Color.RED, ActionType.ROLL, (1, 1)), force=True)
    for resource in ("WOOD", "BRICK"):
        amount = max(0, 2 - game.state.player_state[f"P0_{resource}_IN_HAND"])
        game.state.player_state[f"P0_{resource}_IN_HAND"] += amount
        game.state.resource_freqdeck[RESOURCES.index(resource)] -= amount
    game.step(Action(Color.RED, ActionType.OFFER_TRADE, TradeOffer(
        offered_by=Color.RED,
        audience=frozenset(game.state.colors[1:]),
        give=(1, 0, 0, 0, 0), receive=(0, 0, 0, 0, 1),
    )))
    buildable: Any = {tuple(sorted(edge)) for edge in game.state.board.buildable_edges(Color.RED)}
    edge: Any = next(
        tuple(sorted(edge)) for edge in game.state.board.buildable_subgraph.edges
        if edge not in game.state.board.roads and tuple(sorted(edge)) not in buildable
    )
    assert not game.state.is_initial_build_phase
    assert any(action.action_type == ActionType.CANCEL_TRADE for action in game.state.playable_actions)
    assert any(action.action_type == ActionType.END_TURN for action in game.state.playable_actions)
    state.current_game = game
    state.replay_mode = state.game_running = True
    state.edge_to_edge_map = {"_mismatch": list(edge)}
    state.replay_data = {
        "parsed_actions": [{
            "index": 0, "type": "BUILD_ROAD", "player": 1,
            "colonist_edge": "mismatch",
        }],
        "events": [{}], "colonist_color_to_engine_idx": {"1": 0},
        "end_game_state": {},
    }
    offer_id = game.state.actions[-1].value.id
    state.replay_trade_ledger = {offer_id: {"creator": 1, "responses": {}}}
    before = engine_snapshot(state)
    before_roads = deepcopy(game.state.board.roads)
    before_ledger = deepcopy(state.replay_trade_ledger)
    before_turn = game.state.num_turns
    before_rng = game.rng.getstate()
    before_bank: Any = game.state.resource_freqdeck.copy()
    assert replay_step_logic(state, lambda: None)["status"] == "ok"
    assert game.state.actions[len(before["actions"]):] == [Action(Color.RED, ActionType.BUILD_ROAD, edge)]
    assert [event.event_type for event in game.events[len(before["events"]):]] == ["BUILD_ROAD"]
    assert state.replay_actions_per_step == [1]
    assert game.state.trade_window == before["trade_window"]
    assert state.replay_trade_ledger == before_ledger
    assert game.state.current_player_index == before["current_player_index"]
    assert game.state.current_turn_index == before["current_turn_index"]
    assert game.state.num_turns == before_turn
    assert game.rng.getstate() == before_rng
    for index, resource in enumerate(RESOURCES):
        cost: Any = int(resource in {"WOOD", "BRICK"})
        assert game.state.player_state[f"P0_{resource}_IN_HAND"] == before["hands"][0][index] - cost
        assert game.state.resource_freqdeck[index] == before_bank[index] + cost
    assert [issue["kind"] for issue in state.replay_semantic_issues] == ["forced_replay_overlay"]
    assert "END_TURN" not in repr(state.game_log)
    assert "CANCEL_TRADE" not in repr(state.game_log)
    after = engine_snapshot(state)
    assert replay_undo_logic(state, lambda: None)["actions_undone"] == 1
    assert engine_snapshot(state) == before
    assert game.state.board.roads == before_roads
    assert state.replay_trade_ledger == before_ledger
    assert replay_step_logic(state, lambda: None)["status"] == "ok"
    assert engine_snapshot(state) == after


@pytest.mark.parametrize("overflow", [False, True])
def test_full_offer_snapshot_projects_responses_before_publication(overflow: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    state = ServerState()
    game: Any = GameEngine(
        (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE),
        seed=4, shuffle_players=False, capture_history=True,
        trade_limits=TradeLimits(max_active_root_offers=1, max_offers_per_player=1),
    )
    for _ in range(16):
        game.step(game.state.playable_actions[0])
    game.step(Action(Color.RED, ActionType.ROLL, (1, 1)), force=True)
    for player, resource in ((0, "WOOD"), (1, "BRICK")):
        amount = max(0, 1 - game.state.player_state[f"P{player}_{resource}_IN_HAND"])
        game.state.player_state[f"P{player}_{resource}_IN_HAND"] += amount
        game.state.resource_freqdeck[RESOURCES.index(resource)] -= amount
    game.state.playable_actions = generate_playable_actions(game.state)
    if overflow:
        game.step(Action(Color.RED, ActionType.OFFER_TRADE, TradeOffer(
            offered_by=Color.RED, audience=frozenset(game.state.colors[1:]),
            give=(1, 0, 0, 0, 0), receive=(0, 0, 0, 0, 1),
        )))
    assert any(action.action_type == ActionType.OFFER_TRADE for action in game.state.playable_actions) is not overflow
    raw: Any = trade_offer_event("full-snapshot")
    raw["stateChange"]["tradeState"]["activeOffers"]["full-snapshot"]["playerResponses"] = {
        "2": 1, "3": 2, "4": 0,
    }
    parsed = parse_colonist_events_to_actions([raw])
    assert [action["type"] for action in parsed] == ["OFFER_TRADE"]
    state.current_game = game
    state.replay_mode = state.game_running = True
    state.replay_data = {
        "events": [raw], "parsed_actions": parsed, "end_game_state": {},
        "colonist_color_to_engine_idx": {"1": 0, "2": 1, "3": 2, "4": 3},
    }
    before: Any = engine_snapshot(state)
    original_publish = game.publish_event
    publication_states: list[TradeOffer] = []

    def publish(
        event_type: str,
        actor: Color,
        public_payload: object,
        **kwargs: object,
    ) -> GameEvent:
        if event_type == "OFFER_TRADE":
            publication_states.append(deepcopy(game.state.trade_window.offers["full-snapshot"]))
        return original_publish(event_type, actor, public_payload, **kwargs)

    monkeypatch.setattr(game, "publish_event", publish)
    expected_status: Any = "overlay_applied" if overflow else "ok"
    for attempt in range(2):
        assert replay_step_logic(state, lambda: None)["status"] == expected_status
        offer: Any = game.state.trade_window.offers["full-snapshot"]
        event: Any = game.events[-1]
        assert offer.willing_by == publication_states[-1].willing_by == {Color.BLUE}
        assert offer.declined_by == publication_states[-1].declined_by == {Color.WHITE}
        assert event.public_payload == offer.to_payload()
        assert event.public_payload["willing_by"] == ["BLUE"]
        assert event.public_payload["declined_by"] == ["WHITE"]
        assert state.replay_trade_ledger["full-snapshot"]["responses"] == {
            "2": "accepted", "3": "rejected",
        }
        assert [event.event_type for event in game.events[len(before["events"]):]] == ["OFFER_TRADE"]
        assert state.replay_actions_per_step == [1]
        candidates: Any = [
            action for action in game.state.playable_actions
            if action.action_type == ActionType.CONFIRM_TRADE
            and action.value.offer_id == "full-snapshot"
        ]
        assert len(candidates) == 1
        assert candidates[0].value.counterparty == Color.BLUE
        assert game.is_action_valid(candidates[0])
        for color in game.state.colors:
            assert game.project_events(color)[-1].payload == offer.to_payload()
        if attempt == 0:
            after: Any = engine_snapshot(state)
            assert replay_undo_logic(state, lambda: None)["actions_undone"] == 1
            assert engine_snapshot(state) == before
            assert state.replay_trade_ledger == {}
        else:
            assert engine_snapshot(state) == after


def test_mixed_trade_logs_still_emit_each_source_closure_once() -> None:
    actions = parse_colonist_events_to_actions([
        trade_offer_event(),
        {
            "stateChange": {
                "tradeState": {"activeOffers": {"trade-1": None}},
                "gameLogState": {
                    "confirm": {
                        "text": {
                            "type": 115,
                            "playerColor": 1,
                            "acceptingPlayerColor": 2,
                            "givenCardEnums": [1],
                            "receivedCardEnums": [2],
                        }
                    },
                    "maritime": {
                        "text": {
                            "type": 116,
                            "playerColor": 1,
                            "givenCardEnums": [1, 1, 1, 1],
                            "receivedCardEnums": [2],
                        }
                    },
                },
            }
        },
    ])

    assert [action["type"] for action in actions] == [
        "OFFER_TRADE",
        "CLOSE_TRADE",
        "CONFIRM_TRADE",
        "MARITIME_TRADE",
    ]
    assert sum(action["type"] == "CLOSE_TRADE" for action in actions) == 1
