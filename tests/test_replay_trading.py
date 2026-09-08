import contextlib
from copy import deepcopy
import io
import os
from collections import Counter
from pathlib import Path

import pytest

from cle.game_engine.game import GameEngine
from cle.game_engine.models.actions import generate_playable_actions
from cle.game_engine.models.enums import Action, ActionPrompt, ActionType
from cle.game_engine.models.player import Color
from cle.game_engine.trading import TradeLimits, TradeOffer, TradeWindow
from playground.game_viewer.app import app
from cle.replay.colonist.event_parser import parse_colonist_events_to_actions
from cle.replay.colonist.helpers import validate_resources_match
from cle.replay.runtime.navigation import (
    replay_goto_fast_logic,
    replay_goto_sequential_logic,
    replay_undo_logic,
)
from cle.replay.runtime.step_executor import replay_step_logic
from cle.sandbox.replay import ReplaySandbox
from playground.game_viewer.routes.websocket import broadcast_game_state
from playground.game_viewer.state import ServerState, server_state


RESOURCES = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")


def make_confirmation_state():
    state = ServerState()
    players = [
        Color.RED,
        Color.BLUE,
        Color.WHITE,
        Color.ORANGE,
    ]
    game = GameEngine(players, shuffle_players=False)
    state.current_game = game
    state.current_players = players
    state.replay_mode = True
    state.game_running = True

    game.state.player_state["P0_WOOD_IN_HAND"] = 2
    game.state.player_state["P1_BRICK_IN_HAND"] = 1
    game.state.current_player_index = 2
    game.state.current_turn_index = 2
    game.state.current_prompt = ActionPrompt.BUILD_INITIAL_SETTLEMENT
    window = TradeWindow(
        id="test-window",
        turn_player=Color.RED,
        participants=tuple(players),
    )
    red_offer = window.create_offer(
        TradeOffer(
            id="trade-red",
            offered_by=Color.RED,
            audience=frozenset(players[1:]),
            give=(1, 0, 0, 0, 0),
            receive=(0, 1, 0, 0, 0),
        )
    )
    red_offer.willing_by.add(Color.BLUE)
    orange_offer = window.create_offer(
        TradeOffer(
            id="trade-orange",
            offered_by=Color.ORANGE,
            audience=frozenset({Color.RED, Color.BLUE, Color.WHITE}),
            give=(0, 0, 1, 0, 0),
            receive=(0, 0, 0, 1, 0),
        )
    )
    orange_offer.declined_by.add(Color.WHITE)
    window.create_offer(
        TradeOffer(
            id="counter-white",
            offered_by=Color.WHITE,
            audience=frozenset({Color.RED}),
            give=(0, 0, 0, 1, 0),
            receive=(0, 0, 0, 0, 1),
            parent_offer_id=red_offer.id,
        ),
        allow_duplicate=True,
    )
    game.state.trade_window = window
    game.state.playable_actions = generate_playable_actions(game.state)

    confirmation = {
        "index": 12,
        "type": "CONFIRM_TRADE",
        "player": 1,
        "acceptor": 2,
        "offered": (1, 0, 0, 0, 0),
        "received": (0, 1, 0, 0, 0),
    }
    state.replay_data = {
        "events": [{} for _ in range(13)],
        "parsed_actions": [confirmation],
        "colonist_color_to_engine_idx": {"1": 0, "2": 1},
        "end_game_state": {},
    }
    state.game_log = [{"type": "test", "message": "before confirmation"}]
    state.replay_semantic_issues = [
        {"kind": "existing", "severity": "info", "step": -1}
    ]
    state.first_divergence_step = {"P3": 4}
    state.replay_pending_dev_card = {"keep": "me"}
    return state


def engine_snapshot(state):
    game_state = state.current_game.state
    return {
        "hands": tuple(
            tuple(
                game_state.player_state[f"P{player_idx}_{resource}_IN_HAND"]
                for resource in RESOURCES
            )
            for player_idx in range(len(game_state.colors))
        ),
        "trade_window": deepcopy(game_state.trade_window),
        "current_player_index": game_state.current_player_index,
        "current_turn_index": game_state.current_turn_index,
        "current_prompt": game_state.current_prompt,
        "playable_actions": list(game_state.playable_actions),
        "actions": list(game_state.actions),
        "events": deepcopy(state.current_game.events),
        "player_state": dict(game_state.player_state),
        "game_history_length": len(state.current_game.history),
    }


def test_confirm_trade_undo_restores_the_complete_pre_step_transaction():
    state = make_confirmation_state()
    before_engine = engine_snapshot(state)
    before_log = deepcopy(state.game_log)
    before_issues = deepcopy(state.replay_semantic_issues)
    before_divergence = deepcopy(state.first_divergence_step)
    before_pending_dev = deepcopy(state.replay_pending_dev_card)

    first_result = replay_step_logic(state, lambda: None)

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
        event = state.current_game.project_events(color)[-1]
        assert event.event_type == "CONFIRM_TRADE"
        assert event.payload == {
            "offer_id": None, "turn_player": "RED", "counterparty": "BLUE",
            "give": {"WOOD": 1}, "receive": {"BRICK": 1},
        }
    after_first_confirmation = engine_snapshot(state)

    undo_result = replay_undo_logic(state, lambda: None)

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

    second_result = replay_step_logic(state, lambda: None)

    assert second_result["status"] == "trade_applied"
    assert engine_snapshot(state) == after_first_confirmation


def trade_offer_event(trade_id="trade-1"):
    return {
        "stateChange": {
            "tradeState": {
                "activeOffers": {
                    trade_id: {
                        "creator": 1,
                        "offeredResources": [1],
                        "wantedResources": [2],
                    }
                }
            }
        }
    }


def test_standalone_trade_closure_is_parsed_executed_and_undoable():
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

    state = ServerState()
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

    offer_result = replay_step_logic(state, lambda: None)

    assert offer_result["status"] in {"ok", "overlay_applied"}
    assert state.current_game.state.trade_window.offers["trade-1"].active

    close_result = replay_step_logic(state, lambda: None)

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
    log_text, expected_action_type
):
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
def test_resource_trade_publishes_one_exact_closure_before_exchange(action_type, attached):
    state = make_confirmation_state()
    game = state.current_game
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
    expected_wood = 4 if action_type == "CONFIRM_TRADE" else 1
    assert game.state.player_state["P0_WOOD_IN_HAND"] == expected_wood
    assert game.state.player_state["P0_BRICK_IN_HAND"] == 1
    expected_events = deepcopy(game.events)
    assert replay_undo_logic(state, lambda: None)["status"] == "ok"
    assert len(game.events) == (0 if attached else 1)
    replay_step_logic(state, lambda: None, allow_lookahead=False)
    assert game.events == expected_events


def test_forced_road_repairs_caches_and_publishes_an_undoable_action():
    state = ServerState()
    game = GameEngine(
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


def test_main_game_road_mismatch_does_not_cancel_offers_or_advance_turn():
    state = ServerState()
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
    buildable = {tuple(sorted(edge)) for edge in game.state.board.buildable_edges(Color.RED)}
    edge = next(
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
    before_bank = game.state.resource_freqdeck.copy()
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
        cost = int(resource in {"WOOD", "BRICK"})
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
def test_full_offer_snapshot_projects_responses_before_publication(overflow, monkeypatch):
    state = ServerState()
    game = GameEngine(
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
    raw = trade_offer_event("full-snapshot")
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
    before = engine_snapshot(state)
    original_publish = game.publish_event
    publication_states = []

    def publish(event_type, actor, public_payload, **kwargs):
        if event_type == "OFFER_TRADE":
            publication_states.append(deepcopy(game.state.trade_window.offers["full-snapshot"]))
        return original_publish(event_type, actor, public_payload, **kwargs)

    monkeypatch.setattr(game, "publish_event", publish)
    expected_status = "overlay_applied" if overflow else "ok"
    for attempt in range(2):
        assert replay_step_logic(state, lambda: None)["status"] == expected_status
        offer = game.state.trade_window.offers["full-snapshot"]
        event = game.events[-1]
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
        candidates = [
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
            after = engine_snapshot(state)
            assert replay_undo_logic(state, lambda: None)["actions_undone"] == 1
            assert engine_snapshot(state) == before
            assert state.replay_trade_ledger == {}
        else:
            assert engine_snapshot(state) == after


def test_mixed_trade_logs_still_emit_each_source_closure_once():
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


def test_trade_id_ledger_preserves_same_creator_offers_and_counter_links():
    events = [
        trade_offer_event("trade-1"),
        {
            "stateChange": {
                "tradeState": {
                    "activeOffers": {
                        "trade-2": {
                            "creator": 1,
                            "offeredResources": [3],
                            "wantedResources": [4],
                        }
                    }
                }
            }
        },
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
        },
        {
            "stateChange": {
                "tradeState": {
                    "activeOffers": {
                        "trade-1": {"playerResponses": {"2": 1}}
                    }
                }
            }
        },
        {
            "stateChange": {
                "tradeState": {"activeOffers": {"trade-1": None}}
            }
        },
    ]
    actions = parse_colonist_events_to_actions(events)
    assert [action["type"] for action in actions] == [
        "OFFER_TRADE",
        "OFFER_TRADE",
        "COUNTER_OFFER",
        "ACCEPT_TRADE",
        "CLOSE_TRADE",
    ]

    state = ServerState()
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
        "colonist_color_to_engine_idx": {"1": 0, "2": 1},
        "end_game_state": {},
    }

    replay_step_logic(state, lambda: None)
    replay_step_logic(state, lambda: None)
    replay_step_logic(state, lambda: None)

    assert list(state.replay_trade_ledger) == [
        "trade-1",
        "trade-2",
        "counter-1",
    ]
    assert state.replay_trade_ledger["trade-1"]["creator"] == 1
    assert state.replay_trade_ledger["trade-2"]["creator"] == 1
    assert state.replay_trade_ledger["counter-1"]["counter_offer_to"] == "trade-1"
    window = state.current_game.state.trade_window
    assert set(window.offers) == {"trade-1", "trade-2", "counter-1"}
    assert window.offers["counter-1"].parent_offer_id == "trade-1"

    replay_step_logic(state, lambda: None)

    assert state.replay_trade_ledger["trade-1"]["responses"] == {
        "2": "accepted"
    }
    assert window.offers["trade-1"].willing_by == {Color.BLUE}

    replay_step_logic(state, lambda: None)

    assert list(state.replay_trade_ledger) == ["trade-2", "counter-1"]
    assert not window.offers["trade-1"].active
    assert window.offers["trade-2"].active
    assert window.offers["counter-1"].active

    replay_undo_logic(state, lambda: None)

    assert list(state.replay_trade_ledger) == [
        "trade-1",
        "trade-2",
        "counter-1",
    ]
    restored = state.current_game.state.trade_window
    assert restored.offers["trade-1"].active
    assert restored.offers["trade-1"].willing_by == {Color.BLUE}


def test_fast_navigation_uses_the_authoritative_replay_executor():
    events = [
        trade_offer_event(),
        {
            "stateChange": {
                "tradeState": {"activeOffers": {"trade-1": None}}
            }
        },
    ]
    actions = parse_colonist_events_to_actions(events)

    def make_state():
        state = ServerState()
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
        return state

    sequential_state = make_state()
    for _ in actions:
        replay_step_logic(sequential_state, lambda: None)

    fast_state = make_state()
    result = replay_goto_fast_logic(
        fast_state,
        len(actions),
        lambda: replay_step_logic(fast_state, lambda: None),
        lambda: None,
    )

    assert result["status"] == "ok"
    assert result["event_index"] == len(actions)
    assert engine_snapshot(fast_state) == engine_snapshot(sequential_state)
    assert fast_state.replay_trade_ledger == sequential_state.replay_trade_ledger
    assert fast_state.replay_actions_per_step == sequential_state.replay_actions_per_step
    assert fast_state.replay_semantic_issues == sequential_state.replay_semantic_issues


def make_multi_offer_engine_game():
    game = GameEngine(
        [
            Color.RED,
            Color.BLUE,
            Color.WHITE,
            Color.ORANGE,
        ],
        shuffle_players=False,
    )
    state = game.state
    state.is_initial_build_phase = False
    state.current_player_index = 0
    state.current_turn_index = 0
    state.current_prompt = ActionPrompt.PLAY_TURN
    state.player_state["P0_HAS_ROLLED"] = True
    state.player_state["P0_WOOD_IN_HAND"] = 1
    state.player_state["P1_BRICK_IN_HAND"] = 1
    window = TradeWindow(
        id="multi-offer-window",
        turn_player=Color.RED,
        participants=state.colors,
    )
    red_offer = window.create_offer(
        TradeOffer(
            id="red-offer",
            offered_by=Color.RED,
            audience=frozenset({Color.BLUE, Color.WHITE, Color.ORANGE}),
            give=(1, 0, 0, 0, 0),
            receive=(0, 1, 0, 0, 0),
        )
    )
    red_offer.willing_by.add(Color.BLUE)
    orange_offer = window.create_offer(
        TradeOffer(
            id="orange-offer",
            offered_by=Color.ORANGE,
            audience=frozenset({Color.RED, Color.BLUE, Color.WHITE}),
            give=(0, 0, 1, 0, 0),
            receive=(0, 0, 0, 1, 0),
        )
    )
    orange_offer.willing_by.add(Color.WHITE)
    window.create_offer(
        TradeOffer(
            id="white-counter",
            offered_by=Color.WHITE,
            audience=frozenset({Color.RED}),
            give=(0, 0, 0, 0, 1),
            receive=(1, 0, 0, 0, 0),
            parent_offer_id=red_offer.id,
        )
    )
    state.trade_window = window
    state.playable_actions = generate_playable_actions(state)
    return game


def test_confirm_trade_executes_one_candidate_and_closes_its_window():
    game = make_multi_offer_engine_game()
    window = game.state.trade_window
    candidate = next(
        item
        for item in window.executable_candidates()
        if item.counterparty == Color.BLUE
    )

    game.step(
        Action(Color.RED, ActionType.CONFIRM_TRADE, candidate),
        force=True,
    )

    assert not window.active_offers
    assert window.selected_candidate == candidate
    assert game.state.player_state["P0_WOOD_IN_HAND"] == 0
    assert game.state.player_state["P0_BRICK_IN_HAND"] == 1
    assert game.state.player_state["P1_WOOD_IN_HAND"] == 1
    assert game.state.player_state["P1_BRICK_IN_HAND"] == 0


def test_cancel_trade_preserves_unrelated_offers():
    game = make_multi_offer_engine_game()

    game.step(
        Action(Color.RED, ActionType.CANCEL_TRADE, None),
        force=True,
    )

    assert {offer.id for offer in game.state.trade_window.active_offers} == {
        "orange-offer",
        "white-counter",
    }


def test_trade_response_transitions_replace_and_clear_previous_state():
    events = [
        trade_offer_event(),
        {
            "stateChange": {
                "tradeState": {
                    "activeOffers": {
                        "trade-1": {"playerResponses": {"2": 1}}
                    }
                }
            }
        },
        {
            "stateChange": {
                "tradeState": {
                    "activeOffers": {
                        "trade-1": {"playerResponses": {"2": 2}}
                    }
                }
            }
        },
        {
            "stateChange": {
                "tradeState": {
                    "activeOffers": {
                        "trade-1": {"playerResponses": {"2": 1}}
                    }
                }
            }
        },
        {
            "stateChange": {
                "tradeState": {
                    "activeOffers": {
                        "trade-1": {"playerResponses": {"2": 0}}
                    }
                }
            }
        },
    ]
    actions = parse_colonist_events_to_actions(events)
    assert [action["type"] for action in actions] == [
        "OFFER_TRADE",
        "ACCEPT_TRADE",
        "REJECT_TRADE",
        "ACCEPT_TRADE",
        "CLEAR_TRADE_RESPONSE",
    ]

    state = ServerState()
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
        "colonist_color_to_engine_idx": {"1": 0, "2": 1},
        "end_game_state": {},
    }
    game_state = state.current_game.state
    game_state.is_initial_build_phase = False
    game_state.current_prompt = ActionPrompt.PLAY_TURN
    game_state.player_state["P0_HAS_ROLLED"] = True
    game_state.player_state["P0_WOOD_IN_HAND"] = 1
    game_state.player_state["P1_BRICK_IN_HAND"] = 1
    game_state.playable_actions = generate_playable_actions(game_state)

    replay_step_logic(state, lambda: None)
    replay_step_logic(state, lambda: None)
    offer = game_state.trade_window.offers["trade-1"]
    assert offer.willing_by == {Color.BLUE}
    assert offer.declined_by == set()
    assert state.replay_trade_ledger["trade-1"]["responses"] == {
        "2": "accepted"
    }

    replay_step_logic(state, lambda: None)
    offer = game_state.trade_window.offers["trade-1"]
    assert offer.willing_by == set()
    assert offer.declined_by == {Color.BLUE}
    assert not any(
        action.action_type == ActionType.CONFIRM_TRADE
        for action in game_state.playable_actions
    )

    replay_step_logic(state, lambda: None)
    offer = game_state.trade_window.offers["trade-1"]
    assert offer.willing_by == {Color.BLUE}
    assert offer.declined_by == set()

    replay_step_logic(state, lambda: None)
    offer = game_state.trade_window.offers["trade-1"]
    assert offer.willing_by == set()
    assert offer.declined_by == set()
    assert state.replay_trade_ledger["trade-1"]["responses"] == {}
    assert [event.event_type for event in state.current_game.events] == [
        "OFFER_TRADE", "ACCEPT_TRADE", "REJECT_TRADE", "ACCEPT_TRADE",
        "CLEAR_TRADE_RESPONSE",
    ]
    assert state.current_game.events[-1].public_payload == {"offer_id": "trade-1"}
    assert len(state.current_game.state.actions) == 4
    events_after_clear = deepcopy(state.current_game.events)
    assert replay_undo_logic(state, lambda: None)["actions_undone"] == 0
    assert state.current_game.state.trade_window.offers["trade-1"].willing_by == {Color.BLUE}
    assert len(state.current_game.events) == 4
    replay_step_logic(state, lambda: None)
    assert state.current_game.events == events_after_clear


def test_full_offer_zero_responses_then_independent_rejections_are_preserved():
    events = [
        {
            "stateChange": {
                "tradeState": {
                    "activeOffers": {
                        "trade-1": {
                            "id": "trade-1",
                            "creator": 3,
                            "playerResponses": {"1": 0, "2": 0, "5": 0},
                            "wantedResources": [1],
                            "offeredResources": [5],
                            "counterOfferInResponseToTradeId": None,
                        }
                    }
                }
            }
        },
        {"stateChange": {"tradeState": {"activeOffers": {
            "trade-1": {"playerResponses": {"1": 2}}
        }}}},
        {"stateChange": {"tradeState": {"activeOffers": {
            "trade-1": {"playerResponses": {"5": 2}}
        }}}},
        {"stateChange": {"tradeState": {"activeOffers": {
            "trade-1": {"playerResponses": {"2": 2}}
        }}}},
    ]
    events_before_parse = deepcopy(events)
    actions = parse_colonist_events_to_actions(events)
    assert events == events_before_parse
    assert [action["type"] for action in actions] == [
        "OFFER_TRADE",
        "REJECT_TRADE",
        "REJECT_TRADE",
        "REJECT_TRADE",
    ]

    state = ServerState()
    players = [
        Color.ORANGE,
        Color.BLACK,
        Color.RED,
        Color.BLUE,
    ]
    state.current_game = GameEngine(players, shuffle_players=False)
    state.current_players = players
    state.replay_mode = True
    state.game_running = True
    state.replay_data = {
        "events": events,
        "parsed_actions": actions,
        "colonist_color_to_engine_idx": {
            "3": 0,
            "5": 1,
            "1": 2,
            "2": 3,
        },
        "end_game_state": {},
    }

    for _ in actions:
        replay_step_logic(state, lambda: None)

    assert state.replay_trade_ledger["trade-1"]["responses"] == {
        "1": "rejected",
        "2": "rejected",
        "5": "rejected",
    }


@pytest.mark.parametrize(
    ("resource_key", "error_text"),
    [
        ("P0_WOOD_IN_HAND", "can no longer afford"),
        ("P1_BRICK_IN_HAND", "can no longer afford"),
    ],
)
def test_confirm_trade_revalidates_both_hands_before_transfer(
    resource_key, error_text
):
    game = make_multi_offer_engine_game()
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


def test_backward_navigation_restarts_replay_running_state():
    events = [
        trade_offer_event(),
        {
            "stateChange": {
                "tradeState": {"activeOffers": {"trade-1": None}}
            }
        },
    ]
    actions = parse_colonist_events_to_actions(events)
    state = ServerState()
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


def test_counter_only_replay_state_does_not_broadcast_fake_legacy_offer():
    events = [
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
    state = ServerState()
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
        def __init__(self):
            self.payload = None

        def emit(self, event, payload):
            assert event == "game_state"
            self.payload = payload

    socket = SocketRecorder()
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


def raw_active_offers_by_event(events):
    active_offers = {}
    snapshots = []
    for event in events:
        offer_updates = (
            event.get("stateChange", {})
            .get("tradeState", {})
            .get("activeOffers", {})
        )
        for trade_id, update in offer_updates.items():
            if update is None:
                active_offers.pop(trade_id, None)
                continue

            trade = active_offers.setdefault(trade_id, {})
            for key, value in update.items():
                if key == "playerResponses":
                    trade.setdefault(key, {}).update(value)
                else:
                    trade[key] = value
        snapshots.append(deepcopy(active_offers))
    return snapshots


def assert_replay_ledger_matches_raw(state, expected_raw_trades, context):
    assert set(state.replay_trade_ledger) == set(expected_raw_trades), context
    for trade_id, raw_trade in expected_raw_trades.items():
        replay_trade = state.replay_trade_ledger[trade_id]
        expected_responses = {
            str(player_id): "accepted" if response == 1 else "rejected"
            for player_id, response in raw_trade.get("playerResponses", {}).items()
            if response in (1, 2)
        }
        assert replay_trade["creator"] == raw_trade.get("creator"), context
        assert replay_trade["counter_offer_to"] == raw_trade.get(
            "counterOfferInResponseToTradeId"
        ), context
        assert replay_trade["responses"] == expected_responses, (
            context,
            trade_id,
            replay_trade["responses"],
            expected_responses,
        )


@pytest.mark.skipif(
    os.getenv("RUN_LOCAL_REPLAY_CORPUS") != "1",
    reason="set RUN_LOCAL_REPLAY_CORPUS=1 for the local replay audit",
)
def test_all_local_replays_have_exact_resources_and_trade_lifecycle():
    replay_dir = Path("artifacts/raw/colonist/replays")
    replay_files = sorted(replay_dir.glob("*.json"))
    requested_game_ids = {
        game_id
        for game_id in os.getenv("LOCAL_REPLAY_GAME_IDS", "").split(",")
        if game_id
    }
    requested_game_id = os.getenv("LOCAL_REPLAY_GAME_ID")
    if requested_game_id:
        requested_game_ids.add(requested_game_id)

    if requested_game_ids:
        replay_files = [
            path for path in replay_files if path.stem in requested_game_ids
        ]
        assert {path.stem for path in replay_files} == requested_game_ids
    else:
        expected_count = int(
            os.getenv("EXPECTED_LOCAL_REPLAY_COUNT", "18")
        )
        assert len(replay_files) == expected_count

    total_actions = 0
    trade_counts = Counter()
    trade_types = {
        "OFFER_TRADE",
        "COUNTER_OFFER",
        "ACCEPT_TRADE",
        "REJECT_TRADE",
        "CLEAR_TRADE_RESPONSE",
        "CLOSE_TRADE",
        "CONFIRM_TRADE",
        "MARITIME_TRADE",
    }

    with app.test_client() as client:
        for replay_file in replay_files:
            output = io.StringIO()
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                load_response = client.post(
                    "/api/load-replay",
                    json={"game_id": replay_file.stem},
                )
            assert load_response.status_code == 200, output.getvalue()[-4_000:]

            parsed_actions = server_state.replay_data["parsed_actions"]
            raw_events = server_state.replay_data["events"]
            raw_trade_snapshots = raw_active_offers_by_event(raw_events)
            total_actions += len(parsed_actions)
            trade_counts.update(
                action["type"]
                for action in parsed_actions
                if action.get("type") in trade_types
            )

            for action_index, action_hint in enumerate(parsed_actions):
                output = io.StringIO()
                with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                    step_response = client.post("/api/replay-step")
                assert step_response.status_code == 200, (
                    replay_file.stem,
                    action_index,
                    action_hint.get("index"),
                    action_hint.get("type"),
                    output.getvalue()[-4_000:],
                )

                expected_resources = action_hint.get("expected_resources", {})
                if expected_resources:
                    mismatches = validate_resources_match(
                        server_state.current_sandbox.game_engine,
                        expected_resources,
                        server_state.replay_data[
                            "colonist_color_to_engine_idx"
                        ],
                    )
                    assert not mismatches, (
                        replay_file.stem,
                        action_index,
                        action_hint.get("type"),
                        mismatches,
                    )

                game_state = server_state.current_sandbox.game_engine.state
                for player_idx in range(len(game_state.colors)):
                    hand = [
                        game_state.player_state[
                            f"P{player_idx}_{resource}_IN_HAND"
                        ]
                        for resource in RESOURCES
                    ]
                    assert all(count >= 0 for count in hand), (
                        replay_file.stem,
                        action_index,
                        player_idx,
                        hand,
                    )

                for resource_idx, resource in enumerate(RESOURCES):
                    resource_total = game_state.resource_freqdeck[resource_idx]
                    resource_total += sum(
                        game_state.player_state[
                            f"P{player_idx}_{resource}_IN_HAND"
                        ]
                        for player_idx in range(len(game_state.colors))
                    )
                    assert resource_total == 19, (
                        replay_file.stem,
                        action_index,
                        resource,
                        resource_total,
                    )

                raw_event_index = action_hint["index"]
                next_raw_event_index = (
                    parsed_actions[action_index + 1]["index"]
                    if action_index + 1 < len(parsed_actions)
                    else None
                )
                if next_raw_event_index != raw_event_index:
                    assert_replay_ledger_matches_raw(
                        server_state,
                        raw_trade_snapshots[raw_event_index],
                        (
                            replay_file.stem,
                            action_index,
                            raw_event_index,
                            action_hint.get("type"),
                        ),
                    )

            semantic_errors = [
                issue
                for issue in server_state.replay_semantic_issues
                if issue.get("severity") == "error"
            ]
            assert not semantic_errors, (replay_file.stem, semantic_errors)
            assert server_state.replay_index == len(parsed_actions)

    print(
        f"Audited {len(replay_files)} replays, {total_actions} actions, "
        f"and {sum(trade_counts.values())} trade lifecycle actions: "
        f"{dict(sorted(trade_counts.items()))}"
    )
