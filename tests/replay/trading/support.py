"""Replay trading states, raw trade snapshots, and ledger assertions."""
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from cle.game_engine.game import GameEngine
from cle.game_engine.models.actions import generate_playable_actions
from cle.game_engine.models.enums import ActionPrompt
from cle.game_engine.models.player import Color
from cle.game_engine.trading import TradeOffer, TradeWindow
from playground.game_viewer.state import ServerState

RESOURCES = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")


def make_confirmation_state() -> ServerState:
    state: Any = ServerState()
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


def engine_snapshot(state: ServerState) -> dict[str, Any]:
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


def trade_offer_event(trade_id: str = "trade-1") -> dict[str, Any]:
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


def make_multi_offer_engine_game() -> GameEngine:
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


def raw_active_offers_by_event(
    events: Sequence[dict[str, Any]],
) -> list[dict[str, dict[str, Any]]]:
    active_offers: dict[str, dict[str, Any]] = {}
    snapshots: list[dict[str, dict[str, Any]]] = []
    for event in events:
        offer_updates: Any = (
            event.get("stateChange", {})
            .get("tradeState", {})
            .get("activeOffers", {})
        )
        for trade_id, update in offer_updates.items():
            if update is None:
                active_offers.pop(trade_id, None)
                continue

            trade: Any = active_offers.setdefault(trade_id, {})
            for key, value in update.items():
                if key == "playerResponses":
                    trade.setdefault(key, {}).update(value)
                else:
                    trade[key] = value
        snapshots.append(deepcopy(active_offers))
    return snapshots


def assert_replay_ledger_matches_raw(
    state: ServerState,
    expected_raw_trades: Mapping[str, dict[str, Any]],
    context: object,
) -> None:
    assert set(state.replay_trade_ledger) == set(expected_raw_trades), context
    for trade_id, raw_trade in expected_raw_trades.items():
        replay_trade: Any = state.replay_trade_ledger[trade_id]
        expected_responses: Any = {
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
