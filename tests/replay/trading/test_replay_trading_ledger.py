"""The trade-id ledger preserves creators, counters, and fast navigation."""
from typing import Any

from cle.game_engine.game import GameEngine
from cle.game_engine.models.player import Color
from cle.replay.colonist.event_parser import parse_colonist_events_to_actions
from cle.replay.runtime.navigation import (
    replay_goto_fast_logic,
    replay_undo_logic,
)
from cle.replay.runtime.step_executor import replay_step_logic
from playground.game_viewer.state import ServerState

from .support import (
    engine_snapshot,
    trade_offer_event,
)


def test_trade_id_ledger_preserves_same_creator_offers_and_counter_links() -> None:
    events: Any = [
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
    window: Any = state.current_game.state.trade_window
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
    restored: Any = state.current_game.state.trade_window
    assert restored.offers["trade-1"].active
    assert restored.offers["trade-1"].willing_by == {Color.BLUE}


def test_fast_navigation_uses_the_authoritative_replay_executor() -> None:
    events: Any = [
        trade_offer_event(),
        {
            "stateChange": {
                "tradeState": {"activeOffers": {"trade-1": None}}
            }
        },
    ]
    actions: Any = parse_colonist_events_to_actions(events)

    def make_state() -> ServerState:
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
        return state

    sequential_state = make_state()
    for _ in actions:
        replay_step_logic(sequential_state, lambda: None)

    fast_state = make_state()
    result: Any = replay_goto_fast_logic(
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
