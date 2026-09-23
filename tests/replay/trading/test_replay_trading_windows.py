"""Confirm, cancel, and response transitions close the right windows."""
from copy import deepcopy
from typing import Any

from cle.game_engine.game import GameEngine
from cle.game_engine.models.actions import generate_playable_actions
from cle.game_engine.models.enums import Action, ActionPrompt, ActionType
from cle.game_engine.models.player import Color
from cle.replay.colonist.event_parser import parse_colonist_events_to_actions
from cle.replay.runtime.navigation import (
    replay_undo_logic,
)
from cle.replay.runtime.step_executor import replay_step_logic
from playground.game_viewer.state import ServerState

from .support import (
    make_multi_offer_engine_game,
    trade_offer_event,
)


def test_confirm_trade_executes_one_candidate_and_closes_its_window() -> None:
    game = make_multi_offer_engine_game()
    window: Any = game.state.trade_window
    candidate: Any = next(
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


def test_cancel_trade_preserves_unrelated_offers() -> None:
    game: Any = make_multi_offer_engine_game()

    game.step(
        Action(Color.RED, ActionType.CANCEL_TRADE, None),
        force=True,
    )

    assert {offer.id for offer in game.state.trade_window.active_offers} == {
        "orange-offer",
        "white-counter",
    }


def test_trade_response_transitions_replace_and_clear_previous_state() -> None:
    events: Any = [
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
    game_state: Any = state.current_game.state
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
    assert state.current_game.events[-1].public_payload == {
        "offer_id": "trade-1", "offer": state.current_game.events[0].public_payload,
    }
    assert len(state.current_game.state.actions) == 4
    events_after_clear = deepcopy(state.current_game.events)
    assert replay_undo_logic(state, lambda: None)["actions_undone"] == 0
    assert state.current_game.state.trade_window.offers["trade-1"].willing_by == {Color.BLUE}
    assert len(state.current_game.events) == 4
    replay_step_logic(state, lambda: None)
    assert state.current_game.events == events_after_clear


def test_full_offer_zero_responses_then_independent_rejections_are_preserved() -> None:
    events: Any = [
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

    state: Any = ServerState()
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
