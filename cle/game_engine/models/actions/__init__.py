"""
Move-generation functions (these return a list of actions that can be taken
by current player). Main function is generate_playable_actions.

Every generator stays importable from this package path. generate_playable_actions
resolves the per-prompt generators through this module's globals at call time, so
replacing a name here still steers the composed result.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cle.game_engine.models.decks import freqdeck_contains
from cle.game_engine.models.enums import Action, ActionPrompt, ActionType
from cle.game_engine.models.player import Color
from cle.game_engine.state_functions import (
    get_player_freqdeck,
    player_can_afford_dev_card,
    player_can_play_dev,
    player_has_rolled,
)

from .exchange import (
    MaritimeTrade,
    YearOfPlentyPick,
    inner_maritime_trade_possibilities,
    maritime_trade_possibilities,
    monopoly_possibilities,
    ncr,
    player_trade_possibilities,
    year_of_plenty_possibilities,
)
from .placement import (
    city_possibilities,
    discard_possibilities,
    initial_road_possibilities,
    road_building_possibilities,
    robber_possibilities,
    settlement_possibilities,
    steal_possibilities,
)

if TYPE_CHECKING:
    from cle.game_engine.state import GameState
    from cle.game_engine.trading import TradeWindow

__all__ = [
    "Action",
    "MaritimeTrade",
    "YearOfPlentyPick",
    "city_possibilities",
    "discard_possibilities",
    "generate_playable_actions",
    "initial_road_possibilities",
    "inner_maritime_trade_possibilities",
    "maritime_trade_possibilities",
    "monopoly_possibilities",
    "ncr",
    "player_trade_possibilities",
    "road_building_possibilities",
    "robber_possibilities",
    "settlement_possibilities",
    "steal_possibilities",
    "trade_response_actions",
    "year_of_plenty_possibilities",
]


def trade_response_actions(state: GameState, color: Color) -> list[Action]:
    """Return exact offer responses available to one participant."""
    window: TradeWindow | None = getattr(state, "trade_window", None)
    if window is None or window.status.value != "open":
        return []
    actions: list[Action] = []
    for offer in window.active_offers:
        if (
            offer.offered_by == color
            or color not in offer.audience
            or color in offer.willing_by
            or color in offer.declined_by
        ):
            continue
        actions.append(Action(color, ActionType.REJECT_TRADE, offer.id))
        if offer.parent_offer_id is not None:
            continue
        if freqdeck_contains(get_player_freqdeck(state, color), offer.receive):
            actions.append(Action(color, ActionType.ACCEPT_TRADE, offer.id))
        if (
            window.remaining_counter_slots > 0
            and window.round < window.limits.max_negotiation_rounds
            and sum(
                active_offer.offered_by == color
                for active_offer in window.active_offers
            ) < window.limits.max_offers_per_player
            and sum(get_player_freqdeck(state, color)) > 0
        ):
            actions.append(
                Action(
                    color,
                    ActionType.COUNTER_OFFER,
                    f"COUNTER_OFFER:{offer.id}: supply a named trade_offer",
                )
            )
    return actions


def generate_playable_actions(state: GameState) -> list[Action]:
    action_prompt = state.current_prompt
    color = state.current_color()

    if action_prompt == ActionPrompt.BUILD_INITIAL_SETTLEMENT:
        return settlement_possibilities(state, color, True)
    elif action_prompt == ActionPrompt.BUILD_INITIAL_ROAD:
        return initial_road_possibilities(state, color)
    elif action_prompt == ActionPrompt.MOVE_ROBBER:
        return robber_possibilities(state, color)
    elif action_prompt == ActionPrompt.STEAL:
        return steal_possibilities(state, color)
    elif action_prompt == ActionPrompt.PLAY_TURN:
        if state.is_road_building:
            return road_building_possibilities(state, color, False)
        actions: list[Action] = []
        # Allow playing dev cards before and after rolling
        if player_can_play_dev(state, color, "YEAR_OF_PLENTY"):
            actions.extend(year_of_plenty_possibilities(color, state.resource_freqdeck))
        if player_can_play_dev(state, color, "MONOPOLY"):
            actions.extend(monopoly_possibilities(color))
        if player_can_play_dev(state, color, "KNIGHT"):
            actions.append(Action(color, ActionType.PLAY_KNIGHT_CARD, None))
        if (
            player_can_play_dev(state, color, "ROAD_BUILDING")
            and len(road_building_possibilities(state, color, False)) > 0
        ):
            actions.append(Action(color, ActionType.PLAY_ROAD_BUILDING, None))
        if not player_has_rolled(state, color):
            actions.append(Action(color, ActionType.ROLL, None))
        else:
            actions.append(Action(color, ActionType.END_TURN, None))
            actions.extend(road_building_possibilities(state, color))
            actions.extend(settlement_possibilities(state, color))
            actions.extend(city_possibilities(state, color))

            can_buy_dev_card = (
                player_can_afford_dev_card(state, color)
                and len(state.development_listdeck) > 0
            )
            if can_buy_dev_card:
                actions.append(Action(color, ActionType.BUY_DEVELOPMENT_CARD, None))

            # Trade
            actions.extend(maritime_trade_possibilities(state, color))
            trade_window: TradeWindow | None = getattr(state, "trade_window", None)
            can_open_trade = (
                trade_window is None
                or trade_window.status.value == "closed"
                or (
                    trade_window.remaining_root_slots > 0
                    and trade_window.round < trade_window.limits.max_negotiation_rounds
                    and sum(
                        offer.offered_by == color
                        for offer in trade_window.active_offers
                    ) < trade_window.limits.max_offers_per_player
                )
            )
            if can_open_trade:
                actions.extend(player_trade_possibilities(state, color))

        window: TradeWindow | None = getattr(state, "trade_window", None)
        if window is not None and window.status.value == "open":
            actions.extend(trade_response_actions(state, color))
            actions.extend(
                Action(color, ActionType.CANCEL_TRADE, offer.id)
                for offer in window.active_offers
                if offer.offered_by == color
            )
            if color == window.turn_player:
                for candidate in window.executable_candidates():
                    offer = window.offers[candidate.offer_id]
                    turn_gives = (
                        offer.give
                        if offer.offered_by == color
                        else offer.receive
                    )
                    other_gives = (
                        offer.receive
                        if offer.offered_by == color
                        else offer.give
                    )
                    if freqdeck_contains(get_player_freqdeck(state, color), turn_gives) and freqdeck_contains(
                        get_player_freqdeck(state, candidate.counterparty),
                        other_gives,
                    ):
                        actions.append(
                            Action(
                                color,
                                ActionType.CONFIRM_TRADE,
                                candidate,
                            )
                        )

        return actions
    elif action_prompt == ActionPrompt.DISCARD:
        return discard_possibilities(color)
    else:
        raise RuntimeError("Unknown ActionPrompt: " + str(action_prompt))
