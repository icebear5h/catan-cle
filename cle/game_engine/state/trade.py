"""Maritime trades and the turn-scoped domestic trade window."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from cle.game_engine.models.actions import generate_playable_actions
from cle.game_engine.models.decks import (
    freqdeck_add,
    freqdeck_contains,
    freqdeck_from_listdeck,
    freqdeck_subtract,
)
from cle.game_engine.models.enums import Action, ActionPrompt
from cle.game_engine.models.player import Color
from cle.game_engine.state_functions import (
    get_player_freqdeck,
    player_freqdeck_add,
    player_freqdeck_subtract,
    player_resource_freqdeck_contains,
)
from cle.game_engine.trading import (
    TradeCandidate,
    TradeOffer,
    TradeWindow,
    TradeWindowStatus,
)

if TYPE_CHECKING:
    from cle.game_engine.state.core import GameState


def new_trade_window(state: GameState) -> TradeWindow:
    """Preview a new window without mutating state or consuming an identity."""
    return TradeWindow(
        id=f"turn-{state.num_turns}-trade-{len(state.actions)}",
        turn_player=state.colors[state.current_turn_index],
        participants=state.colors,
        limits=state.trade_limits,
    )


def ensure_trade_window(state: GameState) -> TradeWindow:
    window = state.trade_window
    if window is None or window.status == TradeWindowStatus.CLOSED:
        window = new_trade_window(state)
        state.trade_window = window
    return window


def latest_trade_offer(
    state: GameState,
    offered_by: Color,
    *,
    root: bool | None = None,
) -> TradeOffer | None:
    if state.trade_window is None:
        return None
    offers = [
        offer
        for offer in state.trade_window.active_offers
        if offer.offered_by == offered_by
        and (root is None or (offer.parent_offer_id is None) == root)
    ]
    return offers[-1] if offers else None


def offer_for_response(
    state: GameState, value: str | Color, *, root: bool | None = True
) -> TradeOffer | None:
    """Resolve an offer id, else the latest offer by that color (any depth when root=None)."""
    if isinstance(value, str):
        # No offer is ever offered_by a str, so the id lookup is the only match.
        if state.trade_window is None:
            return None
        offer = state.trade_window.offers.get(value)
        return offer if offer is not None and offer.active else None
    return latest_trade_offer(state, value, root=root)


def active_offer_id(offer: TradeOffer) -> str:
    """Offers held by a window always carry an id; mirror TradeWindow._active otherwise."""
    if offer.id is None:
        raise ValueError(f"Offer {offer.id!r} is not active")
    return offer.id


def reset_trading_state(state: GameState) -> None:
    """Close the turn-scoped offer board."""
    if state.trade_window is not None:
        state.trade_window.close()


def apply_maritime_trade(state: GameState, action: Action, force: bool = False) -> None:
    trade_offer = action.value

    # Support two formats:
    # 1. New format: (given_freqdeck, received_freqdeck) - two 5-tuples for multi-resource trades
    # 2. Old format: (res, res, res, res, res_asked) - single resource trade
    if isinstance(trade_offer, tuple) and len(trade_offer) == 2:
        # New format: multi-resource maritime trade
        offering, asking = trade_offer
        # Ensure they're tuples/lists of length 5
        if not (len(offering) == 5 and len(asking) == 5):
            raise ValueError(f"Invalid maritime trade format: {trade_offer}")
    else:
        # Old format: single resource trade
        offering = freqdeck_from_listdeck(filter(lambda r: r is not None, trade_offer[:-1]))
        asking = freqdeck_from_listdeck(trade_offer[-1:])

    if not player_resource_freqdeck_contains(state, action.color, offering):
        raise ValueError("Trying to trade without money")
    if not freqdeck_contains(state.resource_freqdeck, asking):
        raise ValueError("Bank doenst have those cards")
    player_freqdeck_subtract(state, action.color, offering)
    state.resource_freqdeck = freqdeck_add(state.resource_freqdeck, offering)
    player_freqdeck_add(state, action.color, asking)
    state.resource_freqdeck = freqdeck_subtract(state.resource_freqdeck, asking)

    # state.current_player_index stays the same
    state.current_prompt = ActionPrompt.PLAY_TURN
    state.playable_actions = generate_playable_actions(state)


def apply_offer_trade(state: GameState, action: Action, force: bool = False) -> Action:
    offer = action.value
    if not isinstance(offer, TradeOffer) or offer.parent_offer_id is not None:
        raise ValueError("OFFER_TRADE requires one root TradeOffer")
    if not force and offer.id is not None:
        raise ValueError("Trade offer IDs are assigned by the engine")
    materialized = ensure_trade_window(state).create_offer(
        offer,
        allow_duplicate=force,
    )
    state.current_prompt = ActionPrompt.PLAY_TURN
    state.playable_actions = generate_playable_actions(state)
    return Action(action.color, action.action_type, copy.deepcopy(materialized))


def apply_accept_trade(state: GameState, action: Action, force: bool = False) -> None:
    offer = offer_for_response(state, action.value, root=None)
    if offer is None:
        raise ValueError(f"No active offer for {action.value}")
    window = state.trade_window
    assert window is not None  # offer_for_response only returns offers held by the window
    window.signal_willingness(active_offer_id(offer), action.color)
    state.playable_actions = generate_playable_actions(state)


def apply_reject_trade(state: GameState, action: Action, force: bool = False) -> None:
    offer = offer_for_response(state, action.value, root=None)
    if offer is None:
        raise ValueError(f"No active offer for {action.value}")
    window = state.trade_window
    assert window is not None  # offer_for_response only returns offers held by the window
    window.decline(active_offer_id(offer), action.color)
    state.playable_actions = generate_playable_actions(state)


def apply_confirm_trade(state: GameState, action: Action, force: bool = False) -> None:
    window = state.trade_window
    if window is None:
        raise ValueError("No active trade window")
    candidate = action.value
    if not isinstance(candidate, TradeCandidate):
        raise ValueError("CONFIRM_TRADE requires a typed TradeCandidate")
    offer = window.offers.get(candidate.offer_id)
    if offer is None or not offer.active:
        raise ValueError(f"No active offer {candidate.offer_id}")
    window.select(action.color, candidate)
    counterparty = candidate.counterparty

    if offer.offered_by == action.color:
        actor_gives, actor_receives = offer.give, offer.receive
    else:
        actor_gives, actor_receives = offer.receive, offer.give
    if not freqdeck_contains(get_player_freqdeck(state, action.color), actor_gives):
        raise ValueError(f"{action.color} can no longer afford this trade")
    if not freqdeck_contains(get_player_freqdeck(state, counterparty), actor_receives):
        raise ValueError(f"{counterparty} can no longer afford this trade")

    player_freqdeck_subtract(state, action.color, actor_gives)
    player_freqdeck_add(state, action.color, actor_receives)
    player_freqdeck_subtract(state, counterparty, actor_receives)
    player_freqdeck_add(state, counterparty, actor_gives)
    window.mark_executed()
    state.current_player_index = state.current_turn_index
    state.current_prompt = ActionPrompt.PLAY_TURN
    state.playable_actions = generate_playable_actions(state)


def apply_counter_offer(state: GameState, action: Action, force: bool = False) -> Action:
    offer = action.value
    if not isinstance(offer, TradeOffer) or offer.parent_offer_id is None:
        raise ValueError("COUNTER_OFFER requires one counter TradeOffer")
    if not force and offer.id is not None:
        raise ValueError("Trade offer IDs are assigned by the engine")
    window = ensure_trade_window(state)
    if offer.parent_offer_id not in window.offers:
        raise ValueError(f"No active root offer {offer.parent_offer_id!r}")
    materialized = window.create_offer(offer, allow_duplicate=force)
    state.current_prompt = ActionPrompt.PLAY_TURN
    state.playable_actions = generate_playable_actions(state)
    return Action(action.color, action.action_type, copy.deepcopy(materialized))


def apply_cancel_trade(state: GameState, action: Action, force: bool = False) -> None:
    window = state.trade_window
    if window is not None:
        if isinstance(action.value, str):
            window.withdraw(action.value, action.color)
        else:
            for offer in tuple(window.active_offers):
                if offer.offered_by == action.color:
                    window.withdraw(active_offer_id(offer), action.color)
    state.current_player_index = state.current_turn_index
    state.current_prompt = ActionPrompt.PLAY_TURN
    state.playable_actions = generate_playable_actions(state)
