"""Action admissibility checks against the current engine state."""

from __future__ import annotations

from typing import TypeGuard

from cle.game_engine.models.actions import trade_response_actions
from cle.game_engine.models.enums import Action, ActionPrompt, ActionType
from cle.game_engine.state import GameState, new_trade_window, validate_discard
from cle.game_engine.state_functions import get_player_freqdeck, player_has_rolled
from cle.game_engine.trading import TradeOffer, TradeWindowStatus


def is_valid_action(state: GameState, action: Action) -> bool:
    """True if its a valid action right now. An action is valid
    if its in playable_actions or if its a OFFER_TRADE/COUNTER_OFFER in the right time."""
    if not isinstance(action, Action) or action.color not in state.colors:
        return False
    if action.action_type == ActionType.DISCARD and action.value is not None:
        try:
            validate_discard(state, action)
        except ValueError:
            return False
        return True
    if action.action_type in {ActionType.OFFER_TRADE, ActionType.COUNTER_OFFER}:
        offer: object = action.value
        if not (
            is_valid_trade(offer)
            and offer.id is None
            and state.current_prompt == ActionPrompt.PLAY_TURN
            and offer.offered_by == action.color
        ):
            return False
        window = state.trade_window
        if action.action_type == ActionType.OFFER_TRADE:
            if not (
                state.current_color() == action.color
                and player_has_rolled(state, action.color)
                and offer.parent_offer_id is None
            ):
                return False
            if window is None or window.status == TradeWindowStatus.CLOSED:
                # Match ensure_trade_window without installing a window during validation.
                window = new_trade_window(state)
        elif window is None or offer.parent_offer_id is None:
            return False
        try:
            window.validate_offer(offer)
        except ValueError:
            return False
        return can_fund_offer(state, offer)

    if action.action_type in {
        ActionType.ACCEPT_TRADE,
        ActionType.REJECT_TRADE,
    } and action in trade_response_actions(state, action.color):
        return True

    return action in state.playable_actions


def is_valid_trade(action_value: object) -> TypeGuard[TradeOffer]:
    """Return whether an action carries one canonical typed offer."""
    return isinstance(action_value, TradeOffer)


def can_fund_offer(state: GameState, offer: TradeOffer) -> bool:
    hand = get_player_freqdeck(state, offer.offered_by)
    return (
        all(held >= given for held, given in zip(hand, offer.give))
        and sum(hand) - sum(offer.give) >= offer.give_any
    )
