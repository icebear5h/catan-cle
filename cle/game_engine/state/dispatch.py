"""Route one Action to its handler and record the fully-specified result."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cle.game_engine.models.enums import Action, ActionType
from cle.game_engine.state.building import (
    apply_build_city,
    apply_build_road,
    apply_build_settlement,
    apply_buy_development_card,
)
from cle.game_engine.state.dev_cards import (
    apply_play_knight_card,
    apply_play_monopoly,
    apply_play_road_building,
    apply_play_year_of_plenty,
)
from cle.game_engine.state.robber import apply_discard, apply_move_robber, apply_steal
from cle.game_engine.state.trade import (
    apply_accept_trade,
    apply_cancel_trade,
    apply_confirm_trade,
    apply_counter_offer,
    apply_maritime_trade,
    apply_offer_trade,
    apply_reject_trade,
)
from cle.game_engine.state.turns import apply_end_turn, apply_roll

if TYPE_CHECKING:
    from cle.game_engine.state.core import GameState


def assert_forced_action_is_explicit(action: Action) -> None:
    """Reject forced actions that would otherwise rely on engine randomness."""
    if action.action_type == ActionType.ROLL:
        if not isinstance(action.value, (tuple, list)) or len(action.value) != 2:
            raise ValueError("Forced ROLL requires explicit dice tuple")
    elif action.action_type == ActionType.DISCARD:
        if action.value is None:
            raise ValueError("Forced DISCARD requires explicit discarded cards")
    elif action.action_type == ActionType.BUY_DEVELOPMENT_CARD:
        if action.value is None:
            raise ValueError("Forced BUY_DEVELOPMENT_CARD requires explicit card type")
    elif action.action_type == ActionType.STEAL:
        if (
            not isinstance(action.value, (tuple, list))
            or len(action.value) != 2
            or action.value[0] is None
            or action.value[1] is None
        ):
            raise ValueError("Forced STEAL requires explicit victim and resource")
    elif action.action_type == ActionType.MOVE_ROBBER:
        if action.value is None:
            raise ValueError("Forced MOVE_ROBBER requires explicit tile coordinate")
    elif action.action_type == ActionType.PLAY_YEAR_OF_PLENTY:
        if not action.value or any(resource is None for resource in action.value):
            raise ValueError("Forced PLAY_YEAR_OF_PLENTY requires explicit resources")
    elif action.action_type == ActionType.PLAY_MONOPOLY:
        if action.value is None:
            raise ValueError("Forced PLAY_MONOPOLY requires explicit resource")


def apply_action(state: GameState, action: Action, force: bool = False) -> Action:
    """Main controller call. Follows redux-like pattern and
    routes the given action to the appropiate state-changing calls.

    Responsible for maintaining:
        .current_player_index, .current_turn_index,
        .current_prompt (and similars), .playable_actions.

    Appends given action to the list of actions, as fully-specified action.

    Args:
        state (GameState): GameState to mutate
        action (Action): Action to carry out

    Raises:
        ValueError: If invalid action given

    Returns:
        Action: Fully-specified action
    """

    if force:
        assert_forced_action_is_explicit(action)

    executed_action: Action | None = None

    match action.action_type:
        case ActionType.END_TURN:
            apply_end_turn(state, action, force=force)
        case ActionType.BUILD_SETTLEMENT:
            apply_build_settlement(state, action, force=force)
        case ActionType.BUILD_ROAD:
            apply_build_road(state, action, force=force)
        case ActionType.BUILD_CITY:
            apply_build_city(state, action, force=force)
        case ActionType.BUY_DEVELOPMENT_CARD:
            executed_action = apply_buy_development_card(state, action, force=force)
        case ActionType.ROLL:
            executed_action = apply_roll(state, action, force=force)
        case ActionType.DISCARD:
            executed_action = apply_discard(state, action, force=force)
        case ActionType.MOVE_ROBBER:
            apply_move_robber(state, action, force=force)
        case ActionType.STEAL:
            executed_action = apply_steal(state, action, force=force)
        case ActionType.PLAY_KNIGHT_CARD:
            apply_play_knight_card(state, action, force=force)
        case ActionType.PLAY_YEAR_OF_PLENTY:
            apply_play_year_of_plenty(state, action, force=force)
        case ActionType.PLAY_MONOPOLY:
            apply_play_monopoly(state, action, force=force)
        case ActionType.PLAY_ROAD_BUILDING:
            apply_play_road_building(state, action, force=force)
        case ActionType.MARITIME_TRADE:
            apply_maritime_trade(state, action, force=force)
        case ActionType.OFFER_TRADE:
            executed_action = apply_offer_trade(state, action, force=force)
        case ActionType.ACCEPT_TRADE:
            apply_accept_trade(state, action, force=force)
        case ActionType.REJECT_TRADE:
            apply_reject_trade(state, action, force=force)
        case ActionType.CONFIRM_TRADE:
            apply_confirm_trade(state, action, force=force)
        case ActionType.COUNTER_OFFER:
            executed_action = apply_counter_offer(state, action, force=force)
        case ActionType.CANCEL_TRADE:
            apply_cancel_trade(state, action, force=force)
        case _:
            raise ValueError("Unknown ActionType " + str(action.action_type))

    if executed_action is None:
        executed_action = action

    state.actions.append(executed_action)
    return executed_action
