"""
Module with main GameState class and main apply_action call (game controller).

Split into submodules by concern; every historical public name is reexported here
and GameState keeps this module path for pickled snapshots.
"""

from cle.game_engine.state.building import (
    apply_build_city,
    apply_build_road,
    apply_build_settlement,
    apply_buy_development_card,
    require_available_piece,
    require_build_resources,
)
from cle.game_engine.state.core import (
    CITIES_PER_PLAYER,
    PLAYER_INITIAL_STATE,
    ROADS_PER_PLAYER,
    SETTLEMENTS_PER_PLAYER,
    GameState,
)
from cle.game_engine.state.dev_cards import (
    apply_play_knight_card,
    apply_play_monopoly,
    apply_play_road_building,
    apply_play_year_of_plenty,
)
from cle.game_engine.state.dispatch import (
    apply_action,
    assert_forced_action_is_explicit,
)
from cle.game_engine.state.robber import (
    apply_discard,
    apply_move_robber,
    apply_steal,
    validate_discard,
)
from cle.game_engine.state.trade import (
    active_offer_id,
    apply_accept_trade,
    apply_cancel_trade,
    apply_confirm_trade,
    apply_counter_offer,
    apply_maritime_trade,
    apply_offer_trade,
    apply_reject_trade,
    ensure_trade_window,
    latest_trade_offer,
    new_trade_window,
    offer_for_response,
    reset_trading_state,
)
from cle.game_engine.state.turns import (
    advance_turn,
    apply_end_turn,
    apply_roll,
    next_player_index,
    roll_dice,
    yield_resources,
)

__all__ = [
    "CITIES_PER_PLAYER",
    "GameState",
    "PLAYER_INITIAL_STATE",
    "ROADS_PER_PLAYER",
    "SETTLEMENTS_PER_PLAYER",
    "active_offer_id",
    "advance_turn",
    "apply_accept_trade",
    "apply_action",
    "apply_build_city",
    "apply_build_road",
    "apply_build_settlement",
    "apply_buy_development_card",
    "apply_cancel_trade",
    "apply_confirm_trade",
    "apply_counter_offer",
    "apply_discard",
    "apply_end_turn",
    "apply_maritime_trade",
    "apply_move_robber",
    "apply_offer_trade",
    "apply_play_knight_card",
    "apply_play_monopoly",
    "apply_play_road_building",
    "apply_play_year_of_plenty",
    "apply_reject_trade",
    "apply_roll",
    "apply_steal",
    "assert_forced_action_is_explicit",
    "ensure_trade_window",
    "latest_trade_offer",
    "new_trade_window",
    "next_player_index",
    "offer_for_response",
    "require_available_piece",
    "require_build_resources",
    "reset_trading_state",
    "roll_dice",
    "validate_discard",
    "yield_resources",
]
