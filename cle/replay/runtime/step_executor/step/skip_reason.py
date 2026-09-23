"""Explain why a recorded trade row found no engine action."""

from __future__ import annotations

from collections.abc import Sequence

from cle.game_engine.game import GameEngine
from cle.game_engine.models.enums import Action
from cle.replay.colonist.helpers import format_trade
from cle.replay.colonist.types import ActionHint
from cle.replay.contracts import ReplayRuntimeState

from ..context import seat_index, seating_map

__all__ = ["skip_message", "skip_reason_for"]


def skip_message(
    action_hint: ActionHint,
    action_type: str,
    trade_label: str,
) -> str:
    """The human label for the skipped trade row."""
    if action_type in ("COUNTER_OFFER", "OFFER_TRADE"):
        offered = action_hint.get("offered", ())
        wanted = action_hint.get("wanted", ())
        return f"{trade_label} {action_type} ({format_trade(offered, wanted)})"
    if action_type in ("ACCEPT_TRADE", "REJECT_TRADE", "CLEAR_TRADE_RESPONSE"):
        return f"{trade_label} {action_type}"
    return action_type


def skip_reason_for(
    state: ReplayRuntimeState,
    game: GameEngine,
    action_hint: ActionHint,
    action_type: str,
    playable: Sequence[Action],
) -> str:
    """Determine why the trade was skipped."""
    if action_type in (
        "ACCEPT_TRADE",
        "REJECT_TRADE",
        "CLEAR_TRADE_RESPONSE",
    ):
        seating = seating_map(state)
        responding_id = action_hint.get("player")
        creator_id = action_hint.get("creator")
        resp_idx = seat_index(seating, responding_id)
        creator_idx = seat_index(seating, creator_id)
        if resp_idx is None:
            return f" -- responder player {responding_id} not mapped to engine color"
        if creator_idx is None:
            return f" -- trade creator {creator_id} not mapped to engine color"
        creator_color = game.state.colors[creator_idx]
        active_offerers = [
            offer.offered_by
            for offer in (
                game.state.trade_window.active_offers
                if game.state.trade_window is not None
                else ()
            )
        ]
        if creator_color not in active_offerers:
            return (
                f" -- no active offer from {creator_color} "
                f"(active: {active_offerers})"
            )
        return " -- no matching action in engine playable actions"
    if action_type == "OFFER_TRADE":
        if not action_hint.get("trade_tuple"):
            return " -- no trade_tuple in replay data"
        engine_has_offer = any(
            "OFFER_TRADE" in str(a.action_type)
            for a in playable
            if hasattr(a, 'action_type')
        )
        if not engine_has_offer:
            return " -- engine has no OFFER_TRADE in playable actions"
        return " -- no matching action in engine playable actions"
    if action_type == "COUNTER_OFFER":
        if not action_hint.get("trade_tuple"):
            return " -- no trade_tuple in replay data"
        return " -- no matching action in engine playable actions"
    return " -- no matching action in engine playable actions"
