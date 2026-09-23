"""Exact legal-menu ordering and standalone action descriptions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cle.game_engine.models.enums import Action, ActionType
from cle.game_engine.trading import TradeCandidate, TradeOffer, TradeWindow

if TYPE_CHECKING:
    from . import CatanObservationFormatter, Observation


def format_valid_actions(self: CatanObservationFormatter, obs: Observation) -> str:
    """Format the complete legal menu without changing engine ordering."""
    lines = [
        "EXACT LEGAL ACTION MENU "
        "(all entries are legal and affordable; choose its zero-based index):"
    ]
    if not obs.valid_actions:
        lines.append("  No actions available (waiting for turn)")
        return "\n".join(lines)
    lines.extend(
        f"  {index}. {self._format_single_action(action, obs)}"
        for index, action in enumerate(obs.valid_actions)
    )
    return "\n".join(lines)


def format_single_action(
    self: CatanObservationFormatter,
    action: Action,
    obs: Observation,
    *,
    discard_count: int | None = None,
) -> str:
    """Format a single action in Catan lingo, without needing a format pass."""
    at = action.action_type
    if at == ActionType.END_TURN:
        return "End turn"
    if at == ActionType.ROLL:
        return "Roll dice"
    if at == ActionType.BUILD_SETTLEMENT:
        desc = self._describe_node(action.value, obs)
        return f"Build settlement at {desc}"
    if at == ActionType.BUILD_CITY:
        desc = self._describe_node(action.value, obs)
        return f"Upgrade to city at {desc}"
    if at == ActionType.BUILD_ROAD:
        if isinstance(action.value, tuple) and len(action.value) == 2:
            n1_desc = self._describe_node(action.value[0], obs)
            n2_desc = self._describe_node(action.value[1], obs)
            return f"Build road between {n1_desc} and {n2_desc}"
    if at == ActionType.BUY_DEVELOPMENT_CARD:
        return "Buy development card"
    if at == ActionType.MARITIME_TRADE:
        val = action.value
        if isinstance(val, tuple) and len(val) >= 2:
            offered = [r for r in val[:-1] if r is not None]
            received = val[-1]
            if offered:
                give_name = offered[0].name if hasattr(offered[0], 'name') else str(offered[0])
                get_name = received.name if hasattr(received, 'name') else str(received)
                return f"Trade {len(offered)} {give_name} for 1 {get_name}"
    if at == ActionType.OFFER_TRADE:
        if isinstance(action.value, TradeOffer):
            return (
                f"Offer: give {self._format_resource_tuple_with_any(action.value.give, action.value.give_any)}, "
                f"receive {self._format_resource_tuple_with_any(action.value.receive, action.value.receive_any)}"
            )
        return str(action.value)
    if at in {ActionType.ACCEPT_TRADE, ActionType.REJECT_TRADE}:
        window: TradeWindow | None = obs.trade_window
        offer = window.offers.get(action.value) if window is not None else None
        verb = "Signal willingness for" if at == ActionType.ACCEPT_TRADE else "Decline"
        if offer is None:
            return f"{verb} offer {action.value}"
        give = self._format_resource_tuple_with_any(offer.receive, offer.receive_any)
        receive = self._format_resource_tuple_with_any(offer.give, offer.give_any)
        offered_by = self._color_name(offer.offered_by)
        return f"{verb} {offer.id} from {offered_by}: give {give}, receive {receive}"
    if at == ActionType.COUNTER_OFFER:
        if isinstance(action.value, TradeOffer):
            return (
                f"Counter {action.value.parent_offer_id}: give "
                f"{self._format_resource_tuple_with_any(action.value.give, action.value.give_any)}, "
                f"receive {self._format_resource_tuple_with_any(action.value.receive, action.value.receive_any)}"
            )
        return str(action.value)
    if at == ActionType.CONFIRM_TRADE and isinstance(action.value, TradeCandidate):
        candidate = action.value
        window = obs.trade_window
        offer = window.offers.get(candidate.offer_id) if window is not None else None
        partner = self._color_name(candidate.counterparty)
        if offer is None:
            return f"Confirm {candidate.offer_id} with {partner}"
        if offer.offered_by == candidate.turn_player:
            give_bundle, give_any = offer.give, offer.give_any
            receive_bundle, receive_any = offer.receive, offer.receive_any
        else:
            give_bundle, give_any = offer.receive, offer.receive_any
            receive_bundle, receive_any = offer.give, offer.give_any
        return (
            f"Confirm {candidate.offer_id} with {partner}: give "
            f"{self._format_resource_tuple_with_any(give_bundle, give_any)}, "
            f"receive {self._format_resource_tuple_with_any(receive_bundle, receive_any)}"
        )
    if at == ActionType.CANCEL_TRADE:
        return f"Withdraw offer {action.value}"
    if at == ActionType.PLAY_KNIGHT_CARD:
        return "Play knight card"
    if at == ActionType.PLAY_YEAR_OF_PLENTY:
        if isinstance(action.value, tuple) and len(action.value) in {1, 2}:
            resources = " and ".join(
                resource.name if hasattr(resource, "name") else str(resource)
                for resource in action.value
            )
            return f"Year of Plenty: take {resources}"
        return "Play Year of Plenty"
    if at == ActionType.PLAY_MONOPOLY:
        r = action.value.name if hasattr(action.value, 'name') else str(action.value)
        return f"Monopoly on {r}"
    if at == ActionType.PLAY_ROAD_BUILDING:
        return "Play road building card"
    if at == ActionType.MOVE_ROBBER:
        return f"Move robber to {action.value}"
    if at == ActionType.STEAL:
        if action.value is not None:
            color_name = action.value.name if hasattr(action.value, 'name') else str(action.value)
            return f"Steal from {color_name}"
        return "Steal (no targets)"
    if at == ActionType.DISCARD:
        if discard_count is not None:
            return (
                f"Discard exactly {discard_count} resource cards; "
                "specify named resource counts in <discard>."
            )
        return "Discard resources"
    type_name = at.name if hasattr(at, 'name') else str(at)
    return f"{type_name}: {action.value}"
