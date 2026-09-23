"""Bounded offer board and ordered resource bundle descriptions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cle.game_engine.trading import ResourceBundle, TradeWindow

if TYPE_CHECKING:
    from . import CatanObservationFormatter, Observation


def format_trade_context(self: CatanObservationFormatter, obs: Observation) -> str:
    """Format the bounded offer board."""
    window: TradeWindow | None = getattr(obs, "trade_window", None)
    if window is None or not window.active_offers:
        return ""
    lines = [
        f"TRADING WINDOW {window.id} (round {window.round}):",
        f"  Remaining root slots: {window.remaining_root_slots}",
        f"  Remaining counter slots: {window.remaining_counter_slots}",
    ]
    if any(offer.parent_offer_id for offer in window.active_offers):
        lines.append(
            "  Counteroffers cannot be accepted; the turn player executes one with confirm_trade."
        )
    for offer in window.active_offers:
        give = self._format_resource_tuple_with_any(offer.give, offer.give_any)
        receive = self._format_resource_tuple_with_any(offer.receive, offer.receive_any)
        parent = f" counter to {offer.parent_offer_id}" if offer.parent_offer_id else ""
        line = (
            f"  {offer.id}: {self._color_name(offer.offered_by)} "
            f"gives {give} for {receive}{parent}"
        )
        if offer.willing_by:
            line += " [willing: " + ", ".join(
                self._color_name(color) for color in offer.willing_by
            ) + "]"
        if offer.declined_by:
            line += " [declined: " + ", ".join(
                self._color_name(color) for color in offer.declined_by
            ) + "]"
        lines.append(line)
    return "\n".join(lines)


def format_resource_tuple(resources: ResourceBundle) -> str:
    """Format a 5-tuple of resources into readable string."""
    resource_names = ['WOOD', 'BRICK', 'SHEEP', 'WHEAT', 'ORE']
    parts = []
    for i, count in enumerate(resources):
        if count > 0:
            parts.append(f"{count} {resource_names[i]}")
    return ", ".join(parts) if parts else "nothing"


def format_resource_tuple_with_any(resources: ResourceBundle, any_count: int) -> str:
    """Format a 5-tuple of resources with optional wildcard count."""
    resource_names = ['WOOD', 'BRICK', 'SHEEP', 'WHEAT', 'ORE']
    parts = []
    for i, count in enumerate(resources):
        if count > 0:
            parts.append(f"{count} {resource_names[i]}")
    if any_count > 0:
        parts.append(f"{any_count} ANY")
    return ", ".join(parts) if parts else "nothing"
