"""Decode Colonist ``tradeState.activeOffers`` deltas into parsed rows.

Colonist reports offers, per-player responses, and closures as deltas on one
shared map keyed by trade id, so the tracker carries the merged offer bodies
and the last response seen for every open trade.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

from cle.replay.colonist.helpers import colonist_resources_to_tuple
from cle.replay.colonist.types import ActionHint
from cle.replay.contracts import as_list, as_mapping, list_field, mapping_field

from .resources import ResourceTracker

__all__ = ["TradeTracker"]


class TradeTracker:
    """Live Colonist trades, numbered sequentially for display."""

    def __init__(self) -> None:
        # Track active trades to detect new offers and responses
        # Structure: {trade_id: {full trade data with merged deltas}}
        self.active_offers: dict[str, dict[str, object]] = {}
        # Track which players have responded to each trade (to avoid duplicates)
        self.trade_responses: dict[str, dict[int, object]] = {}
        # Sequential trade numbering for UI display
        self.counter = 0
        self.trade_id_to_number: dict[str, int] = {}

    def decode(
        self,
        index: int,
        offer_updates: Mapping[str, object],
        closure_reason: str,
        resources: ResourceTracker,
    ) -> tuple[list[ActionHint], list[ActionHint]]:
        """Return this event's trade rows and the closures they were built with."""
        actions: list[ActionHint] = []
        closed_trades: list[ActionHint] = []

        for trade_id, offer_data in offer_updates.items():
            if offer_data is None:
                trade_data = self.active_offers.pop(trade_id, {})
                self.trade_responses.pop(trade_id, None)
                closed_trades.append({
                    "trade_id": trade_id,
                    "trade_num": self.trade_id_to_number.get(trade_id, 0),
                    "creator": trade_data.get("creator"),
                    "is_counter_offer": (
                        trade_data.get("counterOfferInResponseToTradeId") is not None
                    ),
                    "counter_offer_to": trade_data.get(
                        "counterOfferInResponseToTradeId"
                    ),
                    "reason": closure_reason,
                })
                continue

            offer = as_mapping(offer_data, f"tradeState.activeOffers.{trade_id}")
            # Check if this is a new trade offer (has full data)
            if "creator" in offer and "offeredResources" in offer and "wantedResources" in offer:
                actions.append(self._open_offer(index, trade_id, offer, resources))
            # Check for response updates (delta)
            elif "playerResponses" in offer:
                actions.extend(self._responses(index, trade_id, offer, resources))

        # Emit every source closure exactly once before any transaction log.
        for closed_trade in closed_trades:
            actions.append({
                "index": index,
                "type": "CLOSE_TRADE",
                "player": closed_trade.get("creator"),
                # Colonist events are atomic and may also contain a later
                # resource-changing action, so no intermediate hand snapshot
                # is authoritative for this synthetic lifecycle row.
                "expected_resources": {},
                **closed_trade,
            })

        return actions, closed_trades

    def _open_offer(
        self,
        index: int,
        trade_id: str,
        offer: Mapping[str, object],
        resources: ResourceTracker,
    ) -> ActionHint:
        """Record a new offer or counter offer and build its parsed row."""
        copied = deepcopy(dict(offer))
        self.active_offers[trade_id] = copied
        initial_responses = {
            int(color_id): response
            for color_id, response in mapping_field(
                offer, "playerResponses"
            ).items()
        }
        self.trade_responses[trade_id] = initial_responses

        offered, offered_any = colonist_resources_to_tuple(
            as_list(offer["offeredResources"], "offeredResources")
        )
        wanted, wanted_any = colonist_resources_to_tuple(
            as_list(offer["wantedResources"], "wantedResources")
        )

        # Check if this is a counter offer
        counter_offer_to = offer.get("counterOfferInResponseToTradeId")
        is_counter = counter_offer_to is not None

        # Assign trade number - counter offers inherit parent's number
        parent_num = (
            self.trade_id_to_number.get(counter_offer_to)
            if isinstance(counter_offer_to, str)
            else None
        )
        if is_counter and parent_num is not None:
            trade_num = parent_num
        else:
            self.counter += 1
            trade_num = self.counter
        self.trade_id_to_number[trade_id] = trade_num

        return {
            "index": index,
            "type": "COUNTER_OFFER" if is_counter else "OFFER_TRADE",
            "player": offer["creator"],
            "trade_id": trade_id,
            "trade_num": trade_num,
            "offered": offered,
            "wanted": wanted,
            "trade_tuple": offered + wanted + (offered_any, wanted_any),
            "is_flexible": bool(offered_any or wanted_any),
            "is_counter_offer": is_counter,
            "counter_offer_to": counter_offer_to,
            "player_responses": {
                str(color_id): response
                for color_id, response in initial_responses.items()
            },
            "expected_resources": resources.snapshot(),
        }

    def _responses(
        self,
        index: int,
        trade_id: str,
        offer: Mapping[str, object],
        resources: ResourceTracker,
    ) -> list[ActionHint]:
        """Build one row per changed response on an already-tracked offer."""
        incoming = as_mapping(offer["playerResponses"], "playerResponses")
        # Merge into tracked trade
        if trade_id in self.active_offers:
            stored = self.active_offers[trade_id]
            merged = stored.get("playerResponses")
            if not isinstance(merged, dict):
                merged = {}
                stored["playerResponses"] = merged
            merged.update(incoming)

        if trade_id not in self.trade_responses:
            self.trade_responses[trade_id] = {}

        rows: list[ActionHint] = []
        tracked = self.trade_responses[trade_id]
        for color_str, response in incoming.items():
            color_id = int(color_str)
            # Only emit action if response changed (avoid duplicates)
            if color_id in tracked and tracked[color_id] == response:
                continue
            previous_response = tracked.get(color_id)
            tracked[color_id] = response

            trade_num = self.trade_id_to_number.get(trade_id, 0)
            # Get the offer creator from active source state.
            trade_data = self.active_offers.get(trade_id, {})
            creator = trade_data.get("creator")
            offered, offered_any = colonist_resources_to_tuple(
                list_field(trade_data, "offeredResources")
            )
            wanted, wanted_any = colonist_resources_to_tuple(
                list_field(trade_data, "wantedResources")
            )
            counter_offer_to = trade_data.get("counterOfferInResponseToTradeId")
            is_counter = counter_offer_to is not None
            if response == 1:  # Accept
                row_type = "ACCEPT_TRADE"
            elif response == 2:  # Reject
                row_type = "REJECT_TRADE"
            else:
                row_type = "CLEAR_TRADE_RESPONSE"
            rows.append({
                "index": index,
                "type": row_type,
                "player": color_id,
                "trade_id": trade_id,
                "trade_num": trade_num,
                "creator": creator,
                "offered": offered,
                "wanted": wanted,
                "trade_tuple": offered + wanted + (offered_any, wanted_any),
                "is_flexible": bool(offered_any or wanted_any),
                "is_counter_offer": is_counter,
                "counter_offer_to": counter_offer_to,
                "previous_response": previous_response,
                "expected_resources": resources.snapshot(),
            })
        return rows
