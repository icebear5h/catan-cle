"""Bounded turn-scoped domestic trade offers."""

from __future__ import annotations

from dataclasses import dataclass, field

from cle.game_engine.models.player import Color
from cle.game_engine.models.trade import RESOURCE_NAMES as RESOURCE_NAMES
from cle.game_engine.models.trade import DealKey, deal_key
from cle.game_engine.models.trade import ResourceBundle as ResourceBundle
from cle.game_engine.models.trade import TradeCandidate as TradeCandidate
from cle.game_engine.models.trade import TradeLimits as TradeLimits
from cle.game_engine.models.trade import TradeOffer as TradeOffer
from cle.game_engine.models.trade import TradeOfferStatus as TradeOfferStatus
from cle.game_engine.models.trade import TradeWindowStatus as TradeWindowStatus


class _TradeCapError(ValueError):
    """A lifecycle limit failure, counted only when creation is attempted."""


@dataclass(slots=True)
class TradeWindow:
    id: str
    turn_player: Color
    participants: tuple[Color, ...]
    limits: TradeLimits = field(default_factory=TradeLimits)
    round: int = 0
    offers: dict[str, TradeOffer] = field(default_factory=dict)
    selected_candidate: TradeCandidate | None = None
    status: TradeWindowStatus = TradeWindowStatus.OPEN
    cap_hits: int = 0
    _next_offer_number: int = 1
    _seen_deals: set[DealKey] = field(default_factory=set)

    @property
    def active_offers(self) -> tuple[TradeOffer, ...]:
        return tuple(offer for offer in self.offers.values() if offer.active)

    @property
    def remaining_root_slots(self) -> int:
        used = sum(
            offer.active and offer.parent_offer_id is None
            for offer in self.offers.values()
        )
        return max(0, self.limits.max_active_root_offers - used)

    @property
    def remaining_counter_slots(self) -> int:
        used = sum(
            offer.active and offer.parent_offer_id is not None
            for offer in self.offers.values()
        )
        return max(0, self.limits.max_active_counteroffers - used)

    def validate_offer(
        self,
        offer: TradeOffer,
        *,
        supersedes_offer_id: str | None = None,
        offer_id: str | None = None,
        allow_duplicate: bool = False,
    ) -> None:
        """Raise ValueError if creation would fail, without changing window or offer."""
        self._require_open()
        if offer.offered_by not in self.participants:
            raise ValueError("Trade offerer must be a window participant")
        if not offer.audience.issubset(self.participants):
            raise ValueError("Trade audience contains a non-participant")
        if (
            not allow_duplicate
            and self.round >= self.limits.max_negotiation_rounds
        ):
            raise _TradeCapError("maximum negotiation rounds reached")

        if offer.parent_offer_id is not None:
            parent = self._active(offer.parent_offer_id)
            if parent.parent_offer_id is not None:
                raise ValueError("A counteroffer cannot be countered")
            if offer.audience != frozenset({self.turn_player}):
                raise ValueError(
                    "Counteroffers must be directed only to the turn player"
                )
            if offer.offered_by == parent.offered_by:
                raise ValueError("A player cannot counter its own offer")
            self._require_audience(parent, offer.offered_by)
            if not allow_duplicate and self.remaining_counter_slots == 0:
                raise _TradeCapError("maximum active counteroffers reached")
        elif (
            not allow_duplicate
            and self.remaining_root_slots == 0
            and supersedes_offer_id is None
        ):
            raise _TradeCapError("maximum active root offers reached")

        active_for_player = [
            active_offer
            for active_offer in self.active_offers
            if active_offer.offered_by == offer.offered_by
            and active_offer.id != supersedes_offer_id
        ]
        if (
            not allow_duplicate
            and len(active_for_player) >= self.limits.max_offers_per_player
        ):
            raise _TradeCapError("maximum active offers for player reached")

        if not allow_duplicate and self._deal_key(offer) in self._seen_deals:
            raise ValueError("Equivalent offer already appeared in this trade window")

        if supersedes_offer_id is not None:
            old = self._active(supersedes_offer_id)
            if old.offered_by != offer.offered_by:
                raise ValueError("Only an offer's creator may supersede it")

        materialized_id = offer_id or offer.id or (
            f"{self.id}:o{self._next_offer_number}"
        )
        if materialized_id in self.offers:
            raise ValueError(f"Offer {materialized_id!r} already exists")

    def create_offer(
        self,
        offer: TradeOffer,
        *,
        supersedes_offer_id: str | None = None,
        offer_id: str | None = None,
        allow_duplicate: bool = False,
    ) -> TradeOffer:
        try:
            self.validate_offer(
                offer,
                supersedes_offer_id=supersedes_offer_id,
                offer_id=offer_id,
                allow_duplicate=allow_duplicate,
            )
        except _TradeCapError:
            self.cap_hits += 1
            raise

        materialized_id = offer_id or offer.id or (
            f"{self.id}:o{self._next_offer_number}"
        )
        materialized = TradeOffer(
            id=materialized_id,
            offered_by=offer.offered_by,
            audience=offer.audience,
            give=offer.give,
            receive=offer.receive,
            give_any=offer.give_any,
            receive_any=offer.receive_any,
            parent_offer_id=offer.parent_offer_id,
            created_round=self.round,
        )
        if supersedes_offer_id is not None:
            self.offers[supersedes_offer_id].status = TradeOfferStatus.WITHDRAWN
        self._next_offer_number += 1
        self.offers[materialized_id] = materialized
        if offer.parent_offer_id is not None:
            parent = self.offers[offer.parent_offer_id]
            parent.willing_by.discard(offer.offered_by)
            parent.declined_by.add(offer.offered_by)
        self._seen_deals.add(self._deal_key(offer))
        return materialized

    def signal_willingness(self, offer_id: str, player: Color) -> None:
        offer = self._active(offer_id)
        self._require_audience(offer, player)
        offer.declined_by.discard(player)
        offer.willing_by.add(player)

    def decline(self, offer_id: str, player: Color) -> None:
        offer = self._active(offer_id)
        self._require_audience(offer, player)
        offer.willing_by.discard(player)
        offer.declined_by.add(player)

    def withdraw(self, offer_id: str, player: Color) -> None:
        offer = self._active(offer_id)
        if offer.offered_by != player:
            raise ValueError("Only an offer's creator may withdraw it")
        offer.status = TradeOfferStatus.WITHDRAWN

    def executable_candidates(self) -> tuple[TradeCandidate, ...]:
        candidates: list[TradeCandidate] = []
        for offer in self.active_offers:
            # Wildcards are proposals only; execution requires an exact counteroffer.
            if offer.give_any or offer.receive_any:
                continue
            offer_id = offer.id
            if offer.parent_offer_id is None and offer.offered_by == self.turn_player:
                assert offer_id is not None, "Candidate offers must have materialized IDs"
                candidates.extend(
                    TradeCandidate(offer_id, self.turn_player, player)
                    for player in offer.willing_by
                )
            elif (
                offer.parent_offer_id is not None
                and self.turn_player in offer.audience
                and self.turn_player not in offer.declined_by
            ):
                assert offer_id is not None, "Candidate offers must have materialized IDs"
                candidates.append(
                    TradeCandidate(
                        offer_id,
                        self.turn_player,
                        offer.offered_by,
                    )
                )
        player_order = {
            color: index for index, color in enumerate(self.participants)
        }
        offer_order = {
            offer_id: index
            for index, offer_id in enumerate(self.offers)
        }
        return tuple(
            sorted(
                candidates,
                key=lambda candidate: (
                    offer_order[candidate.offer_id],
                    player_order[candidate.counterparty],
                ),
            )
        )

    def select(
        self,
        player: Color,
        candidate: TradeCandidate,
    ) -> TradeCandidate:
        if player != self.turn_player or candidate.turn_player != self.turn_player:
            raise ValueError("Only the turn player may select a trade candidate")
        if candidate not in self.executable_candidates():
            raise ValueError("Selected trade candidate is not executable")
        self.selected_candidate = candidate
        return candidate

    def mark_executed(self) -> None:
        if self.selected_candidate is None:
            raise ValueError("No trade candidate has been selected")
        self.offers[self.selected_candidate.offer_id].status = (
            TradeOfferStatus.EXECUTED
        )
        self.close()

    def advance_round(self) -> None:
        self._require_open()
        self.round += 1

    def close(self) -> None:
        for offer in self.active_offers:
            offer.status = TradeOfferStatus.EXPIRED
        self.status = TradeWindowStatus.CLOSED

    def _active(self, offer_id: str) -> TradeOffer:
        offer = self.offers.get(offer_id)
        if offer is None or not offer.active:
            raise ValueError(f"Offer {offer_id!r} is not active")
        return offer

    def _require_open(self) -> None:
        if self.status != TradeWindowStatus.OPEN:
            raise ValueError("Trade window is closed")

    @staticmethod
    def _require_audience(offer: TradeOffer, player: Color) -> None:
        if player == offer.offered_by or player not in offer.audience:
            raise ValueError(f"Player {player} cannot respond to {offer.id}")

    @staticmethod
    def _deal_key(offer: TradeOffer) -> DealKey:
        return deal_key(offer)
