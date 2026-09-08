"""Bounded turn-scoped domestic trade offers."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import Enum

from cle.game_engine.models.player import Color

ResourceBundle = tuple[int, int, int, int, int]
RESOURCE_NAMES = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")


class _TradeCapError(ValueError):
    """A lifecycle limit failure, counted only when creation is attempted."""


class TradeOfferStatus(str, Enum):
    ACTIVE = "active"
    WITHDRAWN = "withdrawn"
    EXECUTED = "executed"
    EXPIRED = "expired"


class TradeWindowStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class TradeLimits:
    max_active_root_offers: int = 4
    max_active_counteroffers: int = 8
    max_offers_per_player: int = 3
    max_negotiation_rounds: int = 3
    max_operations_per_player_round: int = 2

    def __post_init__(self) -> None:
        for item in fields(self):
            if getattr(self, item.name) < 1:
                raise ValueError(f"{item.name} must be positive")


@dataclass(slots=True)
class TradeOffer:
    """One parameterized offer, including who, what, and lifecycle state."""

    offered_by: Color
    audience: frozenset[Color]
    give: ResourceBundle
    receive: ResourceBundle
    give_any: int = 0
    receive_any: int = 0
    parent_offer_id: str | None = None
    id: str | None = None
    created_round: int | None = None
    willing_by: set[Color] = field(default_factory=set)
    declined_by: set[Color] = field(default_factory=set)
    status: TradeOfferStatus = TradeOfferStatus.ACTIVE

    def __post_init__(self) -> None:
        values = (*self.give, *self.receive, self.give_any, self.receive_any)
        if not self.audience or self.offered_by in self.audience:
            raise ValueError("Trade audience must contain other participants")
        if len(self.give) != 5 or len(self.receive) != 5:
            raise ValueError("Trade bundles must have five resource counts")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in values
        ):
            raise ValueError("Trade resource counts must be integers")
        if any(value < 0 for value in values):
            raise ValueError("Trade resource counts cannot be negative")
        if sum(self.give) + self.give_any == 0:
            raise ValueError("Trade give side must be nonempty")
        if sum(self.receive) + self.receive_any == 0:
            raise ValueError("Trade receive side must be nonempty")
        if any(left and right for left, right in zip(self.give, self.receive)):
            raise ValueError("The same resource cannot be given and received")

    @property
    def active(self) -> bool:
        return self.status == TradeOfferStatus.ACTIVE

    @staticmethod
    def _named(bundle: ResourceBundle) -> dict[str, int]:
        return {
            resource: count
            for resource, count in zip(RESOURCE_NAMES, bundle)
            if count
        }

    def to_payload(self) -> dict[str, object]:
        """Return the semantic player/viewer representation."""
        return {
            "id": self.id,
            "offered_by": self.offered_by.value,
            "audience": sorted(color.value for color in self.audience),
            "give": self._named(self.give),
            "receive": self._named(self.receive),
            "give_any": self.give_any,
            "receive_any": self.receive_any,
            "parent_offer_id": self.parent_offer_id,
            "willing_by": sorted(color.value for color in self.willing_by),
            "declined_by": sorted(color.value for color in self.declined_by),
            "status": self.status.value,
        }


@dataclass(frozen=True, slots=True)
class TradeCandidate:
    offer_id: str
    turn_player: Color
    counterparty: Color

    def to_payload(self) -> dict[str, str]:
        return {
            "offer_id": self.offer_id,
            "turn_player": self.turn_player.value,
            "counterparty": self.counterparty.value,
        }


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
    _seen_deals: set[tuple] = field(default_factory=set)

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
        candidates = []
        for offer in self.active_offers:
            # Wildcards are proposals only; execution requires an exact counteroffer.
            if offer.give_any or offer.receive_any:
                continue
            if offer.parent_offer_id is None and offer.offered_by == self.turn_player:
                candidates.extend(
                    TradeCandidate(offer.id, self.turn_player, player)
                    for player in offer.willing_by
                )
            elif (
                offer.parent_offer_id is not None
                and self.turn_player in offer.audience
                and self.turn_player not in offer.declined_by
            ):
                candidates.append(
                    TradeCandidate(
                        offer.id,
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
    def _deal_key(offer: TradeOffer) -> tuple:
        if len(offer.audience) == 1:
            counterparty = next(iter(offer.audience))
            first, second = sorted(
                (offer.offered_by, counterparty),
                key=lambda color: color.value,
            )
            first_gives = offer.give if offer.offered_by == first else offer.receive
            second_gives = offer.receive if offer.offered_by == first else offer.give
            first_any = (
                offer.give_any
                if offer.offered_by == first
                else offer.receive_any
            )
            second_any = (
                offer.receive_any
                if offer.offered_by == first
                else offer.give_any
            )
            return (
                first.value,
                second.value,
                first_gives,
                first_any,
                second_gives,
                second_any,
            )
        return (
            offer.offered_by.value,
            tuple(sorted(color.value for color in offer.audience)),
            offer.give,
            offer.give_any,
            offer.receive,
            offer.receive_any,
        )
