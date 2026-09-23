"""Domestic trade values and stable deal identities."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import Enum
from typing import TypedDict

from cle.game_engine.models.player import Color

ResourceBundle = tuple[int, int, int, int, int]
RESOURCE_NAMES = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")
DealKey = tuple[str, str | tuple[str, ...], ResourceBundle, int, ResourceBundle, int]


class TradeOfferPayload(TypedDict):
    id: str | None
    offered_by: str
    audience: list[str]
    give: dict[str, int]
    receive: dict[str, int]
    give_any: int
    receive_any: int
    parent_offer_id: str | None
    willing_by: list[str]
    declined_by: list[str]
    status: str


class TradeCandidatePayload(TypedDict):
    offer_id: str
    turn_player: str
    counterparty: str


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

    def to_payload(self) -> TradeOfferPayload:
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

    def to_payload(self) -> TradeCandidatePayload:
        return {
            "offer_id": self.offer_id,
            "turn_player": self.turn_player.value,
            "counterparty": self.counterparty.value,
        }


def deal_key(offer: TradeOffer) -> DealKey:
    """Keep the historical tuple form, canonicalizing two-player orientation."""
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


# Historical snapshots and new pickles share the public trading-module identity.
TradeOfferStatus.__module__ = "cle.game_engine.trading"
TradeWindowStatus.__module__ = "cle.game_engine.trading"
TradeLimits.__module__ = "cle.game_engine.trading"
TradeOffer.__module__ = "cle.game_engine.trading"
TradeCandidate.__module__ = "cle.game_engine.trading"
