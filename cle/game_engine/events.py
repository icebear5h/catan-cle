"""Canonical engine events and perspective projections."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from cle.game_engine.models.enums import Action, ActionType
from cle.game_engine.models.player import Color
from cle.game_engine.trading import TradeCandidate, TradeOffer


@dataclass(frozen=True, slots=True)
class GameEvent:
    """One canonical resolved game event stored by the engine."""

    sequence: int
    causation_id: str
    actor: Color
    event_type: str
    public_payload: Any = None
    private_overlays: tuple[tuple[Color, Any], ...] = ()
    visible_to: tuple[Color, ...] | None = None


@dataclass(frozen=True, slots=True)
class PlayerEvent:
    """One event after applying visibility rules for a participant."""

    sequence: int
    causation_id: str
    actor: Color
    event_type: str
    payload: Any = None


@dataclass(frozen=True, slots=True)
class EngineTransition:
    """Result of one strict engine action."""

    before_revision: int
    after_revision: int
    requested_action: Action
    resolved_action: Action
    events: tuple[GameEvent, ...]
    winner: Color | None = None


@dataclass(frozen=True, slots=True)
class GameEngineSnapshot:
    """Explicit restorable engine state and event history."""

    engine_id: str
    seed: int
    vps_to_win: int
    state: Any
    events: tuple[GameEvent, ...]
    capture_history: bool
    communication_limits: Any = None
    commitments: tuple[Any, ...] = field(default_factory=tuple)
    history: tuple[tuple[Any, Action, int, tuple[Any, ...]], ...] = field(
        default_factory=tuple
    )


def event_from_action(action: Action, sequence: int, *, trade_offers: dict[str, TradeOffer] | None = None) -> GameEvent:
    """Build the canonical public payload plus private participant overlays."""
    public_payload = action.value
    private_overlays: tuple[tuple[Color, Any], ...] = ()

    if action.action_type == ActionType.BUY_DEVELOPMENT_CARD:
        public_payload = None
        private_overlays = ((action.color, action.value),)
    elif action.action_type == ActionType.STEAL:
        victim, resource = action.value
        public_payload = (victim, None)
        private_overlays = (
            (action.color, (victim, resource)),
            (victim, (victim, resource)),
        )
    elif action.action_type == ActionType.DISCARD:
        cards = tuple(action.value) if isinstance(action.value, (list, tuple)) else ()
        public_payload = len(cards)
        private_overlays = ((action.color, cards),)
    elif (
        action.action_type
        in {ActionType.OFFER_TRADE, ActionType.COUNTER_OFFER}
        and isinstance(action.value, TradeOffer)
    ):
        public_payload = action.value.to_payload()

    if action.action_type == ActionType.CONFIRM_TRADE and isinstance(action.value, TradeCandidate):
        public_payload = action.value.to_payload()

    # Lifecycle events must remain understandable after the offer window closes
    # and after a recipient has acknowledged the original proposal.
    offers = trade_offers or {}
    if action.action_type in {ActionType.OFFER_TRADE, ActionType.COUNTER_OFFER} and isinstance(action.value, TradeOffer):
        parent = offers.get(action.value.parent_offer_id)
        if parent is not None:
            public_payload["original"] = parent.to_payload()
    elif action.action_type in {ActionType.ACCEPT_TRADE, ActionType.REJECT_TRADE, ActionType.CANCEL_TRADE, ActionType.CONFIRM_TRADE}:
        offer_id = action.value.offer_id if isinstance(action.value, TradeCandidate) else action.value
        offer = offers.get(offer_id) if isinstance(offer_id, str) else None
        if offer is not None:
            public_payload = {"offer": offer.to_payload()}
            if isinstance(action.value, TradeCandidate):
                public_payload.update(action.value.to_payload())

    return GameEvent(
        sequence=sequence,
        causation_id=f"action:{sequence}",
        actor=action.color,
        event_type=action.action_type.value,
        public_payload=deepcopy(public_payload),
        private_overlays=deepcopy(private_overlays),
    )


def project_event(event: GameEvent, color: Color) -> PlayerEvent | None:
    """Project a canonical event for one participant without mutating state."""
    if event.visible_to is not None and color not in event.visible_to:
        return None
    payload = dict(event.private_overlays).get(color, event.public_payload)
    return PlayerEvent(
        sequence=event.sequence,
        causation_id=event.causation_id,
        actor=event.actor,
        event_type=event.event_type,
        payload=deepcopy(payload),
    )
