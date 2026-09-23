"""Canonical engine events and perspective projections."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from cle.game_engine.communication import CommunicationLimits, SocialCommitment
from cle.game_engine.models.enums import Action, ActionType
from cle.game_engine.models.player import Color
from cle.game_engine.trading import TradeCandidate, TradeOffer

if TYPE_CHECKING:
    from cle.game_engine.state import GameState

# Payloads are polymorphic per event type (dice tuples, node ids, trade payload
# dicts, message dicts, ...); consumers narrow them with isinstance.
PrivateOverlays = tuple[tuple[Color, object], ...]
HistoryEntry = tuple["GameState", Action, int, tuple[SocialCommitment, ...]]


@dataclass(frozen=True, slots=True)
class GameEvent:
    """One canonical resolved game event stored by the engine."""

    sequence: int
    causation_id: str
    actor: Color
    event_type: str
    public_payload: object = None
    private_overlays: PrivateOverlays = ()
    visible_to: tuple[Color, ...] | None = None


@dataclass(frozen=True, slots=True)
class PlayerEvent:
    """One event after applying visibility rules for a participant."""

    sequence: int
    causation_id: str
    actor: Color
    event_type: str
    payload: object = None


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
    state: GameState
    events: tuple[GameEvent, ...]
    capture_history: bool
    communication_limits: CommunicationLimits = field(default_factory=CommunicationLimits)
    commitments: tuple[SocialCommitment, ...] = field(default_factory=tuple)
    history: tuple[HistoryEntry, ...] = field(default_factory=tuple)


def event_from_action(
    action: Action, sequence: int, *, trade_offers: dict[str, TradeOffer] | None = None,
) -> GameEvent:
    """Build the canonical public payload plus private participant overlays."""
    value: object = action.value
    public_payload: object = value
    private_overlays: PrivateOverlays = ()

    if action.action_type == ActionType.BUY_DEVELOPMENT_CARD:
        public_payload = None
        private_overlays = ((action.color, value),)
    elif action.action_type == ActionType.STEAL and isinstance(value, tuple):
        victim, resource = value
        public_payload = (victim, None)
        private_overlays = (
            (action.color, (victim, resource)),
            (victim, (victim, resource)),
        )
    elif action.action_type == ActionType.DISCARD:
        cards = tuple(value) if isinstance(value, (list, tuple)) else ()
        public_payload = len(cards)
        private_overlays = ((action.color, cards),)
    elif (
        action.action_type
        in {ActionType.OFFER_TRADE, ActionType.COUNTER_OFFER}
        and isinstance(value, TradeOffer)
    ):
        public_payload = value.to_payload()

    if action.action_type == ActionType.CONFIRM_TRADE and isinstance(value, TradeCandidate):
        public_payload = value.to_payload()

    # Lifecycle events must remain understandable after the offer window closes
    # and after a recipient has acknowledged the original proposal.
    offers = trade_offers or {}
    if action.action_type in {ActionType.OFFER_TRADE, ActionType.COUNTER_OFFER} and isinstance(value, TradeOffer):
        parent_id = value.parent_offer_id
        parent = offers.get(parent_id) if parent_id is not None else None
        if parent is not None:
            offer_payload: dict[str, object] = dict(value.to_payload())
            offer_payload["original"] = parent.to_payload()
            public_payload = offer_payload
    elif action.action_type in {ActionType.ACCEPT_TRADE, ActionType.REJECT_TRADE, ActionType.CANCEL_TRADE, ActionType.CONFIRM_TRADE}:
        offer_id = value.offer_id if isinstance(value, TradeCandidate) else value
        offer = offers.get(offer_id) if isinstance(offer_id, str) else None
        if offer is not None:
            lifecycle_payload: dict[str, object] = {"offer": offer.to_payload()}
            if isinstance(value, TradeCandidate):
                lifecycle_payload.update(value.to_payload())
            public_payload = lifecycle_payload

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
