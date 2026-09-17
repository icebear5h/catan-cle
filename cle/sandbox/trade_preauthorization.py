"""One admitted proposer instruction, bounded to its first response barrier."""

from __future__ import annotations

from dataclasses import dataclass

from cle.game_engine.events import GameEngineSnapshot
from cle.game_engine.game import GameEngine
from cle.game_engine.models.actions import generate_playable_actions
from cle.game_engine.models.enums import Action, ActionType
from cle.game_engine.models.player import Color
from cle.game_engine.trading import ResourceBundle, TradeCandidate, TradeOffer


@dataclass(frozen=True, slots=True)
class TradePreauthorization:
    actor: Color
    window_id: str
    offer_id: str
    turn_number: int
    round: int
    give: ResourceBundle
    receive: ResourceBundle
    audience: frozenset[Color]
    priority: tuple[Color, ...]
    offer_sequence: int
    origin_context_id: str
    provider_response_id: str | None = None
    provider_request_id: str | None = None
    response_revision: int | None = None

    def validate_snapshot(self, snapshot: GameEngineSnapshot) -> None:
        """Validate saved instruction identity, without reinterpreting its policy.

        Material state may be stale: that is resolved as a private pause on step.
        The original immutable event must still attest the authorized terms.
        """
        if (
            not isinstance(self.actor, Color) or self.actor not in snapshot.state.colors
            or not isinstance(self.audience, frozenset)
            or any(not isinstance(c, Color) for c in self.audience)
            or not self.audience.issubset(snapshot.state.colors)
            or not isinstance(self.priority, tuple) or not self.priority
            or any(not isinstance(c, Color) or c not in self.audience for c in self.priority)
            or len(set(self.priority)) != len(self.priority)
            or any(type(n) is not int or n < 0 for n in (self.turn_number, self.round, self.offer_sequence))
            or self.offer_sequence >= len(snapshot.events)
            or (self.response_revision is not None and (
                type(self.response_revision) is not int
                or not self.offer_sequence < self.response_revision <= len(snapshot.events)
            ))
            or any(not isinstance(s, str) or not s for s in (self.window_id, self.offer_id, self.origin_context_id))
        ):
            raise ValueError("Invalid saved trade preauthorization identity")
        original = snapshot.events[self.offer_sequence]
        expected = TradeOffer(self.actor, self.audience, self.give, self.receive, id=self.offer_id).to_payload()
        if (
            original.event_type != "OFFER_TRADE" or original.actor != self.actor
            or not isinstance(original.public_payload, dict)
            or any(original.public_payload.get(key) != expected[key] for key in (
                "id", "offered_by", "audience", "give", "receive", "give_any", "receive_any", "parent_offer_id",
            ))
        ):
            raise ValueError("Saved trade preauthorization does not match its original offer event")

    def invalid_reason(self, engine: GameEngine) -> str | None:
        state = engine.state
        window = state.trade_window
        if (
            state.num_turns != self.turn_number or state.current_color() != self.actor
            or window is None or window.id != self.window_id
            or window.turn_player != self.actor or window.status.value != "open"
        ):
            return "Trade window is no longer current/open"
        offer = window.offers.get(self.offer_id)
        if offer is None or not offer.active:
            return "Original offer was withdrawn, expired, or is no longer active"
        if not self.matches(offer):
            return "Original offer terms or audience changed"
        expected_round = self.round + (self.response_revision is not None)
        if window.round != expected_round:
            return "Authorized response round expired"
        events = engine.events[self.offer_sequence + 1:]
        if any(event.event_type == "COUNTER_OFFER" for event in events):
            return "Counteroffer arrived; proposer must decide again"
        if any(event.event_type not in {"ACCEPT_TRADE", "REJECT_TRADE", "MESSAGE_SENT"} for event in events):
            return "Game changed outside the authorized response window"
        if self.response_revision is not None and any(
            event.event_type != "MESSAGE_SENT"
            for event in engine.events[self.response_revision:]
        ):
            return "Trade responses changed after the admitted response barrier"
        return None

    def matches(self, offer: TradeOffer) -> bool:
        return (
            offer.id == self.offer_id and offer.offered_by == self.actor
            and offer.parent_offer_id is None and offer.created_round == self.round
            and offer.give == self.give and offer.receive == self.receive
            and not offer.give_any and not offer.receive_any
            and offer.audience == self.audience
        )

    def resolve(self, engine: GameEngine) -> tuple[Action | None, str | None]:
        reason = self.invalid_reason(engine)
        if reason:
            return None, reason
        if self.response_revision is None:
            return None, "Authorized response barrier was not completed"
        offer = engine.state.trade_window.offers[self.offer_id]
        responses = {}
        for event in engine.events[self.offer_sequence + 1:self.response_revision]:
            if event.event_type not in {"ACCEPT_TRADE", "REJECT_TRADE"}:
                continue
            payload = event.public_payload
            if not isinstance(payload, dict) or payload.get("offer", {}).get("id") != self.offer_id:
                continue
            if event.actor in responses:
                return None, "Original offer received multiple responses from the same player"
            responses[event.actor] = event.event_type
        if set(responses) != self.audience:
            return None, "Not all original audience members responded in the first batch"
        willing = {color for color, response in responses.items() if response == "ACCEPT_TRADE"}
        if offer.willing_by != willing or offer.declined_by != self.audience - willing:
            return None, "Current willingness no longer matches the admitted response batch"
        partner = next((color for color in self.priority if color in offer.willing_by), None)
        if partner is None:
            return None, "Nobody permitted accepted the exact original offer"
        action = Action(self.actor, ActionType.CONFIRM_TRADE, TradeCandidate(self.offer_id, self.actor, partner))
        # Regenerate from current material state, not a saved/stale legal menu.
        # Do not skip an unfunded preferred acceptor in favor of a lower priority.
        if action not in generate_playable_actions(engine.state) or not engine.is_action_valid(action):
            return None, "Selected exact trade is no longer legal or both hands cannot fund it"
        return action, None


@dataclass(frozen=True, slots=True)
class AutomaticTradeAction:
    """Trace-only causal reference; contains no synthetic response or usage."""

    authorization: TradePreauthorization
    action_sequence: int

    def to_payload(self) -> dict:
        return {
            "kind": "preauthorized_trade_confirmation",
            "origin_context_id": self.authorization.origin_context_id,
            "provider_response_id": self.authorization.provider_response_id,
            "provider_request_id": self.authorization.provider_request_id,
            "offer_id": self.authorization.offer_id,
            "action_sequence": self.action_sequence,
        }
