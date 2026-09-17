"""Durable deterministic continuations, independent of inference policy rebinding."""

from __future__ import annotations

from dataclasses import dataclass

from cle.game_engine.events import GameEngineSnapshot
from cle.game_engine.models.player import Color
from cle.players.action_batches import validate_batch_actions


@dataclass(frozen=True, slots=True)
class PendingActionBatch:
    actor: Color
    actions: tuple[dict, ...]
    next_index: int
    expected_revision: int
    turn_number: int
    phase: str
    origin_sequence: int
    origin_context_id: str
    provider_response_id: str | None = None
    provider_request_id: str | None = None

    def validate_snapshot(self, snapshot: GameEngineSnapshot) -> None:
        validate_batch_actions(self.actions, stored=True)
        if (
            not isinstance(self.actor, Color) or self.actor not in snapshot.state.colors
            or type(self.next_index) is not int or not 1 <= self.next_index < len(self.actions)
            or type(self.expected_revision) is not int
            or type(self.origin_sequence) is not int
            or not 0 <= self.origin_sequence < self.expected_revision <= len(snapshot.events)
            or type(self.turn_number) is not int or self.turn_number < 0
            or not isinstance(self.phase, str) or not self.phase
            or not isinstance(self.origin_context_id, str) or not self.origin_context_id
            or any(value is not None and not isinstance(value, str) for value in (
                self.provider_response_id, self.provider_request_id,
            ))
        ):
            raise ValueError("Invalid saved action batch identity")
        # Every consumed entry has exactly one canonical deterministic action event.
        events = snapshot.events[self.origin_sequence:self.expected_revision]
        names = {"upgrade_city": "BUILD_CITY"}
        if len(events) != self.next_index or any(
            event.actor != self.actor
            or event.event_type != names.get(call["tool"], call["tool"].upper())
            for event, call in zip(events, self.actions)
        ):
            raise ValueError("Saved action batch does not match its committed prefix")


@dataclass(frozen=True, slots=True)
class AutomaticBatchAction:
    """Causal reference only: no duplicated request, response, usage or notes."""

    batch: PendingActionBatch
    action_sequence: int

    def to_payload(self) -> dict:
        return {
            "kind": "deterministic_batch_continuation",
            "origin_context_id": self.batch.origin_context_id,
            "provider_response_id": self.batch.provider_response_id,
            "provider_request_id": self.batch.provider_request_id,
            "origin_sequence": self.batch.origin_sequence,
            "action_sequence": self.action_sequence,
            "action_number": self.batch.next_index + 1,
            "action_count": len(self.batch.actions),
        }
