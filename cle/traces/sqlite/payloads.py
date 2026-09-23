"""Event, observation, context, and whole-step payloads written to a trace."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import fields

from cle.game_engine.events import GameEvent, PlayerEvent
from cle.players.contracts import PlayerAttempt, PlayerContext
from cle.sandbox.communication import CommunicationAdmission
from cle.sandbox.contracts import SandboxStepResult
from cle.traces.sqlite.calls import (
    _attempt_payload,
    _communication_payload,
)

__all__: list[str] = []


def _event_payload(event: GameEvent | PlayerEvent) -> dict[str, object]:
    return {
        "sequence": event.sequence,
        "causation_id": event.causation_id,
        "actor": event.actor,
        "event_type": event.event_type,
        "payload": getattr(event, "public_payload", getattr(event, "payload", None)),
        "private_overlays": getattr(event, "private_overlays", ()),
        "visible_to": getattr(event, "visible_to", None),
    }


def _observation_payload(context: PlayerContext) -> dict[str, object]:
    observation = context.observation
    payload: dict[str, object] = {
        field.name: getattr(observation, field.name)
        for field in fields(observation)
        if field.name not in {"board_map", "valid_actions"}
    }
    payload["board_map"] = "stored in step public_state.game"
    return payload


def _context_payload(context: PlayerContext) -> dict[str, object]:
    return {
        "context_id": context.context_id,
        "actor": context.actor,
        "turn_number": context.turn_number,
        "phase": context.phase,
        "prompt_key": context.prompt_key,
        "observation": _observation_payload(context),
        "events": [_event_payload(event) for event in context.events],
        "recent_messages": [
            _event_payload(event) for event in context.recent_messages
        ],
        "active_commitments": context.active_commitments,
        "discard_count": context.discard_count,
        "legal_actions": list(context.legal_actions),
        "visible_through_sequence": getattr(context, "visible_through_sequence", None),
        "visible_messages": [
            _event_payload(event) for event in getattr(context, "visible_messages", ())
        ],
    }


def _result_payload(
    result: SandboxStepResult,
    rejected_attempts: Iterable[PlayerAttempt],
    communication_attempts: Iterable[CommunicationAdmission],
) -> dict[str, object]:
    return {
        "before_revision": result.before_revision,
        "after_revision": result.after_revision,
        "winner": result.winner,
        "automatic_action": result.automatic_action.to_payload() if result.automatic_action else None,
        "contexts": [_context_payload(context) for context in result.contexts],
        "rejected_attempts": [
            _attempt_payload(attempt, accepted=False)
            for attempt in rejected_attempts
        ],
        "accepted_attempts": [
            _attempt_payload(attempt, accepted=True)
            for attempt in result.attempts
        ],
        "communication_attempts": [
            _communication_payload(record)
            for record in communication_attempts
        ],
        "transitions": [
            {
                "before_revision": transition.before_revision,
                "after_revision": transition.after_revision,
                "requested_action": transition.requested_action,
                "resolved_action": transition.resolved_action,
                "events": [
                    _event_payload(event)
                    for event in transition.events
                ],
                "winner": transition.winner,
            }
            for transition in result.transitions
        ],
        "message_events": [
            _event_payload(event)
            for event in result.messages
        ],
    }
