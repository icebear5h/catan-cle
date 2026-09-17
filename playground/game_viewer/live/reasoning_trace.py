"""Read-only reasoning diagnostics for accepted live sandbox model calls."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, replace
from typing import Any

from cle.harness.board_surface import board_presentation_payload
from cle.harness.models import ModelRequest, ModelResponse
from cle.harness.reasoning import (
    native_reasoning_enabled,
    native_reasoning_returned,
    reasoning_token_count,
)
from cle.players.contracts import CommunicationChoice, PlayerChoice
from cle.players.validation import choice_followup_action
from cle.sandbox.communication import CommunicationAdmission
from cle.sandbox.contracts import SandboxStepResult
from cle.game_engine.json import GameEncoder


def build_live_reasoning_traces(
    sandbox,
    result: SandboxStepResult,
    *,
    communication_attempts: Sequence[CommunicationAdmission] = (),
) -> list[dict[str, Any]]:
    """Project accepted calls; the caller supplies only this step's admissions."""
    traces = []
    transitions = iter(result.transitions)
    for context, attempt in zip(result.contexts, result.attempts, strict=True):
        transition = next(transitions)
        sequence = [transition]
        choice = attempt.choice
        if choice is not None:
            followup = choice_followup_action(context, choice)
            if followup is not None and transition.winner is None:
                movement = next(transitions)
                if movement.requested_action != followup:
                    raise ValueError("Knight trace does not match the committed robber move")
                sequence.append(movement)
        player = sandbox.players[context.actor]
        if player.status().get("kind") != "agent":
            continue
        if choice is None or attempt.validation_error is not None:
            continue

        traces.append({
            **_reasoning_artifacts(choice, attempt.model_request, attempt.model_response),
            "call_kind": "decision",
            "context_id": context.context_id,
            "player_color": context.actor.value,
            "turn_number": context.turn_number,
            "phase": context.phase,
            "prompt_key": context.prompt_key,
            "action_index": choice.action_index,
            "action_type": transition.requested_action.action_type.value,
            "action_sequence": [str(item.resolved_action) for item in sequence],
            "batch_actions": choice.batch_actions,
            "knight_destination": choice.knight_destination,
            "game_plan": choice.game_plan,
        })

    for admission in communication_attempts:
        choice = admission.choice
        if not admission.accepted or admission.validation_error or choice.validation_error:
            continue
        request = choice.model_request
        response = choice.model_response
        if request is None and response is None:
            continue
        opportunity = admission.opportunity
        traces.append({
            **_reasoning_artifacts(choice, request, response),
            "call_kind": "communication",
            "context_id": (
                request.decision_id if request is not None else
                f"{sandbox.game_engine.id}:talk:{opportunity.cause.causation_id}:"
                f"{opportunity.round}:{opportunity.player.value}"
            ),
            "player_color": opportunity.player.value,
            "turn_number": None,
            "phase": None,
            "prompt_key": None,
            "action_index": None,
            "action_type": None,
            "action_sequence": [],
            "game_plan": "",
            "communication_mode": choice.mode.value,
            "trigger_reason": opportunity.reason.value,
            "respondents": choice.respondents,
            "text": choice.text,
        })

    return json.loads(json.dumps(traces, cls=GameEncoder))


def _reasoning_artifacts(
    choice: PlayerChoice | CommunicationChoice,
    request: ModelRequest | None,
    response: ModelResponse | None,
) -> dict[str, Any]:
    # Historical decision receipts may have provider diagnostics only on the choice.
    source = response if response is not None else (
        choice if isinstance(choice, PlayerChoice) else None
    )
    usage = dict(source.usage) if source is not None else {}
    native_reasoning = source.native_reasoning if source is not None else ""
    native_details = source.native_reasoning_details if source is not None else ()
    reasoning_request = dict(source.reasoning_request) if source is not None else {}
    returned = native_reasoning_returned(native_reasoning, native_details, usage)
    requested = bool(reasoning_request) and native_reasoning_enabled(reasoning_request)
    request_payload = None
    if request is not None:
        # Exclude image bytes before recursively projecting the request.
        request_payload = asdict(replace(request, board_presentation=None))
        request_payload["board_presentation"] = board_presentation_payload(
            request.board_presentation, include_text_content=True,
        )
    return {
        "schema": "live-reasoning-trace-v2",
        "accepted": True,
        "notes_update": choice.notes_update,
        "request": request_payload,
        "context_policy": request.context_policy if request is not None else None,
        "memory_revision": request.memory_revision if request is not None else None,
        "input_next_sequence": request.input_next_sequence if request is not None else None,
        "channel": request.channel if request is not None else None,
        "native_reasoning": native_reasoning,
        "native_reasoning_details": list(native_details),
        "native_reasoning_source": "provider_response" if returned else None,
        "native_reasoning_requested": requested,
        "native_reasoning_returned": returned,
        "native_reasoning_missing": requested and not returned,
        "reasoning_request": reasoning_request,
        "reasoning_tokens": reasoning_token_count(usage),
        "finish_reason": response.finish_reason if response is not None else None,
        "provider_native_finish_reason": (
            source.provider_native_finish_reason if source is not None else None
        ),
        "provider_response_id": source.provider_response_id if source is not None else None,
        "provider_request_id": source.provider_request_id if source is not None else None,
        "model": source.model if source is not None else None,
        "latency_ms": source.latency_ms if source is not None else None,
        "usage": usage,
    }
