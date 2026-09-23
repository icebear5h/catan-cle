"""One model call's durable payload: request, response, and admitted choice."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from typing import NotRequired, TypedDict

from cle.harness.board_surface import (
    BoardPresentation,
    board_presentation_payload,
    sanitize_provider_payload,
)
from cle.harness.models import ModelRequest, ModelResponse
from cle.players.contracts import PlayerAttempt
from cle.sandbox.communication import CommunicationAdmission, CommunicationOpportunity

__all__: list[str] = []

# Redacted header names are compared with separators and case removed.
_SECRET_KEYS = frozenset({
    "authorization",
    "proxyauthorization",
    "apikey",
    "xapikey",
    "accesstoken",
    "refreshtoken",
    "clientsecret",
    "secret",
    "password",
    "cookie",
    "setcookie",
})


class ModelCallPayload(TypedDict):
    """One row of the model_calls table before serialization.

    `actor` and `opportunity` are present only for communication calls; decision
    calls recover their actor from the step's contexts.
    """

    call_kind: str
    context_id: str
    accepted: bool
    validation_error: str | None
    choice: object
    model_request: dict[str, object] | None
    model_response: dict[str, object] | None
    actor: NotRequired[str]
    opportunity: NotRequired[CommunicationOpportunity]


def _request_payload(request: ModelRequest | None) -> dict[str, object] | None:
    if request is None:
        return None
    return {
        "decision_id": request.decision_id,
        "session_id": request.session_id,
        "context_policy": getattr(request, "context_policy", None),
        "memory_revision": getattr(request, "memory_revision", None),
        "input_next_sequence": getattr(request, "input_next_sequence", None),
        "prompt_sources": [asdict(source) for source in getattr(request, "prompt_sources", ())],
        "channel": getattr(request, "channel", None),
        "trigger_reason": getattr(request, "trigger_reason", None),
        "messages": [
            {"role": message.role, "content": message.content}
            for message in request.messages
        ],
        "components": [
            {
                "id": component.id,
                "channel": component.channel,
                "template": component.template,
                "value": component.value,
                "rendered": component.rendered,
                "variables": dict(component.variables),
            }
            for component in getattr(request, "components", ())
        ],
        "board_presentation": board_presentation_payload(
            getattr(request, "board_presentation", None),
            include_text_content=True,
        ),
    }


def _provider_payload(
    value: object, board_presentation: BoardPresentation | None = None
) -> object:
    if isinstance(value, Mapping):
        return {
            key: _provider_payload(item, board_presentation)
            for key, item in value.items()
            if str(key).lower().replace("-", "").replace("_", "") not in _SECRET_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_provider_payload(item, board_presentation) for item in value]
    return sanitize_provider_payload(value, board_presentation)


def _response_payload(
    response: ModelResponse | None,
    board_presentation: BoardPresentation | None = None,
) -> dict[str, object] | None:
    if response is None:
        return None
    return {
        "content": response.content,
        "model": response.model,
        "usage": dict(response.usage),
        "latency_ms": response.latency_ms,
        "finish_reason": response.finish_reason,
        "native_reasoning": response.native_reasoning,
        "native_reasoning_details": response.native_reasoning_details,
        "reasoning_request": dict(response.reasoning_request),
        "provider_response_id": response.provider_response_id,
        "provider_request_id": response.provider_request_id,
        "provider_native_finish_reason": response.provider_native_finish_reason,
        "provider_request_payload": _provider_payload(
            response.provider_request_payload,
            board_presentation,
        ),
        "provider_response_payload": _provider_payload(
            response.provider_response_payload,
            board_presentation,
        ),
    }


def _attempt_payload(
    attempt: PlayerAttempt,
    *,
    accepted: bool,
) -> ModelCallPayload:
    return {
        "call_kind": "decision",
        "context_id": attempt.context_id,
        "accepted": accepted,
        "validation_error": attempt.validation_error,
        "choice": attempt.choice,
        "model_request": _request_payload(attempt.model_request),
        "model_response": _response_payload(
            attempt.model_response,
            getattr(attempt.model_request, "board_presentation", None),
        ),
    }


def _communication_payload(record: CommunicationAdmission) -> ModelCallPayload:
    opportunity, choice = record.opportunity, record.choice
    request = choice.model_request
    return {
        "call_kind": "communication",
        "context_id": (
            request.decision_id
            if request is not None
            else f"communication:{opportunity.player.value}:"
            f"{opportunity.visible_through_sequence}:{opportunity.round}"
        ),
        "actor": opportunity.player.value,
        "accepted": record.accepted,
        "validation_error": record.validation_error,
        "choice": {
            "trigger_reason": opportunity.reason.value,
            "mode": choice.mode,
            "text": choice.text,
            "audience": choice.audience,
            "respondents": choice.respondents,
            "commitment": choice.commitment,
            "notes_update": getattr(choice, "notes_update", None),
            "validation_error": getattr(choice, "validation_error", None),
        },
        "model_request": _request_payload(request),
        "model_response": _response_payload(
            choice.model_response,
            getattr(request, "board_presentation", None),
        ),
        "opportunity": opportunity,
    }
