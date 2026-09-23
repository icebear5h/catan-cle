"""Turning one failed live step into the diagnostics the viewer shows."""

import json
from collections.abc import Sequence
from dataclasses import asdict, replace

from flask import current_app

from cle.game_engine.json import GameEncoder
from cle.game_engine.models.player import Color
from cle.game_engine.public_board import JsonValue
from cle.harness.board_surface import board_presentation_payload
from cle.harness.models import ModelRequest
from cle.harness.reasoning import (
    reasoning_token_count,
)
from cle.players.contracts import PlayerAttempt
from cle.sandbox.catan import CatanSandbox
from cle.sandbox.communication import CommunicationAdmission

from ...state import ServerState
from ..websocket import build_game_state_snapshot
from .blueprint import (
    _sync_live_inference,
)

_MODEL_OUTPUT_EXCERPT_LIMIT = 1000


def _last_model_output_excerpt(attempts: Sequence[PlayerAttempt]) -> str | None:
    """Return the most recent non-empty model output, truncated for banner display."""
    for attempt in reversed(tuple(attempts or ())):
        response = getattr(attempt, "model_response", None)
        content = getattr(response, "content", "")
        if isinstance(content, str) and content.strip():
            text = content.strip()
            if len(text) > _MODEL_OUTPUT_EXCERPT_LIMIT:
                return text[:_MODEL_OUTPUT_EXCERPT_LIMIT] + "... [truncated]"
            return text
    return None


def _failed_attempt_payload(attempt: PlayerAttempt) -> dict[str, JsonValue]:
    response = attempt.model_response
    choice = attempt.choice
    model_request = attempt.model_request
    if not isinstance(model_request, ModelRequest):
        model_request = None
    request_payload = None
    if model_request is not None:
        request_payload = asdict(replace(model_request, board_presentation=None))
        request_payload["board_presentation"] = board_presentation_payload(
            model_request.board_presentation, include_text_content=True,
        )
    final_response = getattr(response, "content", "") if response is not None else ""
    usage = dict(getattr(response, "usage", ())) if response is not None else {}
    return {
        "context_id": attempt.context_id,
        "validation_error": attempt.validation_error,
        "action_index": getattr(choice, "action_index", None),
        "accepted": False,
        "notes_update": choice.notes_update if choice is not None else None,
        "request": (
            json.loads(json.dumps(request_payload, cls=GameEncoder))
            if model_request is not None else None
        ),
        "context_policy": model_request.context_policy if model_request is not None else None,
        "memory_revision": model_request.memory_revision if model_request is not None else None,
        "input_next_sequence": model_request.input_next_sequence if model_request is not None else None,
        "channel": model_request.channel if model_request is not None else None,
        "final_response": final_response,
        "model": getattr(response, "model", None),
        "latency_ms": getattr(response, "latency_ms", None),
        "finish_reason": getattr(response, "finish_reason", None),
        "provider_native_finish_reason": getattr(
            response,
            "provider_native_finish_reason",
            None,
        ),
        "usage": usage,
        "reasoning_tokens": reasoning_token_count(usage),
        "native_reasoning": getattr(response, "native_reasoning", ""),
        "native_reasoning_details": list(
            getattr(response, "native_reasoning_details", ())
        ),
        "reasoning_request": dict(getattr(response, "reasoning_request", ())),
        "native_reasoning_chars": len(
            getattr(response, "native_reasoning", "")
        ) if response is not None else 0,
        "provider_response_id": getattr(
            response,
            "provider_response_id",
            None,
        ),
        "provider_request_id": getattr(
            response,
            "provider_request_id",
            None,
        ),
    }


def _safe_failure_traces(
    attempts: Sequence[PlayerAttempt],
    communications: Sequence[CommunicationAdmission],
    error_type: str,
) -> tuple[list[PlayerAttempt], list[CommunicationAdmission]]:
    """Do not publish arbitrary exception bodies embedded by barrier withholding."""
    return (
        [replace(attempt, validation_error=f"Decision withheld: {error_type}") for attempt in attempts],
        [
            record if record.accepted else replace(
                record, validation_error=f"Communication withheld: {error_type}",
            )
            for record in communications
        ],
    )


def _record_live_failure(
    state: ServerState,
    sandbox: CatanSandbox,
    player: Color,
    error_payload: dict[str, JsonValue],
    attempts: Sequence[PlayerAttempt],
    communication_attempts: Sequence[CommunicationAdmission],
    *,
    validation_error: str | None = None,
    persistence_failed: bool = False,
) -> None:
    """Checkpoint the exact paused boundary, including admitted silence and notes."""
    _sync_live_inference(state, sandbox)
    state.step_processing = False
    state.last_live_step_error = error_payload
    error_payload["trace_failure_id"] = None
    error_payload["checkpoint_saved"] = False
    trace_store = getattr(state, "live_trace_store", None)
    if not persistence_failed and trace_store is not None and state.live_trace_game_id is not None:
        try:
            error_payload["checkpoint_saved"] = True
            error_payload["trace_failure_id"] = trace_store.record_failure(
                state.live_trace_game_id,
                revision=sandbox.revision,
                player=player,
                validation_error=(
                    validation_error if validation_error is not None else error_payload["details"]
                ),
                attempts=attempts,
                communication_attempts=communication_attempts,
                snapshot=sandbox.snapshot(),
                public_state=build_game_state_snapshot(state),
            )
        except Exception as exc:
            current_app.logger.error("Could not persist live failure (%s)", type(exc).__name__)
            persistence_failed = True
    if persistence_failed:
        error_payload["checkpoint_saved"] = False
        error_payload["retryable"] = False
        error_payload["details"] = str(error_payload["details"]).replace(
            " Press Step to ask the model again.", "",
        ).replace(" Press Step to retry.", "") + (
            " Current state and failure diagnostics could not be saved. "
            "Do not retry until storage is repaired; loading a saved game may lose admitted changes."
        )
