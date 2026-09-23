"""Keep durable result evidence private without duplicating image/provider secrets."""

from __future__ import annotations

from dataclasses import replace

from cle.harness.board_surface import ImageBoardPresentation
from cle.harness.models import ModelRequest, ModelResponse
from cle.players.contracts import CommunicationChoice, PlayerAttempt
from cle.sandbox.communication import CommunicationAdmission
from cle.traces.journal import CommandOutcome

from .transport import durable_response


def clean_request(request: ModelRequest | None) -> ModelRequest | None:
    if request is not None and isinstance(request.board_presentation, ImageBoardPresentation):
        # Full provenance and image hashes live in call_started.request JSON.
        # Replay of an interrupted call regenerates and verifies that input.
        return replace(request, board_presentation=None)
    return request


def clean_response(
    response: ModelResponse | None, request: ModelRequest | None,
) -> ModelResponse | None:
    if response is not None and request is not None:
        return durable_response(response, request)
    return response


def clean_error(message: str | None, remote_errors: list[str]) -> str | None:
    if message is not None:
        for error in sorted(remote_errors, key=len, reverse=True):
            message = message.replace(error, "[provider failure details omitted]")
    return message


def clean_choice(choice: CommunicationChoice, remote_errors: list[str]) -> CommunicationChoice:
    return replace(
        choice, model_request=clean_request(choice.model_request),
        model_response=clean_response(choice.model_response, choice.model_request),
        validation_error=clean_error(choice.validation_error, remote_errors),
    )


def clean_attempt(attempt: PlayerAttempt, remote_errors: list[str]) -> PlayerAttempt:
    choice = attempt.choice
    if isinstance(choice, CommunicationChoice):
        choice = clean_choice(choice, remote_errors)
    return replace(
        attempt, choice=choice, model_request=clean_request(attempt.model_request),
        model_response=clean_response(attempt.model_response, attempt.model_request),
        validation_error=clean_error(attempt.validation_error, remote_errors),
    )


def clean_admission(
    admission: CommunicationAdmission, remote_errors: list[str],
) -> CommunicationAdmission:
    return replace(
        admission, choice=clean_choice(admission.choice, remote_errors),
        validation_error=clean_error(admission.validation_error, remote_errors),
    )


def clean_outcome(outcome: CommandOutcome, remote_errors: list[str]) -> CommandOutcome:
    result = outcome.result
    if result is not None:
        result = replace(result, attempts=tuple(
            clean_attempt(attempt, remote_errors) for attempt in result.attempts
        ))
    return replace(
        outcome, result=result,
        attempts=tuple(clean_attempt(attempt, remote_errors) for attempt in outcome.attempts),
        communications=tuple(clean_admission(item, remote_errors) for item in outcome.communications),
    )
