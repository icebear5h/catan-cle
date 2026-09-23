"""Pure admission of player-authored speech, silence, and commitments."""

from __future__ import annotations

from cle.game_engine.models.player import Color
from cle.players.contracts import CommitmentProposal, CommunicationChoice, CommunicationMode
from cle.players.notes import MAX_NOTES_CHARS, validate_notes


def validate_communication_choice(
    choice: CommunicationChoice,
    *,
    speaker: Color,
    participants: tuple[Color, ...],
    max_notes_chars: int = MAX_NOTES_CHARS,
) -> None:
    """Reject malformed typed speech before any message or commitment is appended."""
    if not isinstance(choice, CommunicationChoice):
        raise ValueError("Communication choice must be a CommunicationChoice")
    if choice.validation_error is not None:
        if not isinstance(choice.validation_error, str):
            raise ValueError("CommunicationChoice.validation_error must be a string or None")
        raise ValueError(choice.validation_error or "Invalid communication response")
    if choice.notes_update is not None:
        try:
            validate_notes(choice.notes_update, max_notes_chars)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"CommunicationChoice.notes_update: {exc}") from exc
    if not isinstance(choice.mode, CommunicationMode):
        raise ValueError("CommunicationChoice.mode must be a CommunicationMode")
    if not isinstance(choice.text, str):
        raise ValueError("CommunicationChoice.text must be a string")
    if not isinstance(choice.audience, tuple) or any(
        not isinstance(color, Color) for color in choice.audience
    ):
        raise ValueError("CommunicationChoice.audience must be a tuple of player colors")
    if any(color not in participants for color in choice.audience):
        raise ValueError("Message audience contains a non-participant")
    if speaker in choice.audience:
        raise ValueError("Message audience must contain other participants, not the speaker")
    if choice.respondents is not None:
        if not isinstance(choice.respondents, tuple) or len(set(choice.respondents)) != len(choice.respondents) or any(
            not isinstance(color, Color) or color == speaker or color not in participants
            for color in choice.respondents
        ):
            raise ValueError("Respondents must be distinct eligible other participants")
        if choice.mode == CommunicationMode.SAY and set(choice.audience) != set(participants) - {speaker}:
            raise ValueError("Reactive table talk must be public")
        if choice.mode == CommunicationMode.SILENCE and choice.respondents:
            raise ValueError("Pass cannot request respondents")
    if choice.mode == CommunicationMode.SAY and (not choice.text.strip() or not choice.audience):
        raise ValueError("Spoken messages require nonempty text and an audience")
    proposal = choice.commitment
    if proposal is not None:
        if not isinstance(proposal, CommitmentProposal):
            raise ValueError("Communication commitment must be a CommitmentProposal")
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (proposal.condition, proposal.promise)
        ):
            raise ValueError("Commitment condition and promise must be nonempty strings")
        if (
            isinstance(proposal.expires_turn, bool)
            or not isinstance(proposal.expires_turn, int)
            or proposal.expires_turn < 0
        ):
            raise ValueError("Commitment expires_turn must be a non-negative integer")
    if choice.mode == CommunicationMode.SILENCE and (
        choice.text or choice.audience or proposal is not None
    ):
        raise ValueError("Silence cannot carry a message, audience, or commitment")
