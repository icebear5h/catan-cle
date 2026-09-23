"""Validation, commitment construction, and publication of table-talk messages."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from cle.game_engine.communication import SocialCommitment
from cle.game_engine.events import GameEvent
from cle.game_engine.models.player import Color

if TYPE_CHECKING:
    from cle.game_engine.game import GameEngine


def validate_message(
    participants: Sequence[Color],
    *,
    speaker: Color,
    text: str,
    audience: tuple[Color, ...],
    causation_id: str,
    respondents: tuple[Color, ...] | None,
) -> tuple[Color, ...]:
    """Check one message request and return its de-duplicated recipients."""
    if speaker not in participants:
        raise ValueError(f"Speaker {speaker} is not a participant")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Message text must be a nonempty string")
    if not isinstance(audience, (tuple, list)):
        raise ValueError("Message audience must be a sequence of participants")
    if not isinstance(causation_id, str) or not causation_id:
        raise ValueError("Message causation ID must be a nonempty string")
    audience = tuple(audience)
    if any(not isinstance(color, Color) for color in audience):
        raise ValueError("Message audience must contain participant colors")
    recipients = tuple(dict.fromkeys((speaker, *audience)))
    if any(color not in participants for color in recipients):
        raise ValueError("Message audience contains a non-participant")
    if respondents is not None and (
        not isinstance(respondents, tuple) or len(set(respondents)) != len(respondents)
        or any(not isinstance(color, Color) or color == speaker or color not in audience for color in respondents)
        or set(recipients) != set(participants)
    ):
        raise ValueError("Respondents require public speech and distinct eligible other players")
    return recipients


def build_commitment(
    commitment: tuple[str, str, int],
    *,
    speaker: Color,
    audience: tuple[Color, ...],
    sequence: int,
) -> SocialCommitment:
    """Validate one proposed commitment tuple and materialize it."""
    if not isinstance(commitment, (tuple, list)) or len(commitment) != 3:
        raise ValueError("Commitment must contain condition, promise, and expiry")
    condition, promise, expires_turn = commitment
    if not all(isinstance(value, str) and value.strip() for value in (condition, promise)):
        raise ValueError("Commitment condition and promise must be nonempty strings")
    if type(expires_turn) is not int or expires_turn < 0:
        raise ValueError("Commitment expiry must be a non-negative integer")
    return SocialCommitment(
        id=f"commitment:{sequence}",
        proposer=speaker,
        audience=audience,
        condition=condition,
        promise=promise,
        created_sequence=sequence,
        expires_turn=expires_turn,
        source_message_sequence=sequence,
    )


def append_message(
    engine: GameEngine,
    *,
    speaker: Color,
    text: str,
    audience: tuple[Color, ...],
    causation_id: str,
    commitment: tuple[str, str, int] | None,
    respondents: tuple[Color, ...] | None,
) -> GameEvent:
    """Publish one message event and register any commitment it proposes."""
    recipients = validate_message(
        engine.state.colors,
        speaker=speaker,
        text=text,
        audience=audience,
        causation_id=causation_id,
        respondents=respondents,
    )
    audience = tuple(audience)
    sequence = engine.revision
    proposed_commitment = None
    if commitment is not None:
        proposed_commitment = build_commitment(
            commitment, speaker=speaker, audience=audience, sequence=sequence,
        )
    payload: dict[str, object] = {
        "speaker": speaker,
        "text": text,
        "audience": audience,
    }
    if respondents is not None:
        payload["respondents"] = respondents
    is_public = set(recipients) == set(engine.state.colors)
    event = engine.publish_event(
        "MESSAGE_SENT",
        speaker,
        payload if is_public else None,
        causation_id=causation_id,
        private_overlays=(
            ()
            if is_public
            else tuple((color, payload) for color in recipients)
        ),
        visible_to=None if is_public else recipients,
    )
    if proposed_commitment is not None:
        engine.commitments.append(proposed_commitment)
    return event
