"""Parse communication responses for the XML, fresh-JSON and reactive contracts."""

from __future__ import annotations

import re
from typing import cast

from cle.game_engine.models.player import Color
from cle.harness import communication
from cle.harness.communication.suite import CommunicationSuite
from cle.harness.components import parse_strict_json_object
from cle.harness.models import ModelResponse
from cle.harness.public_speech import parse_public_speech
from cle.harness.response_xml import parse_response_fields
from cle.players.contracts import CommunicationChoice, CommunicationMode
from cle.players.notes import validate_notes


def parse_communication_response(
    response: ModelResponse,
    *,
    speaker: Color,
    participants: tuple[Color, ...],
    instruction: str = "",
    suite: CommunicationSuite | None = None,
) -> CommunicationChoice:
    if suite is not None and suite.reactive_speech:
        try:
            return parse_public_speech(
                parse_strict_json_object(response.content or ""), speaker=speaker,
                participants=participants, max_notes_chars=suite.max_notes_chars,
            )
        except (ValueError, TypeError, RecursionError) as exc:
            return CommunicationChoice(validation_error=f"Invalid communication response: {exc}")
    if suite is not None and suite.memory_mode == "fresh_notes":
        return _parse_fresh_communication_response(
            response, speaker=speaker, participants=participants,
            max_notes_chars=suite.max_notes_chars,
        )
    text = response.content or ""
    if instruction and text.lstrip().startswith(instruction):
        text = text.lstrip()[len(instruction):]
    try:
        fields, _ = parse_response_fields(text, instruction=instruction)
    except ValueError:
        return CommunicationChoice()
    if any(len(values) != 1 for values in fields.values()):
        return CommunicationChoice()
    text = fields.get("message", [""])[0]
    if not text or text.strip().upper() == "SILENCE":
        return CommunicationChoice()

    audience_text = fields.get("audience", ["PUBLIC"])[0].upper()
    if audience_text == "PUBLIC":
        audience = tuple(color for color in participants if color != speaker)
    else:
        names = {name.strip() for name in audience_text.split(",")}
        eligible_names = {color.value for color in participants if color != speaker}
        if not names.issubset(eligible_names):
            return CommunicationChoice()
        audience = tuple(
            color
            for color in participants
            if color != speaker and color.value in names
        )

    commitment = None
    condition = fields.get("commitment_condition", [""])[0]
    promise = fields.get("commitment_promise", [""])[0]
    expires = fields.get("commitment_expires_turn", [""])[0]
    if any(name.startswith("commitment_") for name in fields):
        if not condition or not promise or not re.fullmatch(r"[0-9]+", expires):
            return CommunicationChoice()
        try:
            expires_turn = int(expires)
        except ValueError:
            return CommunicationChoice()
        commitment = communication.CommitmentProposal(condition, promise, expires_turn)

    return CommunicationChoice(
        mode=CommunicationMode.SAY,
        text=text,
        audience=audience,
        commitment=commitment,
    )


def _parse_fresh_communication_response(
    response: ModelResponse,
    *,
    speaker: Color,
    participants: tuple[Color, ...],
    max_notes_chars: int,
) -> CommunicationChoice:
    try:
        payload = parse_strict_json_object(response.content or "")
        notes_update = (
            validate_notes(cast("str", payload["notes"]), max_chars=max_notes_chars)
            if "notes" in payload else None
        )
        mode = payload.get("mode")
        if mode == "silence":
            if set(payload) - {"mode", "notes"}:
                raise ValueError("silence permits only mode and optional notes")
            return CommunicationChoice(notes_update=notes_update)
        if mode != "say":
            raise ValueError("mode must be silence or say")
        required = {"mode", "text", "audience"}
        # Historical fresh-JSON responses carried an intent label; it is ignored, never required.
        if not required <= payload.keys() or payload.keys() - required - {"notes", "commitment", "intent"}:
            raise ValueError("say requires text, audience and permits optional notes and commitment")
        text = payload["text"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError("say text must be a nonempty string")
        eligible = tuple(color for color in participants if color != speaker)
        names = payload["audience"]
        if names == "PUBLIC":
            audience = eligible
        elif (
            isinstance(names, list) and names
            and all(isinstance(name, str) for name in names)
            and len(names) == len(set(names))
            and set(names) <= {color.value for color in eligible}
        ):
            audience = tuple(color for color in eligible if color.value in names)
        else:
            raise ValueError("audience must be PUBLIC or a nonempty list of distinct eligible colors")
        if not audience:
            raise ValueError("say requires at least one eligible recipient")
        commitment = None
        if "commitment" in payload:
            proposal = payload["commitment"]
            if not isinstance(proposal, dict) or set(proposal) != {"condition", "promise", "expires_turn"}:
                raise ValueError("commitment requires exactly condition, promise, expires_turn")
            if any(not isinstance(proposal[key], str) or not proposal[key].strip() for key in ("condition", "promise")):
                raise ValueError("commitment condition and promise must be nonempty strings")
            expires = proposal["expires_turn"]
            if type(expires) is not int or expires < 0:
                raise ValueError("commitment expires_turn must be a non-negative integer")
            commitment = communication.CommitmentProposal(
                proposal["condition"].strip(), proposal["promise"].strip(), expires,
            )
        return CommunicationChoice(
            mode=CommunicationMode.SAY, text=text.strip(), audience=audience,
            commitment=commitment, notes_update=notes_update,
        )
    except (ValueError, TypeError, RecursionError) as exc:
        return CommunicationChoice(validation_error=f"Invalid communication response: {exc}")
