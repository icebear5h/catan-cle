"""Strict public-speech choice parsing, independent of action and prompt rendering."""

from typing import cast

from cle.game_engine.models.player import Color
from cle.players.contracts import CommitmentProposal, CommunicationChoice, CommunicationMode
from cle.players.notes import validate_notes


def parse_public_speech(
    payload: dict[str, object], *, speaker: Color, participants: tuple[Color, ...],
    max_notes_chars: int,
) -> CommunicationChoice:
    notes = (
        validate_notes(cast("str", payload["notes"]), max_notes_chars)
        if "notes" in payload else None
    )
    if payload.get("mode") in {"pass", "silence"}:
        if set(payload) - {"mode", "notes"}:
            raise ValueError("Pass permits only mode and optional notes")
        return CommunicationChoice(notes_update=notes, respondents=())
    required = {"mode", "text", "respondents"}
    if payload.get("mode") != "say" or not required <= payload.keys() or (
        payload.keys() - required - {"notes", "commitment", "audience"}
    ):
        raise ValueError("Say requires mode, text, respondents; optional notes and commitment")
    if payload.get("audience", "PUBLIC") != "PUBLIC":
        raise ValueError("Table talk is public; use respondents to address players")
    text = payload["text"]
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Say text must be nonempty")
    eligible = tuple(color for color in participants if color != speaker)
    names = payload["respondents"]
    if (
        isinstance(names, list)
        and all(isinstance(name, str) for name in names)
        and len(names) == len(set(names))
        and set(names) <= {color.value for color in eligible}
    ):
        respondents = tuple(color for color in eligible if color.value in names)
    else:
        raise ValueError("Respondents must be a list of distinct eligible other colors (possibly [])")
    commitment = None
    if "commitment" in payload:
        proposal = payload["commitment"]
        if not isinstance(proposal, dict) or set(proposal) != {"condition", "promise", "expires_turn"}:
            raise ValueError("Commitment requires condition, promise, expires_turn")
        if any(not isinstance(proposal[k], str) or not proposal[k].strip() for k in ("condition", "promise")):
            raise ValueError("Commitment condition and promise must be nonempty strings")
        if type(proposal["expires_turn"]) is not int or proposal["expires_turn"] < 0:
            raise ValueError("Commitment expiry must be a non-negative integer")
        commitment = CommitmentProposal(**proposal)
    return CommunicationChoice(
        mode=CommunicationMode.SAY, text=text.strip(), audience=eligible,
        respondents=respondents, commitment=commitment,
        notes_update=notes,
    )
