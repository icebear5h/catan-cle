"""Audience, commitment, and structural parsing."""

import pytest

from cle.game_engine.models.player import Color
from cle.harness import ModelResponse
from cle.harness.communication import (
    load_communication_suite,
    parse_communication_response,
)
from cle.players.contracts import (
    CommunicationChoice,
    CommunicationMode,
)

from .support import COLORS


def test_communication_parser_ignores_intent_labels_and_keeps_text() -> None:
    """The legacy XML contract asked for an intent tag; it is no longer read or policed."""
    social = parse_communication_response(
        ModelResponse(
            content=(
                "<message>Nice settlement!</message>"
                "<audience>PUBLIC</audience>"
                "<intent>COMMENT</intent>"
            )
        ),
        speaker=Color.RED,
        participants=COLORS,
    )
    trade = parse_communication_response(
        ModelResponse(
            content=(
                "<message>I can give WOOD for ORE.</message>"
                "<audience>BLUE</audience>"
                "<intent>TRADE</intent>"
            )
        ),
        speaker=Color.RED,
        participants=COLORS,
    )

    assert social.mode == CommunicationMode.SAY
    assert social.text == "Nice settlement!"
    assert social.audience == COLORS[1:]
    assert trade.mode == CommunicationMode.SAY
    assert trade.text == "I can give WOOD for ORE."
    assert trade.audience == (Color.BLUE,)


@pytest.mark.parametrize(
    "audience",
    [
        "BLU", "BLACK", "RED", "BLUE,BLACK", "BLUE,RED", "PUBLIC,BLUE",
        "", " ", "BLUE,", "BLUE,,WHITE",
    ],
)
def test_communication_parser_fails_closed_for_invalid_explicit_audience(audience: str) -> None:
    choice = parse_communication_response(
        ModelResponse(
            content=(
                "<message>I hold 3 ORE; this is private.</message>"
                f"<audience>{audience}</audience><intent>TRADE</intent>"
            )
        ),
        speaker=Color.RED,
        participants=COLORS,
    )

    assert choice == CommunicationChoice()


@pytest.mark.parametrize(
    "audience_markup",
    [
        "<audience>BLUE",
        "<audience >BLUE",
        "<audience/>",
        "<audience />",
        "</audience>",
        "< audience>BLUE</audience>",
        '<audience target="BLUE">PUBLIC</audience>',
        "<audience>PUBLIC</audience><audience>BLUE</audience>",
        "<audience>BLUE</audience><audience>PUBLIC</audience>",
        "<audience>PUBLIC</audience><audience>PUBLIC</audience>",
        "<audience>BLUE</audience><audience>BLUE</audience>",
        "<audience>PUBLIC</audience><audience/>",
        "<audience><audience>BLUE</audience></audience>",
    ],
)
def test_communication_parser_fails_closed_for_malformed_or_repeated_audience(audience_markup: str) -> None:
    choice = parse_communication_response(
        ModelResponse(
            content=(
                "<message>I hold 3 ORE; this is private.</message>"
                f"<intent>TRADE</intent>{audience_markup}"
            )
        ),
        speaker=Color.RED,
        participants=COLORS,
    )

    assert choice == CommunicationChoice()


@pytest.mark.parametrize(
    "audience_markup",
    [
        "<audience >BLUE</audience>",
        "<audience\n>BLUE</audience>",
        "<AuDiEnCe\t> blue </AuDiEnCe\n>",
    ],
)
def test_communication_parser_accepts_unambiguous_audience_whitespace(audience_markup: str) -> None:
    choice = parse_communication_response(
        ModelResponse(
            content=(
                "<message>I hold 3 ORE; this is private.</message>"
                f"<intent>TRADE</intent>{audience_markup}"
            )
        ),
        speaker=Color.RED,
        participants=COLORS,
    )

    assert choice.mode == CommunicationMode.SAY
    assert choice.audience == (Color.BLUE,)


@pytest.mark.parametrize(
    "audience, expected",
    [
        (None, COLORS[1:]),
        ("public", COLORS[1:]),
        ("BLUE", (Color.BLUE,)),
        (" white, blue ", (Color.BLUE, Color.WHITE)),
        ("BLUE,BLUE", (Color.BLUE,)),
    ],
)
def test_communication_parser_preserves_valid_audiences(
    audience: str | None, expected: tuple[Color, ...]
) -> None:
    audience_tag = f"<audience>{audience}</audience>" if audience is not None else ""
    choice = parse_communication_response(
        ModelResponse(
            content=(
                "<message>I can trade WOOD for ORE.</message>"
                f"{audience_tag}<intent>TRADE</intent>"
            )
        ),
        speaker=Color.RED,
        participants=COLORS,
    )

    assert choice.mode == CommunicationMode.SAY
    assert choice.audience == expected


@pytest.mark.parametrize("text", [
    "<!-- <message>Trade WOOD for ORE?</message><intent>TRADE</intent> -->",
    "<message>Trade WOOD for ORE?</message><intent>TRADE</intent><intent>TRADE</intent>",
    "<message>Trade WOOD for ORE?</message><message>SILENCE</message><intent>TRADE</intent>",
    '<message kind="private">Trade WOOD for ORE?</message><intent>TRADE</intent>',
    "<message>Trade WOOD for ORE?</message><intent><intent>TRADE</intent></intent>",
    "<message>Trade WOOD for ORE?</message><intent>TRADE</intent><audience>BLUE",
    "<message>Trade WOOD for ORE?</message><intent>TRADE",
])
def test_communication_structural_parser_fails_closed_on_malformed_or_commented_fields(text: str) -> None:
    assert parse_communication_response(
        ModelResponse(content=text), speaker=Color.RED, participants=COLORS
    ) == CommunicationChoice()


def test_communication_comments_do_not_override_real_fields_and_entities_decode() -> None:
    response = ModelResponse(content=(
        "<!-- <message>SILENCE</message><intent>COMMENT</intent><audience>PUBLIC</audience> -->"
        "<message>WOOD &amp; BRICK for ORE?</message><intent>TRADE</intent><audience>BLUE</audience>"
    ))
    choice = parse_communication_response(response, speaker=Color.RED, participants=COLORS)
    assert choice.mode == CommunicationMode.SAY
    assert choice.text == "WOOD & BRICK for ORE?"
    assert choice.audience == (Color.BLUE,)
    assert "&amp;" in response.content


def test_communication_parser_uses_only_explicit_instruction_for_schema_echoes() -> None:
    instruction = load_communication_suite().sections["response_schema"].template
    response = ModelResponse(content=(
        f"{instruction}\n<message>Trade WOOD for ORE?</message>"
        "<intent>TRADE</intent><audience>BLUE</audience>"
    ))
    choice = parse_communication_response(
        response, speaker=Color.RED, participants=COLORS, instruction=instruction
    )
    assert choice.mode == CommunicationMode.SAY
    assert choice.audience == (Color.BLUE,)
    assert parse_communication_response(
        response, speaker=Color.RED, participants=COLORS
    ) == CommunicationChoice()


@pytest.mark.parametrize("fields", [
    "<commitment_condition>Leave BLUE alone</commitment_condition>",
    "<commitment_promise>Offer ORE</commitment_promise>",
    "<commitment_expires_turn>3</commitment_expires_turn>",
    "<commitment_condition>Leave BLUE alone</commitment_condition>"
    "<commitment_promise>Offer ORE</commitment_promise>"
    "<commitment_expires_turn>3.5</commitment_expires_turn>",
    "<commitment_condition>Leave BLUE alone</commitment_condition>"
    "<commitment_promise>Offer ORE</commitment_promise>"
    "<commitment_expires_turn>3</commitment_expires_turn>"
    "<commitment_expires_turn>4</commitment_expires_turn>",
])
def test_communication_parser_rejects_incomplete_invalid_or_duplicate_commitments(fields: str) -> None:
    response = ModelResponse(content=(
        "<message>Leave me alone and I will trade.</message><intent>BRIBE</intent>"
        f"<audience>BLUE</audience>{fields}"
    ))
    assert parse_communication_response(
        response, speaker=Color.RED, participants=COLORS
    ) == CommunicationChoice()
