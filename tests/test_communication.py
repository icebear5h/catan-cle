from dataclasses import dataclass, field

import pytest

from cle.harness import ModelResponse
from cle.harness.communication import (
    CommunicationSuite,
    build_communication_request,
    default_communication_suite_path,
    load_communication_suite,
    parse_communication_response,
)
from cle.players.baseline import FirstLegalPlayer
from cle.players.contracts import (
    CommitmentProposal,
    CommunicationChoice,
    CommunicationMode,
    TalkContext,
)
from cle.sandbox import CatanSandbox
from cle.sandbox.communication import CommunicationPolicy
from cle.game_engine.communication import CommitmentStatus
from cle.game_engine.events import PlayerEvent
from cle.game_engine.game import GameEngine
from cle.game_engine.models.player import Color


COLORS = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)


def test_default_communication_suite_separates_role_from_policy():
    suite = load_communication_suite()
    context = TalkContext(
        context_id="talk:test:BLUE",
        player=Color.BLUE,
        participants=COLORS,
        cause=PlayerEvent(0, "action:0", Color.RED, "BUILD_SETTLEMENT", 21),
        visible_through_sequence=0,
        game_events=(),
        recent_messages=(),
    )
    request = build_communication_request(context, "session:BLUE", suite)
    system, user = request.messages

    assert suite.version == 5
    assert system.content == (
        "You are playing a game of Catan. You are playing as BLUE."
    )
    assert [component.id for component in request.components] == [
        "system.identity",
        "environment.communication_policy",
        "environment.trigger",
        "environment.visible_events",
        "environment.recent_table_talk",
        "environment.commitments",
        "environment.response_schema",
    ]
    assert user.content == "\n\n".join(
        component.rendered for component in request.components[1:]
    )
    assert "SILENCE" not in system.content
    assert "Default to SILENCE" in user.content
    assert "concrete resource exchange" in user.content
    assert "Never send compliments" in user.content
    assert "COMMENT" not in user.content
    assert next(
        component.value
        for component in request.components
        if component.id == "environment.trigger"
    ).startswith("0. RED: BUILD_SETTLEMENT")


def test_legacy_communication_suite_remains_loadable():
    suite = load_communication_suite(
        default_communication_suite_path().with_name("communication_v4.yaml")
    )

    assert suite.version == 4
    assert suite.user_template is not None
    assert suite.sections == {}


def test_component_communication_suite_rejects_invalid_structure_and_strings():
    suite = load_communication_suite()

    unknown = suite.model_dump(mode="python")
    unknown["sections"]["trigger"]["template"] = "{{ private_state }}"
    with pytest.raises(ValueError, match="unknown variables"):
        CommunicationSuite.model_validate(unknown)

    missing = suite.model_dump(mode="python")
    del missing["sections"]["commitments"]
    with pytest.raises(ValueError, match="exactly match"):
        CommunicationSuite.model_validate(missing)

    duplicate = suite.model_dump(mode="python")
    duplicate["order"] = (*duplicate["order"], "trigger")
    with pytest.raises(ValueError, match="fixed component order"):
        CommunicationSuite.model_validate(duplicate)

    oversized = suite.model_dump(mode="python")
    oversized["sections"]["trigger"]["template"] = "x" * 12_001
    with pytest.raises(ValueError, match="at most 12000 characters"):
        CommunicationSuite.model_validate(oversized)


def test_communication_parser_drops_social_comments_but_keeps_trade_messages():
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

    assert social.mode == CommunicationMode.SILENCE
    assert trade.mode == CommunicationMode.SAY
    assert trade.text == "I can give WOOD for ORE."
    assert trade.audience == (Color.BLUE,)
    assert trade.intent == "TRADE"


@pytest.mark.parametrize(
    "audience",
    [
        "BLU", "BLACK", "RED", "BLUE,BLACK", "BLUE,RED", "PUBLIC,BLUE",
        "", " ", "BLUE,", "BLUE,,WHITE",
    ],
)
def test_communication_parser_fails_closed_for_invalid_explicit_audience(audience):
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
def test_communication_parser_fails_closed_for_malformed_or_repeated_audience(audience_markup):
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
def test_communication_parser_accepts_unambiguous_audience_whitespace(audience_markup):
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
def test_communication_parser_preserves_valid_audiences(audience, expected):
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
    "<message>Trade WOOD for ORE?</message><!-- <intent>TRADE</intent> -->",
    "<message>Trade WOOD for ORE?</message><intent>TRADE</intent><intent>TRADE</intent>",
    "<message>Trade WOOD for ORE?</message><message>SILENCE</message><intent>TRADE</intent>",
    '<message kind="private">Trade WOOD for ORE?</message><intent>TRADE</intent>',
    "<message>Trade WOOD for ORE?</message><intent><intent>TRADE</intent></intent>",
    "<message>Trade WOOD for ORE?</message><intent>TRADE</intent><audience>BLUE",
    "<message>Trade WOOD for ORE?</message><intent>TRADE",
])
def test_communication_structural_parser_fails_closed_on_malformed_or_commented_fields(text):
    assert parse_communication_response(
        ModelResponse(content=text), speaker=Color.RED, participants=COLORS
    ) == CommunicationChoice()


def test_communication_comments_do_not_override_real_fields_and_entities_decode():
    response = ModelResponse(content=(
        "<!-- <message>SILENCE</message><intent>COMMENT</intent><audience>PUBLIC</audience> -->"
        "<message>WOOD &amp; BRICK for ORE?</message><intent>TRADE</intent><audience>BLUE</audience>"
    ))
    choice = parse_communication_response(response, speaker=Color.RED, participants=COLORS)
    assert choice.mode == CommunicationMode.SAY
    assert choice.text == "WOOD & BRICK for ORE?"
    assert choice.audience == (Color.BLUE,)
    assert "&amp;" in response.content


def test_communication_parser_uses_only_explicit_instruction_for_schema_echoes():
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
def test_communication_parser_rejects_incomplete_invalid_or_duplicate_commitments(fields):
    response = ModelResponse(content=(
        "<message>Leave me alone and I will trade.</message><intent>BRIBE</intent>"
        f"<audience>BLUE</audience>{fields}"
    ))
    assert parse_communication_response(
        response, speaker=Color.RED, participants=COLORS
    ) == CommunicationChoice()


def test_communication_rendering_preserves_template_syntax_in_message_data():
    engine = GameEngine(COLORS, seed=7, shuffle_players=False)
    text = "Offer {{ wood }} for {{ ore }}"
    engine.append_message(
        speaker=Color.RED,
        text=text,
        audience=(Color.BLUE,),
        intent="TRADE",
        causation_id="literal-template-data",
        commitment=("If {{ wood }} is available", "Offer {{ ore }}", 3),
    )
    messages = engine.project_messages(Color.BLUE)
    context = TalkContext(
        context_id="literal-template-data:BLUE",
        player=Color.BLUE,
        participants=COLORS,
        cause=messages[0],
        visible_through_sequence=messages[0].sequence,
        game_events=engine.project_game_events(Color.BLUE),
        recent_messages=messages,
        active_commitments=engine.active_commitments(Color.BLUE),
    )

    request = build_communication_request(context, "BLUE", load_communication_suite())

    assert text in request.messages[-1].content
    assert "If {{ wood }} is available -> Offer {{ ore }}" in request.messages[-1].content


@dataclass
class TalkPlayer(FirstLegalPlayer):
    messages: list[str] = field(default_factory=list)
    audiences: tuple[Color, ...] = ()
    contexts: list = field(default_factory=list)
    commitment: CommitmentProposal | None = None

    async def communicate(self, context):
        self.contexts.append(context)
        if not self.messages:
            return CommunicationChoice()
        return CommunicationChoice(
            mode=CommunicationMode.SAY,
            text=self.messages.pop(0),
            audience=self.audiences,
            intent="COMMENT",
            commitment=self.commitment,
        )


def _sandbox(players):
    engine = GameEngine(COLORS, seed=7, shuffle_players=False)
    return CatanSandbox(engine, players)


@pytest.mark.asyncio
async def test_broad_build_trigger_emits_message_and_drops_silence():
    blue = TalkPlayer(
        Color.BLUE,
        messages=["That blocks my route."],
        audiences=(Color.RED, Color.WHITE, Color.ORANGE),
    )
    players = {color: FirstLegalPlayer(color) for color in COLORS}
    players[Color.BLUE] = blue
    sandbox = _sandbox(players)

    result = await sandbox.step()

    assert len(result.messages) == 1
    assert result.messages[0].event_type == "MESSAGE_SENT"
    assert result.messages[0].public_payload["text"] == "That blocks my route."
    assert all(event.event_type != "MESSAGE_SENT" for event in sandbox.game_engine.project_game_events(Color.RED))
    assert sandbox.game_engine.project_messages(Color.RED)[0].payload["text"] == "That blocks my route."


@pytest.mark.asyncio
async def test_same_round_players_share_cutoff_and_do_not_see_peer_message():
    blue = TalkPlayer(
        Color.BLUE,
        messages=["Blue message"],
        audiences=(Color.RED, Color.WHITE, Color.ORANGE),
    )
    white = TalkPlayer(
        Color.WHITE,
        messages=["White message"],
        audiences=(Color.RED, Color.BLUE, Color.ORANGE),
    )
    players = {color: FirstLegalPlayer(color) for color in COLORS}
    players[Color.BLUE] = blue
    players[Color.WHITE] = white
    sandbox = _sandbox(players)

    result = await sandbox.step()

    assert [event.actor for event in result.messages[:2]] == [Color.BLUE, Color.WHITE]
    assert blue.contexts[0].visible_through_sequence == white.contexts[0].visible_through_sequence
    assert blue.contexts[0].recent_messages == ()
    assert white.contexts[0].recent_messages == ()


def test_message_events_do_not_retrigger_communication_opportunities():
    engine = GameEngine(COLORS, seed=1, shuffle_players=False)
    message = engine.append_message(
        speaker=Color.BLUE,
        text="I can trade WOOD for ORE.",
        audience=(Color.RED,),
        intent="TRADE",
        causation_id="test",
    )

    opportunities = CommunicationPolicy().after_events(
        engine,
        (message,),
        round_number=1,
    )

    assert opportunities == ()


@pytest.mark.asyncio
async def test_private_message_projects_only_to_speaker_and_audience():
    engine = GameEngine(COLORS, seed=1, shuffle_players=False)
    event = engine.append_message(
        speaker=Color.BLUE,
        text="Private offer",
        audience=(Color.RED,),
        intent="TRADE",
        causation_id="test",
    )

    assert engine.project_messages(Color.RED)[0].sequence == event.sequence
    assert engine.project_messages(Color.BLUE)[0].sequence == event.sequence
    assert engine.project_messages(Color.WHITE) == ()
    assert engine.project_messages(Color.ORANGE) == ()


@pytest.mark.asyncio
async def test_commitment_is_pinned_exactly_and_expires_on_engine_step():
    engine = GameEngine(COLORS, seed=2, shuffle_players=False)
    engine.append_message(
        speaker=Color.BLUE,
        text="Do not rob me and I will trade later.",
        audience=(Color.RED,),
        intent="BRIBE",
        causation_id="robber:1",
        commitment=("RED does not rob BLUE", "BLUE offers ORE", 0),
    )

    active = engine.active_commitments(Color.RED)
    assert active[0].condition == "RED does not rob BLUE"
    assert active[0].promise == "BLUE offers ORE"

    engine.step(engine.state.playable_actions[0])

    assert active[0].status == CommitmentStatus.ACTIVE
    assert engine.commitments[0].status == CommitmentStatus.EXPIRED
    assert engine.active_commitments(Color.RED) == ()


def test_debug_undo_restores_commitment_status_with_engine_state():
    engine = GameEngine(
        COLORS,
        seed=4,
        shuffle_players=False,
        capture_history=True,
    )
    engine.append_message(
        speaker=Color.BLUE,
        text="Promise",
        audience=(Color.RED,),
        intent="BRIBE",
        causation_id="test",
        commitment=("condition", "promise", 0),
    )

    engine.step(engine.state.playable_actions[0])
    assert engine.commitments[0].status == CommitmentStatus.EXPIRED
    engine.undo()

    assert engine.commitments[0].status == CommitmentStatus.ACTIVE
    assert engine.active_commitments(Color.RED)


def test_message_window_is_bounded_but_game_events_are_complete():
    engine = GameEngine(COLORS, seed=3, shuffle_players=False)
    for index in range(15):
        engine.append_message(
            speaker=Color.RED,
            text=f"message-{index}",
            audience=COLORS[1:],
            intent="COMMENT",
            causation_id=f"message:{index}",
        )
    engine.step(engine.state.playable_actions[0])

    assert len(engine.project_messages(Color.BLUE)) == 12
    assert engine.project_messages(Color.BLUE)[0].payload["text"] == "message-3"
    assert len(engine.project_game_events(Color.BLUE)) == 1
