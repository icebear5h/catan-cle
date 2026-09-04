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

    assert active[0].status == CommitmentStatus.EXPIRED
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
