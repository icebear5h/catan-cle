"""Triggering, projection, and commitment lifecycle."""
from typing import Any

import pytest

from cle.game_engine.communication import CommitmentStatus
from cle.game_engine.game import GameEngine
from cle.game_engine.models.player import Color
from cle.players.baseline import FirstLegalPlayer
from cle.sandbox.communication import CommunicationPolicy

from .support import COLORS, TalkPlayer, _sandbox


@pytest.mark.asyncio
async def test_broad_build_trigger_emits_message_and_drops_silence() -> None:
    blue = TalkPlayer(
        Color.BLUE,
        messages=["That blocks my route."],
        audiences=(Color.RED, Color.WHITE, Color.ORANGE),
    )
    players: Any = {color: FirstLegalPlayer(color) for color in COLORS}
    players[Color.BLUE] = blue
    sandbox: Any = _sandbox(players)

    result: Any = await sandbox.step()

    assert len(result.messages) == 1
    assert result.messages[0].event_type == "MESSAGE_SENT"
    assert result.messages[0].public_payload["text"] == "That blocks my route."
    assert all(event.event_type != "MESSAGE_SENT" for event in sandbox.game_engine.project_game_events(Color.RED))
    assert sandbox.game_engine.project_messages(Color.RED)[0].payload["text"] == "That blocks my route."


@pytest.mark.asyncio
async def test_same_round_players_share_cutoff_and_do_not_see_peer_message() -> None:
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
    players: Any = {color: FirstLegalPlayer(color) for color in COLORS}
    players[Color.BLUE] = blue
    players[Color.WHITE] = white
    sandbox = _sandbox(players)

    result = await sandbox.step()

    assert [event.actor for event in result.messages[:2]] == [Color.BLUE, Color.WHITE]
    assert blue.contexts[0].visible_through_sequence == white.contexts[0].visible_through_sequence
    assert blue.contexts[0].recent_messages == ()
    assert white.contexts[0].recent_messages == ()


def test_message_events_do_not_retrigger_communication_opportunities() -> None:
    engine = GameEngine(COLORS, seed=1, shuffle_players=False)
    message = engine.append_message(
        speaker=Color.BLUE,
        text="I can trade WOOD for ORE.",
        audience=(Color.RED,),
        causation_id="test",
    )

    opportunities = CommunicationPolicy().after_events(
        engine,
        (message,),
        round_number=1,
    )

    assert opportunities == ()


@pytest.mark.asyncio
async def test_private_message_projects_only_to_speaker_and_audience() -> None:
    engine = GameEngine(COLORS, seed=1, shuffle_players=False)
    event = engine.append_message(
        speaker=Color.BLUE,
        text="Private offer",
        audience=(Color.RED,),
        causation_id="test",
    )

    assert engine.project_messages(Color.RED)[0].sequence == event.sequence
    assert engine.project_messages(Color.BLUE)[0].sequence == event.sequence
    assert engine.project_messages(Color.WHITE) == ()
    assert engine.project_messages(Color.ORANGE) == ()


@pytest.mark.asyncio
async def test_commitment_is_pinned_exactly_and_expires_on_engine_step() -> None:
    engine = GameEngine(COLORS, seed=2, shuffle_players=False)
    engine.append_message(
        speaker=Color.BLUE,
        text="Do not rob me and I will trade later.",
        audience=(Color.RED,),
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


def test_debug_undo_restores_commitment_status_with_engine_state() -> None:
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
        causation_id="test",
        commitment=("condition", "promise", 0),
    )

    engine.step(engine.state.playable_actions[0])
    assert engine.commitments[0].status == CommitmentStatus.EXPIRED
    engine.undo()

    assert engine.commitments[0].status == CommitmentStatus.ACTIVE
    assert engine.active_commitments(Color.RED)


def test_message_window_is_bounded_but_game_events_are_complete() -> None:
    engine: Any = GameEngine(COLORS, seed=3, shuffle_players=False)
    for index in range(15):
        engine.append_message(
            speaker=Color.RED,
            text=f"message-{index}",
            audience=COLORS[1:],
            causation_id=f"message:{index}",
        )
    engine.step(engine.state.playable_actions[0])

    assert len(engine.project_messages(Color.BLUE)) == 12
    assert engine.project_messages(Color.BLUE)[0].payload["text"] == "message-3"
    assert len(engine.project_game_events(Color.BLUE)) == 1
