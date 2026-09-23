"""Bounded structured communication contracts."""
from typing import Any

import pytest

from cle.game_engine.game import GameEngine
from cle.game_engine.models.player import Color
from cle.harness import ModelResponse, PlayerSession, load_context_suite
from cle.harness.communication import CommunicationSuite, load_communication_suite
from cle.harness.models import ModelRequest
from cle.players import (
    CommunicationMode,
    FirstLegalPlayer,
)
from cle.players.agent import AgentPlayer
from cle.players.contracts import TalkContext
from cle.sandbox import CatanSandbox

from .support import COLORS, AsyncFixedTransport


@pytest.mark.asyncio
async def test_agent_player_communication_uses_bounded_structured_contract() -> None:
    engine = GameEngine(COLORS, seed=9, shuffle_players=False)
    transition = engine.step(engine.state.playable_actions[0])
    transport = AsyncFixedTransport([
        ModelResponse(
            content=(
                "<message>Do not block me and I will trade later.</message>"
                "<audience>RED</audience>"
                "<intent>BRIBE</intent>"
                "<commitment_condition>RED avoids BLUE</commitment_condition>"
                "<commitment_promise>BLUE offers ORE</commitment_promise>"
                "<commitment_expires_turn>3</commitment_expires_turn>"
            )
        )
    ])
    player = AgentPlayer(
        Color.BLUE, transport, session_id="game:BLUE",
        suite=load_context_suite(),
        communication_suite=load_communication_suite(),
    )
    cause = engine.project_events(Color.BLUE)[0]
    context = TalkContext(
        context_id="talk:1:BLUE",
        player=Color.BLUE,
        participants=COLORS,
        cause=cause,
        visible_through_sequence=transition.after_revision - 1,
        game_events=engine.project_game_events(Color.BLUE),
        recent_messages=(),
    )

    choice: Any = await player.communicate(context)

    assert choice.mode == CommunicationMode.SAY
    assert choice.audience == (Color.RED,)
    assert choice.commitment.promise == "BLUE offers ORE"
    assert transport.requests[0].messages[0].content == (
        "You are playing a game of Catan. You are playing as BLUE."
    )
    assert "COMMUNICATION POLICY" in transport.requests[0].messages[-1].content
    assert "COMPLETE VISIBLE GAME EVENTS" in transport.requests[0].messages[-1].content
    assert transport.requests[0].components[1].channel == "environment"


@pytest.mark.asyncio
@pytest.mark.parametrize("echo_dynamic_prompt", [False, True])
async def test_agent_communication_trusts_only_authored_schema_echo_and_keeps_private_audience(echo_dynamic_prompt: bool) -> None:
    engine: Any = GameEngine(COLORS, seed=9, shuffle_players=False)
    engine.append_message(
        speaker=Color.RED, text="DYNAMIC PRIVATE TABLE TALK", audience=(Color.BLUE,),
        causation_id="prior-talk",
    )
    data = load_communication_suite().model_dump()
    instruction = "LOCAL AUTHORED SCHEMA:\n" + data["sections"]["response_schema"]["template"]
    data["sections"]["response_schema"]["template"] = instruction
    suite: Any = CommunicationSuite.model_validate(data)
    requests = []

    class EchoTransport:
        async def complete(self, request: ModelRequest) -> ModelResponse:
            requests.append(request)
            prefix = request.messages[-1].content if echo_dynamic_prompt else instruction
            return ModelResponse(content=(
                f"{prefix}\n<message>PRIVATE: WOOD for ORE?</message>"
                "<audience>RED</audience><intent>TRADE</intent>"
            ))

    players: Any = {color: FirstLegalPlayer(color) for color in COLORS}
    players[Color.BLUE] = AgentPlayer(
        Color.BLUE, EchoTransport(), session_id="private-echo:BLUE",
        suite=load_context_suite(), communication_suite=suite,
    )
    sandbox = CatanSandbox(engine, players)
    result = await sandbox.step()

    assert len(requests) == 1
    assert "DYNAMIC PRIVATE TABLE TALK" in requests[0].messages[-1].content
    schema = next(component for component in requests[0].components if component.id == "environment.response_schema")
    assert schema.template == instruction
    assert len(result.messages) == (0 if echo_dynamic_prompt else 1)
    if not echo_dynamic_prompt:
        assert engine.project_messages(Color.RED)[-1].payload["text"] == "PRIVATE: WOOD for ORE?"
        assert engine.project_messages(Color.BLUE)[-1].payload["audience"] == (Color.RED,)
        assert result.messages[0].public_payload is None
    assert engine.project_messages(Color.WHITE) == ()
    assert engine.project_messages(Color.ORANGE) == ()
    assert players[Color.BLUE].session.messages == []
    assert players[Color.BLUE].session.receipts == {}


def test_player_session_snapshot_restores_identity_and_continuity() -> None:
    session = PlayerSession(Color.RED, "game:RED")
    session.strategic_memory = "build cities"
    snapshot = session.snapshot()
    session.strategic_memory = "changed"

    session.restore(snapshot)

    assert session.strategic_memory == "build cities"
