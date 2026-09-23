"""Pre and post action speech acquisition and failures."""
import asyncio
from typing import Any

import pytest

from cle.game_engine.game import GameEngine
from cle.game_engine.models.enums import Action, ActionType
from cle.game_engine.models.player import Color
from cle.harness.models import ModelResponse
from cle.harness.suite import default_suite_path, load_context_suite
from cle.players.agent import AgentPlayer
from cle.players.baseline import FirstLegalPlayer
from cle.players.contracts import (
    CommunicationChoice,
    CommunicationMode,
    PlayerContext,
    TalkContext,
)
from cle.sandbox import CatanSandbox
from cle.sandbox.catan import PostActionCommunicationError, SandboxError

from .support import COLORS, FixedTransport, _sandbox, _trade_engine, _trade_offer


@pytest.mark.asyncio
async def test_pre_action_speech_rejects_stale_results_before_emitting_or_acknowledging() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    class PendingSpeaker(FirstLegalPlayer):
        async def communicate(self, context: TalkContext) -> CommunicationChoice:
            entered.set()
            await release.wait()
            return CommunicationChoice(
                mode=CommunicationMode.SAY, text="Stale speech", audience=COLORS[1:]
            )

        async def choose(self, context: PlayerContext, feedback: str | None=None) -> None:
            raise AssertionError("Must not choose after a stale communication result")

    engine = _trade_engine()
    players = {color: FirstLegalPlayer(color) for color in COLORS}
    players[Color.RED] = PendingSpeaker(Color.RED)
    sandbox: Any = CatanSandbox(engine, players)
    pending = asyncio.create_task(sandbox.step())
    await asyncio.wait_for(entered.wait(), 1)
    engine.step(
        next(a for a in engine.state.playable_actions if a.action_type == ActionType.END_TURN)
    )
    release.set()

    with pytest.raises(SandboxError, match="Stale context"):
        await pending
    assert engine.project_messages(Color.RED) == ()
    assert players[Color.RED].event_cursor == 0
    assert len(sandbox.communication_trace) == 1
    assert sandbox.communication_trace[0].accepted is False
    assert "Stale context" in sandbox.communication_trace[0].validation_error
    assert engine.revision == 1


@pytest.mark.asyncio
async def test_pre_action_messages_advance_revision_without_invalidating_following_choice() -> None:
    transport = FixedTransport(
        [
            ModelResponse(
                content=(
                    "<message>I can offer WOOD.</message><audience>PUBLIC</audience>"
                    "<intent>TRADE</intent>"
                )
            ),
            ModelResponse(content="<game_plan>wait</game_plan><action>0</action>"),
        ]
    )
    engine = _trade_engine()
    players: Any = {color: FirstLegalPlayer(color) for color in COLORS}
    red = AgentPlayer(
        Color.RED, transport, session_id="pre-action:RED",
        suite=load_context_suite(default_suite_path().with_name("catan_v10.yaml")),
    )
    players[Color.RED] = red
    sandbox = CatanSandbox(engine, players)

    result: Any = await sandbox.step()

    assert result.context.context_id.endswith(":1:RED")
    assert result.before_revision == 1
    assert result.after_revision == 2
    assert len(result.messages) == 1
    assert len(red.session.receipts) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("barrier", [False, True])
@pytest.mark.parametrize("cancel", [False, True])
async def test_post_action_speech_failure_preserves_accepted_agent_history(barrier: bool, cancel: bool) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    engine = _trade_engine() if barrier else GameEngine(COLORS, seed=7, shuffle_players=False)
    if barrier:
        engine.step(Action(Color.RED, ActionType.OFFER_TRADE, _trade_offer()))
    agent_colors: Any = COLORS[1:] if barrier else (Color.RED,)
    expected_revision = 4 if barrier else 1

    class FailingSpeaker(AgentPlayer):
        async def communicate(self, context: TalkContext) -> None:
            entered.set()
            assert engine.revision == expected_revision
            for color in agent_colors:
                assert len(players[color].session.receipts) == 1
                assert len(players[color].session.messages) == 2
            await release.wait()
            raise RuntimeError("speech transport failed")

    players: Any = {color: FirstLegalPlayer(color) for color in COLORS}
    for color in agent_colors:
        transport: Any = FixedTransport(
            [ModelResponse(content=f"<game_plan>{color.value} plan</game_plan><action>0</action>")]
        )
        players[color] = FailingSpeaker(
            color, transport, session_id=f"post:{color.value}",
            suite=load_context_suite(default_suite_path().with_name("catan_v10.yaml")),
        )
    if not barrier:
        players[Color.BLUE] = FailingSpeaker(
            Color.BLUE, FixedTransport([]), session_id="post:BLUE",
            suite=load_context_suite(default_suite_path().with_name("catan_v10.yaml")),
        )
    sandbox = CatanSandbox(engine, players)
    pending = asyncio.create_task(sandbox.step())
    await asyncio.wait_for(entered.wait(), 1)
    if cancel:
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
    else:
        release.set()
        with pytest.raises(PostActionCommunicationError) as caught:
            await pending
        assert caught.value.result.after_revision == expected_revision
        assert len(caught.value.result.transitions) == len(agent_colors)
        assert str(caught.value.__cause__) == "speech transport failed"
    assert engine.revision == expected_revision
    for color in agent_colors:
        assert len(players[color].session.receipts) == 1
        assert players[color].session.strategic_memory == f"{color.value} plan"
    sandbox.restore(sandbox.snapshot())


@pytest.mark.asyncio
async def test_post_action_error_carries_already_emitted_messages() -> None:
    class Speaker(FirstLegalPlayer):
        async def communicate(self, context: TalkContext) -> CommunicationChoice:
            return CommunicationChoice(
                mode=CommunicationMode.SAY,
                text=f"{self.color.value} offer",
                audience=(Color.RED,) if self.color == Color.BLUE else (Color.BLACK,),
            )

    sandbox, players = _sandbox()
    for color in (Color.BLUE, Color.WHITE):
        sandbox.register_player(Speaker(color))

    with pytest.raises(PostActionCommunicationError) as caught:
        await sandbox.step()

    result = caught.value.result
    assert result.after_revision == 1
    assert sandbox.revision == 2
    assert players[Color.RED].accepted_choices == 1
    assert result.messages == (sandbox.game_engine.events[1],)
    assert result.messages[0].actor == Color.BLUE
    assert "non-participant" in str(caught.value.__cause__)
    accepted, rejected, withheld = sandbox.communication_trace
    assert accepted.accepted is True
    assert accepted.validation_error is None
    assert rejected.accepted is False
    assert rejected.validation_error == "Message audience contains a non-participant"
    assert withheld.accepted is False
    assert "withheld after WHITE rejection" in withheld.validation_error
    assert [sandbox.players[color].event_cursor for color in COLORS[1:]] == [1, 0, 0]
