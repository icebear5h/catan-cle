"""Speech barrier limits, withholding, and cancellation."""
import asyncio
from dataclasses import replace
from typing import Any

import pytest

from cle.game_engine.game import GameEngine
from cle.game_engine.models.enums import Action, ActionType
from cle.game_engine.models.player import Color
from cle.players.baseline import FirstLegalPlayer
from cle.players.contracts import (
    CommunicationChoice,
    CommunicationMode,
    PlayerAttempt,
    PlayerContext,
    TalkContext,
)
from cle.sandbox import CatanSandbox, RetryPolicy
from cle.sandbox.catan import PlayerResponseError, PostActionCommunicationError

from .support import COLORS, _sandbox, _trade_engine, _trade_offer


@pytest.mark.asyncio
@pytest.mark.parametrize("limit,orange_silent", [(1, False), (2, False), (1, True)])
async def test_communication_message_limit_counts_each_emitted_message_once(limit: int, orange_silent: bool) -> None:
    class Speaker(FirstLegalPlayer):
        async def communicate(self, context: TalkContext) -> CommunicationChoice:
            return CommunicationChoice(
                mode=CommunicationMode.SAY, text="Trade?", audience=(Color.RED,)
            )

    sandbox, _ = _sandbox()
    sandbox.game_engine.communication_limits = replace(
        sandbox.game_engine.communication_limits, max_messages_per_window=limit
    )
    for color in COLORS[1:]:
        if color != Color.ORANGE or not orange_silent:
            sandbox.register_player(Speaker(color))

    result = await sandbox.step()

    assert [event.actor for event in result.messages] == list(COLORS[1 : limit + 1])
    assert sandbox.revision == 1 + limit
    assert len(sandbox.communication_trace) == 3
    assert [record.accepted for record in sandbox.communication_trace] == [
        True,
        limit >= 2,
        orange_silent,
    ]
    for record in sandbox.communication_trace:
        if record.accepted:
            assert record.validation_error is None
            assert sandbox.players[record.opportunity.player].event_cursor == 1
        else:
            assert "message limit" in record.validation_error
            assert sandbox.players[record.opportunity.player].event_cursor == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_acquired_speech_is_recorded_withheld_when_sibling_request_fails(cancel: bool) -> None:
    acquired = asyncio.Event()
    release = asyncio.Event()
    never = asyncio.Event()

    class Speaker(FirstLegalPlayer):
        async def communicate(self, context: TalkContext) -> CommunicationChoice:
            if self.color == Color.BLUE:
                acquired.set()
                return CommunicationChoice(
                    mode=CommunicationMode.SAY, text="Trade?", audience=(Color.RED,)
                )
            if self.color == Color.WHITE:
                await release.wait()
                raise RuntimeError("speech provider failed")
            await never.wait()

    sandbox, players = _sandbox()
    for color in COLORS[1:]:
        sandbox.register_player(Speaker(color))
    pending = asyncio.create_task(sandbox.step())
    await asyncio.wait_for(acquired.wait(), 1)
    if cancel:
        pending.cancel()
    else:
        release.set()

    with pytest.raises(asyncio.CancelledError if cancel else PostActionCommunicationError):
        await pending

    assert sandbox.revision == 1
    assert players[Color.RED].accepted_choices == 1
    assert len(sandbox.communication_trace) == 1
    record: Any = sandbox.communication_trace[0]
    assert record.opportunity.player == Color.BLUE
    assert record.choice.text == "Trade?"
    assert record.accepted is False
    assert "withheld" in record.validation_error
    assert ("CancelledError" if cancel else "speech provider failed") in record.validation_error
    assert all(sandbox.players[color].event_cursor == 0 for color in COLORS[1:])


@pytest.mark.asyncio
@pytest.mark.parametrize("speech", [False, True])
@pytest.mark.parametrize("cancel", [False, True])
async def test_failed_barrier_cancels_and_awaits_sibling_requests(speech: bool, cancel: bool) -> None:
    entered = asyncio.Event()
    never = asyncio.Event()
    active: set[Color] = set()
    cleaned_up: set[Color] = set()

    async def wait_for_result(color: Color) -> None:
        active.add(color)
        if len(active) == 3:
            entered.set()
        try:
            if color == Color.BLUE and not cancel:
                await entered.wait()
                raise RuntimeError("speech failed") if speech else ValueError("bad choice")
            await never.wait()
        finally:
            await asyncio.sleep(0)
            active.remove(color)
            cleaned_up.add(color)

    class WaitingPlayer(FirstLegalPlayer):
        async def choose(self, context: PlayerContext, feedback: str | None=None) -> PlayerAttempt:
            if not speech:
                try:
                    await wait_for_result(self.color)
                except ValueError as exc:
                    return PlayerAttempt(context.context_id, None, str(exc))
            return await super().choose(context, feedback)

        async def communicate(self, context: TalkContext) -> CommunicationChoice:
            if speech:
                await wait_for_result(self.color)
            return CommunicationChoice()

    engine = GameEngine(COLORS, seed=7, shuffle_players=False) if speech else _trade_engine()
    if not speech:
        engine.step(Action(Color.RED, ActionType.OFFER_TRADE, _trade_offer()))
    players = {Color.RED: FirstLegalPlayer(Color.RED)}
    players.update({color: WaitingPlayer(color) for color in COLORS[1:]})
    sandbox = CatanSandbox(engine, players, retry_policy=RetryPolicy(1))
    pending = asyncio.create_task(sandbox.step())
    await asyncio.wait_for(entered.wait(), 1)
    if cancel:
        pending.cancel()
        expected_error = asyncio.CancelledError
    else:
        expected_error = PostActionCommunicationError if speech else PlayerResponseError
    with pytest.raises(expected_error):
        await pending
    assert active == set()
    assert cleaned_up == set(COLORS[1:])
    assert engine.revision == 1
    assert players[Color.RED].accepted_choices == int(speech)
    sandbox.restore(sandbox.snapshot())
