"""Trade capacity preflight and barrier failure retention."""
import asyncio
import pickle
from typing import Any

import pytest

from cle.game_engine.models.actions import trade_response_actions
from cle.game_engine.models.enums import Action, ActionType
from cle.game_engine.models.player import Color
from cle.game_engine.trading import TradeLimits, TradeOffer
from cle.harness.models import ModelRequest, ModelResponse
from cle.harness.suite import default_suite_path, load_context_suite
from cle.players.agent import AgentPlayer
from cle.players.baseline import FirstLegalPlayer
from cle.players.contracts import (
    PlayerAttempt,
    PlayerChoice,
    PlayerContext,
)
from cle.sandbox import CatanSandbox, RetryPolicy
from cle.sandbox.catan import PlayerResponseError

from .support import COLORS, FixedTransport, NoCommunicationPolicy, _trade_engine, _trade_offer


@pytest.mark.asyncio
@pytest.mark.parametrize("attempt_limit,initial_invalid", [(1, False), (2, False), (2, True)])
async def test_default_trade_capacity_is_preflighted_without_partial_live_actions(
    attempt_limit: int, initial_invalid: bool
) -> None:
    calls = {color: [] for color in COLORS[1:]}

    class CounterPlayer(FirstLegalPlayer):
        async def choose(self, context: PlayerContext, feedback: str | None=None) -> PlayerAttempt:
            calls[self.color].append((context.context_id, feedback))
            if initial_invalid and self.color == Color.ORANGE and len(calls[self.color]) == 3:
                return PlayerAttempt(context.context_id, PlayerChoice(action_index=999))
            if feedback and "maximum active counteroffers" in feedback:
                return await super().choose(context, feedback)
            root = next(
                offer
                for offer in context.observation.trade_window.active_offers
                if offer.parent_offer_id is None and self.color not in offer.declined_by
            )
            index = next(
                index
                for index, action in enumerate(context.legal_actions)
                if action.action_type == ActionType.COUNTER_OFFER
                and action.value.startswith(f"COUNTER_OFFER:{root.id}:")
            )
            return PlayerAttempt(
                context.context_id,
                PlayerChoice(
                    action_index=index,
                    trade_offer=TradeOffer(
                        offered_by=self.color,
                        audience=frozenset({Color.RED}),
                        give=(0, 0, 0, 0, 1),
                        receive=root.give,
                        parent_offer_id=root.id,
                    ),
                ),
            )

    engine = _trade_engine()
    players = {Color.RED: FirstLegalPlayer(Color.RED)}
    players.update({color: CounterPlayer(color) for color in COLORS[1:]})
    sandbox: Any = CatanSandbox(
        engine,
        players,
        communication_policy=NoCommunicationPolicy(),
        retry_policy=RetryPolicy(attempt_limit),
    )
    for amount in (1, 2):
        engine.step(
            Action(Color.RED, ActionType.OFFER_TRADE, _trade_offer(give=(amount, 0, 0, 0, 0)))
        )
        await sandbox.step()
    engine.step(Action(Color.RED, ActionType.OFFER_TRADE, _trade_offer(give=(3, 0, 0, 0, 0))))
    before = engine.snapshot()

    if attempt_limit == 1 or initial_invalid:
        with pytest.raises(PlayerResponseError) as caught:
            await sandbox.step()
        assert caught.value.player == Color.ORANGE
        assert "maximum active counteroffers" in caught.value.attempts[-1].validation_error
        assert engine.revision == len(before.events)
        assert engine.state.actions == before.state.actions
        assert engine.state.trade_window == before.state.trade_window
        assert all(players[color].accepted_choices == 2 for color in COLORS[1:])
    else:
        result = await sandbox.step()
        assert [t.requested_action.action_type for t in result.transitions] == [
            ActionType.COUNTER_OFFER,
            ActionType.COUNTER_OFFER,
            ActionType.REJECT_TRADE,
        ]
        assert all(players[color].accepted_choices == 3 for color in COLORS[1:])
        assert engine.state.trade_window.cap_hits == 0
        assert calls[Color.ORANGE][-1][0] == calls[Color.ORANGE][-2][0]
        assert "maximum active counteroffers" in calls[Color.ORANGE][-1][1]
    assert len(calls[Color.BLUE]) == len(calls[Color.WHITE]) == 3
    assert len(calls[Color.ORANGE]) == 2 + attempt_limit
    withheld = [item for item in sandbox.decision_trace if "withheld" in item.validation_error]
    assert len(withheld) == (2 if attempt_limit == 1 or initial_invalid else 0)
    assert len(sandbox.decision_trace) == 1 + int(initial_invalid) + len(withheld)


@pytest.mark.asyncio
@pytest.mark.parametrize("retry", ["none", "invalid", "valid_then_peer_fails"])
async def test_failed_capacity_barrier_preserves_each_completed_call_once(retry: str) -> None:
    engine: Any = _trade_engine(TradeLimits(max_active_counteroffers=1))
    engine.step(Action(Color.RED, ActionType.OFFER_TRADE, _trade_offer()))
    players: Any = {Color.RED: FirstLegalPlayer(Color.RED)}
    transports: Any = {}
    responses: Any = {}
    for color in COLORS[1:]:
        index: Any = next(
            i for i, action in enumerate(trade_response_actions(engine.state, color))
            if action.action_type == ActionType.COUNTER_OFFER
        )
        contents: Any = [
            f"<action>{index}</action>"
            '<trade_offer>{"give":{"ORE":1},"receive":{"WOOD":2}}</trade_offer>'
        ]
        if color == Color.WHITE and retry != "none":
            contents.append("<action>0</action>" if retry == "valid_then_peer_fails" else "<action>999</action>")
        if color == Color.ORANGE and retry == "valid_then_peer_fails":
            contents.append("<action>999</action>")
        replies: Any = [
            ModelResponse(
                content=content,
                model="offline-audit",
                usage=(("completion_tokens", 10 + number), ("cost", 0.01 * number)),
                provider_response_id=f"{color.value}:{number}",
            )
            for number, content in enumerate(contents, start=1)
        ]
        responses.update((reply.provider_response_id, reply) for reply in replies)
        transports[color] = FixedTransport(replies)
        players[color] = AgentPlayer(
            color, transports[color], session_id=color.value,
            suite=load_context_suite(default_suite_path().with_name("catan_v10.yaml")),
        )
    sandbox: Any = CatanSandbox(
        engine, players, retry_policy=RetryPolicy(1 if retry == "none" else 2),
        communication_policy=NoCommunicationPolicy(),
    )
    sandbox._trade_barrier_contexts()
    before = pickle.dumps(sandbox.snapshot())

    with pytest.raises(PlayerResponseError) as caught:
        await sandbox.step()

    failed_actor = Color.ORANGE if retry == "valid_then_peer_fails" else Color.WHITE
    assert caught.value.player == failed_actor
    assert len(caught.value.attempts) == (1 if retry == "none" else 2)
    assert all(attempt.context_id.endswith(f":{failed_actor.value}") for attempt in caught.value.attempts)
    assert all("withheld" not in attempt.validation_error for attempt in caught.value.attempts)
    assert pickle.dumps(sandbox.snapshot()) == before
    traced: Any = {attempt.model_response.provider_response_id: attempt for attempt in sandbox.decision_trace}
    assert len(traced) == len(sandbox.decision_trace) == len(responses)
    assert set(traced) == set(responses)
    assert all(traced[f"{color.value}:1"].choice is not None for color in COLORS[1:])
    assert "maximum active counteroffers" in traced["WHITE:1"].validation_error
    if retry == "valid_then_peer_fails":
        assert "maximum active counteroffers" in traced["ORANGE:1"].validation_error
    withheld_ids = {key for key, attempt in traced.items() if "withheld" in attempt.validation_error}
    assert withheld_ids == ({"BLUE:1", "WHITE:2"} if retry == "valid_then_peer_fails" else {"BLUE:1", "ORANGE:1"})
    for key, attempt in traced.items():
        assert attempt.validation_error
        assert attempt.model_response == responses[key]
        actor_name, number = key.split(":")
        assert attempt.model_request == transports[Color(actor_name)].requests[int(number) - 1]
        assert attempt.model_response.usage == responses[key].usage
    assert all(player.session.messages == [] and player.session.receipts == {} for player in players.values() if isinstance(player, AgentPlayer))


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["exhausted", "callback", "cancel"])
async def test_barrier_failure_retains_completed_sibling_and_awaits_blocked_children(failure: str) -> None:
    engine: Any = _trade_engine()
    engine.step(Action(Color.RED, ActionType.OFFER_TRADE, _trade_offer()))
    blue_completed, orange_entered, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    never = asyncio.Event()
    active, cleaned = set(), set()
    requests, responses = {}, {}
    callback_error = RuntimeError("WHITE provider failed")

    class ControlledTransport:
        async def complete(self, request: ModelRequest) -> ModelResponse:
            color: Any = Color(request.session_id)
            requests[color] = request
            active.add(color)
            try:
                if color == Color.WHITE:
                    await release.wait()
                    if failure == "callback":
                        raise callback_error
                elif color == Color.ORANGE:
                    orange_entered.set()
                    await never.wait()
                response: Any = ModelResponse(
                    content="<action>999</action>" if color == Color.WHITE else "<action>0</action>",
                    usage=(("completion_tokens", 11), ("cost", 0.01)),
                    provider_response_id=f"completed:{color.value}",
                )
                responses[color] = response
                return response
            finally:
                await asyncio.sleep(0)
                active.remove(color)
                cleaned.add(color)
                if color == Color.BLUE:
                    blue_completed.set()

    transport: Any = ControlledTransport()
    players: Any = {Color.RED: FirstLegalPlayer(Color.RED)}
    players.update({
        color: AgentPlayer(
            color, transport, session_id=color.value,
            suite=load_context_suite(default_suite_path().with_name("catan_v10.yaml")),
        )
        for color in COLORS[1:]
    })
    sandbox: Any = CatanSandbox(
        engine, players, retry_policy=RetryPolicy(1), communication_policy=NoCommunicationPolicy()
    )
    sandbox._trade_barrier_contexts()
    before = pickle.dumps(sandbox.snapshot())
    pending = asyncio.create_task(sandbox.step())
    expected_error = {
        "exhausted": PlayerResponseError, "callback": RuntimeError, "cancel": asyncio.CancelledError,
    }[failure]
    try:
        await asyncio.wait_for(asyncio.gather(blue_completed.wait(), orange_entered.wait()), 1)
        if failure == "cancel":
            pending.cancel()
        else:
            release.set()
        with pytest.raises(expected_error) as caught:
            await asyncio.wait_for(pending, 1)
    finally:
        if not pending.done():
            pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)

    assert active == set() and cleaned == set(COLORS[1:])
    assert pickle.dumps(sandbox.snapshot()) == before
    assert set(requests) == set(COLORS[1:])
    assert set(responses) == ({Color.BLUE, Color.WHITE} if failure == "exhausted" else {Color.BLUE})
    assert len(sandbox.decision_trace) == len(responses)
    for color, response in responses.items():
        attempt: Any = next(item for item in sandbox.decision_trace if item.model_response.provider_response_id == response.provider_response_id)
        assert attempt.model_request == requests[color]
        assert attempt.model_response == response
        assert attempt.model_response.usage == response.usage
        assert ("withheld" in attempt.validation_error) == (color == Color.BLUE)
    if failure == "exhausted":
        assert caught.value.player == Color.WHITE
        assert len(caught.value.attempts) == 1
        assert caught.value.attempts[0].model_response == responses[Color.WHITE]
    elif failure == "callback":
        assert caught.value is callback_error
    else:
        assert "CancelledError" in sandbox.decision_trace[0].validation_error
    await asyncio.sleep(0)
    assert active == set() and len(sandbox.decision_trace) == len(responses)
