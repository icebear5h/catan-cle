"""Player protocol, transport, and prompt boundaries."""
import asyncio
import json
import pickle
from typing import Any

import httpx
import pytest

from cle.game_engine.models.enums import ActionType
from cle.game_engine.models.player import Color
from cle.game_engine.trading import TradeOffer
from cle.harness.communication import default_communication_suite_path
from cle.harness.models import ModelMessage, ModelRequest
from cle.harness.providers import (
    OpenRouterConfig,
    OpenRouterTransport,
    VLLMConfig,
    VLLMTransport,
)
from cle.harness.suite import default_suite_path, load_context_suite
from cle.players.agent import AgentPlayer
from cle.players.baseline import FirstLegalPlayer
from cle.players.contracts import (
    PlayerAttempt,
    PlayerChoice,
    PlayerContext,
)
from cle.sandbox import CatanSandbox
from cle.sandbox.communication import CommunicationPolicy
from cle.sandbox.factory import LiveSandboxConfig, create_live_sandbox

from .support import COLORS, BoundaryDefect, LocalTransport


@pytest.mark.parametrize(
    "kind", ["attempt-none", "choice-dict", "offer-dict", "offer-list"],
)
@pytest.mark.asyncio
async def test_custom_player_protocol_violation_is_rejected_or_retried(
    trade_sandbox: CatanSandbox, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """Protocol-violating returns must now receive typed rejection or retry."""
    sandbox: Any = trade_sandbox
    original_choose = sandbox.players[Color.RED].choose
    feedbacks = []

    async def choose(context: PlayerContext, feedback: str | None=None) -> PlayerAttempt:
        feedbacks.append(feedback)
        if len(feedbacks) > 1:
            assert pickle.dumps(sandbox.game_engine.snapshot()) == before_engine
            assert tuple(player.snapshot() for player in sandbox.players.values()) == before_players
            return await original_choose(context, feedback)
        if kind == "attempt-none":
            return None
        if kind == "choice-dict":
            return PlayerAttempt(context.context_id, {"action_index": 0})
        index: Any = next(
            i
            for i, action in enumerate(context.legal_actions)
            if action.action_type == ActionType.OFFER_TRADE
        )
        offer: Any = (
            {"give": {"WOOD": 1}, "receive": {"ORE": 1}}
            if kind == "offer-dict"
            else TradeOffer(Color.RED, frozenset(COLORS[1:]), (1, 0, 0, 0, 0), (0, 0, 0, 0, 1))
        )
        if kind == "offer-list":
            offer.give = [1, 0, 0, 0, 0]
        return PlayerAttempt(context.context_id, PlayerChoice(index, trade_offer=offer))

    monkeypatch.setattr(sandbox.players[Color.RED], "choose", choose)
    # Materialize lazy observation entries before measuring player-output effects.
    sandbox.decision_context()
    before_engine = pickle.dumps(sandbox.game_engine.snapshot())
    before_players = tuple(player.snapshot() for player in sandbox.players.values())
    result = await sandbox.step()
    assert len(feedbacks) == 2 and feedbacks[1]
    assert len(sandbox.decision_trace) == 1
    assert sandbox.decision_trace[0].validation_error
    assert sandbox.decision_trace[0].choice is None
    assert result.transitions[0].requested_action.action_type == ActionType.END_TURN
    assert sandbox.players[Color.RED].accepted_choices == 1


@pytest.mark.asyncio
async def test_random_mode_must_not_invoke_injected_transport() -> None:
    transport = LocalTransport(["<action>0</action>"])
    sandbox = create_live_sandbox(
        LiveSandboxConfig(
            mode="random",
            seed=7,
            shuffle_players=False,
            palette="canonical_four",
            context_suite_path=str(default_suite_path()),
            communication_suite_path=str(default_communication_suite_path()),
        ),
        transport=transport,
    )
    await sandbox.step()
    assert all(type(sandbox.players[c]) is FirstLegalPlayer for c in COLORS[1:])
    if transport.requests:
        assert len(transport.requests) == 1
        assert isinstance(sandbox.players[Color.RED], AgentPlayer)
        raise BoundaryDefect("Random-mode first step invoked the injected model transport")
    assert type(sandbox.players[Color.RED]) is FirstLegalPlayer


@pytest.mark.asyncio
async def test_decision_prompt_must_retain_visible_table_talk_and_commitments(trade_sandbox: CatanSandbox) -> None:
    sandbox = trade_sandbox
    engine = sandbox.game_engine
    markers = ("AUDIT_PRIVATE_TALK", "AUDIT_COMMITMENT_PROMISE")
    engine.append_message(
        speaker=Color.BLUE,
        text=markers[0],
        audience=(Color.RED,),
        causation_id="audit:private-bribe",
        commitment=("Spare BLUE", markers[1], 9),
    )
    assert engine.project_messages(Color.WHITE) == ()
    assert engine.active_commitments(Color.WHITE) == ()
    transport = LocalTransport([
        "<message>SILENCE</message>",
        '{"game_plan":"wait","tool":"end_turn","arguments":{}}',
    ])
    sandbox.register_player(AgentPlayer(
        Color.RED, transport, session_id="audit:RED", suite=load_context_suite(),
    ))
    sandbox.communication_policy = CommunicationPolicy()

    await sandbox.step()

    assert len(transport.requests) == 2
    talk, decision = transport.requests
    assert ":talk:" in talk.decision_id and ":talk:" not in decision.decision_id
    talk_text = "\n".join(message.content for message in talk.messages)
    decision_text = "\n".join(message.content for message in decision.messages)
    assert all(marker in talk_text for marker in markers)
    if any(marker not in decision_text for marker in markers):
        assert all(marker not in decision_text for marker in markers)
        raise BoundaryDefect("Both private message and active promise disappeared before choice")


@pytest.mark.parametrize("provider", ["openrouter", "vllm"])
@pytest.mark.asyncio
async def test_shared_provider_isolates_sessions_and_preserves_borrowed_client(provider: str) -> None:
    seen = {}
    both_entered, cancel_entered, never = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        session = request.headers["x-session-id"]
        seen[session] = json.loads(request.content)["messages"]
        if session in {"RED", "BLUE"}:
            if {"RED", "BLUE"}.issubset(seen):
                both_entered.set()
            await both_entered.wait()
        if session == "cancel":
            cancel_entered.set()
            await never.wait()
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "<action>0</action>"}}]}
        )

    def request(session: str) -> ModelRequest:
        return ModelRequest(session, session, (ModelMessage("user", f"{session} private context"),))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        transport = (
            OpenRouterTransport(
                OpenRouterConfig(
                    model="audit", endpoint="https://provider.invalid/v1", max_retries=0
                ),
                api_key="audit-not-a-secret",
                client=client,
            )
            if provider == "openrouter"
            else VLLMTransport(
                VLLMConfig(model="audit", base_url="https://provider.invalid/v1", max_retries=0),
                client=client,
            )
        )
        responses = await asyncio.wait_for(
            asyncio.gather(transport.complete(request("RED")), transport.complete(request("BLUE"))),
            2,
        )
        assert [response.content for response in responses] == ["<action>0</action>"] * 2
        pending = asyncio.create_task(transport.complete(request("cancel")))
        try:
            await asyncio.wait_for(cancel_entered.wait(), 2)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        finally:
            if not pending.done():
                pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
        await transport.aclose()
        assert not client.is_closed
        assert (await transport.complete(request("after-cancel"))).content == "<action>0</action>"
        assert seen == {
            session: [{"role": "user", "content": f"{session} private context"}]
            for session in ("RED", "BLUE", "cancel", "after-cancel")
        }
