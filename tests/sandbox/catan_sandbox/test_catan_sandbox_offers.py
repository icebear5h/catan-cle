"""Offer retries and round-limit menu refresh."""
from typing import Any

import pytest

from cle.game_engine.models.actions import generate_playable_actions
from cle.game_engine.models.enums import Action, ActionType
from cle.game_engine.models.player import Color
from cle.game_engine.trading import TradeLimits
from cle.harness.models import ModelResponse
from cle.harness.suite import default_suite_path, load_context_suite
from cle.players.agent import AgentPlayer
from cle.players.baseline import FirstLegalPlayer
from cle.players.contracts import (
    CommunicationChoice,
    TalkContext,
)
from cle.sandbox import CatanSandbox, RetryPolicy
from cle.sandbox.catan import PlayerResponseError

from .support import COLORS, FixedTransport, NoCommunicationPolicy, _trade_engine, _trade_offer


@pytest.mark.asyncio
@pytest.mark.parametrize("attempt_limit", [1, 2])
@pytest.mark.parametrize("give_count,error", [(1, "Equivalent offer"), (9, "affordable")])
async def test_invalid_offer_retries_with_error_and_only_commits_valid_response(
    attempt_limit: int,
    give_count: int,
    error: str,
) -> None:
    engine: Any = _trade_engine()
    offer = engine.step(
        Action(Color.RED, ActionType.OFFER_TRADE, _trade_offer())
    ).resolved_action.value
    for color in COLORS[1:]:
        engine.step(Action(color, ActionType.REJECT_TRADE, offer.id))
    index = next(
        i
        for i, a in enumerate(engine.state.playable_actions)
        if a.action_type == ActionType.OFFER_TRADE
    )
    bad = ModelResponse(
        content=(
            f"<game_plan>rejected</game_plan><action>{index}</action>"
            f'<trade_offer>{{"give":{{"WOOD":{give_count}}},'
            '"receive":{"ORE":1}}</trade_offer>'
        )
    )
    transport = FixedTransport(
        [
            bad,
            ModelResponse(content="<game_plan>accepted</game_plan><action>0</action>"),
        ]
    )
    red = AgentPlayer(
        Color.RED, transport, session_id="duplicate:RED",
        suite=load_context_suite(default_suite_path().with_name("catan_v10.yaml")),
    )
    players: Any = {color: FirstLegalPlayer(color) for color in COLORS}
    players[Color.RED] = red
    sandbox = CatanSandbox(
        engine,
        players,
        communication_policy=NoCommunicationPolicy(),
        retry_policy=RetryPolicy(attempt_limit),
    )
    before = engine.snapshot()

    if attempt_limit == 1:
        with pytest.raises(PlayerResponseError) as caught:
            await sandbox.step()
        assert error in caught.value.validation_error
        assert engine.revision == len(before.events)
        assert engine.state.trade_window == before.state.trade_window
        assert red.session.messages == []
        assert red.session.receipts == {}
    else:
        await sandbox.step()
        assert red.session.strategic_memory == "accepted"
        assert len(red.session.receipts) == 1
        assert len(red.session.messages) == 2
        assert red.session.messages[-1].content == (
            "<game_plan>accepted</game_plan><action>0</action>"
        )
        assert error in transport.requests[1].messages[-1].content
    rejected: Any = sandbox.decision_trace[0]
    assert rejected.model_response == bad
    assert rejected.model_response is not bad
    assert rejected.model_request == transport.requests[0]
    assert rejected.model_request is not transport.requests[0]
    assert error in rejected.validation_error
    assert len(transport.requests) == attempt_limit


@pytest.mark.asyncio
async def test_round_limit_refreshes_menu_and_post_barrier_reactions_share_cutoff() -> None:
    contexts = {}

    class CapturePlayer(FirstLegalPlayer):
        async def communicate(self, context: TalkContext) -> CommunicationChoice:
            contexts[self.color] = context
            return CommunicationChoice()

    engine: Any = _trade_engine(TradeLimits(max_negotiation_rounds=1))
    engine.step(Action(Color.RED, ActionType.OFFER_TRADE, _trade_offer()))
    sandbox = CatanSandbox(engine, {color: CapturePlayer(color) for color in COLORS})

    result = await sandbox.step()

    assert engine.state.trade_window.round == 1
    assert engine.state.playable_actions == generate_playable_actions(engine.state)
    assert all(a.action_type != ActionType.OFFER_TRADE for a in engine.state.playable_actions)
    assert set(contexts) == set(COLORS)
    cutoff = result.transitions[-1].events[-1].sequence
    assert cutoff == 3
    assert all(context.visible_through_sequence == cutoff for context in contexts.values())
    assert all(context.game_events[-1].sequence == cutoff for context in contexts.values())
    assert contexts[Color.RED].cause.actor == Color.BLUE
    assert contexts[Color.BLUE].cause.actor == Color.WHITE
