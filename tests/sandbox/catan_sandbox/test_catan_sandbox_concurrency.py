"""Overlap rejection and stale choice handling."""
import asyncio
from typing import Any

import pytest

from cle.game_engine.models.enums import Action, ActionType
from cle.game_engine.models.player import Color
from cle.harness.models import ModelRequest, ModelResponse
from cle.harness.suite import default_suite_path, load_context_suite
from cle.players.agent import AgentPlayer
from cle.players.baseline import FirstLegalPlayer
from cle.players.contracts import (
    PlayerAttempt,
    PlayerContext,
)
from cle.sandbox import CatanSandbox
from cle.sandbox.catan import SandboxError

from .support import COLORS, NoCommunicationPolicy, _sandbox, _trade_engine, _trade_offer


@pytest.mark.filterwarnings("ignore:Prompt suite .* is deprecated:DeprecationWarning")
@pytest.mark.asyncio
async def test_step_rejects_overlap_and_restore_until_pending_choice_finishes() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    class PendingTransport:
        async def complete(self, request: ModelRequest) -> ModelResponse:
            entered.set()
            await release.wait()
            return ModelResponse(content="<game_plan>opening</game_plan><action>0</action>")

    red = AgentPlayer(
        Color.RED, PendingTransport(), session_id="pending:RED",
        suite=load_context_suite(default_suite_path().with_name("catan_v10.yaml")),
    )
    sandbox, _ = _sandbox(red)
    sandbox.communication_policy = NoCommunicationPolicy()
    snapshot = sandbox.snapshot()
    pending = asyncio.create_task(sandbox.step())
    await asyncio.wait_for(entered.wait(), 1)

    with pytest.raises(SandboxError, match="in flight"):
        await sandbox.step()
    with pytest.raises(SandboxError, match="in flight"):
        sandbox.restore(snapshot)
    with pytest.raises(SandboxError, match="in flight"):
        sandbox.register_player(FirstLegalPlayer(Color.RED))
    assert sandbox.revision == 0
    assert red.session.messages == []

    release.set()
    await pending
    assert sandbox.revision == 1
    assert len(red.session.receipts) == 1
    sandbox.restore(snapshot)
    assert sandbox.revision == 0
    assert red.session.receipts == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("barrier", [False, True])
@pytest.mark.parametrize("mutation", ["message", "restore"])
async def test_stale_choice_is_rejected_even_if_selected_action_remains_legal(barrier: bool, mutation: str) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    engine: Any = _trade_engine()
    if barrier:
        engine.step(Action(Color.RED, ActionType.OFFER_TRADE, _trade_offer()))
        actor = Color.BLUE
    else:
        actor = Color.RED
    contexts = []

    class PendingPlayer(FirstLegalPlayer):
        async def choose(self, context: PlayerContext, feedback: str | None=None) -> PlayerAttempt:
            contexts.append(context)
            entered.set()
            await release.wait()
            return await super().choose(context, feedback)

    players: Any = {color: FirstLegalPlayer(color) for color in COLORS}
    players[actor] = PendingPlayer(actor)
    sandbox = CatanSandbox(engine, players, communication_policy=NoCommunicationPolicy())
    before = engine.revision
    pending = asyncio.create_task(sandbox.step())
    await asyncio.wait_for(entered.wait(), 1)
    if mutation == "message":
        engine.append_message(
            speaker=Color.RED,
            text="New information",
            audience=COLORS[1:],
            causation_id="external",
        )
    else:
        engine.restore(engine.snapshot())
    assert engine.is_action_valid(contexts[0].legal_actions[0])
    release.set()

    with pytest.raises(SandboxError, match="Stale context"):
        await pending
    assert engine.revision == before + int(mutation == "message")
    assert len(engine.state.actions) == before
    assert all(player.accepted_choices == 0 for player in players.values())
    assert "Stale context" in sandbox.decision_trace[-1].validation_error
