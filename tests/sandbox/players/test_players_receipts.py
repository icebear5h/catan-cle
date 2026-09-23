"""Receipt caching, detachment, and reasoning traces."""
import json
import pickle
from types import SimpleNamespace
from typing import Any

import pytest

from cle.game_engine.game import GameEngine
from cle.game_engine.models.player import Color
from cle.game_engine.trading import TradeOffer
from cle.harness import ModelResponse, load_context_suite
from cle.players.agent import AgentPlayer
from cle.players.contracts import PlayerAttempt, PlayerChoice, PlayerContext

from .support import COLORS, AsyncFixedTransport, _context


@pytest.mark.asyncio
async def test_agent_receipts_cached_choices_and_snapshots_are_detached(trade_context: PlayerContext) -> None:
    player: Any = AgentPlayer(
        Color.RED, AsyncFixedTransport([]), session_id="snapshot:RED",
        suite=load_context_suite(),
    )
    choice: Any = PlayerChoice(
        0,
        trade_offer=TradeOffer(Color.RED, frozenset(COLORS[1:]), (1, 0, 0, 0, 0), (0, 0, 0, 0, 1)),
        native_reasoning_details=({"text": "original"},),
    )
    attempt = PlayerAttempt(trade_context.context_id, choice)
    player.accept(attempt, SimpleNamespace(context=trade_context, after_revision=1))
    choice.trade_offer.give = (2, 0, 0, 0, 0)
    choice.native_reasoning_details[0]["text"] = "edited"
    receipt = player.session.receipts[trade_context.context_id]
    assert receipt.choice.trade_offer.give == (1, 0, 0, 0, 0)
    # Receipts keep the decision, never the trace: reasoning lives in model_calls.
    assert receipt.choice.native_reasoning_details == ()
    cached = await player.choose(trade_context)
    assert cached.choice.native_reasoning_details == ()
    cached.choice.trade_offer.give = (3, 0, 0, 0, 0)
    assert receipt.choice.trade_offer.give == (1, 0, 0, 0, 0)
    snapshot: Any = player.snapshot()
    receipt.choice.trade_offer.give = (4, 0, 0, 0, 0)
    player.restore(snapshot)
    assert player.session.receipts[trade_context.context_id].choice.trade_offer.give == (1, 0, 0, 0, 0)
    snapshot.session.receipts[0][1].choice.trade_offer.give = (5, 0, 0, 0, 0)
    assert player.session.receipts[trade_context.context_id].choice.trade_offer.give == (1, 0, 0, 0, 0)


@pytest.mark.asyncio
async def test_agent_detaches_mutable_provider_response_metadata() -> None:
    engine = GameEngine(COLORS, seed=9, shuffle_players=False)
    response: Any = ModelResponse(
        content=json.dumps({
            "game_plan": "expand",
            "tool": "build_settlement",
            "arguments": {"node": f"<N{engine.state.playable_actions[0].value:02d}>"},
        }),
        native_reasoning_details=({"text": "original"},),
    )
    player = AgentPlayer(
        Color.RED, AsyncFixedTransport([response]), session_id="provider:RED",
        suite=load_context_suite(),
    )
    attempt: Any = await player.choose(_context(engine))
    response.native_reasoning_details[0]["text"] = "edited"
    assert attempt.model_response.native_reasoning_details == ({"text": "original"},)
    assert attempt.choice.native_reasoning_details == ({"text": "original"},)


@pytest.mark.asyncio
async def test_receipts_keep_the_decision_and_drop_the_reasoning_trace(trade_context: PlayerContext) -> None:
    player = AgentPlayer(
        Color.RED, AsyncFixedTransport([]), session_id="slim:RED",
        suite=load_context_suite(),
    )
    reasoning = "REASONING-SENTINEL " * 2000
    choice: Any = PlayerChoice(
        0,
        trade_offer=TradeOffer(Color.RED, frozenset(COLORS[1:]), (1, 0, 0, 0, 0), (0, 0, 0, 0, 1)),
        game_plan="expand toward ore",
        notes_update="need brick",
        knight_destination=(0, 1, -1),
        raw_response="{...}",
        native_reasoning=reasoning,
        native_reasoning_details=({"text": reasoning},),
        reasoning_request=(("effort", "high"),),
        usage=(("prompt_tokens", 157147),),
        provider_response_id="gen-1",
    )
    player.accept(
        PlayerAttempt(trade_context.context_id, choice),
        SimpleNamespace(context=trade_context, after_revision=1),
    )

    receipt: Any = player.session.receipts[trade_context.context_id].choice
    assert receipt.action_index == 0
    assert receipt.trade_offer == choice.trade_offer
    assert receipt.game_plan == "expand toward ore"
    assert receipt.notes_update == "need brick"
    assert receipt.knight_destination == (0, 1, -1)
    assert receipt.provider_response_id == "gen-1"
    assert receipt.native_reasoning == ""
    assert receipt.native_reasoning_details == ()
    assert receipt.reasoning_request == ()
    assert receipt.raw_response == ""
    assert receipt.usage == ()

    # Redelivery answers with the same decision, and the session snapshot no
    # longer carries the model's reasoning at all.
    cached: Any = await player.choose(trade_context)
    assert cached.choice.action_index == 0
    assert cached.choice.knight_destination == (0, 1, -1)
    snapshot_bytes = pickle.dumps(player.snapshot())
    assert b"REASONING-SENTINEL" not in snapshot_bytes
    assert len(snapshot_bytes) < 8 * 1024
