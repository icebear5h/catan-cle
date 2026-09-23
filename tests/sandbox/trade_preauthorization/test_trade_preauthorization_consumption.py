"""Consumption, checkpoints, and cancellation retention."""
import asyncio
import pickle
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from cle.game_engine.models.enums import Action, ActionType
from cle.game_engine.models.player import Color
from cle.game_engine.trading import TradeOfferStatus
from cle.sandbox.catan import PlayerResponseError
from cle.traces.sqlite import SQLiteLiveTraceStore

from .support import COLORS, reply, sandbox


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["proposer_hand", "preferred_hand", "terms", "audience", "withdraw", "expire", "round", "cancel", "response", "offer_id", "window_id", "willingness"])
async def test_stale_state_consumes_without_fallback_or_transfer(change: str) -> None:
    game, transports = sandbox(["BLUE", "WHITE"])
    await game.step()
    await game.step()
    engine = game.game_engine
    window: Any = engine.state.trade_window
    offer: Any = window.active_offers[0]
    if change == "proposer_hand":
        engine.state.player_state["P0_WOOD_IN_HAND"] = 0
    elif change == "preferred_hand":
        engine.state.player_state["P1_ORE_IN_HAND"] = 0
    elif change == "terms":
        offer.give = (2, 0, 0, 0, 0)
    elif change == "audience":
        offer.audience = frozenset({Color.BLUE})
    elif change == "withdraw":
        offer.status = TradeOfferStatus.WITHDRAWN
    elif change == "expire":
        window.close()
    elif change == "round":
        window.advance_round()
    elif change == "cancel":
        engine.step(Action(Color.RED, ActionType.CANCEL_TRADE, offer.id))
    elif change == "response":
        # A post-barrier response is a different response window, even if legal.
        engine.step(Action(Color.BLUE, ActionType.REJECT_TRADE, offer.id), force=True)
    elif change == "offer_id":
        window.offers = {"replacement": replace(offer, id="replacement")}
    elif change == "window_id":
        window.id = "replacement-window"
    elif change == "willingness":
        offer.willing_by.remove(Color.BLUE)
        offer.declined_by.add(Color.BLUE)
    before = {c: dict(engine.observe(c).my_resources) for c in COLORS}
    await game.step()
    assert {c: dict(engine.observe(c).my_resources) for c in COLORS} == before
    assert len(transports[Color.RED].requests) == 2
    assert game.snapshot().trade_preauthorization is None
    assert not any(a.action_type == ActionType.CONFIRM_TRADE for a in engine.state.actions)
    assert any(e.event_type == "TRADE_PREAUTHORIZATION_PAUSED" for e in engine.events)


@pytest.mark.asyncio
async def test_pause_is_consumed_even_when_next_model_call_fails_and_restores() -> None:
    game, transports = sandbox(responses=("reject_offer",) * 3)
    await game.step()
    await game.step()
    transports[Color.RED].contents.insert(0, "invalid")
    with pytest.raises(PlayerResponseError):
        await game.step()
    snapshot = pickle.loads(pickle.dumps(game.snapshot()))
    assert snapshot.trade_preauthorization is None
    assert game.players[Color.RED].session.memory_revision == 1
    game.restore(snapshot)
    await game.step()
    assert sum(e.event_type == "TRADE_PREAUTHORIZATION_PAUSED" for e in game.game_engine.events) == 1
    assert not any(a.action_type == ActionType.CONFIRM_TRADE for a in game.game_engine.state.actions)


@pytest.mark.asyncio
async def test_checkpoint_restore_call_accounting_and_consumption(tmp_path: Path) -> None:
    game, transports = sandbox()
    store = SQLiteLiveTraceStore(tmp_path / "trades.sqlite3")
    store.start_game(game.game_engine.id, config={}, snapshot=game.snapshot())
    results: Any = []
    for index in range(3):
        results.append(await game.step())
        assert store.record_step(
            game.game_engine.id, result=results[-1], rejected_attempts=(),
            public_state={}, snapshot=game.snapshot(),
        ) == index
        if index < 2:
            saved = store.load_snapshot(game.game_engine.id, step_index=index)
            restored, _ = sandbox()
            restored.players = game.players
            restored.restore(saved)
            game: Any = restored
    trace: Any = store.get_game(game.game_engine.id)
    assert trace["step_count"] == 3
    assert len(trace["model_calls"]) == 4  # proposer and three responders only
    assert sum(call["response"]["usage"]["prompt_tokens"] for call in trace["model_calls"]) == 40
    automatic: Any = store.get_step(game.game_engine.id, 2)
    assert automatic["model_calls"] == []
    provenance: Any = automatic["step"]["result"]["automatic_action"]
    assert provenance["origin_context_id"] == results[0].attempts[0].context_id
    assert provenance["provider_response_id"] == "RED:1"
    assert [call["context_id"] for call in automatic["origin_calls"]] == [
        provenance["origin_context_id"]
    ]
    snapshot = store.load_snapshot(game.game_engine.id, step_index=2)
    assert snapshot.trade_preauthorization is None
    game.restore(snapshot)
    await game.step()
    assert len(transports[Color.RED].requests) == 2
    assert sum(a.action_type == ActionType.CONFIRM_TRADE for a in game.game_engine.state.actions) == 1
    assert game.players[Color.RED].session.memory_revision == 2  # offer and later end_turn


@pytest.mark.asyncio
async def test_failed_and_cancelled_barrier_preserves_authorization_until_retry() -> None:
    game, transports = sandbox()
    await game.step()
    pending = game.snapshot().trade_preauthorization
    transports[Color.WHITE].contents.insert(0, "invalid")
    with pytest.raises(PlayerResponseError):
        await game.step()
    assert game.snapshot().trade_preauthorization == pending
    snapshot = pickle.loads(pickle.dumps(game.snapshot()))
    game.restore(snapshot)
    for color in (Color.BLUE, Color.ORANGE):
        transports[color].contents.append(reply("accept_offer", {"player": "RED", "give": {"ORE": 1}, "receive": {"WOOD": 1}}))
    started = asyncio.Event()

    async def block() -> None:
        started.set()
        await asyncio.Future()

    transports[Color.WHITE].before_reply = block
    task = asyncio.create_task(game.step())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert game.snapshot().trade_preauthorization == pending
    for color in (Color.BLUE, Color.ORANGE):
        transports[color].contents.append(reply("accept_offer", {"player": "RED", "give": {"ORE": 1}, "receive": {"WOOD": 1}}))
    transports[Color.WHITE].before_reply = None
    await game.step()
    await game.step()
    assert len(transports[Color.RED].requests) == 1
    assert sum(a.action_type == ActionType.CONFIRM_TRADE for a in game.game_engine.state.actions) == 1
