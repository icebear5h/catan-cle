"""Barrier priority, seat order, and control return."""
import asyncio
import pickle
from collections.abc import Awaitable, Callable
from copy import deepcopy
from typing import Any

import pytest

from cle.game_engine.models.enums import ActionType
from cle.game_engine.models.player import Color
from playground.game_viewer.live.reasoning_trace import build_live_reasoning_traces

from .support import COLORS, sandbox, totals


@pytest.mark.asyncio
@pytest.mark.parametrize("priority,expected", [(["ORANGE", "BLUE"], Color.ORANGE), ("ANY", Color.BLUE)])
@pytest.mark.parametrize("completion_order", [COLORS[1:], tuple(reversed(COLORS[1:]))])
async def test_full_barrier_priority_is_independent_of_latency(
    priority: list[str] | str, expected: Color, completion_order: tuple[Color, ...]
) -> None:
    game, transports = sandbox(priority)
    offered = await game.step()
    origin: Any = offered.attempts[0]
    before = pickle.dumps(game.game_engine.snapshot())
    all_started = set()
    done = {color: asyncio.Event() for color in COLORS[1:]}

    def delayed(color: Color) -> Callable[[], Awaitable[None]]:
        async def wait() -> None:
            all_started.add(color)
            while len(all_started) < 3:
                await asyncio.sleep(0)
            index = completion_order.index(color)
            if index:
                await done[completion_order[index - 1]].wait()
            # Replies see the same pre-barrier state, with no early transfer.
            assert pickle.dumps(game.game_engine.snapshot()) == before
            done[color].set()
        return wait

    for color in COLORS[1:]:
        transports[color].before_reply = delayed(color)
    barrier = await game.step()
    assert len(barrier.attempts) == len(barrier.transitions) == 3
    assert all(t.requested_action.action_type == ActionType.ACCEPT_TRADE for t in barrier.transitions)
    assert not any(a.action_type == ActionType.CONFIRM_TRADE for a in game.game_engine.state.actions)
    players_before = deepcopy(game.snapshot().player_states)
    inventory = totals(game.game_engine)
    rng = game.game_engine.rng.getstate()
    result: Any = await game.step()
    transition = result.transitions[0]
    assert transition.requested_action.action_type == ActionType.CONFIRM_TRADE
    assert transition.requested_action.value.counterparty == expected
    assert transition.events[0].event_type == "CONFIRM_TRADE"
    assert result.contexts == result.attempts == ()
    assert result.automatic_action.authorization.origin_context_id == origin.context_id
    assert result.automatic_action.authorization.provider_response_id == "RED:1"
    assert game.snapshot().trade_preauthorization is None
    assert game.snapshot().player_states == players_before
    assert len(transports[Color.RED].requests) == 1
    assert build_live_reasoning_traces(game, result) == []
    assert totals(game.game_engine) == inventory
    assert game.game_engine.rng.getstate() == rng
    assert game.game_engine.observe(Color.RED).my_resources["WOOD"] == 2
    assert game.game_engine.observe(expected).my_resources["WOOD"] == 4
    for color in COLORS:
        event = game.game_engine.project_events(color)[-1]
        assert event.event_type == "CONFIRM_TRADE"
        assert "confirm_if_accepted_by" not in str(event.payload)


@pytest.mark.asyncio
@pytest.mark.parametrize("priority,responses,reason", [
    (None, ("accept_offer",) * 3, None),
    (None, ("reject_offer",) * 3, None),
    ("ANY", ("reject_offer",) * 3, "Nobody permitted"),
    (["BLUE"], ("reject_offer", "accept_offer", "accept_offer"), "Nobody permitted"),
    (["BLUE"], ("accept_offer", "counter_offer", "reject_offer"), "Counteroffer arrived"),
])
async def test_probe_or_unmatched_condition_returns_model_control(
    priority: str | list[str] | None, responses: tuple[str, ...], reason: str | None
) -> None:
    game, transports = sandbox(priority, responses)
    await game.step()
    await game.step()
    result = await game.step()
    assert result.transitions[0].requested_action.action_type == ActionType.END_TURN
    assert len(transports[Color.RED].requests) == 2
    assert not any(a.action_type == ActionType.CONFIRM_TRADE for a in game.game_engine.state.actions)
    paused: Any = [e for e in game.game_engine.events if e.event_type == "TRADE_PREAUTHORIZATION_PAUSED"]
    assert len(paused) == bool(reason)
    if reason:
        assert reason in paused[0].public_payload["reason"]
        assert reason in str(transports[Color.RED].requests[-1].messages)
        assert all(
            not any(e.event_type == "TRADE_PREAUTHORIZATION_PAUSED" for e in game.game_engine.project_events(c))
            for c in COLORS[1:]
        )
    assert game.snapshot().trade_preauthorization is None


@pytest.mark.asyncio
async def test_any_uses_engine_seat_order_not_alphabetical_order() -> None:
    game, _ = sandbox(seat_order=(Color.RED, Color.WHITE, Color.ORANGE, Color.BLUE))
    await game.step()
    await game.step()
    result = await game.step()
    assert result.transitions[0].requested_action.value.counterparty == Color.WHITE


@pytest.mark.asyncio
@pytest.mark.parametrize("priority", [["WHITE"], ["BLUE", "WHITE"]])
async def test_first_permitted_willing_player_is_selected(priority: list[str]) -> None:
    game, _ = sandbox(priority, responses=("reject_offer", "accept_offer", "accept_offer"))
    await game.step()
    await game.step()
    result = await game.step()
    assert result.transitions[0].requested_action.value.counterparty == Color.WHITE
