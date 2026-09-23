"""Continuation authorization and detachment."""
import json
import pickle
from typing import Any

import pytest

from cle.game_engine.models.enums import ActionType
from cle.game_engine.models.player import Color
from cle.harness.shared_suite import load_shared_prompt_suite
from cle.players.agent import AgentPlayer
from cle.sandbox.catan import PostActionCommunicationCancelled

from .support import batch, conversions, main_engine, pair, sandbox


@pytest.mark.asyncio
@pytest.mark.parametrize("rate", [2, 3, 4])
async def test_real_port_and_bank_conversions_then_city(rate: int) -> None:
    engine = main_engine(port_rate=rate, WOOD=rate * 2, WHEAT=1, ORE=2)
    actions = conversions(engine)
    assert len([card for card in actions[0].value[:4] if card]) == rate
    game, transports = sandbox(engine)
    transports[Color.RED].contents = [batch(*actions)]
    before_bank = tuple(engine.state.resource_freqdeck)
    for _ in actions:
        await game.step()
    hand = engine.observe(Color.RED).my_resources
    assert all(count == 0 for count in hand.values())
    assert engine.state.resource_freqdeck[0] == before_bank[0] + rate * 2
    assert len(engine.observe(Color.RED).my_cities) == 1
    assert len(transports[Color.RED].requests) == 1


@pytest.mark.asyncio
async def test_prompt_rebinding_cannot_reinterpret_pending_accepted_plan() -> None:
    game, transports = sandbox()
    settlement, road = pair(game.game_engine)
    transports[Color.RED].contents = [batch(settlement, road)]
    await game.step()
    old: Any = game.players[Color.RED]
    bundle: Any = load_shared_prompt_suite().model_copy(update={"deterministic_batches": False})
    replacement = AgentPlayer(Color.RED, transports[Color.RED], session_id=old.session.session_id,
                              suite=bundle.decision_suite(), communication_suite=bundle.communication_suite())
    replacement.restore(old.snapshot())
    game.register_player(replacement)
    result = await game.step()
    assert result.transitions[0].requested_action == road
    assert len(transports[Color.RED].requests) == 1


@pytest.mark.asyncio
async def test_automatic_result_and_snapshot_are_detached_from_pending_plan() -> None:
    engine = main_engine(WOOD=8, WHEAT=1, ORE=2)
    game, transports = sandbox(engine)
    actions = conversions(engine)
    transports[Color.RED].contents = [batch(*actions)]
    await game.step()
    second: Any = await game.step()
    second.automatic_action.batch.actions[2]["arguments"]["node"] = "<N99>"
    saved: Any = game.snapshot()
    saved.pending_action_batch.actions[2]["arguments"]["node"] = "<N98>"
    third = await game.step()
    assert third.transitions[0].requested_action == actions[2]
    assert len(transports[Color.RED].requests) == 1


@pytest.mark.asyncio
async def test_stale_cached_menu_cannot_authorize_an_unfunded_continuation() -> None:
    engine = main_engine(WOOD=8, WHEAT=1, ORE=2)
    game, transports = sandbox(engine)
    actions = conversions(engine)
    transports[Color.RED].contents = [batch(*actions), json.dumps({"tool": "end_turn", "arguments": {}})]
    await game.step()
    assert actions[1] in engine.state.playable_actions
    # Deliberately stale derived menu: continuation must regenerate from holdings.
    remaining = engine.state.player_state["P0_WOOD_IN_HAND"]
    engine.state.player_state["P0_WOOD_IN_HAND"] = 0
    engine.state.resource_freqdeck[0] += remaining
    result = await game.step()
    assert result.automatic_action is None
    assert engine.observe(Color.RED).my_resources["ORE"] == 2
    assert not engine.observe(Color.RED).my_cities
    assert game.snapshot().pending_action_batch is None
    assert len(transports[Color.RED].requests) == 2


@pytest.mark.asyncio
async def test_cancelled_middle_continuation_restores_next_action_not_previous(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = main_engine(WOOD=8, WHEAT=1, ORE=2)
    game, transports = sandbox(engine)
    actions = conversions(engine)
    transports[Color.RED].contents = [batch(*actions)]
    await game.step()
    original = game._post_action_communication

    async def cancelled(result: object, **kwargs: object) -> None:
        raise PostActionCommunicationCancelled(result)

    monkeypatch.setattr(game, "_post_action_communication", cancelled)
    with pytest.raises(PostActionCommunicationCancelled) as caught:
        await game.step()
    assert caught.value.result.automatic_action.batch.next_index == 1
    saved = pickle.loads(pickle.dumps(game.snapshot()))
    assert saved.pending_action_batch.next_index == 2
    game.restore(saved)
    monkeypatch.setattr(game, "_post_action_communication", original)
    result = await game.step()
    assert result.transitions[0].requested_action == actions[2]
    assert sum(a.action_type == ActionType.MARITIME_TRADE for a in engine.state.actions) == 2
    assert len(transports[Color.RED].requests) == 1
