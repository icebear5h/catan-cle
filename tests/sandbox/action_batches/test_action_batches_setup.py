"""Setup pairs, snake reversal, and unlock ordering."""
import pickle
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from cle.game_engine.models.enums import ActionType
from cle.game_engine.models.player import Color
from cle.traces.sqlite import SQLiteLiveTraceStore
from playground.game_viewer.live.reasoning_trace import build_live_reasoning_traces

from .support import batch, conversions, main_engine, pair, road_unlock, sandbox


@pytest.mark.asyncio
@pytest.mark.parametrize("pair_index", range(8))
async def test_every_setup_pair_has_one_call_and_two_checkpoints(pair_index: int) -> None:
    game, transports = sandbox()
    engine = game.game_engine
    for _ in range(pair_index * 2):
        engine.step(engine.state.playable_actions[0])
    actor = game.current_actor()
    settlement, road = pair(engine)
    expected = deepcopy(engine)
    expected.step(settlement)
    expected.step(road)
    rng = engine.rng.getstate()
    transports[actor].contents = [batch(settlement, road, notes="Build this pair")]
    first: Any = await game.step()
    accepted = deepcopy(game.snapshot().player_states)
    pending = pickle.loads(pickle.dumps(game.snapshot()))
    assert pending.pending_action_batch.next_index == 1
    game.restore(pending)
    game._refresh_players = lambda _: pytest.fail("Queued action must not refresh or call policy")
    second = await game.step()
    assert first.transitions[0].requested_action == settlement
    assert second.transitions[0].requested_action == road
    assert road.value[0] == settlement.value or road.value[1] == settlement.value
    assert len(transports[actor].requests) == 1
    assert second.contexts == second.attempts == ()
    assert game.snapshot().player_states == accepted
    assert game.snapshot().pending_action_batch is None
    assert engine.observe(actor).my_resources == expected.observe(actor).my_resources
    assert engine.observe(actor).my_settlements == expected.observe(actor).my_settlements
    assert engine.observe(actor).my_roads == expected.observe(actor).my_roads
    assert engine.rng.getstate() == rng
    assert build_live_reasoning_traces(game, second) == []
    assert build_live_reasoning_traces(game, first)[0]["batch_actions"] == list(first.attempts[0].choice.batch_actions)
    text = str(transports[actor].requests[0].messages)
    assert "YOUR CURRENTLY LEGAL TOOLS" not in text


@pytest.mark.asyncio
async def test_snake_reversal_does_not_chain_both_setup_pairs() -> None:
    game, transports = sandbox()
    for _ in range(6):
        game.game_engine.step(game.game_engine.state.playable_actions[0])
    actor = game.current_actor()
    first_pair = pair(game.game_engine)
    staged = deepcopy(game.game_engine)
    for action in first_pair:
        staged.step(action)
    assert staged.state.current_color() == actor
    second_pair = pair(staged)
    transports[actor].contents = [batch(*first_pair, *second_pair), batch(*second_pair)]
    await game.step()
    result = await game.step()
    assert game.current_actor() == actor
    assert game.snapshot().pending_action_batch is None
    assert result.after_revision == game.revision
    assert "each pair" in game.game_engine.events[-1].public_payload["reason"]
    assert len(game.game_engine.observe(actor).my_settlements) == 1
    await game.step()
    assert len(transports[actor].requests) == 2
    assert len(game.game_engine.observe(actor).my_settlements) == 2


@pytest.mark.asyncio
async def test_road_unlocks_settlement_and_end_turn_is_terminal() -> None:
    engine = main_engine(WOOD=6, BRICK=6, SHEEP=2, WHEAT=2)
    road, settlement = road_unlock(engine)
    assert settlement not in engine.state.playable_actions
    game, transports = sandbox(engine)
    transports[Color.RED].contents = [batch(road, settlement, {"tool": "end_turn", "arguments": {}})]
    expected = deepcopy(engine)
    expected.step(road)
    expected.step(settlement)
    await game.step()
    await game.step()
    assert engine.observe(Color.RED).my_resources == expected.observe(Color.RED).my_resources
    end = await game.step()
    assert end.transitions[0].requested_action.action_type == ActionType.END_TURN
    assert game.current_actor() == Color.BLUE
    assert game.snapshot().pending_action_batch is None
    assert len(transports[Color.RED].requests) == 1


@pytest.mark.asyncio
async def test_conversions_city_persist_once_and_count_one_model_call(tmp_path: Path) -> None:
    engine: Any = main_engine(WOOD=8, WHEAT=1, ORE=2)
    actions: Any = conversions(engine)
    assert not any(a.action_type == ActionType.BUILD_CITY for a in engine.state.playable_actions)
    game, transports = sandbox(engine)
    transports[Color.RED].contents = [batch(*actions, notes="One update")]
    store: Any = SQLiteLiveTraceStore(tmp_path / "batches.sqlite3")
    store.start_game(engine.id, config={}, snapshot=game.snapshot())
    expected: Any = deepcopy(engine)
    first_context: Any = None
    for index, action in enumerate(actions):
        result: Any = await game.step()
        expected.step(action)
        assert engine.observe(Color.RED).my_resources == expected.observe(Color.RED).my_resources
        store.record_step(engine.id, result=result, rejected_attempts=(), public_state={}, snapshot=game.snapshot())
        game.restore(store.load_snapshot(engine.id, step_index=index))
        if index == 0:
            first_context = result.attempts[0].context_id
        else:
            provenance: Any = store.get_step(engine.id, index)["step"]["result"]["automatic_action"]
            assert provenance["origin_context_id"] == first_context
            assert provenance["provider_request_id"] == "request:1"
            assert provenance["action_number"] == index + 1
    trace: Any = store.get_game(engine.id)
    assert trace["step_count"] == 3
    assert len(trace["model_calls"]) == 1
    assert trace["model_calls"][0]["response"]["usage"]["prompt_tokens"] == 10
    player: Any = game.players[Color.RED]
    assert player.session.memory_revision == 1
    assert player.session.strategic_memory == "One update"
    assert player.session.action_next_sequence == 16
    assert game.snapshot().pending_action_batch is None
