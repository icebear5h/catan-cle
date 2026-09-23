"""Queue interruption, cancellation, and prefix safety."""
import json
import pickle

import pytest

from cle.game_engine.models.actions import generate_playable_actions
from cle.game_engine.models.enums import ActionPrompt, ActionType
from cle.game_engine.models.player import Color
from cle.sandbox.catan import PlayerResponseError, PostActionCommunicationCancelled

from .support import batch, call, conversions, main_engine, pair, sandbox


@pytest.mark.asyncio
async def test_invalid_second_action_preserves_prefix_and_consumes_before_retry_restore() -> None:
    game, transports = sandbox()
    settlement, road = pair(game.game_engine)
    wrong_road = {"tool": "build_road", "arguments": {"edge": "<E98_99>"}}
    transports[Color.RED].contents = [batch(settlement, wrong_road), "bad response", json.dumps(call(road))]
    await game.step()
    with pytest.raises(PlayerResponseError):
        await game.step()
    assert len(game.game_engine.state.actions) == 1
    assert game.snapshot().pending_action_batch is None
    assert game.players[Color.RED].session.memory_revision == 1
    game.restore(pickle.loads(pickle.dumps(game.snapshot())))
    await game.step()
    assert len(game.game_engine.state.actions) == 2
    assert sum(e.event_type == "ACTION_BATCH_PAUSED" for e in game.game_engine.events) == 1
    assert "Committed prefix remains" in str(transports[Color.RED].requests[-1].messages)
    assert all(e.event_type != "ACTION_BATCH_PAUSED" for e in game.game_engine.project_events(Color.BLUE))


@pytest.mark.asyncio
async def test_new_speech_interrupts_queue_without_acknowledging_unseen_events() -> None:
    game, transports = sandbox()
    settlement, road = pair(game.game_engine)
    transports[Color.RED].contents = [batch(settlement, road), json.dumps(call(road))]
    await game.step()
    assert game.players[Color.RED].session.action_next_sequence == 0
    game.game_engine.append_message(speaker=Color.BLUE, text="Please reconsider", audience=(Color.RED, Color.WHITE, Color.ORANGE), causation_id="external-speech")
    result = await game.step()
    assert result.automatic_action is None
    assert len(transports[Color.RED].requests) == 2
    assert "Please reconsider" in str(transports[Color.RED].requests[-1].messages)
    assert game.players[Color.RED].session.action_next_sequence == 3


@pytest.mark.asyncio
async def test_winning_city_discards_remainder_immediately() -> None:
    engine = main_engine(WOOD=8, WHEAT=1, ORE=2)
    engine.vps_to_win = 3
    game, transports = sandbox(engine)
    transports[Color.RED].contents = [batch(*conversions(engine), {"tool": "end_turn", "arguments": {}})]
    await game.step()
    await game.step()
    won = await game.step()
    assert won.winner == Color.RED
    assert game.snapshot().pending_action_batch is None
    assert engine.state.actions[-1].action_type == ActionType.BUILD_CITY
    assert len(transports[Color.RED].requests) == 1


@pytest.mark.asyncio
async def test_post_commit_cancellation_keeps_queue_and_never_repeats_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    game, transports = sandbox()
    settlement, road = pair(game.game_engine)
    transports[Color.RED].contents = [batch(settlement, road)]
    original = game._post_action_communication

    async def cancelled(result: object, **kwargs: object) -> None:
        raise PostActionCommunicationCancelled(result)

    monkeypatch.setattr(game, "_post_action_communication", cancelled)
    with pytest.raises(PostActionCommunicationCancelled):
        await game.step()
    game.restore(pickle.loads(pickle.dumps(game.snapshot())))
    monkeypatch.setattr(game, "_post_action_communication", original)
    await game.step()
    assert len(game.game_engine.state.actions) == 2
    assert len(transports[Color.RED].requests) == 1
    assert game.players[Color.RED].session.memory_revision == 1


@pytest.mark.asyncio
async def test_paid_prefix_failure_never_charges_or_replays_twice() -> None:
    engine = main_engine(WOOD=1, BRICK=1)
    road = next(a for a in engine.state.playable_actions if a.action_type == ActionType.BUILD_ROAD)
    game, transports = sandbox(engine)
    # Repeating an occupied edge is syntactically valid, but fails after action 1.
    transports[Color.RED].contents = [batch(road, road), json.dumps({"tool": "end_turn", "arguments": {}})]
    roads_before = len(engine.observe(Color.RED).my_roads)
    await game.step()
    result = await game.step()
    assert result.automatic_action is None
    assert engine.observe(Color.RED).my_resources["WOOD"] == 0
    assert engine.observe(Color.RED).my_resources["BRICK"] == 0
    assert len(engine.observe(Color.RED).my_roads) == roads_before + 1
    assert len(transports[Color.RED].requests) == 2
    assert "Action 2" in str(transports[Color.RED].requests[-1].messages)


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["actor", "phase"])
async def test_external_actor_and_phase_changes_consume_queue(boundary: str) -> None:
    game, transports = sandbox()
    settlement, road = pair(game.game_engine)
    transports[Color.RED].contents = [batch(settlement, road), "invalid"]
    await game.step()
    if boundary == "actor":
        game.game_engine.step(road)
        transports[Color.BLUE].contents = ["invalid"]
    else:
        game.game_engine.state.current_prompt = ActionPrompt.MOVE_ROBBER
        game.game_engine.state.playable_actions = generate_playable_actions(game.game_engine.state)
    with pytest.raises(PlayerResponseError):
        await game.step()
    assert game.snapshot().pending_action_batch is None
    assert sum(a.action_type == ActionType.BUILD_ROAD for a in game.game_engine.state.actions) == (boundary == "actor")
