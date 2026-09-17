"""Real engine + shared parser + durable one-call deterministic continuations."""

import asyncio
import json
import pickle
from copy import deepcopy
from dataclasses import fields, replace

import pytest

from cle.game_engine.board_tokens import edge_token, node_token
from cle.game_engine.game import GameEngine
from cle.game_engine.models.actions import generate_playable_actions
from cle.game_engine.models.enums import ActionPrompt, ActionType
from cle.game_engine.models.player import Color
from cle.game_engine.trading import RESOURCE_NAMES
from cle.harness.context import PlayerResponseParseError, PlayerResponseParser
from cle.harness.models import ModelResponse
from cle.harness.shared_suite import load_shared_prompt_suite
from cle.harness.suite import load_context_suite
from cle.players.agent import AgentPlayer
from cle.players.contracts import PlayerChoice
from cle.sandbox import CatanSandbox, RetryPolicy
from cle.sandbox.catan import PlayerResponseError, PostActionCommunicationCancelled
from cle.traces.sqlite import SQLiteLiveTraceStore
from playground.game_viewer.live.reasoning_trace import build_live_reasoning_traces


COLORS = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)


def call(action):
    tool = {ActionType.BUILD_CITY: "upgrade_city"}.get(action.action_type, action.action_type.value.lower())
    if action.action_type == ActionType.BUILD_ROAD:
        arguments = {"edge": edge_token(action.value)}
    elif action.action_type in {ActionType.BUILD_SETTLEMENT, ActionType.BUILD_CITY}:
        arguments = {"node": node_token(action.value)}
    elif action.action_type == ActionType.MARITIME_TRADE:
        given = [card for card in action.value[:4] if card is not None]
        arguments = {"give": {given[0]: len(given)}, "receive": {action.value[-1]: 1}}
    else:
        arguments = {}
    return {"tool": tool, "arguments": arguments}


def batch(*actions, **extra):
    return json.dumps({"actions": [call(a) if not isinstance(a, dict) else a for a in actions], **extra})


class Replies:
    def __init__(self):
        self.contents = []
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        return ModelResponse(
            self.contents.pop(0), usage=(("prompt_tokens", 10), ("completion_tokens", 5)),
            provider_response_id=f"response:{len(self.requests)}",
            provider_request_id=f"request:{len(self.requests)}",
        )


def sandbox(engine=None):
    engine = engine or GameEngine(COLORS, seed=7, shuffle_players=False)
    transports = {color: Replies() for color in COLORS}
    players = {color: AgentPlayer(color, transports[color], session_id=f"batch:{color.value}") for color in COLORS}
    return CatanSandbox(engine, players, retry_policy=RetryPolicy(1)), transports


def main_engine(*, port_rate=None, **hand):
    engine = GameEngine(COLORS, seed=7, shuffle_players=False)
    while engine.state.is_initial_build_phase:
        action = engine.state.playable_actions[0]
        if port_rate is not None and action.color == Color.RED and action.action_type == ActionType.BUILD_SETTLEMENT:
            ports = engine.state.board.map.port_nodes
            unwanted = ports.get("WOOD", set()) | ports.get(None, set())
            nodes = (
                ports["WOOD"] if port_rate == 2 else ports[None]
            ) if not engine.observe(Color.RED).my_settlements and port_rate != 4 else set(range(54)) - unwanted
            action = next(a for a in engine.state.playable_actions if a.value in nodes)
        engine.step(action)
    engine.state.player_state["P0_HAS_ROLLED"] = True
    for resource in RESOURCE_NAMES:
        key = f"P0_{resource}_IN_HAND"
        count = hand.get(resource, 0)
        engine.state.resource_freqdeck[RESOURCE_NAMES.index(resource)] -= count - engine.state.player_state[key]
        engine.state.player_state[key] = count
    engine.state.playable_actions = generate_playable_actions(engine.state)
    return engine


def pair(engine):
    staged = deepcopy(engine)
    settlement = staged.state.playable_actions[0]
    assert settlement.action_type == ActionType.BUILD_SETTLEMENT
    staged.step(settlement)
    road = staged.state.playable_actions[0]
    assert road.action_type == ActionType.BUILD_ROAD
    return settlement, road


@pytest.mark.asyncio
@pytest.mark.parametrize("pair_index", range(8))
async def test_every_setup_pair_has_one_call_and_two_checkpoints(pair_index):
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
    first = await game.step()
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
async def test_snake_reversal_does_not_chain_both_setup_pairs():
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


def road_unlock(engine):
    # Find a genuine extension which unlocks a distance-rule-compliant settlement.
    for _ in range(3):
        existing = {a.value for a in engine.state.playable_actions if a.action_type == ActionType.BUILD_SETTLEMENT}
        roads = [a for a in engine.state.playable_actions if a.action_type == ActionType.BUILD_ROAD]
        for road in roads:
            staged = deepcopy(engine)
            staged.step(road)
            new = next((a for a in staged.state.playable_actions if a.action_type == ActionType.BUILD_SETTLEMENT and a.value not in existing), None)
            if new:
                return road, new
        engine.step(roads[0])
    raise AssertionError("Fixture could not find a road opening a new settlement")


@pytest.mark.asyncio
async def test_road_unlocks_settlement_and_end_turn_is_terminal():
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


def conversions(engine):
    staged = deepcopy(engine)
    actions = []
    for resource in ("WHEAT", "ORE"):
        trade = next(a for a in staged.state.playable_actions if a.action_type == ActionType.MARITIME_TRADE and a.value[0] == "WOOD" and a.value[-1] == resource)
        actions.append(trade)
        staged.step(trade)
    city = next(a for a in staged.state.playable_actions if a.action_type == ActionType.BUILD_CITY)
    return (*actions, city)


@pytest.mark.asyncio
async def test_conversions_city_persist_once_and_count_one_model_call(tmp_path):
    engine = main_engine(WOOD=8, WHEAT=1, ORE=2)
    actions = conversions(engine)
    assert not any(a.action_type == ActionType.BUILD_CITY for a in engine.state.playable_actions)
    game, transports = sandbox(engine)
    transports[Color.RED].contents = [batch(*actions, notes="One update")]
    store = SQLiteLiveTraceStore(tmp_path / "batches.sqlite3")
    store.start_game(engine.id, config={}, snapshot=game.snapshot())
    expected = deepcopy(engine)
    first_context = None
    for index, action in enumerate(actions):
        result = await game.step()
        expected.step(action)
        assert engine.observe(Color.RED).my_resources == expected.observe(Color.RED).my_resources
        store.record_step(engine.id, result=result, rejected_attempts=(), public_state={}, snapshot=game.snapshot())
        game.restore(store.load_snapshot(engine.id, step_index=index))
        if index == 0:
            first_context = result.attempts[0].context_id
        else:
            provenance = store.get_step(engine.id, index)["step"]["result"]["automatic_action"]
            assert provenance["origin_context_id"] == first_context
            assert provenance["provider_request_id"] == "request:1"
            assert provenance["action_number"] == index + 1
    trace = store.get_game(engine.id)
    assert trace["step_count"] == 3
    assert len(trace["model_calls"]) == 1
    assert trace["model_calls"][0]["response"]["usage"]["prompt_tokens"] == 10
    player = game.players[Color.RED]
    assert player.session.memory_revision == 1
    assert player.session.strategic_memory == "One update"
    assert player.session.action_next_sequence == 16
    assert game.snapshot().pending_action_batch is None


@pytest.mark.asyncio
async def test_invalid_second_action_preserves_prefix_and_consumes_before_retry_restore():
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
async def test_new_speech_interrupts_queue_without_acknowledging_unseen_events():
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
async def test_winning_city_discards_remainder_immediately():
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
async def test_post_commit_cancellation_keeps_queue_and_never_repeats_prefix(monkeypatch):
    game, transports = sandbox()
    settlement, road = pair(game.game_engine)
    transports[Color.RED].contents = [batch(settlement, road)]
    original = game._post_action_communication

    async def cancelled(result, **kwargs):
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


@pytest.mark.parametrize("tail", [
    {"tool": "say", "arguments": {}},
    {"tool": "offer_trade", "arguments": {"confirm_if_accepted_by": "ANY"}},
    {"tool": "confirm_trade", "arguments": {}},
    {"tool": "buy_development_card", "arguments": {}},
    {"tool": "roll_dice", "arguments": {}},
    {"tool": "play_knight", "arguments": {"tile": "<T00>"}},
    {"tool": "build_road", "arguments": {"edge": "<E00_01>", "notes": "buried"}},
    {"tool": "build_road", "arguments": {"edge": 1}},
    {"tool": "maritime_trade", "arguments": {"give": {"WOOD": True}, "receive": {"ORE": 1}}},
    {"tool": "maritime_trade", "arguments": {"give": {"WOOD": 4}, "receive": {"ORE": 2}}},
])
def test_whole_batch_syntax_is_checked_before_first_action(tail):
    game, _ = sandbox()
    first, _ = pair(game.game_engine)
    parser = PlayerResponseParser(load_shared_prompt_suite().decision_suite())
    with pytest.raises(PlayerResponseParseError):
        parser.parse(game.decision_context(), ModelResponse(batch(first, tail)))
    assert game.revision == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("notes", [None, 3, {}, "x" * 4001])
async def test_invalid_notes_reject_entire_batch_before_state_or_memory(notes):
    game, transports = sandbox()
    transports[Color.RED].contents = [batch(*pair(game.game_engine), notes=notes)]
    before = pickle.dumps(game.snapshot())
    with pytest.raises(PlayerResponseError):
        await game.step()
    assert pickle.dumps(game.snapshot()) == before


@pytest.mark.parametrize("payload", [
    {"actions": []}, {"actions": [{}] * 5}, {"actions": {}},
    {"actions": [{"tool": "end_turn", "arguments": {}}, {"tool": "end_turn", "arguments": {}}]},
    {"actions": [{"tool": "end_turn", "arguments": {}, "notes": "buried"}]},
    {"actions": [{"tool": "end_turn", "arguments": {}}], "tool": "end_turn", "arguments": {}},
])
def test_strict_envelope(payload):
    game, _ = sandbox()
    with pytest.raises(PlayerResponseParseError):
        PlayerResponseParser(load_shared_prompt_suite().decision_suite()).parse(game.decision_context(), ModelResponse(json.dumps(payload)))


def test_historical_contracts_do_not_admit_batches_and_old_slots_default_empty():
    game, _ = sandbox()
    response = ModelResponse(batch(*pair(game.game_engine)))
    suites = [load_context_suite("cle/harness/suites/catan_v11.yaml"), load_context_suite("cle/harness/suites/catan_v10.yaml")]
    old_shared = load_shared_prompt_suite().model_copy(update={"deterministic_batches": False}).decision_suite()
    for suite in [*suites, old_shared]:
        with pytest.raises(PlayerResponseParseError):
            PlayerResponseParser(suite).parse(game.decision_context(), response)
    for value, name, expected in [(game.snapshot(), "pending_action_batch", None), (PlayerChoice(0), "batch_actions", ())]:
        historical = [getattr(value, field.name) for field in fields(value) if field.name != name]
        restored = object.__new__(type(value))
        restored.__setstate__(historical)
        assert getattr(restored, name) == expected


@pytest.mark.asyncio
async def test_bad_saved_queue_rejected_without_mutating_game():
    game, transports = sandbox()
    transports[Color.RED].contents = [batch(*pair(game.game_engine))]
    await game.step()
    saved = game.snapshot()
    before = pickle.dumps(saved)
    for bad in [replace(saved.pending_action_batch, next_index=2), replace(saved.pending_action_batch, actor=Color.BLUE)]:
        with pytest.raises(ValueError):
            game.restore(replace(saved, pending_action_batch=bad))
        assert pickle.dumps(game.snapshot()) == before


@pytest.mark.asyncio
async def test_cancelled_inference_admits_no_plan():
    game, transports = sandbox()
    started = asyncio.Event()

    async def wait(request):
        started.set()
        await asyncio.Future()

    transports[Color.RED].complete = wait
    task = asyncio.create_task(game.step())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert game.snapshot().pending_action_batch is None
    assert game.revision == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("rate", [2, 3, 4])
async def test_real_port_and_bank_conversions_then_city(rate):
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
async def test_paid_prefix_failure_never_charges_or_replays_twice():
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
async def test_external_actor_and_phase_changes_consume_queue(boundary):
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


def test_duplicate_keys_escaped_tokens_and_buried_notes_rejected():
    game, _ = sandbox()
    response = batch(*pair(game.game_engine))
    parser = PlayerResponseParser(load_shared_prompt_suite().decision_suite())
    malformed = [
        response.replace('"actions":', '"actions":[],"actions":'),
        response.replace('"edge":', '"edg\\u0065":'),
        response.replace('"<E', '"\\u003cE'),
        response.replace('"arguments":', '"notes":"buried","arguments":', 1),
    ]
    for text in malformed:
        with pytest.raises(PlayerResponseParseError):
            parser.parse(game.decision_context(), ModelResponse(text))


@pytest.mark.asyncio
async def test_prompt_rebinding_cannot_reinterpret_pending_accepted_plan():
    game, transports = sandbox()
    settlement, road = pair(game.game_engine)
    transports[Color.RED].contents = [batch(settlement, road)]
    await game.step()
    old = game.players[Color.RED]
    bundle = load_shared_prompt_suite().model_copy(update={"deterministic_batches": False})
    replacement = AgentPlayer(Color.RED, transports[Color.RED], session_id=old.session.session_id,
                              suite=bundle.decision_suite(), communication_suite=bundle.communication_suite())
    replacement.restore(old.snapshot())
    game.register_player(replacement)
    result = await game.step()
    assert result.transitions[0].requested_action == road
    assert len(transports[Color.RED].requests) == 1


@pytest.mark.asyncio
async def test_automatic_result_and_snapshot_are_detached_from_pending_plan():
    engine = main_engine(WOOD=8, WHEAT=1, ORE=2)
    game, transports = sandbox(engine)
    actions = conversions(engine)
    transports[Color.RED].contents = [batch(*actions)]
    await game.step()
    second = await game.step()
    second.automatic_action.batch.actions[2]["arguments"]["node"] = "<N99>"
    saved = game.snapshot()
    saved.pending_action_batch.actions[2]["arguments"]["node"] = "<N98>"
    third = await game.step()
    assert third.transitions[0].requested_action == actions[2]
    assert len(transports[Color.RED].requests) == 1


@pytest.mark.asyncio
async def test_stale_cached_menu_cannot_authorize_an_unfunded_continuation():
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
async def test_cancelled_middle_continuation_restores_next_action_not_previous(monkeypatch):
    engine = main_engine(WOOD=8, WHEAT=1, ORE=2)
    game, transports = sandbox(engine)
    actions = conversions(engine)
    transports[Color.RED].contents = [batch(*actions)]
    await game.step()
    original = game._post_action_communication

    async def cancelled(result, **kwargs):
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
