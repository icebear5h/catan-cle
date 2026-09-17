"""Offline end-to-end preauthorization: real parsers, barriers, engine and store."""

import asyncio
import json
import pickle
from copy import deepcopy
from dataclasses import fields, replace

import pytest

from cle.game_engine.game import GameEngine
from cle.game_engine.models.actions import generate_playable_actions
from cle.game_engine.models.enums import Action, ActionType
from cle.game_engine.models.player import Color
from cle.game_engine.trading import RESOURCE_NAMES, TradeOfferStatus
from cle.harness.action_tools import parse_tool_choice
from cle.harness.context import PlayerResponseParseError, PlayerResponseParser
from cle.harness.models import ModelResponse
from cle.harness.shared_suite import load_shared_prompt_suite
from cle.harness.suite import load_context_suite
from cle.players.agent import AgentPlayer
from cle.players.contracts import PlayerChoice
from cle.players.validation import action_from_choice
from cle.sandbox import CatanSandbox, RetryPolicy
from cle.sandbox.catan import PlayerResponseError, PostActionCommunicationCancelled
from cle.sandbox.contracts import SandboxSnapshot
from cle.traces.sqlite import SQLiteLiveTraceStore
from playground.game_viewer.live.reasoning_trace import build_live_reasoning_traces


COLORS = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)
TERMS = {"give": {"WOOD": 1}, "receive": {"ORE": 1}}


def reply(tool, arguments, **extra):
    return json.dumps({"tool": tool, "arguments": arguments, **extra})


class Replies:
    def __init__(self, color, contents):
        self.color = color
        self.contents = list(contents)
        self.requests = []
        self.before_reply = None

    async def complete(self, request):
        self.requests.append(request)
        if self.before_reply is not None:
            await self.before_reply()
        content = self.contents.pop(0)
        return ModelResponse(
            content, usage=(("prompt_tokens", 10), ("completion_tokens", 5)),
            provider_response_id=f"{self.color.value}:{len(self.requests)}",
        )


def sandbox(priority="ANY", responses=("accept_offer",) * 3, seat_order=COLORS):
    engine = GameEngine(seat_order, seed=7, shuffle_players=False)
    while engine.state.is_initial_build_phase:
        engine.step(engine.state.playable_actions[0])
    engine.state.player_state["P0_HAS_ROLLED"] = True
    for index, color in enumerate(seat_order):
        for resource in ("WOOD", "ORE"):
            key = f"P{index}_{resource}_IN_HAND"
            engine.state.resource_freqdeck[RESOURCE_NAMES.index(resource)] -= 3 - engine.state.player_state[key]
            engine.state.player_state[key] = 3
    engine.state.playable_actions = generate_playable_actions(engine.state)
    arguments = dict(TERMS)
    if priority is not None:
        arguments["confirm_if_accepted_by"] = priority
    transports = {Color.RED: Replies(Color.RED, [
        reply("offer_trade", arguments, notes="Admitted original plan"),
        reply("end_turn", {}, notes="Model regained control"),
    ])}
    original = {"give": {"ORE": 1}, "receive": {"WOOD": 1}}
    for color, tool in zip(COLORS[1:], responses, strict=True):
        args = {"player": "RED", **original}
        if tool == "counter_offer":
            args = {"player": "RED", "original": original, "proposed": {
                "give": {"ORE": 1}, "receive": {"WOOD": 2},
            }}
        transports[color] = Replies(color, [reply(tool, args)])
    players = {color: AgentPlayer(color, transports[color], session_id=f"trade:{color.value}") for color in COLORS}
    return CatanSandbox(engine, players, retry_policy=RetryPolicy(1)), transports


def totals(engine):
    return tuple(
        engine.state.resource_freqdeck[i] + sum(
            engine.state.player_state[f"P{seat}_{resource}_IN_HAND"] for seat in range(4)
        ) for i, resource in enumerate(RESOURCE_NAMES)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("priority,expected", [(["ORANGE", "BLUE"], Color.ORANGE), ("ANY", Color.BLUE)])
@pytest.mark.parametrize("completion_order", [COLORS[1:], tuple(reversed(COLORS[1:]))])
async def test_full_barrier_priority_is_independent_of_latency(priority, expected, completion_order):
    game, transports = sandbox(priority)
    offered = await game.step()
    origin = offered.attempts[0]
    before = pickle.dumps(game.game_engine.snapshot())
    all_started = set()
    done = {color: asyncio.Event() for color in COLORS[1:]}

    def delayed(color):
        async def wait():
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
    result = await game.step()
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
async def test_probe_or_unmatched_condition_returns_model_control(priority, responses, reason):
    game, transports = sandbox(priority, responses)
    await game.step()
    await game.step()
    result = await game.step()
    assert result.transitions[0].requested_action.action_type == ActionType.END_TURN
    assert len(transports[Color.RED].requests) == 2
    assert not any(a.action_type == ActionType.CONFIRM_TRADE for a in game.game_engine.state.actions)
    paused = [e for e in game.game_engine.events if e.event_type == "TRADE_PREAUTHORIZATION_PAUSED"]
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
@pytest.mark.parametrize("change", ["proposer_hand", "preferred_hand", "terms", "audience", "withdraw", "expire", "round", "cancel", "response", "offer_id", "window_id", "willingness"])
async def test_stale_state_consumes_without_fallback_or_transfer(change):
    game, transports = sandbox(["BLUE", "WHITE"])
    await game.step()
    await game.step()
    engine = game.game_engine
    window = engine.state.trade_window
    offer = window.active_offers[0]
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
async def test_any_uses_engine_seat_order_not_alphabetical_order():
    game, _ = sandbox(seat_order=(Color.RED, Color.WHITE, Color.ORANGE, Color.BLUE))
    await game.step()
    await game.step()
    result = await game.step()
    assert result.transitions[0].requested_action.value.counterparty == Color.WHITE


@pytest.mark.asyncio
@pytest.mark.parametrize("priority", [["WHITE"], ["BLUE", "WHITE"]])
async def test_first_permitted_willing_player_is_selected(priority):
    game, _ = sandbox(priority, responses=("reject_offer", "accept_offer", "accept_offer"))
    await game.step()
    await game.step()
    result = await game.step()
    assert result.transitions[0].requested_action.value.counterparty == Color.WHITE


@pytest.mark.asyncio
async def test_pause_is_consumed_even_when_next_model_call_fails_and_restores():
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
async def test_checkpoint_restore_call_accounting_and_consumption(tmp_path):
    game, transports = sandbox()
    store = SQLiteLiveTraceStore(tmp_path / "trades.sqlite3")
    store.start_game(game.game_engine.id, config={}, snapshot=game.snapshot())
    results = []
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
            game = restored
    trace = store.get_game(game.game_engine.id)
    assert trace["step_count"] == 3
    assert len(trace["model_calls"]) == 4  # proposer and three responders only
    assert sum(call["response"]["usage"]["prompt_tokens"] for call in trace["model_calls"]) == 40
    automatic = store.get_step(game.game_engine.id, 2)
    assert automatic["model_calls"] == []
    provenance = automatic["step"]["result"]["automatic_action"]
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
async def test_failed_and_cancelled_barrier_preserves_authorization_until_retry():
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

    async def block():
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


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_after", ["offer", "barrier"])
async def test_post_action_cancellation_and_policy_rebinding_retain_admitted_instruction(monkeypatch, cancel_after):
    game, transports = sandbox(["ORANGE", "BLUE"])
    original = game._run_communication
    if cancel_after == "barrier":
        await game.step()

    async def cancel(*args, **kwargs):
        raise asyncio.CancelledError()

    monkeypatch.setattr(game, "_run_communication", cancel)
    with pytest.raises(PostActionCommunicationCancelled):
        await game.step()
    assert game.players[Color.RED].session.memory_revision == 1
    snapshot = pickle.loads(pickle.dumps(game.snapshot()))
    game.restore(snapshot)
    monkeypatch.setattr(game, "_run_communication", original)

    def rebind(runtime):
        old = runtime.players[Color.RED]
        replacement = AgentPlayer(Color.RED, transports[Color.RED], session_id=old.session.session_id)
        replacement.restore(old.snapshot())
        runtime.players[Color.RED] = replacement

    game._refresh_players = rebind
    if cancel_after == "offer":
        await game.step()
    else:
        rebind(game)

    def must_not_refresh(runtime):
        raise AssertionError("Automatic execution must not reinterpret the admitted model call")

    game._refresh_players = must_not_refresh
    result = await game.step()
    assert result.transitions[0].requested_action.value.counterparty == Color.ORANGE
    assert game.players[Color.RED].session.memory_revision == 1
    assert len(transports[Color.RED].requests) == 1


@pytest.mark.asyncio
async def test_saved_authorization_must_match_original_offer_event():
    game, _ = sandbox()
    await game.step()
    snapshot = game.snapshot()
    before = pickle.dumps(snapshot)
    altered = replace(snapshot.trade_preauthorization, give=(2, 0, 0, 0, 0))
    with pytest.raises(ValueError, match="original offer event"):
        game.restore(replace(snapshot, trade_preauthorization=altered))
    assert pickle.dumps(game.snapshot()) == before


@pytest.mark.parametrize("bad", [[], ["BLUE", "blue"], ["RED"], ["GOLD"], ["BLUE", 1], "BLUE", "any", "if BLUE accepts", None, {}, True])
def test_shared_parser_rejects_invalid_authorization_without_mutation(bad):
    game, _ = sandbox()
    before = pickle.dumps(game.snapshot())
    parser = PlayerResponseParser(load_shared_prompt_suite().decision_suite())
    with pytest.raises(PlayerResponseParseError):
        parser.parse(game.decision_context(), ModelResponse(reply("offer_trade", {**TERMS, "confirm_if_accepted_by": bad})))
    assert pickle.dumps(game.snapshot()) == before


def test_authorization_is_shared_only_exact_root_offer_and_old_slots_restore():
    game, _ = sandbox()
    context = game.decision_context()
    args = {**TERMS, "confirm_if_accepted_by": ["BLUE"]}
    with pytest.raises(ValueError):
        parse_tool_choice(context, "offer_trade", args)
    historical = PlayerResponseParser(load_context_suite("cle/harness/suites/catan_v11.yaml"))
    with pytest.raises(PlayerResponseParseError):
        historical.parse(context, ModelResponse(reply("offer_trade", args)))
    for args in ({**args, "give_any": 1}, {**args, "receive_any": 1}, {**args, "condition": "nobody accepts"}):
        with pytest.raises(ValueError):
            parse_tool_choice(context, "offer_trade", args, shared=True)
    choice = parse_tool_choice(context, "offer_trade", TERMS, shared=True)
    narrow = replace(choice.trade_offer, audience=frozenset({Color.BLUE}))
    narrow_context = replace(context, legal_actions=(Action(Color.RED, ActionType.OFFER_TRADE, narrow),))
    with pytest.raises(ValueError, match="audience"):
        action_from_choice(narrow_context, PlayerChoice(0, confirm_if_accepted_by=(Color.WHITE,)))
    old = PlayerChoice(0)
    restored = object.__new__(PlayerChoice)
    restored.__setstate__([getattr(old, field.name) for field in fields(old)[:-1]])
    assert restored.confirm_if_accepted_by is None
    old_snapshot = game.snapshot()
    restored_snapshot = object.__new__(SandboxSnapshot)
    restored_snapshot.__setstate__([getattr(old_snapshot, field.name) for field in fields(old_snapshot)[:-1]])
    assert restored_snapshot.trade_preauthorization is None
