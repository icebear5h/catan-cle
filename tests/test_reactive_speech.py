"""Reactive speech against real engine, fresh model parsing and checkpoint storage."""

import asyncio
import json
import pickle
from dataclasses import replace

import pytest

from cle.game_engine.board_tokens import node_token, tile_token
from cle.game_engine.communication import CommunicationLimits
from cle.game_engine.events import EngineTransition
from cle.game_engine.game import GameEngine
from cle.game_engine.models.actions import generate_playable_actions
from cle.game_engine.models.enums import Action, ActionPrompt, ActionType
from cle.game_engine.models.player import Color
from cle.game_engine.state_functions import player_key
from cle.harness.models import ModelResponse
from cle.players.agent import AgentPlayer
from cle.players.contracts import CommunicationChoice
from cle.sandbox.catan import CatanSandbox, PlayerResponseError
from cle.sandbox.contracts import RetryPolicy, SandboxStepResult
from cle.traces import SQLiteLiveTraceStore
from playground.game_viewer.live.reasoning_trace import build_live_reasoning_traces


COLORS = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)


class Transport:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        reply = self.replies.pop(0)
        if callable(reply):
            reply = reply(request)
        if isinstance(reply, BaseException):
            raise reply
        return ModelResponse(json.dumps(reply), usage=(("prompt_tokens", 17), ("completion_tokens", 5)))


def sandbox(engine=None):
    engine = engine or GameEngine(COLORS, seed=7, shuffle_players=False)
    transports = {color: Transport() for color in COLORS}
    players = {color: AgentPlayer(color, transports[color], session_id=f"reactive:{color.value}") for color in COLORS}
    return CatanSandbox(engine, players, retry_policy=RetryPolicy(1)), transports


def say(respondents, *, text="Leave this spot open?", notes="private plan"):
    return {"tool": "say", "arguments": {"text": text, "respondents": respondents}, "notes": notes}


def reply(respondents, *, text="Will you leave mine open?", notes="private reply"):
    return {"mode": "say", "text": text, "respondents": respondents, "notes": notes}


def build(engine):
    return {"tool": "build_settlement", "arguments": {"node": node_token(engine.state.playable_actions[0].value)}}


@pytest.mark.asyncio
async def test_standalone_public_choice_keeps_board_pending_and_records_one_notes_commit(tmp_path):
    sb, ts = sandbox()
    engine = sb.game_engine
    before = pickle.dumps(engine.state)
    decision = build(engine)
    store = SQLiteLiveTraceStore(tmp_path / "speech.sqlite3")
    store.start_game(str(engine.id), config={}, snapshot=sb.snapshot())

    def pass_after_hearing(request):
        assert pickle.dumps(engine.state) == before
        assert engine.state.actions == []
        assert request.trigger_reason == "addressed_speech"
        assert "Leave this spot open?" in request.messages[-1].content
        assert "private plan" not in request.messages[-1].content
        assert sb.players[Color.RED].session.memory_revision == 1
        return {"mode": "pass", "notes": "Remember RED asked"}

    def act_after_reply(request):
        assert pickle.dumps(engine.state) == before
        assert "Standalone say is unavailable" in request.messages[-1].content
        assert "private plan" in request.messages[-1].content
        return decision

    ts[Color.RED].replies = [say(["BLUE"]), act_after_reply]
    ts[Color.BLUE].replies = [pass_after_hearing]
    result = await sb.step()
    assert len(result.transitions) == len(engine.state.actions) == 1
    assert len(result.messages) == 1
    assert not ts[Color.WHITE].requests and not ts[Color.ORANGE].requests
    for color in COLORS:
        assert engine.project_events(color)[0].payload["text"] == "Leave this spot open?"
    assert sb.players[Color.BLUE].session.strategic_memory == "Remember RED asked"
    assert sb.players[Color.BLUE].session.memory_revision == 1
    assert sb.players[Color.RED].session.memory_revision == 2
    traces = build_live_reasoning_traces(sb, result, communication_attempts=sb.communication_trace)
    assert len(traces) == 3
    spoken = next(t for t in traces if t.get("trigger_reason") == "standalone_speech")
    assert spoken["communication_mode"] == "say" and spoken["channel"] == "action"
    assert spoken["usage"]["prompt_tokens"] == 17
    index = store.record_step(str(engine.id), result=result, rejected_attempts=(),
                              communication_attempts=sb.communication_trace, public_state={}, snapshot=sb.snapshot())
    assert index == 0
    saved = store.get_game(str(engine.id))
    calls = saved["model_calls"]
    assert len(calls) == 3
    spoken_call = next(call for call in calls if call["choice"].get("trigger_reason") == "standalone_speech")
    assert spoken_call["call_kind"] == "communication"
    assert spoken_call["choice"]["text"] == "Leave this spot open?"
    assert spoken_call["request"]["channel"] == "action"
    assert json.loads(spoken_call["response"]["content"])["tool"] == "say"
    restored, _ = sandbox()
    restored.restore(store.load_resume_point(str(engine.id)).snapshot)
    assert restored.players[Color.RED].session.memory_revision == 2


@pytest.mark.asyncio
async def test_failure_after_say_resume_preserves_pending_action_and_cannot_say_twice():
    sb, ts = sandbox()
    decision = build(sb.game_engine)
    ts[Color.RED].replies = [say([]), say([])]
    with pytest.raises(PlayerResponseError, match="failed to choose"):
        await sb.step()
    assert not sb.game_engine.state.actions
    saved = pickle.loads(pickle.dumps(sb.snapshot()))
    assert saved.speech_used and saved.speech_calls_remaining == 11
    restored, rt = sandbox()
    restored.restore(saved)
    rt[Color.RED].replies = [decision]
    result = await restored.step()
    assert len(restored.game_engine.state.actions) == 1
    assert not restored.communication_trace
    assert len([e for e in restored.game_engine.events if e.event_type == "MESSAGE_SENT"]) == 1
    assert result.attempts[0].model_request.memory_revision == 1


@pytest.mark.asyncio
async def test_broadcast_replies_are_bounded_and_pass_does_not_start_another_branch():
    sb, ts = sandbox()
    sb.game_engine.communication_limits = CommunicationLimits(max_messages_per_window=4)
    ts[Color.RED].replies = [say(["BLUE", "WHITE", "ORANGE"]), build(sb.game_engine)]
    for color in COLORS[1:]:
        ts[color].replies = [reply([c.value for c in COLORS if c != color])]
    result = await sb.step()
    assert len(result.messages) == 4
    assert sum(len(t.requests) for t in ts.values()) == 5
    assert len(sb.game_engine.state.actions) == 1
    assert sb.snapshot().pending_reactions == ()


@pytest.mark.asyncio
async def test_pending_reactions_resume_without_recalling_accepted_pass():
    sb, ts = sandbox()
    ts[Color.RED].replies = [say(["BLUE", "WHITE", "ORANGE"])]
    ts[Color.BLUE].replies = [{"mode": "pass", "notes": "accepted blue"}]
    ts[Color.WHITE].replies = [RuntimeError("offline failure")]
    with pytest.raises(RuntimeError, match="offline failure"):
        await sb.step()
    restored, rt = sandbox()
    restored.restore(pickle.loads(pickle.dumps(sb.snapshot())))
    rt[Color.WHITE].replies = [{"mode": "pass"}]
    rt[Color.ORANGE].replies = [{"mode": "pass"}]
    rt[Color.RED].replies = [build(restored.game_engine)]
    await restored.step()
    assert not rt[Color.BLUE].requests
    assert restored.players[Color.BLUE].session.strategic_memory == "accepted blue"
    assert restored.players[Color.BLUE].session.memory_revision == 1


def main_engine():
    engine = GameEngine(COLORS, seed=7, shuffle_players=False)
    while engine.state.is_initial_build_phase:
        engine.step(engine.state.playable_actions[0])
    return engine


@pytest.mark.asyncio
@pytest.mark.parametrize("discards", [False, True])
async def test_seven_window_after_all_discards_once_across_retry_and_snapshot(discards):
    engine = main_engine()
    for color in COLORS[:2] if discards else ():
        key = player_key(engine.state, color)
        engine.state.player_state[f"{key}_WOOD_IN_HAND"] = 10
    engine.step(Action(Color.RED, ActionType.ROLL, (3, 4)), force=True)
    sb, ts = sandbox(engine)
    while engine.state.current_prompt == ActionPrompt.DISCARD:
        actor = engine.state.current_color()
        count = sb.decision_context().discard_count
        ts[actor].replies = [{"tool": "discard", "arguments": {"cards": {"WOOD": count}}}]
        await sb.step()
        assert not sb.communication_trace
    assert engine.state.current_prompt == ActionPrompt.MOVE_ROBBER
    for color in COLORS[1:]:
        ts[color].replies = [{"mode": "pass", "notes": f"{color.value} considered robber"}]
    ts[Color.RED].replies = [{"tool": "invalid", "arguments": {}}]
    with pytest.raises(PlayerResponseError):
        await sb.step()
    assert len(sb.communication_trace) == 3
    assert all(r.opportunity.reason.value == "pre_robber" for r in sb.communication_trace)
    for color in COLORS[1:]:
        assert "discards are complete" in ts[color].requests[-1].messages[-1].content
    restored, rt = sandbox()
    restored.restore(pickle.loads(pickle.dumps(sb.snapshot())))
    target = restored.game_engine.state.playable_actions[0].value
    tile = restored.game_engine.state.board.map.land_tiles[target]
    rt[Color.RED].replies = [{"tool": "move_robber", "arguments": {"tile": tile_token(tile.id)}}]
    await restored.step()
    assert not restored.communication_trace
    assert all(not rt[color].requests for color in COLORS[1:])


@pytest.mark.asyncio
async def test_routine_events_never_poll_or_acknowledge_and_are_delivered_later():
    sb, ts = sandbox()
    for event_type in (
        "END_TURN", "ROLL", "RESOURCE_PAYOUT", "DISCARD", "BUY_DEVELOPMENT_CARD",
        "BUILD_CITY", "BUILD_ROAD", "BUILD_SETTLEMENT", "MARITIME_TRADE",
        "PLAY_YEAR_OF_PLENTY", "STEAL", "CONFIRM_TRADE", "MOVE_ROBBER",
    ):
        event = sb.game_engine.publish_event(event_type, Color.RED, (2, 3) if event_type == "ROLL" else None)
        # Exercise the actual post-action scheduler with a public engine event.
        transition = EngineTransition(
            requested_action=Action(Color.RED, ActionType.END_TURN, None),
            resolved_action=Action(Color.RED, ActionType.END_TURN, None),
            before_revision=event.sequence, after_revision=event.sequence + 1,
            events=(event,), winner=None,
        )
        await sb._post_action_communication(SandboxStepResult((), (), (transition,)), max_rounds=2)
    assert all(not transport.requests for transport in ts.values())
    assert all(player.event_cursor == 0 for player in sb.players.values())
    ts[Color.RED].replies = [build(sb.game_engine)]
    await sb.step()
    text = ts[Color.RED].requests[0].messages[-1].content
    assert "END_TURN" in text and "BUILD_CITY" in text
    assert all(not ts[color].requests for color in COLORS[1:])


@pytest.mark.asyncio
async def test_parser_rejects_private_or_bundled_speech_atomically():
    sb, ts = sandbox()
    ctx = sb.decision_context()
    valid = say(["BLUE"])
    ts[Color.RED].replies = [valid]
    attempt = await sb.players[Color.RED].choose(ctx)
    assert isinstance(attempt.choice, CommunicationChoice)
    invalid = [
        {**valid, "action": {"tool": "build_settlement"}},
        {**valid, "arguments": {**valid["arguments"], "audience": ["BLUE"]}},
        {**valid, "arguments": {**valid["arguments"], "respondents": ["RED"]}},
        {**valid, "notes": None},
    ]
    ts[Color.RED].replies = invalid
    for _ in invalid:
        attempt = await sb.players[Color.RED].choose(ctx)
        assert attempt.choice is None and attempt.validation_error
    assert not sb.game_engine.events and sb.players[Color.RED].session.memory_revision == 0
    ts[Color.RED].replies = [valid]
    rejected = await sb.players[Color.RED].choose(replace(ctx, speech_allowed=False))
    assert rejected.choice is None


@pytest.mark.asyncio
async def test_reply_round_bound_and_received_trigger_skip():
    sb, ts = sandbox()
    ts[Color.RED].replies = [say(["BLUE"]), reply(["BLUE"]), build(sb.game_engine)]
    ts[Color.BLUE].replies = [reply(["RED"])]
    result = await sb.step()
    assert len(result.messages) == 3
    assert len(ts[Color.RED].requests) == 3 and len(ts[Color.BLUE].requests) == 1
    # This already-heard event must not start a new call even in a new queue.
    sb._pending_reactions = sb.communication_policy.addressed(sb.game_engine, (result.messages[0],))
    assert await sb._run_reactive() == ()
    assert len(ts[Color.BLUE].requests) == 1


@pytest.mark.asyncio
async def test_required_trade_replies_remain_simultaneous_private_and_no_extra_speech():
    engine = main_engine()
    engine.state.player_state["P0_HAS_ROLLED"] = True
    engine.state.player_state["P0_WOOD_IN_HAND"] = 3
    engine.state.playable_actions = generate_playable_actions(engine.state)
    sb, ts = sandbox(engine)
    ts[Color.RED].replies = [{"tool": "offer_trade", "arguments": {"give": {"WOOD": 1}, "receive": {"ORE": 1}}}]
    await sb.step()
    assert not sb.communication_trace
    original_actions = len(engine.state.actions)
    ready = asyncio.Event()
    entered = []

    async def reject(request):
        entered.append(request)
        if len(entered) == 3:
            ready.set()
        await asyncio.wait_for(ready.wait(), 2)
        assert len(engine.state.actions) == original_actions
        assert "peer secret" not in request.messages[-1].content
        assert "Standalone say is unavailable" in request.messages[-1].content
        return ModelResponse(json.dumps({"tool": "reject_offer", "arguments": {
            "player": "RED", "give": {"ORE": 1}, "receive": {"WOOD": 1},
        }, "notes": "peer secret"}))

    for color in COLORS[1:]:
        ts[color].complete = reject
    result = await sb.step()
    assert len(entered) == len(result.transitions) == 3
    assert all(request.channel == "action" for request in entered)
    assert len({request.input_next_sequence for request in entered}) == 1
    assert not sb.communication_trace
    assert all(sb.players[color].session.memory_revision == 1 for color in COLORS[1:])


@pytest.mark.asyncio
async def test_shared_knight_remains_atomic_without_a_speech_await():
    engine = main_engine()
    engine.state.player_state["P0_KNIGHT_IN_HAND"] = 1
    engine.state.player_state["P0_KNIGHT_OWNED_AT_START"] = True
    engine.state.playable_actions = generate_playable_actions(engine.state)
    sb, ts = sandbox(engine)
    tile = next(tile for coord, tile in engine.state.board.map.land_tiles.items() if coord != engine.state.board.robber_coordinate)
    ts[Color.RED].replies = [{"tool": "play_knight", "arguments": {"tile": tile_token(tile.id)}, "notes": "Knight committed"}]
    result = await sb.step()
    assert [t.requested_action.action_type for t in result.transitions] == [ActionType.PLAY_KNIGHT_CARD, ActionType.MOVE_ROBBER]
    assert sum(len(t.requests) for t in ts.values()) == 1
    assert not sb.communication_trace
    assert sb.players[Color.RED].session.memory_revision == 1


def test_all_shorthand_is_rejected_in_favor_of_explicit_colors():
    from cle.harness.public_speech import parse_public_speech
    for shorthand in ("ALL", ["ALL"]):
        with pytest.raises(ValueError, match="distinct eligible other colors"):
            parse_public_speech(
                {"mode": "say", "text": "Trade?", "respondents": shorthand},
                speaker=Color.RED, participants=COLORS, max_notes_chars=4000,
            )
