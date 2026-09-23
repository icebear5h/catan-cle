"""Trigger windows, routine events, and audiences."""
import asyncio
import json
import pickle
from dataclasses import replace
from typing import Any

import pytest

from cle.game_engine.board_tokens import tile_token
from cle.game_engine.events import EngineTransition
from cle.game_engine.models.actions import generate_playable_actions
from cle.game_engine.models.enums import Action, ActionPrompt, ActionType
from cle.game_engine.models.player import Color
from cle.game_engine.state_functions import player_key
from cle.harness.models import ModelRequest, ModelResponse
from cle.harness.public_speech import parse_public_speech
from cle.players.contracts import CommunicationChoice
from cle.sandbox.catan import PlayerResponseError
from cle.sandbox.contracts import SandboxStepResult

from .support import COLORS, build, main_engine, reply, sandbox, say


@pytest.mark.asyncio
@pytest.mark.parametrize("discards", [False, True])
async def test_seven_window_after_all_discards_once_across_retry_and_snapshot(discards: bool) -> None:
    engine = main_engine()
    for color in COLORS[:2] if discards else ():
        key = player_key(engine.state, color)
        engine.state.player_state[f"{key}_WOOD_IN_HAND"] = 10
    engine.step(Action(Color.RED, ActionType.ROLL, (3, 4)), force=True)
    sb, ts = sandbox(engine)
    while engine.state.current_prompt == ActionPrompt.DISCARD:
        # Every discarder of one 7 is prompted in the same step.
        for color in COLORS[:2]:
            count = sb.decision_context(color, (Action(color, ActionType.DISCARD, None),)).discard_count
            ts[color].replies = [{"tool": "discard", "arguments": {"cards": {"WOOD": count}}}]
        result = await sb.step()
        assert [context.actor for context in result.contexts] == list(COLORS[:2])
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
async def test_routine_events_never_poll_or_acknowledge_and_are_delivered_later() -> None:
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
async def test_parser_rejects_private_or_bundled_speech_atomically() -> None:
    sb, ts = sandbox()
    ctx = sb.decision_context()
    valid: Any = say(["BLUE"])
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
async def test_reply_round_bound_and_received_trigger_skip() -> None:
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
async def test_required_trade_replies_remain_simultaneous_private_and_no_extra_speech() -> None:
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
    entered: list[ModelRequest] = []

    async def reject(request: ModelRequest) -> ModelResponse:
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
async def test_shared_knight_remains_atomic_without_a_speech_await() -> None:
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


def test_all_shorthand_is_rejected_in_favor_of_explicit_colors() -> None:
    for shorthand in ("ALL", ["ALL"]):
        with pytest.raises(ValueError, match="distinct eligible other colors"):
            parse_public_speech(
                {"mode": "say", "text": "Trade?", "respondents": shorthand},
                speaker=Color.RED, participants=COLORS, max_notes_chars=4000,
            )


@pytest.mark.asyncio
async def test_targeted_offer_prompts_only_its_audience() -> None:
    engine = main_engine()
    key = player_key(engine.state, Color.RED)
    engine.state.player_state[f"{key}_WOOD_IN_HAND"] = 2
    engine.step(Action(Color.RED, ActionType.ROLL, (3, 5)), force=True)
    sb, ts = sandbox(engine)
    ts[Color.RED].replies = [{
        "tool": "offer_trade",
        "arguments": {"give": {"WOOD": 1}, "receive": {"ORE": 1}, "player": "BLUE"},
    }]
    ts[Color.BLUE].replies = [{"tool": "reject_offer", "arguments": {"give": {"ORE": 1}, "receive": {"WOOD": 1}, "player": "RED"}}]

    offered = await sb.step()
    assert offered.transitions[0].resolved_action.value.audience == frozenset({Color.BLUE})

    responded = await sb.step()
    assert [context.actor for context in responded.contexts] == [Color.BLUE]
    assert all(not ts[color].requests for color in COLORS[2:])
