"""Standalone choices, resume, and reply bounds."""
import json
import pickle
from pathlib import Path
from typing import Any

import pytest

from cle.game_engine.communication import CommunicationLimits
from cle.game_engine.models.player import Color
from cle.harness.models import ModelRequest
from cle.sandbox.catan import PlayerResponseError
from cle.traces import SQLiteLiveTraceStore
from playground.game_viewer.live.reasoning_trace import build_live_reasoning_traces

from .support import COLORS, build, reply, sandbox, say


@pytest.mark.asyncio
async def test_standalone_public_choice_keeps_board_pending_and_records_one_notes_commit(tmp_path: Path) -> None:
    sb, ts = sandbox()
    engine: Any = sb.game_engine
    before = pickle.dumps(engine.state)
    decision = build(engine)
    store = SQLiteLiveTraceStore(tmp_path / "speech.sqlite3")
    store.start_game(str(engine.id), config={}, snapshot=sb.snapshot())

    def pass_after_hearing(request: ModelRequest) -> dict[str, Any]:
        assert pickle.dumps(engine.state) == before
        assert engine.state.actions == []
        assert request.trigger_reason == "addressed_speech"
        assert "Leave this spot open?" in request.messages[-1].content
        assert "private plan" not in request.messages[-1].content
        assert sb.players[Color.RED].session.memory_revision == 1
        return {"mode": "pass", "notes": "Remember RED asked"}

    def act_after_reply(request: ModelRequest) -> dict[str, Any]:
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
    spoken: Any = next(t for t in traces if t.get("trigger_reason") == "standalone_speech")
    assert spoken["communication_mode"] == "say" and spoken["channel"] == "action"
    assert spoken["usage"]["prompt_tokens"] == 17
    index = store.record_step(str(engine.id), result=result, rejected_attempts=(),
                              communication_attempts=sb.communication_trace, public_state={}, snapshot=sb.snapshot())
    assert index == 0
    saved: Any = store.get_game(str(engine.id))
    calls: Any = saved["model_calls"]
    assert len(calls) == 3
    spoken_call: Any = next(call for call in calls if call["choice"].get("trigger_reason") == "standalone_speech")
    assert spoken_call["call_kind"] == "communication"
    assert spoken_call["choice"]["text"] == "Leave this spot open?"
    assert spoken_call["request"]["channel"] == "action"
    assert json.loads(spoken_call["response"]["content"])["tool"] == "say"
    restored, _ = sandbox()
    restored.restore(store.load_resume_point(str(engine.id)).snapshot)
    assert restored.players[Color.RED].session.memory_revision == 2


@pytest.mark.asyncio
async def test_failure_after_say_resume_preserves_pending_action_and_cannot_say_twice() -> None:
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
    result: Any = await restored.step()
    assert len(restored.game_engine.state.actions) == 1
    assert not restored.communication_trace
    assert len([e for e in restored.game_engine.events if e.event_type == "MESSAGE_SENT"]) == 1
    assert result.attempts[0].model_request.memory_revision == 1


@pytest.mark.asyncio
async def test_broadcast_replies_are_bounded_and_pass_does_not_start_another_branch() -> None:
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
async def test_pending_reactions_resume_without_recalling_accepted_pass() -> None:
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
