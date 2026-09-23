"""Admission outcomes and the event index survive every trace serializer."""
import asyncio
import sqlite3
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from cle.game_engine.events import GameEvent
from cle.game_engine.game import GameEngine
from cle.game_engine.models.player import Color
from cle.harness import (
    ModelResponse,
    default_suite_path,
    load_context_suite,
)
from cle.harness.communication import default_communication_suite_path, load_communication_suite
from cle.players.agent import AgentPlayer
from cle.players.baseline import FirstLegalPlayer
from cle.players.contracts import (
    CommunicationChoice,
    CommunicationMode,
    TalkContext,
)
from cle.sandbox import CatanSandbox
from cle.sandbox.catan import PostActionCommunicationError
from cle.traces import SQLiteLiveTraceStore

from .support import COLORS, SequenceTransport


@pytest.mark.filterwarnings("ignore:Prompt suite .* is deprecated:DeprecationWarning")
@pytest.mark.parametrize("model_backed", [False, True])
def test_communication_admission_outcomes_survive_all_trace_serializers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_backed: bool
) -> None:
    engine: Any = GameEngine(COLORS, seed=4, shuffle_players=False)
    response: Any = ModelResponse(
        content="<message>I can offer WOOD.</message><audience>RED</audience><intent>TRADE</intent>",
        model="test/model",
        native_reasoning="private speech reasoning",
    )

    class Speaker(FirstLegalPlayer):
        async def communicate(self, context: TalkContext) -> CommunicationChoice:
            return CommunicationChoice(
                mode=CommunicationMode.SAY, text="I can offer WOOD.",
                audience=(Color.RED,),
            )

    if model_backed:
        players: Any = {
            color: AgentPlayer(
                color,
                SequenceTransport([
                    ModelResponse(content="<game_plan>opening</game_plan><action>0</action>")
                    if color == Color.RED
                    else replace(response, provider_response_id=f"speech-{color.value}")
                ]),
                session_id=f"{engine.id}:{color.value}",
                suite=load_context_suite(default_suite_path().with_name("catan_v10.yaml")),
                communication_suite=load_communication_suite(default_communication_suite_path()),
            )
            for color in COLORS
        }
    else:
        players = {Color.RED: FirstLegalPlayer(Color.RED)}
        players.update({color: Speaker(color) for color in COLORS[1:]})
    sandbox = CatanSandbox(engine, players)
    append_message = engine.append_message

    def reject_white(**kwargs: object) -> GameEvent:
        if kwargs["speaker"] == Color.WHITE:
            raise ValueError("Message admission denied for WHITE")
        return append_message(**kwargs)

    monkeypatch.setattr(engine, "append_message", reject_white)
    store = SQLiteLiveTraceStore(tmp_path / "speech-admission.sqlite3")
    store.start_game(engine.id, config={}, snapshot=sandbox.snapshot())

    with pytest.raises(PostActionCommunicationError) as caught:
        asyncio.run(sandbox.step())

    result = caught.value.result
    assert len(result.transitions) == 1
    assert len(result.messages) == 1
    assert sandbox.revision == 2
    if model_backed:
        assert len(players[Color.RED].session.receipts) == 1
        assert players[Color.RED].session.strategic_memory == "opening"
    else:
        assert players[Color.RED].accepted_choices == 1

    store.record_step(
        engine.id, result=result, rejected_attempts=(),
        communication_attempts=sandbox.communication_trace,
        public_state={"revision": sandbox.revision}, snapshot=sandbox.snapshot(),
    )
    store.record_failure(
        engine.id, revision=sandbox.revision, player=Color.WHITE,
        validation_error=str(caught.value.__cause__), attempts=(),
        communication_attempts=iter(sandbox.communication_trace),
    )
    store = SQLiteLiveTraceStore(store.path)
    trace: Any = store.get_game(engine.id)
    saved_step: Any = trace["steps"][0]["result"]
    communications: Any = saved_step["communication_attempts"]
    assert saved_step["accepted_attempts"][0]["accepted"] is True
    assert [call["actor"] for call in communications] == [color.value for color in COLORS[1:]]
    assert [call["accepted"] for call in communications] == [True, False, False]
    assert communications[0]["validation_error"] is None
    assert communications[1]["validation_error"] == "Message admission denied for WHITE"
    assert "withheld after WHITE rejection" in communications[2]["validation_error"]
    assert trace["failures"][0]["communication_attempts"] == communications
    assert [event["event_type"] for event in trace["events"]] == [
        "BUILD_SETTLEMENT", "MESSAGE_SENT",
    ]
    message: Any = trace["events"][1]
    assert message["actor"] == "BLUE"
    assert message["event"] == saved_step["message_events"][0]
    assert message["event"]["payload"] is None
    assert message["event"]["visible_to"] == ["BLUE", "RED"]
    assert [color for color, _ in message["event"]["private_overlays"]] == ["BLUE", "RED"]
    model_calls: Any = [call for call in trace["model_calls"] if call["call_kind"] == "communication"]
    if model_backed:
        assert [call["accepted"] for call in model_calls] == [True, False, False]
        for call, admission in zip(model_calls, communications):
            assert call["validation_error"] == admission["validation_error"]
            assert call["response"]["content"] == response.content
            assert call["response"]["native_reasoning"] == response.native_reasoning
            assert call["response"]["provider_response_id"] == f"speech-{call['actor']}"
            assert call["request"]["session_id"] == f"{engine.id}:{call['actor']}"
    else:
        assert model_calls == []
    assert len(store.load_snapshot(engine.id).engine.events) == 2


def test_trace_event_index_merges_ordered_unique_events_without_backfilling(tmp_path: Path) -> None:
    engine = GameEngine(COLORS, seed=4, shuffle_players=False)
    sandbox = CatanSandbox(engine, {color: FirstLegalPlayer(color) for color in COLORS})
    store = SQLiteLiveTraceStore(tmp_path / "event-index.sqlite3")
    store.start_game(engine.id, config={}, snapshot=sandbox.snapshot())
    first = asyncio.run(sandbox.step())
    private = engine.append_message(
        speaker=Color.BLUE, text="Private offer", audience=(Color.RED,),
        causation_id="private-offer",
    )
    second = asyncio.run(sandbox.step())
    public = engine.append_message(
        speaker=Color.RED, text="Public reply", audience=COLORS,
        causation_id="public-reply",
    )
    # Exercise overlapping event sources and non-sequential batch enumeration.
    result = replace(
        first,
        contexts=first.contexts + second.contexts,
        attempts=first.attempts + second.attempts,
        transitions=(
            replace(first.transitions[0], events=(private, *first.transitions[0].events)),
            replace(second.transitions[0], events=(public, *second.transitions[0].events)),
        ),
        messages=(public, private, private),
    )
    store.record_step(
        engine.id, result=result, rejected_attempts=(), public_state={},
        snapshot=sandbox.snapshot(),
    )

    trace: Any = SQLiteLiveTraceStore(store.path).get_game(engine.id)
    indexed: Any = trace["events"]
    assert [event["sequence"] for event in indexed] == [0, 1, 2, 3]
    assert [event["step_index"] for event in indexed] == [0, 0, 0, 0]
    assert [event["event_type"] for event in indexed] == [
        "BUILD_SETTLEMENT", "MESSAGE_SENT", "BUILD_ROAD", "MESSAGE_SENT",
    ]
    saved_messages: Any = trace["steps"][0]["result"]["message_events"]
    assert indexed[1]["event"] == saved_messages[1]
    assert indexed[3]["event"] == saved_messages[0]
    assert indexed[1]["event"]["payload"] is None
    assert indexed[1]["event"]["visible_to"] == ["BLUE", "RED"]
    assert indexed[3]["event"]["payload"]["text"] == "Public reply"
    assert indexed[3]["event"]["private_overlays"] == []
    assert indexed[3]["event"]["visible_to"] is None

    with sqlite3.connect(store.path) as connection:
        assert connection.execute(
            "SELECT sequence FROM game_events ORDER BY rowid"
        ).fetchall() == [(0,), (1,), (2,), (3,)]
        # Reproduce a historical incomplete index only in this temporary store.
        connection.execute("DELETE FROM game_events WHERE event_type = 'MESSAGE_SENT'")
        stored_step = connection.execute("SELECT * FROM live_steps").fetchall()
    reopened: Any = SQLiteLiveTraceStore(store.path)
    assert [event["sequence"] for event in reopened.get_game(engine.id)["events"]] == [0, 2]
    assert reopened.get_step(engine.id, 0)["step"]["result"]["message_events"] == saved_messages
    assert reopened.load_snapshot(engine.id).engine.events == tuple(engine.events)
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("SELECT * FROM live_steps").fetchall() == stored_step
        assert connection.execute("SELECT COUNT(*) FROM game_events").fetchone()[0] == 2
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 5
