import asyncio
import json
import pickle
import sqlite3
from dataclasses import dataclass, field, replace
from uuid import UUID

import pytest

from cle.harness import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    PromptComponent,
)
from cle.harness.catan_board_surface import ImageBoardPresenter
from cle.players.agent import AgentPlayer
from cle.players.baseline import FirstLegalPlayer
from cle.players.contracts import (
    CommunicationChoice,
    CommunicationMode,
    PlayerAttempt,
    PlayerChoice,
)
from cle.sandbox import CatanSandbox
from cle.sandbox.catan import PlayerResponseError, PostActionCommunicationError
from cle.sandbox.communication import CommunicationAdmission, CommunicationOpportunity, ReactionReason
from cle.sandbox.contracts import RetryPolicy
from cle.traces import SQLiteLiveTraceStore
from cle.game_engine.events import PlayerEvent
from cle.game_engine.game import GameEngine
from cle.game_engine.models.actions import generate_playable_actions
from cle.game_engine.models.enums import ActionPrompt
from cle.game_engine.models.player import Color
from playground.game_viewer.routes.websocket import build_game_state_snapshot
from playground.game_viewer.state import ServerState

COLORS = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)


@dataclass
class SequenceTransport:
    responses: list[ModelResponse]
    requests: list = field(default_factory=list)

    async def complete(self, request):
        self.requests.append(request)
        return self.responses.pop(0)


def test_sqlite_trace_store_persists_full_attempts_and_restorable_snapshot(tmp_path):
    invalid = ModelResponse(
        content="<action>999</action>",
        model="test/model",
        provider_request_payload={"messages": ["invalid"]},
        provider_response_payload={"id": "gen-invalid"},
    )
    accepted = ModelResponse(
        content=(
            "<game_plan>expand toward ore</game_plan>"
            "<action>0</action>"
        ),
        model="test/model",
        usage=(("completion_tokens_details", {"reasoning_tokens": 7}),),
        finish_reason="stop",
        native_reasoning="private chain",
        native_reasoning_details=({"type": "reasoning.text"},),
        reasoning_request=(("effort", "high"), ("exclude", False)),
        provider_response_id="gen-accepted",
        provider_request_id="req-accepted",
        provider_native_finish_reason="stop",
        provider_request_payload={"model": "test/model", "messages": ["full"]},
        provider_response_payload={"id": "gen-accepted", "choices": [{}]},
    )
    transport = SequenceTransport([invalid, accepted])
    engine = GameEngine(COLORS, seed=4, shuffle_players=False)
    red = AgentPlayer(Color.RED, transport, session_id=f"{engine.id}:RED")
    players = {Color.RED: red}
    players.update({color: FirstLegalPlayer(color) for color in COLORS[1:]})
    sandbox = CatanSandbox(engine, players)
    store = SQLiteLiveTraceStore(tmp_path / "traces.sqlite3")
    game_id = str(engine.id)
    store.start_game(
        game_id,
        config={"mode": "llm_vs_random", "seed": 4},
        snapshot=sandbox.snapshot(),
        display_name="  Opening study  ",
    )

    rejected_cursor = len(sandbox.decision_trace)
    result = asyncio.run(sandbox.step())
    state = ServerState()
    state.current_sandbox = sandbox
    state.game_running = True
    communication_request = ModelRequest(
        decision_id="talk:0:RED",
        session_id="talk-session",
        messages=(ModelMessage("user", "say something"),),
        components=(
            PromptComponent(
                id="environment.trigger",
                channel="environment",
                template="TRIGGER:\n{{ value }}",
                value="a build occurred",
                rendered="TRIGGER:\na build occurred",
                variables=(("value", "a build occurred"),),
            ),
        ),
    )
    communication_response = ModelResponse(
        content="<say>hello</say>",
        model="test/model",
        provider_response_id="gen-talk",
        provider_request_payload={"messages": ["say something"]},
        provider_response_payload={"id": "gen-talk"},
    )
    opportunity = CommunicationOpportunity(
        player=Color.RED,
        cause=PlayerEvent(0, "action:0", Color.RED, "BUILD_SETTLEMENT", 0),
        visible_through_sequence=0,
        reason=ReactionReason.MAJOR_BUILD,
        round=0,
    )
    communication_choice = CommunicationChoice(
        mode=CommunicationMode.SAY,
        text="hello",
        audience=COLORS,
        model_request=communication_request,
        model_response=communication_response,
    )
    step_index = store.record_step(
        game_id,
        result=result,
        rejected_attempts=sandbox.decision_trace[rejected_cursor:],
        public_state=build_game_state_snapshot(state),
        snapshot=sandbox.snapshot(),
        communication_attempts=(
            CommunicationAdmission(opportunity, communication_choice, accepted=True),
        ),
    )

    assert step_index == 0
    trace = store.get_game(game_id)
    assert trace is not None
    assert trace["status"] == "running"
    assert trace["display_name"] == "Opening study"
    assert trace["step_count"] == 1
    assert trace["failures"] == []
    assert trace["steps"][0]["before_revision"] == 0
    assert trace["steps"][0]["after_revision"] == 1
    assert len(trace["events"]) == 1
    assert [call["accepted"] for call in trace["model_calls"]] == [
        False,
        True,
        True,
    ]
    rejected, completed, communication = trace["model_calls"]
    assert rejected["validation_error"].startswith("Choose an action index")
    assert rejected["response"]["provider_response_payload"] == {
        "id": "gen-invalid"
    }
    assert completed["request"]["messages"][-1]["role"] == "user"
    assert completed["request"]["components"][0]["id"] == "system.identity"
    assert completed["request"]["board_presentation"]["kind"] == "text"
    assert completed["request"]["board_presentation"]["format"] == (
        "indexed_tile_rows/v3"
    )
    assert "CATAN FULL PUBLIC GRAPH V1" in (
        completed["request"]["board_presentation"]["content"]
    )
    assert completed["request"]["components"][1]["channel"] == "environment"
    assert completed["response"]["native_reasoning"] == "private chain"
    assert completed["response"]["provider_response_id"] == "gen-accepted"
    assert completed["response"]["provider_request_payload"]["model"] == "test/model"
    assert "authorization" not in {
        key.lower()
        for key in completed["response"]["provider_request_payload"]
    }
    assert "rationale" not in completed["choice"]
    assert communication["call_kind"] == "communication"
    assert communication["response"]["provider_response_id"] == "gen-talk"
    assert communication["choice"]["text"] == "hello"
    assert communication["request"]["components"] == [
        {
            "id": "environment.trigger",
            "channel": "environment",
            "template": "TRIGGER:\n{{ value }}",
            "value": "a build occurred",
            "rendered": "TRIGGER:\na build occurred",
            "variables": {"value": "a build occurred"},
        }
    ]

    checkpoint = store.get_step(game_id, 0)
    assert checkpoint["latest_step_index"] == 0
    assert checkpoint["step"]["after_revision"] == 1
    assert checkpoint["step"]["public_state"]["game"] is not None
    assert [call["call_kind"] for call in checkpoint["model_calls"]] == [
        "decision",
        "decision",
        "communication",
    ]
    assert store.get_step(game_id, 1) is None

    with pytest.raises(sqlite3.IntegrityError):
        store.record_step(
            game_id,
            result=result,
            rejected_attempts=(),
            public_state=build_game_state_snapshot(state),
            snapshot=sandbox.snapshot(),
        )
    assert store.get_game(game_id)["step_count"] == 1

    resume_point = store.load_resume_point(game_id)
    assert resume_point.step_index == 0
    assert resume_point.display_name == "Opening study"
    assert resume_point.public_state["running"] is True
    assert resume_point.config["mode"] == "llm_vs_random"

    snapshot = store.load_snapshot(game_id, step_index=0)
    sandbox.restore(snapshot)
    assert sandbox.revision == 1
    assert red.session.strategic_memory == "expand toward ore"

    assert store.rename_game(game_id, "  Named game  ") is True
    assert store.get_game(game_id)["display_name"] == "Named game"
    assert store.rename_game("missing", "No game") is False
    with pytest.raises(ValueError, match="at most 80"):
        store.rename_game(game_id, "x" * 81)

    store.mark_game_status(game_id, status="reset")
    assert store.get_game(game_id)["status"] == "reset"

    with sqlite3.connect(store.path) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 4


@pytest.mark.parametrize("model_backed", [False, True])
def test_communication_admission_outcomes_survive_all_trace_serializers(
    tmp_path, monkeypatch, model_backed
):
    engine = GameEngine(COLORS, seed=4, shuffle_players=False)
    response = ModelResponse(
        content="<message>I can offer WOOD.</message><audience>RED</audience><intent>TRADE</intent>",
        model="test/model",
        native_reasoning="private speech reasoning",
    )

    class Speaker(FirstLegalPlayer):
        async def communicate(self, context):
            return CommunicationChoice(
                mode=CommunicationMode.SAY, text="I can offer WOOD.",
                audience=(Color.RED,), intent="TRADE",
            )

    if model_backed:
        players = {
            color: AgentPlayer(
                color,
                SequenceTransport([
                    ModelResponse(content="<game_plan>opening</game_plan><action>0</action>")
                    if color == Color.RED
                    else replace(response, provider_response_id=f"speech-{color.value}")
                ]),
                session_id=f"{engine.id}:{color.value}",
            )
            for color in COLORS
        }
    else:
        players = {Color.RED: FirstLegalPlayer(Color.RED)}
        players.update({color: Speaker(color) for color in COLORS[1:]})
    sandbox = CatanSandbox(engine, players)
    append_message = engine.append_message

    def reject_white(**kwargs):
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
    trace = store.get_game(engine.id)
    saved_step = trace["steps"][0]["result"]
    communications = saved_step["communication_attempts"]
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
    message = trace["events"][1]
    assert message["actor"] == "BLUE"
    assert message["event"] == saved_step["message_events"][0]
    assert message["event"]["payload"] is None
    assert message["event"]["visible_to"] == ["BLUE", "RED"]
    assert [color for color, _ in message["event"]["private_overlays"]] == ["BLUE", "RED"]
    model_calls = [call for call in trace["model_calls"] if call["call_kind"] == "communication"]
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


def test_trace_event_index_merges_ordered_unique_events_without_backfilling(tmp_path):
    engine = GameEngine(COLORS, seed=4, shuffle_players=False)
    sandbox = CatanSandbox(engine, {color: FirstLegalPlayer(color) for color in COLORS})
    store = SQLiteLiveTraceStore(tmp_path / "event-index.sqlite3")
    store.start_game(engine.id, config={}, snapshot=sandbox.snapshot())
    first = asyncio.run(sandbox.step())
    private = engine.append_message(
        speaker=Color.BLUE, text="Private offer", audience=(Color.RED,),
        intent="TRADE", causation_id="private-offer",
    )
    second = asyncio.run(sandbox.step())
    public = engine.append_message(
        speaker=Color.RED, text="Public reply", audience=COLORS,
        intent=None, causation_id="public-reply",
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

    trace = SQLiteLiveTraceStore(store.path).get_game(engine.id)
    indexed = trace["events"]
    assert [event["sequence"] for event in indexed] == [0, 1, 2, 3]
    assert [event["step_index"] for event in indexed] == [0, 0, 0, 0]
    assert [event["event_type"] for event in indexed] == [
        "BUILD_SETTLEMENT", "MESSAGE_SENT", "BUILD_ROAD", "MESSAGE_SENT",
    ]
    saved_messages = trace["steps"][0]["result"]["message_events"]
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
    reopened = SQLiteLiveTraceStore(store.path)
    assert [event["sequence"] for event in reopened.get_game(engine.id)["events"]] == [0, 2]
    assert reopened.get_step(engine.id, 0)["step"]["result"]["message_events"] == saved_messages
    assert reopened.load_snapshot(engine.id).engine.events == tuple(engine.events)
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("SELECT * FROM live_steps").fetchall() == stored_step
        assert connection.execute("SELECT COUNT(*) FROM game_events").fetchone()[0] == 2
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 4


@pytest.mark.parametrize("discarding", [False, True])
def test_stored_context_retains_only_visible_messages_commitments_and_discard_facts(
    tmp_path, discarding
):
    engine = GameEngine(COLORS, seed=4, shuffle_players=False)
    if discarding:
        engine.state.is_initial_build_phase = False
        engine.state.is_discarding = True
        engine.state.current_prompt = ActionPrompt.DISCARD
        engine.state.player_state["P0_WOOD_IN_HAND"] = 8
        engine.state.resource_freqdeck[0] -= 8
        engine.state.playable_actions = generate_playable_actions(engine.state)
    visible = engine.append_message(
        speaker=Color.BLUE, text="Visible promise", audience=(Color.RED,),
        intent="TRADE", causation_id="visible",
        commitment=("if you offer ore", "I will give wood", 3),
    )
    engine.append_message(
        speaker=Color.WHITE, text="Private to orange", audience=(Color.ORANGE,),
        intent="TRADE", causation_id="hidden",
        commitment=("hidden condition", "hidden promise", 3),
    )
    players = {color: FirstLegalPlayer(color) for color in COLORS}
    players[Color.RED] = AgentPlayer(
        Color.RED,
        SequenceTransport([ModelResponse(
            content='<action>0</action><discard>{"WOOD":4}</discard>'
            if discarding else "<action>0</action>",
        )]),
        session_id=f"{engine.id}:RED",
    )
    sandbox = CatanSandbox(engine, players, retry_policy=RetryPolicy(1))
    store = SQLiteLiveTraceStore(tmp_path / "context.sqlite3")
    store.start_game(engine.id, config={}, snapshot=sandbox.snapshot())
    result = asyncio.run(sandbox.step())
    store.record_step(
        engine.id, result=result, rejected_attempts=(), public_state={},
        snapshot=sandbox.snapshot(),
    )

    stored = store.get_step(engine.id, 0)["step"]["result"]
    context = stored["contexts"][0]
    assert context["actor"] == "RED"
    assert context["events"] == []
    assert context["discard_count"] == (4 if discarding else 0)
    assert len(context["recent_messages"]) == 1
    message = context["recent_messages"][0]
    assert message["sequence"] == visible.sequence
    assert message["payload"]["text"] == "Visible promise"
    assert message["private_overlays"] == []
    assert len(context["active_commitments"]) == 1
    assert context["active_commitments"][0]["promise"] == "I will give wood"
    assert "hidden" not in json.dumps(context)
    assert "Private to orange" not in json.dumps(context)
    choice = stored["accepted_attempts"][0]["choice"]
    assert choice["discard_cards"] == (["WOOD"] * 4 if discarding else None)
    sandbox.restore(store.load_snapshot(engine.id))
    receipt = players[Color.RED].session.receipts[result.context.context_id]
    assert receipt.choice.discard_cards == (("WOOD",) * 4 if discarding else None)


@pytest.mark.parametrize("completed_steps", [0, 1])
def test_trace_store_failures_survive_reopen_and_success_without_changing_resume(
    tmp_path, monkeypatch, completed_steps
):
    accepted = ModelResponse(content="<action>0</action>", model="test/model")
    invalid = ModelResponse(content="<action>999</action>", model="test/model")
    exhausted = ModelResponse(
        content="", model="test/model", finish_reason="length",
        native_reasoning="still thinking",
    )
    transport = SequenceTransport(
        [accepted] * completed_steps + [invalid, exhausted] * 3 + [accepted]
    )
    engine = GameEngine(COLORS, seed=4, shuffle_players=False)
    red = AgentPlayer(Color.RED, transport, session_id=f"{engine.id}:RED")
    players = {Color.RED: red}
    players.update({color: FirstLegalPlayer(color) for color in COLORS[1:]})
    sandbox = CatanSandbox(engine, players, retry_policy=RetryPolicy(2))
    store = SQLiteLiveTraceStore(tmp_path / "failures.sqlite3")
    game_id = str(engine.id)
    store.start_game(
        game_id, config={"seed": 4}, snapshot=sandbox.snapshot(),
        display_name="Failure study",
    )
    for _ in range(completed_steps):
        result = asyncio.run(sandbox.step())
        store.record_step(
            game_id, result=result, rejected_attempts=(),
            public_state={"revision": sandbox.revision}, snapshot=sandbox.snapshot(),
        )

    before = store.get_game(game_id)
    listed_before = store.list_games()
    resume_before = store.load_resume_point(game_id)
    with sqlite3.connect(store.path) as connection:
        game_row = connection.execute("SELECT * FROM live_games").fetchall()
        step_rows = connection.execute("SELECT * FROM live_steps").fetchall()

    failure_ids = []
    recorded_times = [
        "2026-09-07T12:00:00+00:00",
        "2026-09-07T12:00:00+00:00",
        "2026-09-07T11:00:00+00:00",
    ]
    with monkeypatch.context() as patch:
        times = iter(recorded_times)
        patch.setattr("cle.traces.sqlite._utc_now", lambda: next(times))
        for _ in range(3):
            with pytest.raises(PlayerResponseError) as error:
                asyncio.run(sandbox.step())
            exc = error.value
            failure_ids.append(store.record_failure(
                game_id,
                revision=sandbox.revision,
                player=exc.player,
                validation_error=exc.validation_error,
                attempts=iter(exc.attempts),
            ))
            store = SQLiteLiveTraceStore(store.path)
            assert [row["failure_id"] for row in store.get_game(game_id)["failures"]] == (
                failure_ids
            )

    assert len(set(failure_ids)) == 3
    assert all(UUID(failure_id).version == 4 for failure_id in failure_ids)
    trace = store.get_game(game_id)
    failures = trace["failures"]
    assert [row["recorded_at"] for row in failures] == recorded_times
    assert {**trace, "failures": []} == before
    assert store.list_games() == listed_before
    assert store.get_step(game_id, completed_steps) is None
    for failure in failures:
        assert failure["game_id"] == game_id
        assert failure["revision"] == completed_steps
        assert failure["actor"] == "RED"
        assert failure["validation_error"] == exc.validation_error
        assert failure["communication_attempts"] == []
        attempts = failure["attempts"]
        assert len(attempts) == 2
        assert all(attempt["accepted"] is False for attempt in attempts)
        assert all(attempt["validation_error"] for attempt in attempts)
        assert attempts[0]["model_response"]["content"] == invalid.content
        assert attempts[1]["model_response"]["content"] == ""
        assert attempts[1]["model_response"]["native_reasoning"] == "still thinking"
        assert attempts[1]["model_response"]["finish_reason"] == "length"
        assert attempts[1]["choice"] is None

    with sqlite3.connect(store.path) as connection:
        assert connection.execute("SELECT * FROM live_games").fetchall() == game_row
        assert connection.execute("SELECT * FROM live_steps").fetchall() == step_rows
    resume = store.load_resume_point(game_id)
    assert resume.step_index == resume_before.step_index
    assert resume.public_state == resume_before.public_state
    assert resume.config == resume_before.config
    assert resume.status == resume_before.status
    assert resume.winner == resume_before.winner
    assert resume.display_name == resume_before.display_name
    assert resume.snapshot.player_states == resume_before.snapshot.player_states
    assert store.load_snapshot(game_id).engine.events == resume_before.snapshot.engine.events
    sandbox.restore(resume.snapshot)
    assert sandbox.revision == completed_steps
    assert engine.state.rng.getstate() == resume_before.snapshot.engine.state.rng.getstate()

    result = asyncio.run(sandbox.step())
    assert store.record_step(
        game_id, result=result, rejected_attempts=(),
        public_state={"revision": sandbox.revision}, snapshot=sandbox.snapshot(),
    ) == completed_steps
    store = SQLiteLiveTraceStore(store.path)
    trace = store.get_game(game_id)
    assert trace["failures"] == failures
    assert trace["step_count"] == completed_steps + 1
    assert trace["steps"][-1]["before_revision"] == completed_steps
    assert len(trace["model_calls"]) == completed_steps + 1
    resume = store.load_resume_point(game_id)
    assert resume.step_index == completed_steps
    assert resume.public_state == {"revision": completed_steps + 1}
    sandbox.restore(resume.snapshot)
    assert sandbox.revision == completed_steps + 1


def test_trace_store_failure_requires_existing_game(tmp_path):
    store = SQLiteLiveTraceStore(tmp_path / "missing-game.sqlite3")
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        store.record_failure(
            "missing", revision=0, player=Color.RED,
            validation_error="invalid action", attempts=(),
        )
    assert store.get_game("missing") is None
    assert store.list_games() == []
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM live_failures").fetchone()[0] == 0


def test_trace_store_failure_preserves_raw_calls_with_sanitization(tmp_path):
    response = ModelResponse(
        content="<game_plan>expand</game_plan><action>999</action>",
        model="vision/model",
        usage=(("completion_tokens_details", {"reasoning_tokens": 7}),),
        latency_ms=123,
        finish_reason="stop",
        native_reasoning="private reasoning, not the final answer",
        native_reasoning_details=({"type": "reasoning.text", "text": "private detail"},),
        reasoning_request=(("effort", "high"), ("exclude", False)),
        provider_response_id="gen-failed",
        provider_request_id="req-failed",
        provider_native_finish_reason="stop",
        provider_request_payload={
            "headers": {"Authorization": "Bearer SECRET_AUTH", "x-request-id": "req-failed"},
            "api_key": "SECRET_KEY",
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,RAW_IMAGE_BYTES"},
                }, {"type": "text", "text": "choose an action"}],
            }],
        },
        provider_response_payload={
            "id": "gen-failed",
            "choices": [{"message": {"content": "<action>999</action>"}}],
            "metadata": ({"X-API-Key": "SECRET_NESTED", "access_token": "SECRET_TOKEN"},),
            "headers": {"Set-Cookie": "SECRET_COOKIE", "x-request-id": "req-failed"},
        },
    )
    engine = GameEngine(COLORS, seed=8, shuffle_players=False)
    red = AgentPlayer(
        Color.RED, SequenceTransport([response]), session_id=f"{engine.id}:RED",
        board_presenter=ImageBoardPresenter(image_size=512),
    )
    players = {Color.RED: red}
    players.update({color: FirstLegalPlayer(color) for color in COLORS[1:]})
    sandbox = CatanSandbox(engine, players)
    store = SQLiteLiveTraceStore(tmp_path / "sanitized-failures.sqlite3")
    game_id = str(engine.id)
    store.start_game(game_id, config={}, snapshot=sandbox.snapshot())
    attempt = asyncio.run(red.choose(sandbox.decision_context(Color.RED)))
    without_model = PlayerAttempt(
        context_id=attempt.context_id,
        choice=PlayerChoice(
            action_index=999, raw_response=response.content, rationale="legacy rationale",
            native_reasoning=response.native_reasoning,
        ),
        validation_error="outside the legal menu",
    )
    opportunity = CommunicationOpportunity(
        player=Color.BLUE,
        cause=PlayerEvent(0, "action:0", Color.RED, "BUILD_SETTLEMENT", 0),
        visible_through_sequence=0,
        reason=ReactionReason.MAJOR_BUILD,
        round=0,
    )
    communication = CommunicationChoice(
        mode=CommunicationMode.SAY, text="I need ore", audience=COLORS, intent="trade",
        model_request=replace(attempt.model_request, decision_id="talk:0:BLUE"),
        model_response=replace(response, content="<say>I need ore</say>"),
    )
    store.record_failure(
        game_id, revision=0, player=Color.RED, validation_error="outside the legal menu",
        attempts=iter((attempt, without_model)),
        communication_attempts=iter((
            CommunicationAdmission(opportunity, communication, accepted=True),
        )),
    )

    failure = SQLiteLiveTraceStore(store.path).get_game(game_id)["failures"][0]
    decision, no_model = failure["attempts"]
    talk = failure["communication_attempts"][0]
    assert no_model["model_request"] is None
    assert no_model["model_response"] is None
    assert no_model["choice"]["raw_response"] == response.content
    assert no_model["choice"]["native_reasoning"] == response.native_reasoning
    assert "rationale" not in no_model["choice"]
    assert decision["call_kind"] == "decision"
    assert decision["accepted"] is False
    assert talk["call_kind"] == "communication"
    assert talk["actor"] == "BLUE"
    assert talk["accepted"] is True
    assert talk["choice"]["text"] == "I need ore"
    assert talk["choice"]["audience"] == [color.value for color in COLORS]
    assert talk["opportunity"]["cause"]["event_type"] == "BUILD_SETTLEMENT"
    for call, original_request, content in (
        (decision, attempt.model_request, response.content),
        (talk, communication.model_request, communication.model_response.content),
    ):
        request = call["model_request"]
        assert request["decision_id"] == original_request.decision_id
        assert request["session_id"] == original_request.session_id
        assert request["messages"] == [
            {"role": message.role, "content": message.content}
            for message in original_request.messages
        ]
        assert request["components"] == [
            {
                "id": component.id, "channel": component.channel,
                "template": component.template, "value": component.value,
                "rendered": component.rendered, "variables": dict(component.variables),
            }
            for component in original_request.components
        ]
        board = request["board_presentation"]
        assert board["kind"] == "image"
        assert board["data"] is None
        assert board["byte_length"] > 0
        persisted = call["model_response"]
        assert persisted == {
            "content": content, "model": response.model,
            "usage": dict(response.usage), "latency_ms": 123, "finish_reason": "stop",
            "native_reasoning": response.native_reasoning,
            "native_reasoning_details": list(response.native_reasoning_details),
            "reasoning_request": dict(response.reasoning_request),
            "provider_response_id": "gen-failed", "provider_request_id": "req-failed",
            "provider_native_finish_reason": "stop",
            "provider_request_payload": {
                "headers": {"x-request-id": "req-failed"},
                "messages": [{
                    "role": "user",
                    "content": [{
                        "type": "image_url",
                        "image_url": {
                            "url": f"local-board-image://sha256/{board['content_sha256']}",
                        },
                    }, {"type": "text", "text": "choose an action"}],
                }],
            },
            "provider_response_payload": {
                "id": "gen-failed",
                "choices": response.provider_response_payload["choices"],
                "metadata": [{}], "headers": {"x-request-id": "req-failed"},
            },
        }
    with sqlite3.connect(store.path) as connection:
        serialized = connection.execute("SELECT payload_json FROM live_failures").fetchone()[0]
    assert json.loads(serialized)["attempts"] == failure["attempts"]
    assert "data:image" not in serialized
    assert "RAW_IMAGE_BYTES" not in serialized
    assert "SECRET_" not in serialized
    assert attempt.model_request.board_presentation.data.hex() not in serialized
    assert response.provider_request_payload["api_key"] == "SECRET_KEY"
    assert response.provider_response_payload["metadata"][0]["X-API-Key"] == "SECRET_NESTED"


def test_trace_store_persists_image_metadata_without_image_bytes(tmp_path):
    response = ModelResponse(
        content=(
            "<game_plan>expand</game_plan>"
            "<action>0</action>"
        ),
        model="vision/model",
        provider_request_payload={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,RAW_IMAGE_BYTES"
                            },
                        }
                    ],
                }
            ]
        },
    )
    transport = SequenceTransport([response])
    engine = GameEngine(COLORS, seed=8, shuffle_players=False)
    red = AgentPlayer(
        Color.RED,
        transport,
        session_id=f"{engine.id}:RED",
        board_presenter=ImageBoardPresenter(image_size=512),
    )
    players = {Color.RED: red}
    players.update({color: FirstLegalPlayer(color) for color in COLORS[1:]})
    sandbox = CatanSandbox(engine, players)
    store = SQLiteLiveTraceStore(tmp_path / "image-traces.sqlite3")
    game_id = str(engine.id)
    store.start_game(
        game_id,
        config={"mode": "llm_vs_random", "board_surface": "image"},
        snapshot=sandbox.snapshot(),
    )

    result = asyncio.run(sandbox.step())
    store.record_step(
        game_id,
        result=result,
        rejected_attempts=(),
        public_state={},
        snapshot=sandbox.snapshot(),
    )

    call = store.get_game(game_id)["model_calls"][0]
    board = call["request"]["board_presentation"]
    assert board["kind"] == "image"
    assert board["media_type"] == "image/png"
    assert board["data"] is None
    assert board["byte_length"] > 0
    persisted_url = call["response"]["provider_request_payload"]["messages"][0][
        "content"
    ][0]["image_url"]["url"]
    assert persisted_url == f"local-board-image://sha256/{board['content_sha256']}"

    with sqlite3.connect(store.path) as connection:
        serialized = "\n".join(
            value
            for row in connection.execute(
                "SELECT request_json, response_json FROM model_calls"
            )
            for value in row
            if value
        )
    assert "data:image" not in serialized
    assert "RAW_IMAGE_BYTES" not in serialized
    assert transport.requests[0].board_presentation.data.hex() not in serialized


def test_trace_store_loads_snapshots_from_legacy_engine_namespace(tmp_path):
    engine = GameEngine(COLORS, seed=4, shuffle_players=False)
    sandbox = CatanSandbox(
        engine,
        {color: FirstLegalPlayer(color) for color in COLORS},
    )
    snapshot = sandbox.snapshot()
    payload = pickle.dumps(snapshot, protocol=0).replace(
        b"ccle.game_engine",
        b"cgame_engine",
    )
    assert b"cgame_engine" in payload

    store = SQLiteLiveTraceStore(tmp_path / "legacy.sqlite3")
    game_id = str(engine.id)
    store.start_game(
        game_id,
        config={"mode": "random", "seed": 4},
        snapshot=snapshot,
    )
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "UPDATE live_games SET initial_snapshot = ? WHERE game_id = ?",
            (payload, game_id),
        )

    restored = store.load_resume_point(game_id).snapshot
    assert restored.engine.state.colors == snapshot.engine.state.colors
    assert restored.engine.state.rng.getstate() == snapshot.engine.state.rng.getstate()


@pytest.mark.parametrize("schema_version", [2, 3])
def test_trace_store_migrates_legacy_database_without_changing_checkpoints(tmp_path, schema_version):
    path = tmp_path / f"v{schema_version}.sqlite3"
    engine = GameEngine(COLORS, seed=4, shuffle_players=False)
    sandbox = CatanSandbox(engine, {color: FirstLegalPlayer(color) for color in COLORS})
    game_id = str(engine.id)
    initial_snapshot = pickle.dumps(sandbox.snapshot())
    result = asyncio.run(sandbox.step())
    step_snapshot = pickle.dumps(sandbox.snapshot())
    result_json = json.dumps({
        "before_revision": result.before_revision,
        "after_revision": result.after_revision,
    })
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE live_games (
                game_id TEXT PRIMARY KEY,
                schema_version INTEGER NOT NULL,
                started_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                ended_at TEXT,
                status TEXT NOT NULL,
                winner TEXT,
                config_json TEXT NOT NULL,
                initial_snapshot BLOB NOT NULL
            )
            """
        )
        if schema_version == 3:
            connection.execute("ALTER TABLE live_games ADD COLUMN display_name TEXT")
        connection.execute(
            """
            INSERT INTO live_games (
                game_id, schema_version, started_at, updated_at, status,
                config_json, initial_snapshot
            ) VALUES (?, ?, 'started', 'updated', 'running', '{"seed":4}', ?)
            """,
            (game_id, schema_version, initial_snapshot),
        )
        connection.execute(
            """
            CREATE TABLE live_steps (
                game_id TEXT NOT NULL,
                step_index INTEGER NOT NULL,
                recorded_at TEXT NOT NULL,
                before_revision INTEGER NOT NULL,
                after_revision INTEGER NOT NULL,
                winner TEXT,
                result_json TEXT NOT NULL,
                public_state_json TEXT NOT NULL,
                sandbox_snapshot BLOB NOT NULL,
                PRIMARY KEY (game_id, step_index),
                UNIQUE (game_id, after_revision),
                FOREIGN KEY (game_id) REFERENCES live_games(game_id)
            )
            """
        )
        connection.execute(
            """
            INSERT INTO live_steps (
                game_id, step_index, recorded_at, before_revision, after_revision,
                result_json, public_state_json, sandbox_snapshot
            ) VALUES (?, 0, 'recorded', 0, 1, ?, '{"revision":1}', ?)
            """,
            (game_id, result_json, step_snapshot),
        )
        connection.execute(f"PRAGMA user_version = {schema_version}")

    store = SQLiteLiveTraceStore(path)
    trace = store.get_game(game_id)
    assert trace["failures"] == []
    assert trace["step_count"] == 1
    assert trace["steps"][0]["result"] == json.loads(result_json)
    failure_id = store.record_failure(
        game_id, revision=1, player=Color.RED,
        validation_error="invalid action", attempts=(),
    )
    store = SQLiteLiveTraceStore(path)
    assert store.get_game(game_id)["failures"][0]["failure_id"] == failure_id
    assert store.get_step(game_id, 0)["step"] == trace["steps"][0]
    resume = store.load_resume_point(game_id)
    assert resume.step_index == 0
    assert resume.config == {"seed": 4}
    assert resume.public_state == {"revision": 1}
    sandbox.restore(resume.snapshot)
    assert sandbox.revision == 1

    with sqlite3.connect(path) as connection:
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(live_games)")
        }
        assert "display_name" in columns
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 4
        assert connection.execute(
            "SELECT initial_snapshot, schema_version FROM live_games"
        ).fetchone() == (initial_snapshot, schema_version)
        assert connection.execute(
            "SELECT sandbox_snapshot FROM live_steps"
        ).fetchone()[0] == step_snapshot
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
