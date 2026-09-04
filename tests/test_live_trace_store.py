import asyncio
import sqlite3
from dataclasses import dataclass, field

import pytest

from cle.harness import ModelMessage, ModelRequest, ModelResponse
from cle.players.agent import AgentPlayer
from cle.players.baseline import FirstLegalPlayer
from cle.players.contracts import CommunicationChoice, CommunicationMode
from cle.sandbox import CatanSandbox
from cle.sandbox.communication import CommunicationOpportunity, ReactionReason
from cle.traces import SQLiteLiveTraceStore
from cle.game_engine.events import PlayerEvent
from cle.game_engine.game import GameEngine
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
            "<rationale>take the exact first action</rationale>"
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
        communication_attempts=((opportunity, communication_choice),),
    )

    assert step_index == 0
    trace = store.get_game(game_id)
    assert trace is not None
    assert trace["status"] == "running"
    assert trace["display_name"] == "Opening study"
    assert trace["step_count"] == 1
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
    assert completed["response"]["native_reasoning"] == "private chain"
    assert completed["response"]["provider_response_id"] == "gen-accepted"
    assert completed["response"]["provider_request_payload"]["model"] == "test/model"
    assert "authorization" not in {
        key.lower()
        for key in completed["response"]["provider_request_payload"]
    }
    assert completed["choice"]["rationale"] == "take the exact first action"
    assert communication["call_kind"] == "communication"
    assert communication["response"]["provider_response_id"] == "gen-talk"
    assert communication["choice"]["text"] == "hello"

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
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 3


def test_trace_store_migrates_v2_database_for_game_names(tmp_path):
    path = tmp_path / "v2.sqlite3"
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
        connection.execute("PRAGMA user_version = 2")

    SQLiteLiveTraceStore(path)

    with sqlite3.connect(path) as connection:
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(live_games)")
        }
        assert "display_name" in columns
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 3
