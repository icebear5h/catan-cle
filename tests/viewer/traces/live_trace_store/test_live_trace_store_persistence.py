"""The SQLite trace store persists full attempts and restorable snapshots."""
import asyncio
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from cle.game_engine.events import PlayerEvent
from cle.game_engine.game import GameEngine
from cle.game_engine.models.player import Color
from cle.harness import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    PromptComponent,
    default_suite_path,
    load_context_suite,
)
from cle.harness.communication import default_communication_suite_path, load_communication_suite
from cle.players.agent import AgentPlayer
from cle.players.baseline import FirstLegalPlayer
from cle.players.contracts import (
    CommunicationChoice,
    CommunicationMode,
)
from cle.sandbox import CatanSandbox
from cle.sandbox.communication import (
    CommunicationAdmission,
    CommunicationOpportunity,
    ReactionReason,
)
from cle.traces import SQLiteLiveTraceStore
from playground.game_viewer.routes.websocket import build_game_state_snapshot
from playground.game_viewer.state import ServerState

from .support import COLORS, SequenceTransport


@pytest.mark.filterwarnings("ignore:Prompt suite .* is deprecated:DeprecationWarning")
def test_sqlite_trace_store_persists_full_attempts_and_restorable_snapshot(tmp_path: Path) -> None:
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
    red = AgentPlayer(
        Color.RED, transport, session_id=f"{engine.id}:RED",
        suite=load_context_suite(default_suite_path().with_name("catan_v10.yaml")),
        communication_suite=load_communication_suite(default_communication_suite_path()),
    )
    players: Any = {Color.RED: red}
    players.update({color: FirstLegalPlayer(color) for color in COLORS[1:]})
    sandbox: Any = CatanSandbox(engine, players)
    store: Any = SQLiteLiveTraceStore(tmp_path / "traces.sqlite3")
    game_id: Any = str(engine.id)
    store.start_game(
        game_id,
        config={"mode": "llm_vs_random", "seed": 4},
        snapshot=sandbox.snapshot(),
        display_name="  Opening study  ",
    )

    rejected_cursor: Any = len(sandbox.decision_trace)
    result: Any = asyncio.run(sandbox.step())
    state: Any = ServerState()
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
    opportunity: Any = CommunicationOpportunity(
        player=Color.RED,
        cause=PlayerEvent(0, "action:0", Color.RED, "BUILD_SETTLEMENT", 0),
        visible_through_sequence=0,
        reason=ReactionReason.MAJOR_BUILD,
        round=0,
    )
    communication_choice: Any = CommunicationChoice(
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
    trace: Any = store.get_game(game_id)
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
    usage: Any = store.get_usage(game_id)
    assert usage["step_count"] == 1
    assert usage["failure_calls"] == []
    assert [row["accepted"] for row in usage["calls"]] == [0, 1, 1]
    assert usage["calls"][1]["usage"] == {"completion_tokens_details": {"reasoning_tokens": 7}}
    assert all(set(row) == {"step_index", "call_index", "call_kind", "accepted", "usage"}
               for row in usage["calls"])
    for call in (rejected, completed, communication):
        assert call["request"]["context_policy"] is None
        assert call["request"]["memory_revision"] is None
        assert call["request"]["input_next_sequence"] is None
        assert call["request"]["channel"] is None
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

    checkpoint: Any = store.get_step(game_id, 0)
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

    resume_point: Any = store.load_resume_point(game_id)
    assert resume_point.step_index == 0
    assert resume_point.display_name == "Opening study"
    assert resume_point.public_state["running"] is True
    assert resume_point.config["mode"] == "llm_vs_random"

    snapshot: Any = store.load_snapshot(game_id, step_index=0)
    assert snapshot.pending_decision_revision is None
    session = dict(snapshot.player_states)[Color.RED].session
    assert session.context_policy == "legacy"
    assert session.action_next_sequence == 0
    assert session.talk_next_sequence == 0
    assert session.memory_revision == 0
    assert session.communication_receipts == ()
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
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 5
