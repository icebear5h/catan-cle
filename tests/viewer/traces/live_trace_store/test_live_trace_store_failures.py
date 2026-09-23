"""Recorded failures need an existing game and sanitize raw provider calls."""
import asyncio
import json
import sqlite3
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from cle.game_engine.events import PlayerEvent
from cle.game_engine.game import GameEngine
from cle.game_engine.models.player import Color
from cle.harness import (
    ModelResponse,
    default_suite_path,
    load_context_suite,
)
from cle.harness.catan_board_surface import ImageBoardPresenter
from cle.harness.communication import default_communication_suite_path, load_communication_suite
from cle.players.agent import AgentPlayer
from cle.players.baseline import FirstLegalPlayer
from cle.players.contracts import (
    CommunicationChoice,
    CommunicationMode,
    PlayerAttempt,
    PlayerChoice,
)
from cle.sandbox import CatanSandbox
from cle.sandbox.communication import (
    CommunicationAdmission,
    CommunicationOpportunity,
    ReactionReason,
)
from cle.traces import SQLiteLiveTraceStore

from .support import COLORS, SequenceTransport


def test_recorded_step_public_state_accepts_late_step_labels(tmp_path: Path) -> None:
    engine = GameEngine(COLORS, seed=7, shuffle_players=False)
    sandbox: Any = CatanSandbox(engine, {color: FirstLegalPlayer(color) for color in COLORS})
    store: Any = SQLiteLiveTraceStore(tmp_path / "relabel.sqlite3")
    game_id: Any = str(engine.id)
    store.start_game(game_id, config={"seed": 7}, snapshot=sandbox.snapshot())
    result = asyncio.run(sandbox.step())
    step_index: Any = store.record_step(
        game_id, result=result, rejected_attempts=(),
        public_state={"game_log": [{"type": "message", "details": {"sequence": 3}}]},
        snapshot=sandbox.snapshot(),
    )

    stamped: Any = {
        "game_log": [
            {
                "type": "message",
                "step_index": step_index,
                "details": {"sequence": 3, "step_index": step_index},
            }
        ]
    }
    assert store.update_step_public_state(game_id, step_index, stamped) is True
    assert store.get_step(game_id, step_index)["step"]["public_state"] == stamped
    assert store.load_resume_point(game_id).public_state == stamped
    # Only the addressed step moves, and an unrecorded step is a no-op.
    assert store.update_step_public_state(game_id, step_index + 1, stamped) is False
    assert store.update_step_public_state("missing-game", step_index, stamped) is False
    assert store.get_game(game_id)["step_count"] == 1
    assert store.get_game(game_id)["steps"][0]["after_revision"] == sandbox.revision


def test_trace_store_failure_requires_existing_game(tmp_path: Path) -> None:
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


@pytest.mark.filterwarnings("ignore:Prompt suite .* is deprecated:DeprecationWarning")
def test_trace_store_failure_preserves_raw_calls_with_sanitization(tmp_path: Path) -> None:
    response: Any = ModelResponse(
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
        suite=load_context_suite(default_suite_path().with_name("catan_v10.yaml")),
        communication_suite=load_communication_suite(default_communication_suite_path()),
        board_presenter=ImageBoardPresenter(image_size=512),
    )
    players: Any = {Color.RED: red}
    players.update({color: FirstLegalPlayer(color) for color in COLORS[1:]})
    sandbox = CatanSandbox(engine, players)
    store: Any = SQLiteLiveTraceStore(tmp_path / "sanitized-failures.sqlite3")
    game_id: Any = str(engine.id)
    store.start_game(game_id, config={}, snapshot=sandbox.snapshot())
    attempt: Any = asyncio.run(red.choose(sandbox.decision_context(Color.RED)))
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
    communication: Any = CommunicationChoice(
        mode=CommunicationMode.SAY, text="I need ore", audience=COLORS,
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

    failure: Any = SQLiteLiveTraceStore(store.path).get_game(game_id)["failures"][0]
    usage: Any = store.get_usage(game_id)
    assert usage["calls"] == []
    assert usage["step_count"] == 0
    assert [(row["call_kind"], row["accepted"]) for row in usage["failure_calls"]] == [
        ("decision", 0), ("communication", 1),
    ]
    assert all(row["usage"] == dict(response.usage) for row in usage["failure_calls"])
    assert all(set(row) == {"failure_id", "call_index", "call_kind", "accepted", "usage"}
               for row in usage["failure_calls"])
    decision, no_model = failure["attempts"]
    talk: Any = failure["communication_attempts"][0]
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
        request: Any = call["model_request"]
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
        board: Any = request["board_presentation"]
        assert board["kind"] == "image"
        assert board["data"] is None
        assert board["byte_length"] > 0
        persisted: Any = call["model_response"]
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
        serialized: Any = connection.execute("SELECT payload_json FROM live_failures").fetchone()[0]
    assert json.loads(serialized)["attempts"] == failure["attempts"]
    assert "data:image" not in serialized
    assert "RAW_IMAGE_BYTES" not in serialized
    assert "SECRET_" not in serialized
    assert attempt.model_request.board_presentation.data.hex() not in serialized
    assert response.provider_request_payload["api_key"] == "SECRET_KEY"
    assert response.provider_response_payload["metadata"][0]["X-API-Key"] == "SECRET_NESTED"
