"""Stored contexts keep only visible facts and resume unchanged after reopen."""
import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from cle.game_engine.game import GameEngine
from cle.game_engine.models.actions import generate_playable_actions
from cle.game_engine.models.enums import ActionPrompt
from cle.game_engine.models.player import Color
from cle.harness import (
    ModelResponse,
    default_suite_path,
    load_context_suite,
)
from cle.harness.communication import default_communication_suite_path, load_communication_suite
from cle.players.agent import AgentPlayer
from cle.players.baseline import FirstLegalPlayer
from cle.sandbox import CatanSandbox
from cle.sandbox.catan import PlayerResponseError
from cle.sandbox.contracts import RetryPolicy
from cle.traces import SQLiteLiveTraceStore

from .support import COLORS, SequenceTransport


@pytest.mark.parametrize("discarding", [False, True])
def test_stored_context_retains_only_visible_messages_commitments_and_discard_facts(
    tmp_path: Path, discarding: bool
) -> None:
    engine: Any = GameEngine(COLORS, seed=4, shuffle_players=False)
    if discarding:
        engine.state.is_initial_build_phase = False
        engine.state.is_discarding = True
        engine.state.current_prompt = ActionPrompt.DISCARD
        engine.state.player_state["P0_WOOD_IN_HAND"] = 8
        engine.state.resource_freqdeck[0] -= 8
        engine.state.playable_actions = generate_playable_actions(engine.state)
    visible: Any = engine.append_message(
        speaker=Color.BLUE, text="Visible promise", audience=(Color.RED,),
        causation_id="visible",
        commitment=("if you offer ore", "I will give wood", 3),
    )
    engine.append_message(
        speaker=Color.WHITE, text="Private to orange", audience=(Color.ORANGE,),
        causation_id="hidden",
        commitment=("hidden condition", "hidden promise", 3),
    )
    players: Any = {color: FirstLegalPlayer(color) for color in COLORS}
    players[Color.RED] = AgentPlayer(
        Color.RED,
        SequenceTransport([ModelResponse(
            content='<action>0</action><discard>{"WOOD":4}</discard>'
            if discarding else "<action>0</action>",
        )]),
        session_id=f"{engine.id}:RED",
        suite=load_context_suite(default_suite_path().with_name("catan_v10.yaml")),
        communication_suite=load_communication_suite(default_communication_suite_path()),
    )
    sandbox = CatanSandbox(engine, players, retry_policy=RetryPolicy(1))
    store: Any = SQLiteLiveTraceStore(tmp_path / "context.sqlite3")
    store.start_game(engine.id, config={}, snapshot=sandbox.snapshot())
    result: Any = asyncio.run(sandbox.step())
    store.record_step(
        engine.id, result=result, rejected_attempts=(), public_state={},
        snapshot=sandbox.snapshot(),
    )

    stored: Any = store.get_step(engine.id, 0)["step"]["result"]
    context: Any = stored["contexts"][0]
    assert context["actor"] == "RED"
    assert context["events"] == []
    assert context["discard_count"] == (4 if discarding else 0)
    assert context["visible_through_sequence"] == 1
    assert len(context["recent_messages"]) == 1
    message: Any = context["recent_messages"][0]
    assert message["sequence"] == visible.sequence
    assert message["payload"]["text"] == "Visible promise"
    assert message["private_overlays"] == []
    assert context["visible_messages"] == context["recent_messages"]
    assert len(context["active_commitments"]) == 1
    assert context["active_commitments"][0]["promise"] == "I will give wood"
    assert "hidden" not in json.dumps(context)
    assert "Private to orange" not in json.dumps(context)
    choice: Any = stored["accepted_attempts"][0]["choice"]
    assert choice["discard_cards"] == (["WOOD"] * 4 if discarding else None)
    sandbox.restore(store.load_snapshot(engine.id))
    receipt = players[Color.RED].session.receipts[result.context.context_id]
    assert receipt.choice.discard_cards == (("WOOD",) * 4 if discarding else None)


@pytest.mark.parametrize("completed_steps", [0, 1])
def test_trace_store_failures_survive_reopen_and_success_without_changing_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, completed_steps: int
) -> None:
    accepted = ModelResponse(content="<action>0</action>", model="test/model")
    invalid: Any = ModelResponse(content="<action>999</action>", model="test/model")
    exhausted = ModelResponse(
        content="", model="test/model", finish_reason="length",
        native_reasoning="still thinking",
    )
    transport = SequenceTransport(
        [accepted] * completed_steps + [invalid, exhausted] * 3 + [accepted]
    )
    engine = GameEngine(COLORS, seed=4, shuffle_players=False)
    red = AgentPlayer(
        Color.RED, transport, session_id=f"{engine.id}:RED",
        suite=load_context_suite(default_suite_path().with_name("catan_v10.yaml")),
        communication_suite=load_communication_suite(default_communication_suite_path()),
    )
    players: Any = {Color.RED: red}
    players.update({color: FirstLegalPlayer(color) for color in COLORS[1:]})
    sandbox: Any = CatanSandbox(engine, players, retry_policy=RetryPolicy(2))
    store: Any = SQLiteLiveTraceStore(tmp_path / "failures.sqlite3")
    game_id: Any = str(engine.id)
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

    before: Any = store.get_game(game_id)
    listed_before = store.list_games()
    resume_before = store.load_resume_point(game_id)
    with sqlite3.connect(store.path) as connection:
        game_row = connection.execute("SELECT * FROM live_games").fetchall()
        step_rows = connection.execute("SELECT * FROM live_steps").fetchall()

    failure_ids: Any = []
    recorded_times: Any = [
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
            exc: Any = error.value
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
    trace: Any = store.get_game(game_id)
    failures: Any = trace["failures"]
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
        attempts: Any = failure["attempts"]
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
    usage: Any = store.get_usage(game_id)
    assert len(usage["calls"]) == completed_steps + 1
    assert len(usage["failure_calls"]) == 6
    assert {row["failure_id"] for row in usage["failure_calls"]} == set(failure_ids)
    assert len({(row["failure_id"], row["call_kind"], row["call_index"])
                for row in usage["failure_calls"]}) == 6
    resume = store.load_resume_point(game_id)
    assert resume.step_index == completed_steps
    assert resume.public_state == {"revision": completed_steps + 1}
    sandbox.restore(resume.snapshot)
    assert sandbox.revision == completed_steps + 1
