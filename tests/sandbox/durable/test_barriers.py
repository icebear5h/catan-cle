import asyncio
import json
from pathlib import Path

import pytest

from cle.game_engine.models.enums import Action, ActionPrompt, ActionType
from cle.game_engine.models.player import Color
from cle.sandbox.durable import DurableSandbox
from cle.traces.journal import SQLiteSandboxJournal

from .faults import FaultJournal, PersistenceFault
from .support import (
    POLICY,
    ScriptedTransport,
    agent,
    assert_checkpoint,
    assert_engine,
    drain,
    entry_kinds,
    fresh_sandbox,
    make_sandbox,
    seven_engine,
    successful,
)


@pytest.mark.asyncio
async def test_reversed_discard_completions_commit_in_seat_order_even_after_recovery(
    tmp_path: Path,
) -> None:
    oracle = seven_engine(discards=True)
    before = oracle.snapshot()
    red = ScriptedTransport(
        '{"tool":"discard","arguments":{"cards":{"WOOD":4}},"notes":"red kept four"}',
        gated=True,
    )
    white = ScriptedTransport(
        '{"tool":"discard","arguments":{"cards":{"WOOD":5}},"notes":"white kept five"}',
        gated=True,
    )
    sandbox, _ = make_sandbox(before, {Color.RED: red, Color.WHITE: white})
    initial = sandbox.snapshot()
    journal = FaultJournal(tmp_path / "discard.sqlite3", "before_settle")
    runner = DurableSandbox.start(sandbox, journal, policy_identity=POLICY)
    task = asyncio.create_task(runner.step("discard-barrier"))
    try:
        # Both seats must acquire concurrently; a sequential implementation stalls.
        await asyncio.wait_for(asyncio.gather(red.entered.wait(), white.entered.wait()), 3)
        white.release.set()
        await asyncio.wait_for(white.completed.wait(), 3)
        assert not task.done()
        assert_engine(sandbox.game_engine, before)
        assert agent(sandbox, Color.WHITE).session.memory_revision == 0
        entries = journal.entries(oracle.id)
        starts = [entry for entry in entries if entry.kind == "call_started"]
        assert len(starts) == 2
        sessions: list[str] = []
        for entry in starts:
            payload: object = json.loads(entry.payload_json)
            assert isinstance(payload, dict)
            request_json: object = payload["request_json"]
            assert isinstance(request_json, str)
            request: object = json.loads(request_json)
            assert isinstance(request, dict)
            session: object = request["session_id"]
            assert isinstance(session, str)
            sessions.append(session)
        assert sessions == ["durable:RED", "durable:WHITE"]
        assert [entry.call_key for entry in entries if entry.kind == "call_completed"] == [
            starts[1].call_key,
        ]
        red.release.set()
        with pytest.raises(PersistenceFault):
            await asyncio.wait_for(task, 3)
    finally:
        await drain(task)

    completed = [
        entry.call_key for entry in journal.entries(oracle.id) if entry.kind == "call_completed"
    ]
    assert completed == [starts[1].call_key, starts[0].call_key]
    assert_checkpoint(sandbox, initial)
    assert journal.attempted_outcome is not None
    attempted = journal.attempted_outcome.result
    assert attempted is not None
    assert [transition.resolved_action.color for transition in attempted.transitions] == [
        Color.RED, Color.WHITE,
    ]

    reopened = SQLiteSandboxJournal(journal.path)
    fresh, scripts = fresh_sandbox(reopened, oracle.id)
    resumed = DurableSandbox.resume(
        fresh, reopened, policy_identity=POLICY, recover_pending=True,
    )
    result = successful(await resumed.step("discard-barrier"))
    assert [context.actor for context in result.contexts] == [Color.RED, Color.WHITE]
    assert [context.discard_count for context in result.contexts] == [4, 5]
    assert [transition.resolved_action for transition in result.transitions] == [
        Action(Color.RED, ActionType.DISCARD, ("WOOD",) * 4),
        Action(Color.WHITE, ActionType.DISCARD, ("WOOD",) * 5),
    ]
    for transition in result.transitions:
        oracle.step(transition.requested_action)
    assert_engine(fresh.game_engine, oracle.snapshot())
    assert fresh.game_engine.state.current_prompt == ActionPrompt.MOVE_ROBBER
    assert fresh.game_engine.rng.getstate() == before.state.rng.getstate()
    assert agent(fresh, Color.RED).session.strategic_memory == "red kept four"
    assert agent(fresh, Color.WHITE).session.strategic_memory == "white kept five"
    assert agent(fresh, Color.RED).session.memory_revision == 1
    assert agent(fresh, Color.WHITE).session.memory_revision == 1
    assert len(red.requests) == len(white.requests) == 1
    assert not any(script.requests for script in scripts.values())
    kinds = entry_kinds(reopened, oracle.id)
    assert kinds.count("call_started") == kinds.count("call_completed") == 2
    assert kinds.count("step_started") == kinds.count("step_succeeded") == 1
    assert "call_retried" not in kinds
