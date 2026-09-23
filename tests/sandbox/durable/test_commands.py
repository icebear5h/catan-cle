import asyncio
from pathlib import Path

import pytest

from cle.game_engine.models.player import Color
from cle.sandbox.durable import DurableSandbox, DurableStepCancelled
from cle.traces.journal import CommandRecord, JournalConflict, SQLiteSandboxJournal

from .support import (
    POLICY,
    ScriptedTransport,
    SetupPair,
    assert_checkpoint,
    assert_engine,
    assert_result,
    drain,
    entry_kinds,
    fresh_sandbox,
    make_sandbox,
    successful,
)


@pytest.mark.asyncio
async def test_old_command_replays_after_advancement_without_rewinding(
    tmp_path: Path, pair: SetupPair,
) -> None:
    transport = ScriptedTransport(pair.response())
    sandbox, _ = make_sandbox(pair.before, {Color.RED: transport})
    journal = SQLiteSandboxJournal(tmp_path / "commands.sqlite3")
    runner = DurableSandbox.start(sandbox, journal, policy_identity=POLICY)
    first = await runner.step("settlement", expected_sequence=0)
    first_result = successful(first)
    assert first_result.transitions[0].requested_action == pair.settlement
    second = await runner.step("road", expected_sequence=first.finished_sequence)
    assert successful(second).transitions[0].requested_action == pair.road
    latest = sandbox.snapshot()
    entries = journal.entries(sandbox.game_engine.id)

    for expected in (None, 0):
        replay = await runner.step("settlement", expected_sequence=expected)
        assert replay.started_sequence == first.started_sequence
        assert replay.finished_sequence == first.finished_sequence
        assert_result(successful(replay), first_result)
        assert runner.checkpoint_sequence == second.finished_sequence
        assert runner.pending_command_id is None
        assert_checkpoint(sandbox, latest)
    with pytest.raises(JournalConflict):
        await runner.step("settlement", expected_sequence=runner.checkpoint_sequence)
    with pytest.raises(JournalConflict):
        await runner.step("stale-new-command", expected_sequence=0)
    assert journal.entries(sandbox.game_engine.id) == entries
    assert len(transport.requests) == 1
    assert_engine(sandbox.game_engine, pair.after_road)

    reopened = SQLiteSandboxJournal(journal.path)
    fresh, scripts = fresh_sandbox(reopened, sandbox.game_engine.id)
    resumed = DurableSandbox.resume(fresh, reopened, policy_identity=POLICY)
    assert_result(successful(await resumed.step("settlement")), first_result)
    assert resumed.checkpoint_sequence == second.finished_sequence
    assert_checkpoint(fresh, latest)
    assert not any(script.requests for script in scripts.values())


@pytest.mark.asyncio
async def test_concurrent_duplicate_waits_for_one_execution_and_one_commit(
    tmp_path: Path, pair: SetupPair,
) -> None:
    transport = ScriptedTransport(pair.response(), gated=True)
    sandbox, _ = make_sandbox(pair.before, {Color.RED: transport})
    journal = SQLiteSandboxJournal(tmp_path / "concurrent.sqlite3")
    runner = DurableSandbox.start(sandbox, journal, policy_identity=POLICY)
    first = asyncio.create_task(runner.step("same-command"))
    duplicate_entered = asyncio.Event()

    async def duplicate() -> CommandRecord:
        duplicate_entered.set()
        return await runner.step("same-command")

    # Queue duplicate delivery while the original is demonstrably in inference.
    tasks = [first]
    try:
        await asyncio.wait_for(transport.entered.wait(), timeout=3)
        second = asyncio.create_task(duplicate())
        tasks.append(second)
        await asyncio.wait_for(duplicate_entered.wait(), timeout=3)
        assert not first.done() and not second.done()
        assert sandbox.revision == 0
        assert journal.head(sandbox.game_engine.id).pending_command_id == "same-command"
        assert len(transport.requests) == 1
        transport.release.set()
        one, two = await asyncio.wait_for(asyncio.gather(first, second), timeout=3)
    finally:
        await drain(*tasks)

    assert_result(successful(one), successful(two))
    assert one.started_sequence == two.started_sequence
    assert one.finished_sequence == two.finished_sequence == runner.checkpoint_sequence
    assert_engine(sandbox.game_engine, pair.after_settlement)
    assert len(transport.requests) == 1
    assert entry_kinds(journal, sandbox.game_engine.id) == [
        "genesis", "step_started", "call_started", "call_completed", "step_succeeded",
    ]


@pytest.mark.asyncio
async def test_cancellation_has_durable_record_and_redelivery_never_reinvokes(
    tmp_path: Path, pair: SetupPair,
) -> None:
    transport = ScriptedTransport(pair.response(), gated=True)
    sandbox, _ = make_sandbox(pair.before, {Color.RED: transport})
    journal = SQLiteSandboxJournal(tmp_path / "cancelled.sqlite3")
    runner = DurableSandbox.start(sandbox, journal, policy_identity=POLICY)
    task = asyncio.create_task(runner.step("cancelled"))
    try:
        await asyncio.wait_for(transport.entered.wait(), timeout=3)
        task.cancel()
        with pytest.raises(DurableStepCancelled) as caught:
            await asyncio.wait_for(task, timeout=3)
    finally:
        await drain(task)
    record = caught.value.record
    assert record.outcome is not None
    assert record.outcome.status == "cancelled"
    assert record.outcome.error_type == "CancelledError"
    assert record.outcome.result is None
    assert runner.pending_command_id is None
    assert record.finished_sequence == runner.checkpoint_sequence
    assert_checkpoint(sandbox, record.outcome.snapshot)
    assert_engine(sandbox.game_engine, pair.before)
    assert entry_kinds(journal, sandbox.game_engine.id) == [
        "genesis", "step_started", "call_started", "step_cancelled",
    ]

    reopened = SQLiteSandboxJournal(journal.path)
    fresh, scripts = fresh_sandbox(reopened, sandbox.game_engine.id)
    resumed = DurableSandbox.resume(fresh, reopened, policy_identity=POLICY)
    replay = await resumed.step("cancelled")
    assert replay.finished_sequence == record.finished_sequence
    assert replay.outcome is not None and replay.outcome.status == "cancelled"
    assert not any(script.requests for script in scripts.values())
    assert len(transport.requests) == 1
    assert_checkpoint(fresh, record.outcome.snapshot)
