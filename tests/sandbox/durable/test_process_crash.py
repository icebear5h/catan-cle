"""A real process death cannot run Python cleanup or flush an in-memory result."""

import asyncio
import multiprocessing
import os
from pathlib import Path

import pytest

from cle.game_engine.models.player import Color
from cle.sandbox.durable import DurableSandbox
from cle.traces.journal import CommandOutcome, CommandRecord, CommandTicket, SQLiteSandboxJournal

from .support import (
    POLICY,
    ScriptedTransport,
    SetupPair,
    assert_engine,
    fresh_sandbox,
    make_sandbox,
    successful,
)


class CrashBeforeSettlement(SQLiteSandboxJournal):
    def settle(self, ticket: CommandTicket, outcome: CommandOutcome) -> CommandRecord:
        os._exit(17)


def crash_worker(path: str, pair: SetupPair) -> None:
    sandbox, _ = make_sandbox(pair.before, {Color.RED: ScriptedTransport(pair.response())})
    runner = DurableSandbox.start(sandbox, CrashBeforeSettlement(path), policy_identity=POLICY)
    asyncio.run(runner.step("placement"))


@pytest.mark.asyncio
async def test_killed_worker_replays_receipt_and_commits_once(tmp_path: Path, pair: SetupPair) -> None:
    path = tmp_path / "crashed.sqlite3"
    worker = multiprocessing.get_context("spawn").Process(target=crash_worker, args=(str(path), pair))
    worker.start()
    try:
        worker.join(timeout=15)
        assert worker.exitcode == 17
    finally:
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=5)
        worker.close()
    store = SQLiteSandboxJournal(path)
    head = store.head(pair.before.engine_id)
    assert head.checkpoint_sequence == 0 and head.pending_command_id == "placement"
    assert [entry.kind for entry in store.entries(head.sandbox_id)] == [
        "genesis", "step_started", "call_started", "call_completed",
    ]
    sandbox, scripts = fresh_sandbox(store, head.sandbox_id)
    resumed = DurableSandbox.resume(sandbox, store, policy_identity=POLICY, recover_pending=True)
    record = await resumed.step("placement")
    assert successful(record).transitions[0].requested_action == pair.settlement
    assert_engine(sandbox.game_engine, pair.after_settlement)
    assert not any(script.requests for script in scripts.values())
    assert store.head(head.sandbox_id).pending_command_id is None
    assert [entry.kind for entry in store.entries(head.sandbox_id)].count("step_succeeded") == 1
