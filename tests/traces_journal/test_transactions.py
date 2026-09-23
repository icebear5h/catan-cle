import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from threading import Barrier

import pytest

from cle.harness.models import ModelResponse
from cle.sandbox import CatanSandbox
from cle.traces.journal import CommandRecord, JournalConflict, SQLiteSandboxJournal, database
from tests.traces_journal.support import POLICY, REQUEST_JSON, REQUEST_SHA, advance, ticket_for


def test_two_instances_racing_begin_allow_one_pending_command(
    journal: SQLiteSandboxJournal, sandbox: CatanSandbox,
) -> None:
    other = SQLiteSandboxJournal(journal.path)
    ticket = ticket_for(journal, sandbox)
    barrier = Barrier(2)

    def begin(index: int) -> CommandRecord | JournalConflict:
        barrier.wait(timeout=10)
        try:
            return (journal, other)[index].begin(replace(ticket, command_id=f"racer-{index}"))
        except JournalConflict as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = tuple(pool.map(begin, (0, 1)))
    assert sum(isinstance(result, JournalConflict) for result in results) == 1
    winner = next(result for result in results if isinstance(result, CommandRecord))
    assert journal.head(ticket.sandbox_id).pending_command_id == winner.ticket.command_id
    assert len(other.entries(ticket.sandbox_id)) == 2
    other.resume(ticket.sandbox_id, POLICY, recover_pending=True)
    with pytest.raises(JournalConflict, match="generation"):
        journal.settle(winner.ticket, advance(sandbox))


def test_racing_duplicate_settlement_has_one_durable_result(
    journal: SQLiteSandboxJournal, sandbox: CatanSandbox,
) -> None:
    other = SQLiteSandboxJournal(journal.path)
    ticket = ticket_for(journal, sandbox)
    journal.begin(ticket)
    outcome = advance(sandbox)
    barrier = Barrier(2)

    def settle(index: int) -> CommandRecord:
        barrier.wait(timeout=10)
        return (journal, other)[index].settle(ticket, outcome)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = tuple(pool.map(settle, (0, 1)))
    assert [result.finished_sequence for result in results] == [2, 2]
    assert journal.head(ticket.sandbox_id).checkpoint_sequence == 2
    assert len(journal.entries(ticket.sandbox_id)) == 3


def test_settlement_trigger_failure_rolls_back_log_command_and_checkpoint(
    journal: SQLiteSandboxJournal, sandbox: CatanSandbox,
) -> None:
    ticket = ticket_for(journal, sandbox)
    journal.begin(ticket)
    outcome = advance(sandbox)
    entries = journal.entries(ticket.sandbox_id)
    with closing(sqlite3.connect(journal.path)) as connection, connection:
        connection.execute("""CREATE TRIGGER reject_checkpoint
            BEFORE UPDATE OF checkpoint_sequence ON sandbox_journal_heads
            BEGIN SELECT RAISE(ABORT, 'injected checkpoint fault'); END""")
    with pytest.raises(sqlite3.IntegrityError, match="injected checkpoint fault"):
        journal.settle(ticket, outcome)
    reopened = SQLiteSandboxJournal(journal.path)
    assert reopened.entries(ticket.sandbox_id) == entries
    head = reopened.head(ticket.sandbox_id)
    assert head.checkpoint_sequence == 0 and head.pending_command_id == ticket.command_id
    assert head.snapshot.engine.events == ()
    assert reopened.command(ticket.sandbox_id, ticket.command_id).outcome is None
    with closing(sqlite3.connect(journal.path)) as connection, connection:
        connection.execute("DROP TRIGGER reject_checkpoint")
    assert reopened.settle(ticket, outcome).finished_sequence == 2
    assert reopened.head(ticket.sandbox_id).snapshot.engine.events == outcome.snapshot.engine.events


def test_call_completion_fault_leaves_unknown_receipt_and_no_completion_entry(
    journal: SQLiteSandboxJournal, sandbox: CatanSandbox,
) -> None:
    ticket = ticket_for(journal, sandbox)
    journal.begin(ticket)
    journal.begin_call(ticket, "call", REQUEST_SHA, REQUEST_JSON)
    entries = journal.entries(ticket.sandbox_id)
    with closing(sqlite3.connect(journal.path)) as connection, connection:
        connection.execute("""CREATE TRIGGER reject_response
            BEFORE UPDATE OF response ON sandbox_journal_calls
            BEGIN SELECT RAISE(ABORT, 'injected response fault'); END""")
    with pytest.raises(sqlite3.IntegrityError, match="injected response fault"):
        journal.finish_call(ticket, "call", ModelResponse("lost"))
    assert SQLiteSandboxJournal(journal.path).entries(ticket.sandbox_id) == entries
    with closing(sqlite3.connect(journal.path)) as connection:
        assert connection.execute("SELECT response FROM sandbox_journal_calls").fetchone() == (None,)


def test_atomic_schema_preserves_existing_database_and_user_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "coexisting.sqlite3"
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("CREATE TABLE live_games (game_id TEXT PRIMARY KEY)")
        connection.execute("INSERT INTO live_games VALUES ('existing-game')")
        connection.execute("PRAGMA user_version = 73")
    schema = database._SCHEMA
    monkeypatch.setattr(database, "_SCHEMA", (*schema[:2], "INVALID SCHEMA SQL"))
    with pytest.raises(sqlite3.OperationalError):
        SQLiteSandboxJournal(path)
    with closing(sqlite3.connect(path)) as connection:
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE name LIKE 'sandbox_journal_%'"
        ).fetchall() == []
    monkeypatch.setattr(database, "_SCHEMA", schema)
    store = SQLiteSandboxJournal(path)
    with closing(sqlite3.connect(path)) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (73,)
        assert connection.execute("PRAGMA journal_mode").fetchone() == ("wal",)
        assert connection.execute("SELECT * FROM live_games").fetchall() == [("existing-game",)]
    with database.transaction(store.path, write=True) as connection:
        assert connection.execute("PRAGMA synchronous").fetchone()[0] == 2
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connection.execute("SELECT 1")


def test_sequences_are_per_sandbox_and_paginate_without_timestamp_ordering(
    journal: SQLiteSandboxJournal, sandbox: CatanSandbox,
) -> None:
    ticket = ticket_for(journal, sandbox)
    journal.begin(ticket)
    journal.settle(ticket, advance(sandbox))
    journal.resume(ticket.sandbox_id, POLICY)
    second = replace(sandbox.snapshot(), engine=replace(sandbox.snapshot().engine, engine_id="other"))
    journal.initialize("other", second, POLICY)
    with closing(sqlite3.connect(journal.path)) as connection, connection:
        connection.execute("""UPDATE sandbox_journal_entries SET recorded_at =
            CASE WHEN sequence = 0 THEN '2099-01-01' ELSE '1900-01-01' END""")
    entries = journal.entries(ticket.sandbox_id)
    assert [entry.sequence for entry in entries] == [0, 1, 2, 3]
    assert journal.entries(ticket.sandbox_id, limit=2) == entries[:2]
    assert journal.entries(ticket.sandbox_id, after_sequence=1, limit=2) == entries[2:]
    assert journal.entries(ticket.sandbox_id, after_sequence=3) == ()
    assert [entry.sequence for entry in journal.entries("other")] == [0]
    assert journal.entries("absent") == ()
    assert journal.head(ticket.sandbox_id).checkpoint_sequence == 2
