"""Namespaced schema and short, explicitly closed SQLite transactions."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

_SCHEMA = (
    """CREATE TABLE IF NOT EXISTS sandbox_journal_heads (
        sandbox_id TEXT PRIMARY KEY,
        generation INTEGER NOT NULL CHECK (generation >= 1),
        checkpoint_sequence INTEGER NOT NULL,
        next_sequence INTEGER NOT NULL CHECK (next_sequence >= 0),
        snapshot BLOB NOT NULL,
        pending_command_id TEXT,
        policy_identity TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS sandbox_journal_commands (
        sandbox_id TEXT NOT NULL,
        command_id TEXT NOT NULL,
        generation INTEGER NOT NULL,
        expected_sequence INTEGER NOT NULL,
        policy_identity TEXT NOT NULL,
        started_sequence INTEGER NOT NULL,
        finished_sequence INTEGER,
        outcome BLOB,
        PRIMARY KEY (sandbox_id, command_id),
        FOREIGN KEY (sandbox_id) REFERENCES sandbox_journal_heads(sandbox_id),
        CHECK ((finished_sequence IS NULL) = (outcome IS NULL))
    )""",
    """CREATE TABLE IF NOT EXISTS sandbox_journal_calls (
        sandbox_id TEXT NOT NULL,
        command_id TEXT NOT NULL,
        call_key TEXT NOT NULL,
        request_sha256 TEXT NOT NULL,
        request_json TEXT NOT NULL,
        response BLOB,
        PRIMARY KEY (sandbox_id, command_id, call_key),
        FOREIGN KEY (sandbox_id, command_id)
            REFERENCES sandbox_journal_commands(sandbox_id, command_id)
    )""",
    """CREATE TABLE IF NOT EXISTS sandbox_journal_entries (
        sandbox_id TEXT NOT NULL,
        sequence INTEGER NOT NULL CHECK (sequence >= 0),
        kind TEXT NOT NULL,
        command_id TEXT,
        call_key TEXT,
        recorded_at TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        PRIMARY KEY (sandbox_id, sequence),
        FOREIGN KEY (sandbox_id) REFERENCES sandbox_journal_heads(sandbox_id)
    )""",
)


@contextmanager
def transaction(path: Path, *, write: bool = False) -> Iterator[sqlite3.Connection]:
    with closing(sqlite3.connect(path, timeout=30, isolation_level=None)) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA synchronous = FULL")
        connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
        with connection:
            yield connection


def initialize_schema(path: Path) -> None:
    with closing(sqlite3.connect(path, timeout=30, isolation_level=None)) as connection:
        mode = connection.execute("PRAGMA journal_mode = WAL").fetchone()
        if mode is None or mode[0] != "wal":
            raise RuntimeError("Sandbox journal requires a file-backed WAL database")
    # Do not use executescript: it implicitly commits a pre-existing transaction.
    with transaction(path, write=True) as connection:
        for statement in _SCHEMA:
            connection.execute(statement)


def head_row(connection: sqlite3.Connection, sandbox_id: str) -> sqlite3.Row:
    row = connection.execute(
        "SELECT * FROM sandbox_journal_heads WHERE sandbox_id = ?", (sandbox_id,)
    ).fetchone()
    if row is None:
        raise KeyError(sandbox_id)
    return cast(sqlite3.Row, row)


def command_row(
    connection: sqlite3.Connection, sandbox_id: str, command_id: str
) -> sqlite3.Row | None:
    return cast(sqlite3.Row | None, connection.execute(
        "SELECT * FROM sandbox_journal_commands WHERE sandbox_id = ? AND command_id = ?",
        (sandbox_id, command_id),
    ).fetchone())


def append(
    connection: sqlite3.Connection,
    sandbox_id: str,
    kind: str,
    payload: Mapping[str, str | int | None],
    *,
    command_id: str | None = None,
    call_key: str | None = None,
) -> int:
    sequence = cast(int, head_row(connection, sandbox_id)["next_sequence"])
    connection.execute(
        """INSERT INTO sandbox_journal_entries
           (sandbox_id, sequence, kind, command_id, call_key, recorded_at, payload_json)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (sandbox_id, sequence, kind, command_id, call_key,
         datetime.now(timezone.utc).isoformat(), json.dumps(payload, sort_keys=True)),
    )
    connection.execute(
        "UPDATE sandbox_journal_heads SET next_sequence = ? WHERE sandbox_id = ?",
        (sequence + 1, sandbox_id),
    )
    return sequence
