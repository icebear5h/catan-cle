"""Durable external-call receipts; transport invocation belongs to the runtime."""

from __future__ import annotations

import json
import re
import sqlite3
from typing import TYPE_CHECKING, cast

from cle.harness.models import ModelResponse
from cle.traces.journal.codec import encode, read_call
from cle.traces.journal.contracts import (
    CallRecord,
    CommandTicket,
    IndeterminateCall,
    JournalConflict,
)
from cle.traces.journal.database import append, transaction
from cle.traces.journal.validation import identifier, pending

if TYPE_CHECKING:
    from cle.traces.journal.store import SQLiteSandboxJournal


def _call_row(
    connection: sqlite3.Connection, ticket: CommandTicket, call_key: str
) -> sqlite3.Row | None:
    return cast(sqlite3.Row | None, connection.execute(
        """SELECT * FROM sandbox_journal_calls
           WHERE sandbox_id = ? AND command_id = ? AND call_key = ?""",
        (ticket.sandbox_id, ticket.command_id, call_key),
    ).fetchone())


def begin_call(
    self: SQLiteSandboxJournal,
    ticket: CommandTicket,
    call_key: str,
    request_sha256: str,
    request_json: str,
    *,
    retry_unknown: bool = False,
) -> CallRecord:
    identifier(call_key, "call_key")
    if re.fullmatch(r"[0-9a-fA-F]{64}", request_sha256) is None:
        raise ValueError("request_sha256 must be exactly 64 hexadecimal characters")
    json.loads(request_json)
    with transaction(self.path, write=True) as connection:
        pending(connection, ticket)
        row = _call_row(connection, ticket, call_key)
        if row is not None:
            if row["request_sha256"] != request_sha256 or row["request_json"] != request_json:
                raise JournalConflict("Call identity differs from its original request")
            if row["response"] is not None:
                return read_call(row)
            if not retry_unknown:
                raise IndeterminateCall(f"Call {call_key!r} has no durable response")
            append(connection, ticket.sandbox_id, "call_retried", {
                "reason": "unknown-outcome", "request_sha256": request_sha256,
            }, command_id=ticket.command_id, call_key=call_key)
            return read_call(row)
        append(connection, ticket.sandbox_id, "call_started", {
            "request_sha256": request_sha256, "request_json": request_json,
        }, command_id=ticket.command_id, call_key=call_key)
        connection.execute(
            """INSERT INTO sandbox_journal_calls
               (sandbox_id, command_id, call_key, request_sha256, request_json)
               VALUES (?, ?, ?, ?, ?)""",
            (ticket.sandbox_id, ticket.command_id, call_key, request_sha256, request_json),
        )
        return CallRecord(call_key, request_sha256, None)


def finish_call(
    self: SQLiteSandboxJournal, ticket: CommandTicket, call_key: str, response: ModelResponse
) -> None:
    identifier(call_key, "call_key")
    with transaction(self.path, write=True) as connection:
        pending(connection, ticket)
        row = _call_row(connection, ticket, call_key)
        if row is None:
            raise JournalConflict("Call has not begun")
        blob = encode(response)
        if row["response"] is not None:
            if row["response"] != blob:
                raise JournalConflict("Completed response differs from immutable stored bytes")
            return
        append(connection, ticket.sandbox_id, "call_completed", {
            "request_sha256": row["request_sha256"],
        }, command_id=ticket.command_id, call_key=call_key)
        connection.execute(
            """UPDATE sandbox_journal_calls SET response = ?
               WHERE sandbox_id = ? AND command_id = ? AND call_key = ?""",
            (blob, ticket.sandbox_id, ticket.command_id, call_key),
        )
