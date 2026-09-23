"""Trusted-local blobs and typed projections of journal rows."""

from __future__ import annotations

import pickle
import sqlite3
import zlib
from typing import TypeVar

from cle.harness.models import ModelResponse
from cle.sandbox.contracts import SandboxSnapshot
from cle.traces.journal.contracts import (
    CallRecord,
    CommandOutcome,
    CommandRecord,
    CommandTicket,
    JournalEntry,
    JournalHead,
)

_Stored = TypeVar("_Stored", SandboxSnapshot, CommandOutcome, ModelResponse)


def encode(value: SandboxSnapshot | CommandOutcome | ModelResponse) -> bytes:
    return zlib.compress(pickle.dumps(value, protocol=5))


def decode(blob: bytes, expected: type[_Stored]) -> _Stored:
    value: object = pickle.loads(zlib.decompress(blob))
    if not isinstance(value, expected):
        raise TypeError(f"Invalid journal blob: expected {expected.__name__}")
    return value


def read_head(row: sqlite3.Row) -> JournalHead:
    return JournalHead(
        sandbox_id=row["sandbox_id"], generation=row["generation"],
        checkpoint_sequence=row["checkpoint_sequence"],
        snapshot=decode(row["snapshot"], SandboxSnapshot),
        pending_command_id=row["pending_command_id"],
    )


def read_command(row: sqlite3.Row) -> CommandRecord:
    return CommandRecord(
        ticket=CommandTicket(
            sandbox_id=row["sandbox_id"], command_id=row["command_id"],
            generation=row["generation"], expected_sequence=row["expected_sequence"],
            policy_identity=row["policy_identity"],
        ),
        started_sequence=row["started_sequence"],
        finished_sequence=row["finished_sequence"],
        outcome=decode(row["outcome"], CommandOutcome) if row["outcome"] is not None else None,
    )


def read_call(row: sqlite3.Row) -> CallRecord:
    return CallRecord(
        call_key=row["call_key"], request_sha256=row["request_sha256"],
        response=decode(row["response"], ModelResponse) if row["response"] is not None else None,
    )


def read_entry(row: sqlite3.Row) -> JournalEntry:
    return JournalEntry(
        sandbox_id=row["sandbox_id"], sequence=row["sequence"], kind=row["kind"],
        command_id=row["command_id"], call_key=row["call_key"],
        recorded_at=row["recorded_at"], payload_json=row["payload_json"],
    )
