"""Admission rules shared by command and external-call transactions."""

from __future__ import annotations

import sqlite3

from cle.sandbox.contracts import SandboxSnapshot
from cle.traces.journal.contracts import CommandOutcome, CommandTicket, JournalConflict
from cle.traces.journal.database import command_row, head_row


def identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > 512 or "\x00" in value:
        raise ValueError(f"{name} must be a nonempty string of at most 512 characters without NUL")


def owner(connection: sqlite3.Connection, ticket: CommandTicket) -> sqlite3.Row:
    identifier(ticket.sandbox_id, "sandbox_id")
    identifier(ticket.command_id, "command_id")
    head = head_row(connection, ticket.sandbox_id)
    if ticket.generation != head["generation"]:
        raise JournalConflict("Stale writer generation")
    return head


def identity(row: sqlite3.Row, ticket: CommandTicket) -> None:
    if (row["expected_sequence"] != ticket.expected_sequence
            or row["policy_identity"] != ticket.policy_identity):
        raise JournalConflict("Command identity differs from its original invocation")


def pending(connection: sqlite3.Connection, ticket: CommandTicket) -> sqlite3.Row:
    head = owner(connection, ticket)
    row = command_row(connection, ticket.sandbox_id, ticket.command_id)
    if row is None or row["outcome"] is not None:
        raise JournalConflict("Command is not pending")
    identity(row, ticket)
    if head["pending_command_id"] != ticket.command_id:
        raise JournalConflict("A different command owns the pending checkpoint")
    if head["checkpoint_sequence"] != ticket.expected_sequence:
        raise JournalConflict("Expected checkpoint no longer matches the head")
    return head


def snapshot_identity(sandbox_id: str, snapshot: SandboxSnapshot) -> None:
    if snapshot.engine.engine_id != sandbox_id:
        raise JournalConflict("Snapshot engine identity differs from sandbox_id")
    if any(event.sequence != index for index, event in enumerate(snapshot.engine.events)):
        raise JournalConflict("Snapshot event sequences must be contiguous from zero")


def settlement(sandbox_id: str, before: SandboxSnapshot, outcome: CommandOutcome) -> None:
    if outcome.status not in ("succeeded", "failed", "cancelled"):
        raise ValueError("Unknown command outcome status")
    snapshot_identity(sandbox_id, outcome.snapshot)
    events = before.engine.events
    if outcome.before_revision != len(events):
        raise JournalConflict("Outcome before_revision differs from the checkpoint revision")
    if outcome.snapshot.engine.events[:len(events)] != events:
        raise JournalConflict("Outcome must preserve the checkpoint's entire event history")
