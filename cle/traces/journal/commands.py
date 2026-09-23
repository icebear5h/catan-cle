"""Atomic command admission and settlement, independent of runtime execution."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cle.traces.journal.codec import encode, read_command, read_head
from cle.traces.journal.contracts import (
    CommandOutcome,
    CommandRecord,
    CommandTicket,
    JournalConflict,
)
from cle.traces.journal.database import append, command_row, transaction
from cle.traces.journal.validation import identity, owner, pending, settlement

if TYPE_CHECKING:
    from cle.traces.journal.store import SQLiteSandboxJournal


def begin(self: SQLiteSandboxJournal, ticket: CommandTicket) -> CommandRecord:
    with transaction(self.path, write=True) as connection:
        head = owner(connection, ticket)
        row = command_row(connection, ticket.sandbox_id, ticket.command_id)
        if row is not None:
            identity(row, ticket)
            return read_command(row)
        if head["pending_command_id"] is not None:
            raise JournalConflict("Another command is already pending")
        if head["checkpoint_sequence"] != ticket.expected_sequence:
            raise JournalConflict("Expected checkpoint no longer matches the head")
        sequence = append(connection, ticket.sandbox_id, "step_started", {
            "generation": ticket.generation, "expected_sequence": ticket.expected_sequence,
            "policy_identity": ticket.policy_identity,
        }, command_id=ticket.command_id)
        connection.execute(
            """INSERT INTO sandbox_journal_commands
               (sandbox_id, command_id, generation, expected_sequence, policy_identity,
                started_sequence) VALUES (?, ?, ?, ?, ?, ?)""",
            (ticket.sandbox_id, ticket.command_id, ticket.generation, ticket.expected_sequence,
             ticket.policy_identity, sequence),
        )
        connection.execute(
            "UPDATE sandbox_journal_heads SET pending_command_id = ? WHERE sandbox_id = ?",
            (ticket.command_id, ticket.sandbox_id),
        )
        return CommandRecord(ticket, sequence, None, None)


def settle(
    self: SQLiteSandboxJournal, ticket: CommandTicket, outcome: CommandOutcome
) -> CommandRecord:
    with transaction(self.path, write=True) as connection:
        owner(connection, ticket)
        row = command_row(connection, ticket.sandbox_id, ticket.command_id)
        if row is None:
            raise JournalConflict("Command has not begun")
        identity(row, ticket)
        blob = encode(outcome)
        if row["outcome"] is not None:
            if row["outcome"] != blob:
                raise JournalConflict("Settled outcome differs from immutable stored bytes")
            return read_command(row)
        head = read_head(pending(connection, ticket))
        settlement(ticket.sandbox_id, head.snapshot, outcome)
        sequence = append(connection, ticket.sandbox_id, f"step_{outcome.status}", {
            "before_revision": outcome.before_revision,
            "after_revision": len(outcome.snapshot.engine.events),
            "error_type": outcome.error_type, "error_message": outcome.error_message,
        }, command_id=ticket.command_id)
        connection.execute(
            """UPDATE sandbox_journal_commands SET finished_sequence = ?, outcome = ?
               WHERE sandbox_id = ? AND command_id = ?""",
            (sequence, blob, ticket.sandbox_id, ticket.command_id),
        )
        connection.execute(
            """UPDATE sandbox_journal_heads SET checkpoint_sequence = ?, snapshot = ?,
               pending_command_id = NULL WHERE sandbox_id = ?""",
            (sequence, encode(outcome.snapshot), ticket.sandbox_id),
        )
        settled = command_row(connection, ticket.sandbox_id, ticket.command_id)
        assert settled is not None
        return read_command(settled)
