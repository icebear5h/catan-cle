"""Durable sandbox ownership, checkpoint reads, and ordered journal pagination."""

from __future__ import annotations

from pathlib import Path

from cle.sandbox.contracts import SandboxSnapshot
from cle.traces.journal import calls, commands
from cle.traces.journal.codec import encode, read_call, read_command, read_entry, read_head
from cle.traces.journal.contracts import (
    CallRecord,
    CommandRecord,
    JournalConflict,
    JournalEntry,
    JournalHead,
)
from cle.traces.journal.database import (
    append,
    command_row,
    head_row,
    initialize_schema,
    transaction,
)
from cle.traces.journal.validation import identifier, snapshot_identity


class SQLiteSandboxJournal:
    """A trusted-local file store; no connections survive an API operation.

    Serialized outcomes and responses are immutable, with duplicate delivery
    defined by exact protocol-5 pickle bytes, not Python dataclass equality.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        initialize_schema(self.path)

    def initialize(
        self, sandbox_id: str, snapshot: SandboxSnapshot, policy_identity: str
    ) -> JournalHead:
        identifier(sandbox_id, "sandbox_id")
        snapshot_identity(sandbox_id, snapshot)
        with transaction(self.path, write=True) as connection:
            if connection.execute(
                "SELECT 1 FROM sandbox_journal_heads WHERE sandbox_id = ?", (sandbox_id,)
            ).fetchone() is not None:
                raise JournalConflict("Sandbox is already initialized")
            connection.execute(
                """INSERT INTO sandbox_journal_heads
                   (sandbox_id, generation, checkpoint_sequence, next_sequence,
                    snapshot, policy_identity) VALUES (?, 1, 0, 0, ?, ?)""",
                (sandbox_id, encode(snapshot), policy_identity),
            )
            append(connection, sandbox_id, "genesis", {
                "generation": 1, "policy_identity": policy_identity,
                "revision": len(snapshot.engine.events),
            })
            return read_head(head_row(connection, sandbox_id))

    def resume(
        self, sandbox_id: str, policy_identity: str, *, recover_pending: bool = False
    ) -> JournalHead:
        with transaction(self.path, write=True) as connection:
            head = read_head(head_row(connection, sandbox_id))
            if head.pending_command_id is not None:
                row = command_row(connection, sandbox_id, head.pending_command_id)
                if not recover_pending:
                    raise JournalConflict("Pending command requires explicit recovery")
                if row is None or row["policy_identity"] != policy_identity:
                    raise JournalConflict("Recovery requires the pending command's original policy")
            connection.execute(
                """UPDATE sandbox_journal_heads SET generation = ?, policy_identity = ?
                   WHERE sandbox_id = ?""",
                (head.generation + 1, policy_identity, sandbox_id),
            )
            append(connection, sandbox_id,
                   "recovery" if head.pending_command_id is not None else "resume", {
                       "generation": head.generation + 1, "policy_identity": policy_identity,
                   }, command_id=head.pending_command_id)
            return read_head(head_row(connection, sandbox_id))

    def head(self, sandbox_id: str) -> JournalHead:
        with transaction(self.path) as connection:
            return read_head(head_row(connection, sandbox_id))

    def command(self, sandbox_id: str, command_id: str) -> CommandRecord:
        with transaction(self.path) as connection:
            row = command_row(connection, sandbox_id, command_id)
            if row is None:
                raise KeyError((sandbox_id, command_id))
            return read_command(row)

    def entries(
        self, sandbox_id: str, *, after_sequence: int = -1, limit: int = 1000
    ) -> tuple[JournalEntry, ...]:
        if after_sequence < -1 or limit < 1:
            raise ValueError("after_sequence must be >= -1 and limit must be positive")
        with transaction(self.path) as connection:
            return tuple(read_entry(row) for row in connection.execute(
                """SELECT * FROM sandbox_journal_entries WHERE sandbox_id = ? AND sequence > ?
                   ORDER BY sequence LIMIT ?""", (sandbox_id, after_sequence, limit),
            ))

    def get_call(self, sandbox_id: str, command_id: str, call_key: str) -> CallRecord:
        """Read one canonical receipt without invoking or reopening its command."""
        with transaction(self.path) as connection:
            row = connection.execute(
                """SELECT * FROM sandbox_journal_calls
                   WHERE sandbox_id = ? AND command_id = ? AND call_key = ?""",
                (sandbox_id, command_id, call_key),
            ).fetchone()
            if row is None:
                raise KeyError((sandbox_id, command_id, call_key))
            return read_call(row)

    begin = commands.begin
    settle = commands.settle
    begin_call = calls.begin_call
    finish_call = calls.finish_call
