"""An owning, fenced execution boundary around one ordinary CatanSandbox."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import replace
from hashlib import sha256
from typing import Literal

from cle.players.agent import AgentPlayer
from cle.sandbox.catan import (
    CatanSandbox,
    PostActionCommunicationCancelled,
    PostActionCommunicationError,
)
from cle.sandbox.contracts import SandboxSnapshot, SandboxStepResult
from cle.traces.journal import (
    CommandOutcome,
    CommandRecord,
    CommandTicket,
    JournalConflict,
    JournalHead,
    SQLiteSandboxJournal,
)
from cle.traces.sqlite.blobs import _snapshot_blob

from .evidence import clean_outcome
from .transport import CallSession, JournalTransport, active_calls


class DurableStepCancelled(asyncio.CancelledError):
    """Cancellation with its already-durable outcome, including partial effects."""

    def __init__(self, record: CommandRecord) -> None:
        self.record = record
        super().__init__("Sandbox cancellation checkpointed")


class DurableSandbox:
    """Serialize steps through commit; stable command IDs replay saved outcomes.

    Own the supplied sandbox exclusively. In-flight recovery re-drives only
    local snapshot-backed engine/player work. Arbitrary external side effects
    in player callbacks are outside this contract.
    """

    def __init__(
        self, sandbox: CatanSandbox, journal: SQLiteSandboxJournal,
        head: JournalHead, policy_identity: str,
        restore_checkpoint: Callable[[SandboxSnapshot], None] | None = None,
    ) -> None:
        if sandbox.game_engine.id != head.sandbox_id:
            raise JournalConflict("Sandbox does not match the journal head")
        self.sandbox = sandbox
        self.journal = journal
        self.policy_identity = policy_identity
        self._head = head
        self._lock = asyncio.Lock()
        self._blocked = False
        self._restore_checkpoint = restore_checkpoint or sandbox.restore
        self._install_transports()
        self._fingerprint = self._state_fingerprint()

    @classmethod
    def start(
        cls, sandbox: CatanSandbox, journal: SQLiteSandboxJournal, *, policy_identity: str,
    ) -> DurableSandbox:
        head = journal.initialize(sandbox.game_engine.id, sandbox.snapshot(), policy_identity)
        return cls(sandbox, journal, head, policy_identity)

    @classmethod
    def resume(
        cls, sandbox: CatanSandbox, journal: SQLiteSandboxJournal, *,
        policy_identity: str, recover_pending: bool = False,
        restore_checkpoint: Callable[[SandboxSnapshot], None] | None = None,
    ) -> DurableSandbox:
        # Validate snapshot compatibility before taking ownership from another
        # writer. Re-read under the fenced claim in case the head moved meanwhile.
        current = journal.head(sandbox.game_engine.id)
        restore = restore_checkpoint or sandbox.restore
        restore(current.snapshot)
        head = journal.resume(
            sandbox.game_engine.id, policy_identity, recover_pending=recover_pending,
        )
        restore(head.snapshot)
        return cls(sandbox, journal, head, policy_identity, restore)

    @property
    def checkpoint_sequence(self) -> int:
        return self._head.checkpoint_sequence

    @property
    def pending_command_id(self) -> str | None:
        return self._head.pending_command_id

    def _install_transports(self) -> None:
        for player in self.sandbox.players.values():
            if isinstance(player, AgentPlayer):
                delegate = player.transport
                if isinstance(delegate, JournalTransport):
                    delegate = delegate.delegate
                player.transport = JournalTransport(delegate, self.policy_identity)

    def _state_fingerprint(self) -> str:
        # Runtime-local mutation guard, not a cross-version snapshot identity.
        return sha256(_snapshot_blob(self.sandbox.snapshot())).hexdigest()

    def _ticket(self, command_id: str, expected_sequence: int | None) -> CommandTicket:
        policy = self.policy_identity
        try:
            previous = self.journal.command(self._head.sandbox_id, command_id)
        except KeyError:
            sequence = self.checkpoint_sequence if expected_sequence is None else expected_sequence
        else:
            sequence = previous.ticket.expected_sequence if expected_sequence is None else expected_sequence
            if previous.outcome is not None:
                policy = previous.ticket.policy_identity
        return CommandTicket(
            self._head.sandbox_id, command_id, self._head.generation, sequence, policy,
        )

    async def step(
        self, command_id: str, *, expected_sequence: int | None = None,
        retry_unknown_calls: bool = False,
    ) -> CommandRecord:
        """Return a settled success/failure or replay it without advancing again.

        Use a new ID for intentional advancement/retry, the same ID for delivery
        retry. Cooperative cancellation is persisted and re-raised; redelivery
        of that ID returns its cancelled record. Persistence uncertainty blocks
        this runner until explicit resume/recovery on a new owner.
        """
        async with self._lock:
            if self._blocked:
                raise JournalConflict("Runner needs explicit recovery after persistence uncertainty")
            if self._state_fingerprint() != self._fingerprint:
                raise JournalConflict("Sandbox was mutated outside its durable runner")
            ticket = self._ticket(command_id, expected_sequence)
            try:
                record = self.journal.begin(ticket)
            except (JournalConflict, ValueError):
                raise
            except BaseException:
                self._blocked = True
                raise
            if record.outcome is not None:
                return record
            self._head = replace(self._head, pending_command_id=command_id)
            return await self._execute(record, ticket, retry_unknown_calls)

    async def _execute(
        self, record: CommandRecord, ticket: CommandTicket, retry_unknown: bool,
    ) -> CommandRecord:
        sandbox = self.sandbox
        decision_start, talk_start = len(sandbox.decision_trace), len(sandbox.communication_trace)
        session = CallSession(self.journal, ticket, record.started_sequence, retry_unknown)
        token = active_calls.set(session)
        result: SandboxStepResult | None = None
        error: BaseException | None = None
        try:
            try:
                result = await sandbox.step()
            except BaseException as exc:
                error = exc
                if isinstance(exc, (PostActionCommunicationError, PostActionCommunicationCancelled)):
                    result = exc.result
            if session.persistence_error is not None:
                raise session.persistence_error
            if error is None:
                session.check_replayed_calls()
            status: Literal["succeeded", "failed", "cancelled"] = "succeeded" if error is None else (
                "cancelled" if isinstance(error, asyncio.CancelledError) else "failed"
            )
            outcome = CommandOutcome(
                status=status,
                snapshot=sandbox.snapshot(),
                before_revision=len(self._head.snapshot.engine.events),
                result=result,
                error_type=type(error).__name__ if error is not None else None,
                # Provider exception strings can contain credentials/URLs. Keep
                # the class; rejected-attempt diagnostics retain action errors.
                error_message="Step failed; inspect recorded attempts" if error is not None else None,
                attempts=tuple(sandbox.decision_trace[decision_start:]),
                communications=tuple(sandbox.communication_trace[talk_start:]),
            )
            settled = self.journal.settle(ticket, clean_outcome(outcome, session.remote_errors))
            assert settled.finished_sequence is not None
            self._head = replace(
                self._head, checkpoint_sequence=settled.finished_sequence,
                snapshot=outcome.snapshot, pending_command_id=None,
            )
            self._fingerprint = self._state_fingerprint()
        except BaseException:
            self._blocked = True
            # Quarantine speculative local effects. A commit acknowledgment may
            # have been lost; only the new fenced owner's head resolves that.
            self._restore_checkpoint(self._head.snapshot)
            del sandbox.decision_trace[decision_start:]
            del sandbox.communication_trace[talk_start:]
            raise
        finally:
            active_calls.reset(token)
        if isinstance(error, asyncio.CancelledError):
            raise DurableStepCancelled(settled) from error
        if error is not None and not isinstance(error, Exception):
            raise error
        return settled
