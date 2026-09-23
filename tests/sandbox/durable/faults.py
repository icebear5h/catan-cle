"""Inject uncertainty at real SQLite operation boundaries, keeping stored receipts."""

from pathlib import Path
from typing import Literal

from cle.harness.models import ModelResponse
from cle.traces.journal import (
    CommandOutcome,
    CommandRecord,
    CommandTicket,
    SQLiteSandboxJournal,
)

Fault = Literal["before_settle", "after_settle", "before_response", "after_response"]


class PersistenceFault(OSError):
    """An injected storage outage or lost commit acknowledgment."""


class FaultJournal(SQLiteSandboxJournal):
    def __init__(self, path: Path, fault: Fault) -> None:
        super().__init__(path)
        self.fault = fault
        self.attempted_outcome: CommandOutcome | None = None

    def settle(self, ticket: CommandTicket, outcome: CommandOutcome) -> CommandRecord:
        self.attempted_outcome = outcome
        if self.fault == "before_settle":
            raise PersistenceFault(self.fault)
        record = super().settle(ticket, outcome)
        if self.fault == "after_settle":
            raise PersistenceFault(self.fault)
        return record

    def finish_call(
        self, ticket: CommandTicket, call_key: str, response: ModelResponse,
    ) -> None:
        if self.fault == "before_response":
            raise PersistenceFault(self.fault)
        super().finish_call(ticket, call_key, response)
        if self.fault == "after_response":
            raise PersistenceFault(self.fault)
