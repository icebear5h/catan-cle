"""Trusted-local per-sandbox command journal contracts.

Journal sequence orders durable observations, not concurrent provider execution.
Engine event revisions remain a separate, authoritative gameplay sequence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from cle.harness.models import ModelResponse
from cle.players.contracts import PlayerAttempt
from cle.sandbox.communication import CommunicationAdmission
from cle.sandbox.contracts import SandboxSnapshot, SandboxStepResult


class JournalConflict(RuntimeError):
    """A command identity, checkpoint, or writer generation does not match."""


class IndeterminateCall(RuntimeError):
    """An external invocation exists without a durable result; do not resend."""


@dataclass(frozen=True)
class JournalHead:
    sandbox_id: str
    generation: int
    checkpoint_sequence: int
    snapshot: SandboxSnapshot
    pending_command_id: str | None


@dataclass(frozen=True)
class CommandTicket:
    sandbox_id: str
    command_id: str
    generation: int
    expected_sequence: int
    policy_identity: str


@dataclass(frozen=True)
class CommandOutcome:
    status: Literal["succeeded", "failed", "cancelled"]
    snapshot: SandboxSnapshot
    before_revision: int
    result: SandboxStepResult | None = None
    error_type: str | None = None
    error_message: str | None = None
    attempts: tuple[PlayerAttempt, ...] = ()
    communications: tuple[CommunicationAdmission, ...] = ()


@dataclass(frozen=True)
class CommandRecord:
    ticket: CommandTicket
    started_sequence: int
    finished_sequence: int | None
    outcome: CommandOutcome | None


@dataclass(frozen=True)
class CallRecord:
    call_key: str
    request_sha256: str
    response: ModelResponse | None


@dataclass(frozen=True)
class JournalEntry:
    sandbox_id: str
    sequence: int
    kind: str
    command_id: str | None
    call_key: str | None
    recorded_at: str
    payload_json: str
