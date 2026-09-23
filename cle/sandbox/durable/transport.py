"""Persist model responses before admission; replay them on command recovery."""

from __future__ import annotations

from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, field, replace
from hashlib import sha256

from cle.harness.models import CompletionTransport, ModelRequest, ModelResponse
from cle.traces.journal import CommandTicket, JournalConflict, SQLiteSandboxJournal
from cle.traces.sqlite.blobs import _json_text
from cle.traces.sqlite.calls import _provider_payload, _request_payload


@dataclass
class CallSession:
    journal: SQLiteSandboxJournal
    ticket: CommandTicket
    started_sequence: int
    retry_unknown: bool = False
    counters: dict[tuple[str, str | None], int] = field(default_factory=dict)
    seen: set[str] = field(default_factory=set)
    remote_errors: list[str] = field(default_factory=list)
    persistence_error: BaseException | None = None

    def next_key(self, request: ModelRequest) -> str:
        # Independent seat/channel ordinals survive reversed barrier completion
        # order. The journal separately records the observed global append order.
        scope = (request.session_id, request.channel)
        ordinal = self.counters.get(scope, 0)
        self.counters[scope] = ordinal + 1
        digest = sha256(_json_text(scope).encode()).hexdigest()
        key = f"{digest}:{ordinal}"
        self.seen.add(key)
        return key

    def check_replayed_calls(self) -> None:
        """Do not silently abandon old acquisitions after changing control flow."""
        cursor = self.started_sequence
        while entries := self.journal.entries(self.ticket.sandbox_id, after_sequence=cursor):
            for entry in entries:
                if entry.command_id == self.ticket.command_id and entry.kind == "call_started":
                    if entry.call_key not in self.seen:
                        raise JournalConflict("Recovery skipped an originally recorded model call")
            cursor = entries[-1].sequence


active_calls: ContextVar[CallSession | None] = ContextVar("durable_sandbox_calls", default=None)


def request_evidence(request: ModelRequest, policy_identity: str) -> str:
    """Bind the full request contract, including limits and image content hashes."""
    payload = _request_payload(request)
    assert payload is not None
    payload.update(
        journal_request_version=1,
        policy_identity=policy_identity,
        reasoning_request=request.reasoning_request,
        max_tokens=request.max_tokens,
    )
    return _json_text(payload)


def durable_response(response: ModelResponse, request: ModelRequest) -> ModelResponse:
    """Reuse trace redaction; never persist transport headers or inline images."""
    return deepcopy(replace(
        response,
        provider_request_payload=_provider_payload(
            response.provider_request_payload, request.board_presentation,
        ),
        provider_response_payload=_provider_payload(
            response.provider_response_payload, request.board_presentation,
        ),
    ))


class JournalTransport:
    """Transport receipts scoped by the owning durable step, not by provider ID."""

    def __init__(self, delegate: CompletionTransport, policy_identity: str) -> None:
        self.delegate = delegate
        self.policy_identity = policy_identity

    async def complete(self, request: ModelRequest) -> ModelResponse:
        session = active_calls.get()
        if session is None:
            raise JournalConflict("Journal transport requires a durable sandbox command")
        if session.ticket.policy_identity != self.policy_identity:
            error = JournalConflict("Transport policy differs from the command policy")
            session.persistence_error = error
            raise error
        key = session.next_key(request)
        evidence = request_evidence(request, self.policy_identity)
        try:
            call = session.journal.begin_call(
                session.ticket, key, sha256(evidence.encode()).hexdigest(), evidence,
                retry_unknown=session.retry_unknown,
            )
        except BaseException as exc:
            session.persistence_error = exc
            raise
        if call.response is not None:
            return deepcopy(call.response)

        # Remote exceptions are not proof the provider did no work. The started
        # receipt stays unresolved; a recovered command must explicitly consent
        # before resending. A handled failed command itself can still settle.
        try:
            response = durable_response(await self.delegate.complete(deepcopy(request)), request)
        except BaseException as exc:
            if str(exc):
                session.remote_errors.append(str(exc))
            raise
        try:
            session.journal.finish_call(session.ticket, key, response)
        except BaseException as exc:
            session.persistence_error = exc
            raise
        return deepcopy(response)
