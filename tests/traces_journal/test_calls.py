import json
from dataclasses import replace

import pytest

from cle.harness.models import ModelResponse
from cle.sandbox import CatanSandbox
from cle.traces.journal import (
    CommandOutcome,
    IndeterminateCall,
    JournalConflict,
    SQLiteSandboxJournal,
)

from .support import POLICY, REQUEST_JSON, REQUEST_SHA, ticket_for


def test_unknown_call_requires_explicit_retry_even_after_reopen_and_recovery(
    journal: SQLiteSandboxJournal, sandbox: CatanSandbox,
) -> None:
    ticket = ticket_for(journal, sandbox)
    journal.begin(ticket)
    call = journal.begin_call(ticket, "decision-0", REQUEST_SHA, REQUEST_JSON)
    assert call.response is None
    assert json.loads(journal.entries(ticket.sandbox_id)[-1].payload_json)["request_json"] == REQUEST_JSON
    other = SQLiteSandboxJournal(journal.path)
    with pytest.raises(IndeterminateCall):
        other.begin_call(ticket, "decision-0", REQUEST_SHA, REQUEST_JSON)
    assert len(other.entries(ticket.sandbox_id)) == 3
    head = other.resume(ticket.sandbox_id, POLICY, recover_pending=True)
    current = replace(ticket, generation=head.generation)
    with pytest.raises(JournalConflict, match="generation"):
        journal.begin_call(ticket, "decision-0", REQUEST_SHA, REQUEST_JSON, retry_unknown=True)
    with pytest.raises(JournalConflict, match="generation"):
        journal.finish_call(ticket, "decision-0", ModelResponse("late response"))
    with pytest.raises(IndeterminateCall):
        other.begin_call(current, "decision-0", REQUEST_SHA, REQUEST_JSON)
    retried = other.begin_call(current, "decision-0", REQUEST_SHA, REQUEST_JSON, retry_unknown=True)
    assert retried == call
    entry = other.entries(ticket.sandbox_id)[-1]
    assert entry.kind == "call_retried" and entry.sequence == 4
    assert json.loads(entry.payload_json)["reason"] == "unknown-outcome"
    assert other.head(ticket.sandbox_id).checkpoint_sequence == 0


def test_response_cache_keeps_private_data_and_rejects_changed_request_or_response(
    journal: SQLiteSandboxJournal, sandbox: CatanSandbox,
) -> None:
    ticket = ticket_for(journal, sandbox)
    journal.begin(ticket)
    journal.begin_call(ticket, "call", REQUEST_SHA, REQUEST_JSON)
    response = ModelResponse(
        "chosen action", model="provider-model", native_reasoning="private reasoning",
        usage=(("completion_tokens", 17),),
        provider_request_payload={"request": ["private prompt"]},
        provider_response_payload={"provider_fields": {"logprob": -0.2}},
    )
    journal.finish_call(ticket, "call", response)
    before_entries = journal.entries(ticket.sandbox_id)
    journal.finish_call(ticket, "call", response)
    other = SQLiteSandboxJournal(journal.path)
    cached = other.begin_call(ticket, "call", REQUEST_SHA, REQUEST_JSON)
    assert cached.response == response
    assert other.begin_call(ticket, "call", REQUEST_SHA, REQUEST_JSON, retry_unknown=True) == cached
    with pytest.raises(JournalConflict, match="identity"):
        other.begin_call(ticket, "call", "f" * 64, REQUEST_JSON)
    with pytest.raises(JournalConflict, match="identity"):
        other.begin_call(ticket, "call", REQUEST_SHA, REQUEST_JSON + " ")
    with pytest.raises(JournalConflict, match="immutable"):
        other.finish_call(ticket, "call", replace(response, native_reasoning="different"))
    assert other.entries(ticket.sandbox_id) == before_entries
    assert cached.response is not None
    payload = cached.response.provider_response_payload
    assert isinstance(payload, dict)
    payload["tampered"] = True
    assert other.begin_call(ticket, "call", REQUEST_SHA, REQUEST_JSON).response == response


def test_calls_require_pending_command_and_exact_ticket_identity(
    journal: SQLiteSandboxJournal, sandbox: CatanSandbox,
) -> None:
    ticket = ticket_for(journal, sandbox)
    with pytest.raises(JournalConflict, match="pending"):
        journal.begin_call(ticket, "call", REQUEST_SHA, REQUEST_JSON)
    journal.begin(ticket)
    with pytest.raises(JournalConflict, match="not begun"):
        journal.finish_call(ticket, "call", ModelResponse("response"))
    for wrong in (replace(ticket, expected_sequence=1), replace(ticket, policy_identity="wrong")):
        with pytest.raises(JournalConflict, match="identity"):
            journal.begin_call(wrong, "call", REQUEST_SHA, REQUEST_JSON)
        with pytest.raises(JournalConflict, match="identity"):
            journal.finish_call(wrong, "call", ModelResponse("response"))
    journal.begin_call(ticket, "call", REQUEST_SHA, REQUEST_JSON)
    journal.finish_call(ticket, "call", ModelResponse("response"))
    journal.settle(ticket, CommandOutcome("failed", sandbox.snapshot(), 0))
    with pytest.raises(JournalConflict, match="pending"):
        journal.begin_call(ticket, "call", REQUEST_SHA, REQUEST_JSON)
    with pytest.raises(JournalConflict, match="pending"):
        journal.finish_call(ticket, "call", ModelResponse("response"))


@pytest.mark.parametrize("digest", ["", "a" * 63, "a" * 65, "g" * 64, " " * 64])
def test_digest_syntax_is_checked_without_writes(
    journal: SQLiteSandboxJournal, sandbox: CatanSandbox, digest: str,
) -> None:
    ticket = ticket_for(journal, sandbox)
    journal.begin(ticket)
    with pytest.raises(ValueError, match="64 hexadecimal"):
        journal.begin_call(ticket, "call", digest, REQUEST_JSON)
    with pytest.raises(ValueError):
        journal.begin_call(ticket, "call", REQUEST_SHA, "not-json")
    assert len(journal.entries(ticket.sandbox_id)) == 2
