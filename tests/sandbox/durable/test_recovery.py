from pathlib import Path

import pytest

from cle.game_engine.models.player import Color
from cle.sandbox.durable import DurableSandbox
from cle.traces.journal import (
    CommandRecord,
    CommandTicket,
    IndeterminateCall,
    JournalConflict,
    SQLiteSandboxJournal,
)

from .faults import Fault, FaultJournal, PersistenceFault
from .support import (
    POLICY,
    ScriptedTransport,
    SetupPair,
    agent,
    assert_checkpoint,
    assert_engine,
    assert_result,
    entry_kinds,
    fresh_sandbox,
    make_sandbox,
    successful,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["before_settle", "after_settle"])
async def test_settlement_uncertainty_fails_closed_and_disk_resolves_recovery(
    tmp_path: Path, pair: SetupPair, fault: Fault,
) -> None:
    transport = ScriptedTransport(pair.response())
    sandbox, _ = make_sandbox(pair.before, {Color.RED: transport})
    initial = sandbox.snapshot()
    journal = FaultJournal(tmp_path / "settlement.sqlite3", fault)
    runner = DurableSandbox.start(sandbox, journal, policy_identity=POLICY)
    with pytest.raises(PersistenceFault, match=fault):
        await runner.step("placement")

    # The engine and agent did commit locally before the durable boundary failed.
    attempted = journal.attempted_outcome
    assert attempted is not None and attempted.status == "succeeded"
    assert attempted.result is not None
    assert attempted.result.transitions[0].requested_action == pair.settlement
    assert attempted.snapshot.pending_action_batch is not None
    assert_checkpoint(sandbox, initial)
    assert sandbox.decision_trace == []
    assert sandbox.communication_trace == []
    for command in ("placement", "other"):
        with pytest.raises(JournalConflict, match="recovery"):
            await runner.step(command)
    assert len(transport.requests) == 1

    reopened = SQLiteSandboxJournal(journal.path)
    head = reopened.head(sandbox.game_engine.id)
    pending = fault == "before_settle"
    assert head.pending_command_id == ("placement" if pending else None)
    assert (reopened.command(head.sandbox_id, "placement").outcome is None) == pending
    if pending:
        assert head.checkpoint_sequence == 0
        candidate, _ = fresh_sandbox(reopened, head.sandbox_id)
        with pytest.raises(JournalConflict, match="explicit recovery"):
            DurableSandbox.resume(candidate, reopened, policy_identity=POLICY)
    else:
        assert head.snapshot.engine.events == pair.after_settlement.events

    fresh, scripts = fresh_sandbox(reopened, head.sandbox_id)
    resumed = DurableSandbox.resume(
        fresh, reopened, policy_identity=POLICY, recover_pending=pending,
    )
    replay = await resumed.step("placement")
    assert_result(successful(replay), attempted.result)
    assert resumed.pending_command_id is None
    assert resumed.checkpoint_sequence == replay.finished_sequence
    assert_engine(fresh.game_engine, pair.after_settlement)
    assert agent(fresh, Color.RED).session.memory_revision == 1
    assert agent(fresh, Color.RED).session.strategic_memory == "Keep the admitted setup pair"
    assert not any(script.requests for script in scripts.values())
    kinds = entry_kinds(reopened, head.sandbox_id)
    for kind in ("step_started", "call_started", "call_completed", "step_succeeded"):
        assert kinds.count(kind) == 1
    assert "call_retried" not in kinds


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["before_response", "after_response"])
async def test_response_uncertainty_requires_consent_only_when_receipt_is_missing(
    tmp_path: Path, pair: SetupPair, fault: Fault,
) -> None:
    original = ScriptedTransport(pair.response())
    sandbox, _ = make_sandbox(pair.before, {Color.RED: original})
    initial = sandbox.snapshot()
    journal = FaultJournal(tmp_path / "response.sqlite3", fault)
    runner = DurableSandbox.start(sandbox, journal, policy_identity=POLICY)
    with pytest.raises(PersistenceFault, match=fault):
        await runner.step("placement")
    assert len(original.requests) == 1
    assert journal.attempted_outcome is None
    assert_checkpoint(sandbox, initial)
    with pytest.raises(JournalConflict, match="recovery"):
        await runner.step("placement", retry_unknown_calls=True)

    reopened = SQLiteSandboxJournal(journal.path)
    sandbox_id = sandbox.game_engine.id
    fresh, scripts = fresh_sandbox(reopened, sandbox_id)
    recovered = DurableSandbox.resume(
        fresh, reopened, policy_identity=POLICY, recover_pending=True,
    )
    if fault == "before_response":
        before = reopened.entries(sandbox_id)
        with pytest.raises(IndeterminateCall):
            await recovered.step("placement")
        assert not any(script.requests for script in scripts.values())
        assert reopened.entries(sandbox_id) == before
        assert reopened.command(sandbox_id, "placement").outcome is None
        assert_checkpoint(fresh, initial)
        with pytest.raises(JournalConflict, match="recovery"):
            await recovered.step("placement", retry_unknown_calls=True)

        # A refusal is fail-closed too: a new fenced owner must opt into resending.
        reopened = SQLiteSandboxJournal(journal.path)
        retry = ScriptedTransport(pair.response())
        fresh, scripts = fresh_sandbox(reopened, sandbox_id, {Color.RED: retry})
        recovered = DurableSandbox.resume(
            fresh, reopened, policy_identity=POLICY, recover_pending=True,
        )
        record = await recovered.step("placement", retry_unknown_calls=True)
        assert len(retry.requests) == 1
        assert retry.requests[0] == original.requests[0]
    else:
        record = await recovered.step("placement")
        assert not any(script.requests for script in scripts.values())

    result = successful(record)
    assert result.transitions[0].requested_action == pair.settlement
    assert_engine(fresh.game_engine, pair.after_settlement)
    kinds = entry_kinds(reopened, sandbox_id)
    assert kinds.count("call_retried") == (1 if fault == "before_response" else 0)
    for kind in ("call_started", "call_completed", "step_started", "step_succeeded"):
        assert kinds.count(kind) == 1
    assert recovered.pending_command_id is None


@pytest.mark.asyncio
@pytest.mark.parametrize("after_commit", [False, True], ids=["before-begin", "lost-begin-ack"])
async def test_admission_persistence_uncertainty_also_requires_a_fresh_owner(
    tmp_path: Path, pair: SetupPair, monkeypatch: pytest.MonkeyPatch, after_commit: bool,
) -> None:
    transport = ScriptedTransport(pair.response())
    sandbox, _ = make_sandbox(pair.before, {Color.RED: transport})
    initial = sandbox.snapshot()
    journal = SQLiteSandboxJournal(tmp_path / "admission.sqlite3")
    runner = DurableSandbox.start(sandbox, journal, policy_identity=POLICY)

    def uncertain_begin(ticket: CommandTicket) -> CommandRecord:
        if after_commit:
            SQLiteSandboxJournal.begin(journal, ticket)
        raise PersistenceFault("uncertain command admission")

    with monkeypatch.context() as patch:
        patch.setattr(journal, "begin", uncertain_begin)
        with pytest.raises(PersistenceFault, match="uncertain command admission"):
            await runner.step("placement")
    assert_checkpoint(sandbox, initial)
    assert not transport.requests
    head = journal.head(sandbox.game_engine.id)
    assert head.pending_command_id == ("placement" if after_commit else None)

    # Storage being reachable again cannot silently unquarantine this owner.
    with pytest.raises(JournalConflict, match="recovery"):
        await runner.step("placement")
    assert not transport.requests
    assert_checkpoint(sandbox, initial)

    reopened = SQLiteSandboxJournal(journal.path)
    fresh_transport = ScriptedTransport(pair.response())
    fresh, _ = fresh_sandbox(reopened, head.sandbox_id, {Color.RED: fresh_transport})
    resumed = DurableSandbox.resume(
        fresh, reopened, policy_identity=POLICY, recover_pending=after_commit,
    )
    result = successful(await resumed.step("placement"))
    assert result.transitions[0].requested_action == pair.settlement
    assert len(fresh_transport.requests) == 1
    assert_engine(fresh.game_engine, pair.after_settlement)
    kinds = entry_kinds(reopened, head.sandbox_id)
    assert kinds.count("step_started") == kinds.count("step_succeeded") == 1
