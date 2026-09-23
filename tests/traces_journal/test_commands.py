from dataclasses import replace
from pathlib import Path

import pytest

from cle.players.agent import AgentPlayerSnapshot
from cle.sandbox import CatanSandbox
from cle.traces.journal import (
    CommandOutcome,
    JournalConflict,
    SQLiteSandboxJournal,
)

from .support import POLICY, advance, ticket_for, with_notes


def test_initialize_is_insert_only_and_validates_identity(
    journal: SQLiteSandboxJournal, sandbox: CatanSandbox,
) -> None:
    sandbox_id = sandbox.game_engine.id
    initial = journal.head(sandbox_id)
    assert (initial.generation, initial.checkpoint_sequence, initial.pending_command_id) == (1, 0, None)
    assert [(entry.sequence, entry.kind) for entry in journal.entries(sandbox_id)] == [(0, "genesis")]
    with pytest.raises(JournalConflict, match="already initialized"):
        journal.initialize(sandbox_id, sandbox.snapshot(), POLICY)
    with pytest.raises(JournalConflict, match="engine identity"):
        journal.initialize("wrong-engine", sandbox.snapshot(), POLICY)
    with pytest.raises(KeyError):
        journal.head("wrong-engine")
    with pytest.raises(KeyError):
        journal.command(sandbox_id, "absent")
    assert len(journal.entries(sandbox_id)) == 1


def test_stable_duplicate_delivery_never_reorders_or_rewinds_head(
    journal: SQLiteSandboxJournal, sandbox: CatanSandbox,
) -> None:
    first = ticket_for(journal, sandbox)
    started = journal.begin(first)
    assert journal.begin(first) == started
    with pytest.raises(JournalConflict, match="identity"):
        journal.begin(replace(first, policy_identity="changed"))
    with pytest.raises(JournalConflict, match="identity"):
        journal.begin(replace(first, expected_sequence=99))
    with pytest.raises(JournalConflict, match="pending"):
        journal.begin(replace(first, command_id="other"))
    outcome = advance(sandbox)
    settled = journal.settle(first, outcome)
    assert settled.finished_sequence == 2
    second = ticket_for(journal, sandbox, "step-2")
    journal.begin(second)
    journal.settle(second, advance(sandbox))
    entries = journal.entries(first.sandbox_id)
    duplicate = journal.begin(first)
    assert duplicate.ticket == first
    assert duplicate.finished_sequence == settled.finished_sequence
    assert duplicate.outcome is not None and duplicate.outcome.before_revision == 0
    assert journal.settle(first, outcome).finished_sequence == 2
    with pytest.raises(JournalConflict, match="immutable"):
        journal.settle(first, replace(outcome, error_message="different delivery"))
    assert journal.entries(first.sandbox_id) == entries
    head = journal.head(first.sandbox_id)
    assert head.checkpoint_sequence == 4 and head.pending_command_id is None
    assert head.snapshot.engine.events == sandbox.snapshot().engine.events
    with pytest.raises(JournalConflict, match="checkpoint"):
        journal.begin(replace(first, command_id="stale-checkpoint"))


@pytest.mark.parametrize("status", ["failed", "cancelled"])
def test_same_revision_notes_supersede_checkpoint_and_survive_resume(
    tmp_path: Path, sandbox: CatanSandbox, status: str,
) -> None:
    store = SQLiteSandboxJournal(tmp_path / "notes.sqlite3")
    initial = with_notes(sandbox.snapshot(), "initial notes", 0)
    store.initialize(sandbox.game_engine.id, initial, POLICY)
    first = ticket_for(store, sandbox)
    store.begin(first)
    success = advance(sandbox)
    checkpoint = with_notes(success.snapshot, "initial notes", 0)
    store.settle(first, replace(success, snapshot=checkpoint))
    second = ticket_for(store, sandbox, "same-revision-failure")
    store.begin(second)
    outcome = CommandOutcome(
        "failed" if status == "failed" else "cancelled",
        with_notes(checkpoint, "accepted silence and updated plan", 1), sandbox.revision,
        error_type="RuntimeError", error_message="after accepted silence",
    )
    settled = store.settle(second, outcome)
    store = SQLiteSandboxJournal(store.path)
    resumed = store.resume(first.sandbox_id, POLICY)
    assert resumed.checkpoint_sequence == settled.finished_sequence == 4
    assert len(resumed.snapshot.engine.events) == sandbox.revision
    player = resumed.snapshot.player_states[0][1]
    assert isinstance(player, AgentPlayerSnapshot)
    assert player.session.strategic_memory == "accepted silence and updated plan"
    assert player.session.memory_revision == 1
    assert [entry.kind for entry in store.entries(first.sandbox_id)] == [
        "genesis", "step_started", "step_succeeded", "step_started", f"step_{status}", "resume",
    ]
    store.begin(ticket_for(store, sandbox, "after-resume"))
    assert store.entries(first.sandbox_id)[-1].sequence == 6
    assert store.head(first.sandbox_id).checkpoint_sequence == 4


def test_resume_requires_explicit_original_policy_and_fences_every_mutation(
    journal: SQLiteSandboxJournal, sandbox: CatanSandbox,
) -> None:
    original = ticket_for(journal, sandbox)
    journal.begin(original)
    for recovery, policy in ((False, POLICY), (True, "new-policy")):
        with pytest.raises(JournalConflict):
            journal.resume(original.sandbox_id, policy, recover_pending=recovery)
    assert journal.head(original.sandbox_id).generation == 1
    assert len(journal.entries(original.sandbox_id)) == 2
    recovered = journal.resume(original.sandbox_id, POLICY, recover_pending=True)
    assert recovered.generation == 2 and recovered.checkpoint_sequence == 0
    assert recovered.pending_command_id == original.command_id
    assert recovered.snapshot.engine.events == ()
    assert journal.command(original.sandbox_id, original.command_id).ticket == original
    with pytest.raises(JournalConflict, match="generation"):
        journal.begin(original)
    with pytest.raises(JournalConflict, match="generation"):
        journal.settle(original, CommandOutcome("failed", sandbox.snapshot(), 0))
    current = replace(original, generation=recovered.generation)
    assert journal.begin(current).ticket == original
    outcome = advance(sandbox)
    assert journal.settle(current, outcome).finished_sequence == 3
    with pytest.raises(JournalConflict, match="generation"):
        journal.settle(original, outcome)
    head = journal.resume(original.sandbox_id, "new-policy")
    assert head.generation == 3 and head.checkpoint_sequence == 3


def test_settlement_rejects_regression_rewrites_gaps_and_wrong_engine(
    journal: SQLiteSandboxJournal, sandbox: CatanSandbox,
) -> None:
    initial = sandbox.snapshot()
    first = ticket_for(journal, sandbox)
    journal.begin(first)
    journal.settle(first, advance(sandbox))
    second = ticket_for(journal, sandbox, "step-2")
    journal.begin(second)
    valid = advance(sandbox)
    events = valid.snapshot.engine.events
    invalid_snapshots = (
        initial,
        replace(valid.snapshot, engine=replace(valid.snapshot.engine, engine_id="other")),
        replace(valid.snapshot, engine=replace(valid.snapshot.engine, events=(
            replace(events[0], causation_id="rewritten"), *events[1:],
        ))),
        replace(valid.snapshot, engine=replace(valid.snapshot.engine, events=(
            *events[:-1], replace(events[-1], sequence=999),
        ))),
    )
    before_entries = journal.entries(first.sandbox_id)
    for invalid in invalid_snapshots:
        with pytest.raises(JournalConflict):
            journal.settle(second, replace(valid, snapshot=invalid))
    with pytest.raises(JournalConflict, match="before_revision"):
        journal.settle(second, replace(valid, before_revision=0))
    assert journal.entries(first.sandbox_id) == before_entries
    assert journal.head(first.sandbox_id).checkpoint_sequence == 2
    journal.settle(second, valid)


@pytest.mark.parametrize("command_id", ["", "  ", "x" * 513, "bad\x00id"])
def test_invalid_command_ids_are_rejected_without_writes(
    journal: SQLiteSandboxJournal, sandbox: CatanSandbox, command_id: str,
) -> None:
    with pytest.raises(ValueError, match="command_id"):
        journal.begin(ticket_for(journal, sandbox, command_id))
    assert len(journal.entries(sandbox.game_engine.id)) == 1
