# Ordered durable sandbox execution

`DurableSandbox` owns one `CatanSandbox` through execution **and persistence**.
`SQLiteSandboxJournal` assigns one monotonically increasing sequence per sandbox,
shared across all seats, command starts, model calls/results, recovery, and step
outcomes. Different games have independent sequences. This is a command journal,
not a new Catan rules engine or an inference scheduler.

## What is ordered

```text
sequence 0   genesis             initial sandbox checkpoint
sequence 1   step_started        command "discard-17"
sequence 2   call_started        RED's frozen request
sequence 3   call_started        WHITE's frozen request
sequence 4   call_completed      WHITE's provider response
sequence 5   call_completed      RED's provider response
sequence 6   step_succeeded      RED then WHITE applied; checkpoint committed
```

The append order records when results became durable. It does not turn response
latency into seat priority: the ordinary sandbox still admits barriers in seat
order. Canonical engine-event sequences, journal sequences, and legacy viewer
step indexes are separate identifiers. A step can produce several events or
change accepted notes without producing any event.

## Start and advance

```python
import asyncio

from cle.sandbox.durable import create_durable_sandbox
from cle.sandbox.factory import LiveSandboxConfig
from cle.traces.journal import SQLiteSandboxJournal


async def main() -> None:
    journal = SQLiteSandboxJournal(".cle/sandbox_journal.sqlite3")
    config = LiveSandboxConfig(mode="random", seed=7)
    runner = create_durable_sandbox(config, journal)
    sandbox_id = runner.sandbox.game_engine.id

    record = await runner.step("setup-1", expected_sequence=0)
    assert record.outcome is not None
    print(sandbox_id, record.finished_sequence, record.outcome.status)

    # Delivery retry: returns the saved outcome, without advancing or inference.
    replay = await runner.step("setup-1", expected_sequence=0)
    assert replay.finished_sequence == record.finished_sequence

    # Intentional next step gets a new command ID and the current checkpoint.
    await runner.step("setup-2", expected_sequence=runner.checkpoint_sequence)


asyncio.run(main())
```

For model-backed games, use the normal `LiveSandboxConfig(mode="llm", ...)`.
The factory freezes authored prompt sources and binds the resolved model and
provider route into the policy identity. Agent transports automatically record
their request before dispatch and their sanitized response before parsing or
admission. An injected transport must keep its settings/weights fixed during
pending recovery. Pin model/checkpoint revisions externally; a mutable remote
model alias cannot prove that a provider kept its weights unchanged.

Persist your caller's command ID before issuing it and reuse that ID after a
lost reply. A new ID means a new operation. A conflicting `expected_sequence`
rejects stale callers. Omitting it uses the runner's current checkpoint for new
commands and the original checkpoint for existing IDs. Old-command redelivery
never rewinds a game that has advanced.

To wrap an existing headless sandbox directly, use
`DurableSandbox.start(sandbox, journal, policy_identity="immutable-policy-v1")`.
The identity must cover code, model revision, sampling, and prompt policy needed
to reproduce an interrupted command. Own that sandbox exclusively: advance it
through the runner and treat `runner.sandbox` as inspection-only.

## Recovery

```python
runner = create_durable_sandbox(
    config, journal, sandbox_id=sandbox_id, recover_pending=True,
)
if runner.pending_command_id is not None:
    record = await runner.step(runner.pending_command_id)
```

Resuming claims a new writer generation. Older workers cannot persist another
receipt or outcome after the claim. This fences writes; it does not kill their
processes or cancel an already-sent remote inference request.

For an interrupted command, recovery restores the **last settled checkpoint**
and re-drives uncommitted local engine/player work. Completed provider responses
come from the journal, matched against the exact request fingerprint. Per-seat,
per-channel call ordinals allow concurrent siblings to finish in different orders.
Changed request content, missing previously recorded acquisitions, changed policy,
or conflicting command identity fails closed.

If a call was started but its result was not saved, `IndeterminateCall` blocks
automatic resubmission. After checking the provider/worker, explicitly reclaim
the pending command and use `retry_unknown_calls=True` only if another remote
invocation is acceptable. The journal records `call_retried`; the original may
already have completed or been billed. Recorded successful responses are reused
even when this flag is enabled.

On database/commit uncertainty the runner is quarantined. Restore/reclaim using
a new runner; do not retry against the uncertain in-memory owner. Reopening the
journal resolves whether settlement actually committed.

## Failures, cancellation, and partial effects

- Ordinary execution failures return a record with `outcome.status == "failed"`.
  Inspect its snapshot and attempts; failure does not imply no gameplay occurred.
- Post-action speech failures retain the committed gameplay result.
- Cancellation saves `status == "cancelled"`, then raises `DurableStepCancelled`
  carrying `.record`. Redelivery returns that saved cancellation outcome.
- Accepted silence, notes, channel cursors, speech budgets, and pending reactions
  are checkpointed even when the engine revision is unchanged.
- An automatic road/build/trade continuation gets its own command but no new
  inference receipt. A restored queue consumes its next action once.
- A deliberate retry after a settled failure uses a new command ID. Reusing the
  failed command ID returns its existing failure.

All local effects become durably visible together at settlement. Re-driving an
uncommitted step can recompute local transitions from the previous RNG state; it
does not append them twice to the committed history. External side effects in
custom player callbacks are outside this contract and need their own idempotent
tool receipts. This is **not** an exactly-once remote-inference guarantee.

## Reading evidence and deployment scope

`journal.entries(sandbox_id, after_sequence=cursor, limit=...)` supports ordered
pagination. `journal.command(sandbox_id, command_id)` returns a detached outcome;
`journal.get_call(sandbox_id, command_id, call_key)` reads one canonical model
receipt, including usage. Count each receipt once, not every delivery/replay.
Private evidence remains private; these reads are operator APIs, not player views.

Invocation JSON preserves exact messages, sources, reasoning settings, limits,
and board/image hashes. Outcomes omit raw board-image data from model requests;
the authoritative provenance remains in the invocation record. Provider secrets
and exception text copied into withheld sibling diagnostics are sanitized.

This is an **opt-in rollout API**. The viewer and `scripts/time_live_game.py`
continue using their existing trace/checkpoint path; starting or loading a viewer
game does not silently claim a journal. Legacy trace chronology is not fabricated.
The namespaced journal tables may coexist in the trace database, but the journal
is the authoritative resume source for journaled runs. Use a trusted local file:
checkpoints/results contain pickle blobs. SQLite permits many readers and one
writer; it is not a cross-node distributed database.

Tests cover actual process death after response persistence, lost commit replies,
fencing, duplication, concurrent discard completion, and same-revision notes.
See `tests/sandbox/durable/` and `tests/traces_journal/`.
