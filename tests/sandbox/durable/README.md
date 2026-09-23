# Durable sandbox integration tests

These tests run actual `GameEngine`, `AgentPlayer`, `DurableSandbox`, and
file-backed `SQLiteSandboxJournal` instances. Scripted transports are network-free,
reject unexpected inference, and obtain legal setup pairs from a separate strict
engine oracle. Event gates control concurrency without timing-based sleeps.
Fresh-owner recovery reconstructs the store, engine, agents, and transports.

Coverage:

- Old command redelivery after advancement, without checkpoint or player rewind.
- Concurrent duplicate delivery: one inference and one settlement.
- Settlement failures before persistence and after a committed acknowledgment loss.
- Cached response recovery, and explicit retry for an unresolved call receipt.
- Discard responses completed in reverse order, replayed from cached receipts, and
  committed in seat order with correct hands, bank, RNG, and private notes.
- A pending automatic road recovered after an unpersisted local execution, including
  snake reversal and the final setup pair; one inference for the settlement/road pair.
- Factory-created agent and batch continuity after reopening the journal.
- Cancellation records and duplicate cancelled-command delivery.
- Accepted silent notes and remaining speech recovery at an unchanged engine revision.
- Fail-closed command admission before commit and after a lost acknowledgment.
- Actual spawned-process death after acquisition and before settlement.
- Policy identity drift, settled delivery across a policy change, private error
  redaction, and image provenance without duplicate image payloads.

## Verification

```sh
uv run --no-sync python -m pytest tests/sandbox/durable -q
uv run --no-sync python -m ruff check tests/sandbox/durable
uv run --no-sync python -m mypy --strict --follow-imports=silent tests/sandbox/durable
```

The admission fault tests caught an unquarantined `journal.begin()` failure path;
the runner now blocks that owner too. These remain ordinary regression tests,
without skips or expected failures. Run the full repository quality gate separately
from these focused behavioral checks.
