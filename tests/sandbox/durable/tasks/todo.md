# Durable sandbox integration tests

- [x] Read runtime, transport, factory, journal, and adjacent integration contracts.
- [x] Build typed network-free fixtures using an independent strict engine oracle.
- [x] Verify command idempotency and concurrent duplicate serialization.
- [x] Inject response/settlement persistence faults and recover with fresh owners.
- [x] Verify discard seat ordering, queued setup continuation, and cancellation.
- [x] Verify notes-only same-revision progress survives a failed decision.
- [x] Check the admission boundary also quarantines a runner after storage uncertainty.
- [x] Run focused pytest, scoped Ruff/mypy, and the required repository quality gate.
- [x] Review changes and report implementation bugs without weakening assertions.

All authored changes stay in `tests/sandbox/durable/`.

## Review

- 15 cases: 13 pass; both admission-fault cases expose the same missing runner
  quarantine around `journal.begin()`. Assertions remain unmodified and unskipped.
- Scoped Ruff and strict mypy pass across nine Python files.
- Repository quality fails on existing out-of-scope structure/Ruff/mypy errors.
- All source files remain under 300 physical lines; ten direct authored files.
- Recovery assertions check gameplay events, material state, RNG, player notes and
  cursors, automatic-action provenance, and exact journal operation counts.
- `README.md` documents the reproduction and the distinction between an unresolved
  provider call and a durably cached response.
