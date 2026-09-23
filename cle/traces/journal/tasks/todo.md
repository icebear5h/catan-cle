# SQLite sandbox journal

- [x] Inspect contracts, snapshot/event semantics, and existing SQLite patterns.
- [x] Implement namespaced atomic schema, explicit connection lifetime, and typed reads.
- [x] Implement fenced claims, command settlement, and durable call receipts.
- [x] Exercise real-file races, redelivery, recovery, validation, rollback, and reopen.
- [x] Run targeted pytest, scoped Ruff/mypy, and repository quality gate; review results.

## Design

Use `BEGIN IMMEDIATE` for writes and one connection per operation. A sandbox-local
counter orders entries; checkpoint sequence advances only on settlement. Keep the
original command ticket immutable when a new generation claims pending work.

Choose compressed protocol-5 pickle bytes for immutable outcome/response identity,
rather than dataclass equality (engine state compares by identity) or introducing
a second canonical object serializer. Never reserialize a stored value to compare
it: compare the new bytes directly with the stored bytes.

## Review

Implemented the full public API and documented byte-identity/recovery caveats in
the package README. The real-file integration suite covers competing writers,
duplicate settlement, stale-generation fencing, private response caching,
same-revision notes checkpoints, sequence pagination, and fault-injected rollback
for schema creation, settlement, and call completion.

The repository-wide quality gate was run. It remains red outside this scope:
`playground/frontend/src/App.tsx` has 1550 lines, Ruff reports 115 existing
violations elsewhere, and mypy reports widespread unrelated typing failures.
Final scoped verification passes: 24 pytest cases, Ruff, and strict mypy across
all 14 journal/test Python files. All scoped files/folders satisfy the 300-line
and 15-direct-file limits. The same-revision test verifies failed/cancelled notes
supersede a successful checkpoint and survive reopening and resume.
