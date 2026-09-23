# SQLite sandbox journal

`SQLiteSandboxJournal(path)` opens a trusted-local, file-backed SQLite journal.
Each operation owns and explicitly closes its connection. Writers use short
`BEGIN IMMEDIATE` transactions, WAL, and `synchronous=FULL`; no transaction spans
runtime/model execution. Schema creation is atomic and uses only
`sandbox_journal_*` tables, leaving the database's global `user_version` alone.

- Initialize with `sandbox_id == snapshot.engine.engine_id`. Genesis is sequence
  zero, checkpoint zero, generation one. Reinitialization conflicts.
- Resume increments generation and logs `resume` or `recovery`. Pending work
  requires explicit recovery and its original policy identity. Reconstruct its
  ticket with the new generation: stored command tickets remain the original
  invocation. Claims never advance the settled checkpoint.
- Journal sequences are sandbox-local counters, independent of timestamps and
  engine revisions. Settlements, including failures/cancellations with only
  notes/cursor changes, advance the checkpoint to their journal sequence.
- Duplicate command admission returns the original record, even after later
  settlements. Generation fencing still applies. Pending call APIs require the
  current owner and exact original command identity.
- Unknown calls are blocked by default. Explicit `retry_unknown=True` records
  `call_retried` with reason `unknown-outcome`; a provider may have completed the
  original invocation. A durable response is cached without adding entries.
- Request SHA-256 strings must have 64 hexadecimal characters. JSON is syntax
  checked but stored exactly as supplied, including whitespace. The runtime owns
  fingerprint construction; the store does not recompute it or mint secrets.
- Sandbox/command/call identifiers are nonblank, at most 512 characters, without
  NUL. Missing heads/commands raise `KeyError`; missing entry streams return `()`.

Snapshots, outcomes, and full model responses use zlib-compressed protocol-5
pickle. Only open trusted local files. Duplicate settlement/response equality is
**exact serialized-byte identity** against the original stored bytes. Equivalent
objects with different pickle representations (including aliasing, dict order,
or runtime-version differences) conflict rather than overwrite evidence. Prefer
`begin()` to retrieve a settled outcome instead of rebuilding and resettling it.
Returned data is detached by deserialization; mutating it cannot modify storage.
