# Local live trace store

Live games use a local SQLite database at `.cle/live_traces.sqlite3` by default.
Override it with `CATAN_LIVE_TRACE_DB`.

SQLite is the normal local-development choice for LLM observability here: WAL
allows readers while the single sandbox writer commits one completed step in a
transaction. Normalized game, step, model-call, and event rows support queries;
JSON columns retain evolving context/provider schemas. Each step also stores a
trusted-local `SandboxSnapshot` blob for exact restore. Never unpickle a trace
database from an untrusted source.

Every completed live step records:

- game configuration and lifecycle status;
- complete perspective-projected player contexts and exact legal menus;
- canonical transitions, public/private event overlays, and table messages;
- accepted and rejected attempts with validation errors;
- exact assembled model messages, typed board-presentation provenance, and
  parsed choices;
- normalized reasoning, usage, finish reasons, and provider IDs;
- provider request/response bodies, excluding authorization headers/secrets and
  replacing inline board-image data with content-addressed metadata;
- the JSON-safe public viewer state and a restorable sandbox snapshot.

Preauthorized trade confirmations are engine-only completed steps with no model
call rows. Their result JSON and live Step response expose `automatic_action`:
`kind`, `origin_context_id`, optional provider response/request IDs, the exact
`offer_id`, and canonical `action_sequence`. Join the origin context to its
accepted decision call (provider IDs disambiguate where available); do not copy
that call or its token usage into the automatic step. Pending and consumed state
is checkpointed. A paused condition is proposer-private engine feedback, not an
LLM response. The existing UI shows the ordinary trade action; provenance is
available through these API/result records rather than a fabricated reasoning card.

Deterministic batch continuations use the same engine-only checkpoint path.
`automatic_action.kind` is `deterministic_batch_continuation`, with original
context/provider IDs, `origin_sequence`, current `action_sequence`, and one-based
`action_number`/`action_count`. The original parsed choice records `batch_actions`;
the original exact response remains unchanged. Only that first step has a model
call/usage row. Saved trace UI shows the requested batch and each automatic step's
causal reference separately, without claiming later actions committed at admission.
Pending queues and consumed/paused state live in normal snapshot blobs; existing
failure snapshots preserve them if a later provider call or post-action callback
fails. No database schema migration is required.

Inspection and live-resume endpoints:

```text
GET   /api/live-traces?limit=50
GET   /api/live-traces/<game_id>
GET   /api/live-traces/<game_id>/steps/<step_index>
PATCH /api/live-traces/<game_id>       {"name":"Opening study"}
POST  /api/live-traces/<game_id>/load
```

The saved-games bar lists optional names alongside stable game IDs. The
browse-only checkpoint navigator renders a selected historical public board and
all model calls for that step—including rejected decisions, communication,
exact messages, selected actions, and provider-native reasoning—without mutating the
live sandbox. Loading restores the latest durable engine/player checkpoint,
including model-session continuity, and subsequent steps append to the same
trace. The live Start and Step responses also return `trace_game_id`; Step
returns the transactional `trace_step_index`.

## Storage size

Large columns are stored zlib-packed behind a small magic prefix: the game's
`initial_snapshot`, each step's `sandbox_snapshot`, `result_json` and
`public_state_json`, each model call's `request_json`, and failure snapshots
and public states. Readers unpack transparently and rows written before packing
(raw pickle, raw JSON text) keep loading, so no migration is required.
`model_calls.response_json` and `live_failures.payload_json` stay plain text
because `get_usage` runs `json_extract`/`json_each` over them in SQL.

Snapshots used to grow with game length: every agent's session carries one
`ChoiceReceipt` per accepted decision so an already-accepted context can be
answered again, and those receipts deep-copied the whole `PlayerChoice`,
including provider reasoning that `model_calls` already records once per
call. Re-pickled into every later step, that made a 400-step game cost about
0.85 GB in snapshots alone. Receipts now hold only the decision
(`receipt_choice` in `cle/harness/models.py`); reasoning, raw responses,
reasoning requests and usage are read from `model_calls`.

To pack history written before this change and reclaim the file:

```text
python -m scripts.compact_live_traces                 # report projected saving
python -m scripts.compact_live_traces --apply         # pack legacy rows in place
python -m scripts.compact_live_traces --apply --vacuum
```

Packing runs in short transactions and is safe to re-run; stop the viewer
before `--vacuum`, which rebuilds the file and needs free disk equal to its size.

## Failure-boundary checkpoints

Schema v5 optionally stores an exact sandbox snapshot and matching public view
with each failure. This preserves accepted pre-action speech/private notes even
when a later decision fails or valid silence leaves the engine revision unchanged.
Default resume selects the newest inserted snapshot-bearing failure on the latest
successful-step baseline; a later successful step supersedes it. Explicit
step-index loads remain historical checkpoint loads. Old failure rows without
snapshots remain diagnostics only.

Successful continuation backfills unindexed canonical events after the initial
checkpoint, so speech accepted before a failed decision is not lost from the event
index. Original event payloads and prior step associations are not overwritten.
Post-action cancellation retains the already accepted action's model call through
the applied-step path. Failure snapshot blobs are never exposed by listing APIs.

Fresh requests record context policy, base memory revision, input cutoff, channel,
and component variables alongside exact messages. Notes inputs/proposals and
acceptance status are distinct; native reasoning remains a separate diagnostic.
Each new live request also records exact `prompt_sources` (kind, identity,
version, SHA-256 and source text). Stored game configuration is historical only:
loading resumes gameplay/notes with the current runtime inference settings and
active suites, never sources or paths from that saved configuration. Editor
changes and replay previews do not overwrite accepted memory or earlier traces.

An unsaved checkpoint is reported explicitly with `checkpoint_saved: false` and
`retryable: false`; an already applied action is never described as unapplied.
Recovery guarantees the last durable checkpoint, not exactly-once remote inference
after arbitrary process or network failure.

Reactive standalone `say` is recorded as an accepted communication call with its
actual selected text/respondents, original **action-channel** request and exact
provider response/usage. It is not a synthetic game step. Reaction requests carry
`trigger_reason`; communication choices also retain the trigger reason, including
`standalone_speech`, `addressed_speech`, and `pre_robber`. Existing live/saved call
cards and token aggregation include these records. A failure before gameplay saves
admitted speech and its resumable queue in the failure checkpoint; successful
continuation backfills the canonical message event without duplicating inference.
