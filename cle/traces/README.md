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
- exact assembled model messages and parsed choices;
- normalized reasoning, usage, finish reasons, and provider IDs;
- provider request/response bodies, excluding authorization headers/secrets;
- the JSON-safe public viewer state and a restorable sandbox snapshot.

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
exact messages, rationale, and provider-native reasoning—without mutating the
live sandbox. Loading restores the latest durable engine/player checkpoint,
including model-session continuity, and subsequent steps append to the same
trace. The live Start and Step responses also return `trace_game_id`; Step
returns the transactional `trace_step_index`.
