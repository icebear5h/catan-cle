# Catan agent-player harness

`AgentPlayer` owns one player's model-visible continuity and uses the harness
package for prompt assembly, parsing, validated YAML suites, and shared async
provider transports.

## Shared components first

New live games and interactive replay previews default to
[`suites/shared_v1.yaml`](suites/shared_v1.yaml), one self-contained authored
component bundle. Decision and speech compositions reference the same definitions;
editing a shared definition changes both consumers. Components declare their
inputs, template, channel, and empty-input behavior. Compositions select and order
references independently. Validation checks reference closure, available inputs,
required facts, and phase coverage, not one global section order or message count.

`components.py` renders typed values once; inserted notes and table talk are data,
not templates to expand again. `shared_suite.py` compiles the two consumers in
memory without creating duplicate editable YAML sources. The sandbox supplies
privacy-projected facts and exact legal actions; authored prose cannot change
visibility, action legality, or memory acceptance.

The shared resolver uses `CATAN_SHARED_SUITE` or an explicit shared path, then a
local `shared.yaml` override under `CATAN_PROMPT_SUITE_DIR` (default
`.cle/prompt_suites`), then the built-in. Explicit historical sources and existing
legacy override pairs remain supported. Mixed shared/legacy selections fail rather
than guessing. Each live request records its complete source, identity and SHA-256;
the runtime rereads the active selection at safe inference boundaries.

Every file under `suites/` declares `status: active | legacy | deprecated`.
Exactly one bundle is active (`shared_v1.yaml`). `catan_v11.yaml` and
`communication_v5.yaml` are `legacy`: still the defaults for the historical
pair path, not for new games. Everything else is `deprecated` and carries a
banner comment naming its successor. Loading a deprecated file by path emits
`DeprecationWarning`; parsing embedded trace sources does not. Status is
reported in `PromptSuiteDocument` and the `/api/prompt-suite` identity block.
Pinned by `tests/test_suite_status.py`.

The existing Prompt Studio edits each definition once and supports independent
composition reference order. Validate rerenders candidate components from a
coherent captured context. Saves/resets use optimistic source hashes and are
allowed during live games and blocked while a replay is loaded. In-flight batches
keep their original contract through admission; the next batch uses saved edits.
A failed reset preserves the override.

## Fresh context and private notes

`fresh_notes` supplies current authoritative observation, newly visible game
events/table talk, current commitments, and the seat's accepted private
notes. Prior user/assistant packets and native reasoning are not replayed. The
current full board presentation is attached at the transport boundary. Exact old
requests remain in traces; trace retention is not model-context retention.

Each player owns accepted notes, a memory revision, separate action/talk delivery
cursors, the reaction cursor, stable session identity, and acceptance receipts.
Speech cannot consume gameplay delivery. All unseen visible messages are supplied
before their channel cursor advances, even beyond the historical recent-talk
window. The model may intentionally omit older facts from its notes; the engine
retains the complete underlying event log.

Single-action responses contain `tool`, `arguments`, and optional `notes`.
Opted-in deterministic batches instead contain `actions` and optional `notes`.
Reactive speech
uses strict JSON `mode: pass` (also accepts `silence`) or `mode: say` with text,
respondents, and an optional commitment. Notes are never broadcast.
For example:

```json
{"tool":"end_turn","arguments":{},"notes":"Reconsider expansion if BLUE contests the port."}
```

`notes` omitted keeps memory; `""` clears it; a string replaces it. The default
ceiling is 4,000 Unicode characters, configurable downward in the active bundle.
Surrounding whitespace is trimmed after validating the raw size. Null, nonstring,
duplicate, unknown, or oversized fields reject the response without truncation.
Notes are fallible private reminders, not authoritative state or native reasoning.
There is no extra summarizer call and no automatic notes-repair inference.

Preparation and inference never commit memory. The sandbox checks the exact
context ID, player, channel, cutoff, and base memory revision before action/speech
admission. Only admitted responses update notes and their matching input cursor.
Valid explicit silence may update notes without emitting a game event; malformed
speech is rejected rather than normalized into accepted silence. Retries retain
accepted notes and pending events, never a rejected proposed notebook.

After accepted pre-action speech, a failed decision resumes at that decision,
not by repeating speech. Snapshots persist this continuation and the player state;
failure checkpoints preserve accepted speech even when engine revision is
unchanged. See [`../traces/README.md`](../traces/README.md) for restore behavior.

Interactive replay is a non-mutating cold preview: current causal inputs and
empty notes unless explicitly seeded for the matching seat. It does not claim
game-long policy continuity or promote returned notes into subsequent previews.

### Deterministic action batches

Built-in shared v4 and RL v3 enable `deterministic_batches: true`. Omitted/false
retains the previous parser contract; historical indexed/v11 and older authored
shared sources do not silently acquire batch admission. The flag is allowed only
with shared/fresh JSON. Existing single-tool responses are unchanged.

```json
{
  "actions": [
    {"tool": "build_settlement", "arguments": {"node": "<N00>"}},
    {"tool": "build_road", "arguments": {"edge": "<E00_01>"}}
  ],
  "notes": "Optional complete replacement, accepted once."
}
```

This is a syntax example, not a claim that these locations are legal. `actions`
has **1–4** entries, each containing only `tool` and `arguments`. Only
`build_settlement`, `build_road`, `upgrade_city`, `maritime_trade`, and `end_turn`
are allowed; `end_turn` must be last. No nested notes, confirmation instructions,
speech, player trades, wildcard exchange, dev-card purchase/play, dice, robber or
theft. Those retain their standalone interfaces. No speculative Knight expansion.

Useful batches include one setup settlement and its attached road, a normal road
that opens a new settlement site, and bank/port conversions funding a build.
Stable tool signatures supply syntax, not a legal-action list. The complete
envelope, arguments and notes are checked before admission, but **future legality
is not checked against the initial menu**. Each entry binds afresh to current
actor, phase, topology, pieces, bank, port rate and holdings immediately before
strict engine application.

The first Step commits action 1 and accepts notes/input cutoff once. Each later
Step commits one queued action with **zero model inference**, preserving ordinary
checkpoints and navigation. Auto-play uses these same steps. A later illegal
action consumes the remainder and publishes actor-private `ACTION_BATCH_PAUSED`
feedback; the next inference sees current state and the committed prefix. There
is no automatic substitute, repeated prefix, or whole-batch rollback.

Continuation ends at one setup pair even if snake reversal keeps the same actor.
Second-settlement resource grants are deterministic and do not interrupt the road.
Actor/turn/phase changes, new engine events (including speech or random outcomes),
pending speech/player-trade response barriers and victory also stop the remainder.
No unseen post-action events are acknowledged merely by executing queued actions.
An admitted plan survives prompt rebinding and save/load; edits apply at the next
actual inference. A source override must opt in explicitly to offer this syntax.

Live/saved traces show the requested envelope separately from committed actions.
Continuation records link the original context/provider IDs, action number/count,
and canonical event sequences, without synthetic responses or duplicated usage.
Read-only replay previews still validate only the first action; they do not run a
live queue or claim the future prefix has already executed.

### Reactive public speech

The built-in shared bundle opts into `reactive_speech: true` (since v3; RL v2).
Omitted/false retains historical scheduling and parsing, including explicit old
fresh suites. No provider reasoning settings change.

On a normal sandbox decision, including placement, the model may choose a game
tool or a separate public speech choice:

```json
{"tool":"say","arguments":{"text":"BLUE, will you leave this spot open?","respondents":["BLUE"]},"notes":"Remember this negotiation."}
```

Say is a typed communication choice, not an engine Action. It appends a public
event but does not consume a game action, turn, or placement. The same `step()`
continues through bounded reactions to the still-pending game decision. There is
one actor-initiated say per pending decision, then a game action is required.
Trade-response barriers and read-only action-comparison previews require game
tools; the prompt explicitly states when standalone speech is unavailable.

All participants hear the event on their next actual observation. `respondents`
selects only immediate reaction calls: a list of distinct other colors, or `[]`.
Optional `audience` may only be `"PUBLIC"`. No text/NLP addressing heuristic.
Replies use `{"mode":"say","text":"...","respondents":[]}`;
`{"mode":"pass","notes":"Updated belief"}` ends that branch and commits notes
once without emitting an event. Malformed speech is rejected, never treated as pass.

Ordinary rolls/payouts, discards, completed builds/buys/bank or player trades,
theft, turn completion, and silence do not cause speech polling. Required trade
decision calls already deliver offers and counters; no redundant speech call is
added. Events remain canonical and unacknowledged until a real accepted input.
Sevens open one window after all discards, before the destination decision, for
the other players. There is no mid-Knight await or scheduled off-turn setup poll.

Reactions run in deterministic table/queue order, with the existing maximum
general reaction depth (default 2) and one shared call/message budget (default
12) for the pending decision, counting the initiating say. Replies and retries
do not replenish it; failed reaction calls also consume a slot. Already-delivered
addressed triggers are skipped. The queue, budget, initiating-speech marker and
seven-window marker are checkpointed. Trade responses retain their original
simultaneous frozen-context barrier. Active prompts refresh only at safe actual
inference boundaries; skipped polls do not apply or acknowledge model input.

### Shared tools and negotiation

The shared composition supplies stable definitions for every tool, not a filtered
legal menu. Setup ordinals/anchors, free-road counts, private development inventory
and actual VP come from current observations. Development-card playability uses
the authoritative action set, including new-card and one-card-per-turn restrictions.
Strategic prose is labeled guidance. No trade-window block is composed into action
or speech requests; new negotiation events include original terms even after closure.

Shared accept/reject/confirm/cancel use `player`, `give`, and `receive`. Every bundle
is from the **acting player's** perspective, including the original counter terms:

```json
{"tool":"accept_offer","arguments":{"player":"BLUE","give":{"WOOD":1},"receive":{"ORE":1}}}
{"tool":"counter_offer","arguments":{"player":"BLUE","original":{"give":{"WOOD":1},"receive":{"ORE":1}},"proposed":{"give":{"WOOD":1},"receive":{"ORE":2}}}}
```

Accept/reject/cancel and each counter term object also support `give_any` and
`receive_any` (default zero). Wildcards describe nonexecutable proposals; acceptance
does not transfer resources. Only the turn player confirms an exact exchange.
Cancellation withdraws your whole matching offer, including all its recipients.
New offers address the other seats; counters address the turn player only.
Resolution checks current visible active offers before filtering by legality, so
multiple equal-term offers remain ambiguous even if only one is executable. No
model-facing IDs or guessed tie-breaking; the legacy ID parser remains separate.

### One-shot trade preauthorization

Shared/fresh `offer_trade` has one optional structured argument:

```json
{"tool":"offer_trade","arguments":{"give":{"WOOD":1},"receive":{"ORE":1},"confirm_if_accepted_by":["BLUE","ORANGE"]}}
```

- Omit `confirm_if_accepted_by` for a **probe**: collect normal responses and
  return control without automatic confirmation.
- `["BLUE"]` authorizes only BLUE. An ordered list selects its first willing
  player; `"ANY"` permits the original audience in the engine's fixed seat/turn
  order (`state.colors`, filtered to that audience). Latency never breaks ties.
- This is the **proposer's confirmation authorization**. `accept_offer` remains
  responder willingness and does not itself transfer cards.
- Only exact original root-offer terms qualify. Wildcards, empty/duplicate lists,
  self/outside-audience colors, null, prose conditions, and extra fields reject
  the whole call. Player colors are case-insensitive; `"ANY"` is literal uppercase.
- The window is the first complete simultaneous response batch after this offer,
  not a wall-clock timeout. All original recipients must have responded in that
  batch. If other pending offers diverted responses, the instruction pauses
  rather than waiting through more batches.
- **Any counteroffer arriving in that window pauses**, including a counter to
  another offer, even when a permitted player accepted the original. Nobody
  permitted accepting also pauses. There is no automatic rejection/renegotiation
  branch and no arbitrary conditional language.
- After the barrier, the next sandbox step rechecks the original offer ID, window,
  turn/round, exact terms/audience, current willingness, both hands, and freshly
  generated legality. If the selected player cannot fund it, pause; do not try
  a lower-priority player or change terms. Withdrawals, expiry, replaced offers,
  later response changes and intervening gameplay invalidate authorization.

Success performs one ordinary atomic `CONFIRM_TRADE`, consuming authorization.
Failure consumes it and appends proposer-private `TRADE_PREAUTHORIZATION_PAUSED`
feedback before the next model decision. Ordinary observers see the normal trade
events; priority lists and failure feedback are not table talk. Notes and delivery
cursors update only for actual accepted model results, never for automatic work.

Pending authorization survives save/load, failed/cancelled response acquisition,
and safe prompt rebinding as the exact admitted instruction. A failed barrier
commits no responses; retry still targets that first batch. Restoring a consumed
checkpoint cannot repeat the exchange. Explicitly restoring an earlier checkpoint
creates an earlier game branch, as with other engine actions. Historical v11 and
indexed parsers keep their existing contracts.

## Historical contracts

The low-level `load_context_suite()` / `load_communication_suite()` defaults remain
the historical v11/v5 contracts for existing callers. Use `load_shared_prompt_suite()`
and its `decision_suite()` / `communication_suite()` methods when explicitly
constructing fresh suites. Bare `AgentPlayer` already selects the shared bundle;
the live factory uses the shared resolver. Historical sources, accepted
trajectories, plans, and trusted pickle receipts remain loadable. Saved sources do
not govern active restored sessions: current runtime selection does. Mode changes
preserve notes and receipts; legacy-to-fresh redelivers events from zero for each
channel rather than treating mixed legacy acknowledgments as delivery. Invalid
notes limits or action/speech policy pairs reject the entire rebind atomically.

Malformed or out-of-menu attempts remain in diagnostics. Rejection does not apply
a gameplay penalty or automatic fallback; exhausting configured attempts leaves
the decision unapplied.

The historical v11 suite preserves v10's strategy guidance, two-round placement order, and
perspective-safe social context. Its action interface is one strict JSON tool
call, with optional durable `game_plan` and no model-facing menu indices:

```json
{"game_plan":"Block the leading player.","tool":"play_knight","arguments":{"tile":"<T05>"}}
```

Location arguments use the literal trained atlas vocabulary: `<N00>` through
`<N53>`, canonical endpoint edges such as `<E00_01>`, and `<T00>` through `<T18>`.
They are data, not XML tags. JSON preserves the angle brackets without escaping
or inventing sequential edge IDs. Text-board labels such as `N00` refer to the
same canonical location as `<N00>`. The 154-token training inventory is unchanged;
shared spatial spelling helpers now belong to `cle.game_engine.board_tokens`.

| Tool | Arguments |
| --- | --- |
| `build_settlement`, `upgrade_city` | `node`: trained node token |
| `build_road` | `edge`: trained endpoint edge token |
| `play_knight`, `move_robber` | `tile`: trained tile token |
| `steal_from` | `player`: visible eligible color |
| `play_year_of_plenty` | `take`: resource-count object |
| `play_monopoly` | `resource`: resource name |
| `maritime_trade`, `offer_trade` | `give`, `receive`: resource-count objects |
| `discard` | `cards`: resource-count object |
| `accept_offer`, `reject_offer`, `cancel_trade` | `offer_id`: exact opaque ID |
| `counter_offer` | `offer_id`, `give`, `receive` |
| `confirm_trade` | `offer_id`, `counterparty`: player color |
| `roll_dice`, `buy_development_card`, `play_road_building`, `end_turn` | Empty `{}` |

Only currently available tools are advertised. Named counts must be positive
integers using WOOD, BRICK, SHEEP, WHEAT, or ORE, without duplicate resource keys.
Year of Plenty allows two cards or an engine-permitted singleton; maritime trade
uses the exact best port/bank rate for one different card; discard requires the
exact count from holdings. Domestic offers/counters also allow non-negative
`give_any`/`receive_any` proposal counts. Acceptance is willingness only; the turn
player selects the final counterparty. Cancellation identifies one exact offer.
When a context supplies concrete offers, their exact terms and audiences are
shown; optional `audience: ["BLUE"]` distinguishes otherwise identical concrete
alternatives. Parameterized offers must omit `audience`: roots address the other
seats and counters address only the turn player.

`play_knight` includes the destination in the same decision. The sandbox
preflights both strict engine actions before consuming the card, executes them
without intervening callbacks, and accepts one receipt at the final revision.
It retains separate canonical Knight and robber-movement events. Immediate
Largest Army victory ends the game before movement; any victim choice remains a
subsequent `steal_from`, with the resource chosen by engine RNG. Road Building
still uses subsequent `build_road` decisions.

Semantic calls resolve to the frozen canonical menu internally. Invalid JSON,
extra fields, duplicate keys, unavailable tools, and illegal parameters fail
closed. Historical v10 XML/indexed suites, recorded source hashes, accepted
messages, and old pickled receipts remain loadable; existing saved games and
local overrides are not silently upgraded. No engine replay action migration is
needed. The provider still receives an ordinary completion request, not a
provider-specific function-calling schema.

Replay policy-diff reports retain the parsed tool receipt and requested Knight
sequence. Their agreement metric scores the primary engine action only; a
bundled Knight comparison is explicitly coarse and its destination is unscored.
It must not be read as agreement on the complete Knight decision.

The v7 suite introduced each formatter field as a typed `PromptComponent`. Stable
identity is the sole system component; strategic memory, visible events, phase,
board, resources, opponents, trades, legal actions, phase facts, decision
request, and response schema are separate environment components. Providers
receive their deterministic join under the portable chat `user` role. Older
suite versions remain loadable as reproducible prompt baselines.

The physical board is a separate typed `BoardPresentation` attached only to the
current user turn. New games default to the complete, lossless
`indexed_tile_rows/v3` text projection used by the strict board eval; an
ordinary unannotated PNG is the alternative image projection. Both originate
from an immutable canonical snapshot of public board facts, carry state and
content hashes plus identifier-space/renderer provenance, and never include
hidden hands. Provider adapters encode the selected projection. Image bytes and
base64 data URLs are excluded from durable traces, which retain only bounded
metadata and hashes; historical games recorded before this contract restore in
legacy semantic-board mode.

Historical communication uses `suites/communication_v5.yaml`.
Every call receives complete visible game events, a bounded recent-message
window, and exact unresolved non-binding commitments. `SILENCE` is the default
and never enters the game log. Catan Lab may save validated static string-only
overrides under `.cle/prompt_suites/`; active games retain their recorded suite
sources and hashes.

OpenRouter, Cerebras, Groq, and vLLM transports are asynchronous and provider-independent.
The complete active context is sent every time. Session affinity and provider KV
caching are optional optimizations, never continuity storage.

### Cerebras

A live model id of `cerebras/<id>` (for example `cerebras/qwen-3.8-27b`) routes
that game to `CerebrasTransport` and sends the bare `<id>` upstream; it needs
`CEREBRAS_API_KEY`. The prefix wins over `VLLM_BASE_URL` and
`OPENROUTER_API_KEY`, so provider choice stays per game and every other model
string keeps the env-selected transport. The transport is text-only (the image
board surface is rejected) and implements the native-reasoning channel:

| harness `reasoning` | Cerebras `reasoning_effort` |
|---|---|
| `enabled: false` | `none` |
| `minimal`, `low` | `low` |
| `medium` | `medium` |
| `high`, `xhigh`, `max` | `high` |

`reasoning.max_tokens` is rejected because Cerebras has no reasoning budget.
`max_tokens` is sent as `max_completion_tokens`, which reasoning tokens count
against. Rate-limit retries honor `retry-after` (capped at 30s).
`scripts/time_live_game.py` plays one headless game on the default suite through
the viewer's own `/api/start-game` and `/api/step` routes, so it is saved to the
live trace database (loadable from the viewer's saved games, mid-run or after),
then prints wall time, call latency, tokens, and tok/s from the recorded calls.

The decision response contains only durable strategic memory, an action selection
(or an opted-in deterministic plan), and applicable structured parameters. It never asks the
model to author a rationale. Reasoning is stored only from a distinct provider
response channel (`reasoning`, `reasoning_content`, or `reasoning_details`) or
explicit provider token evidence. The current live factory sends configurable
native-reasoning requests through OpenRouter or Cerebras and rejects enabled requests on
transports that do not implement that channel instead of silently faking it.
