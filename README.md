# Catan Learning Environment (CLE)

RL training framework for teaching LLMs to play Settlers of Catan through self-play and fine-tuning.

See [the sandbox guide](cle/sandbox/README.md) for the active runtime and
[the engine guide](docs/engine/ENGINE_DOCS.md) for deterministic rules.

## Architecture

```text
CatanSandbox
|- game_engine: GameEngine
`- players: dict[Color, SandboxPlayer]
```

`GameEngine` owns rules, mutable state, RNG, legal actions, canonical events,
privacy projections, trades, and snapshots. `CatanSandbox` is the asynchronous
single-writer orchestrator. The v11 model interface uses semantic JSON tools and
trained spatial tokens such as `<N00>`, `<E00_01>`, and `<T05>`, resolved internally
against the engine's legal actions. Knight includes its robber destination in
one model decision. Historical indexed suites remain loadable. See the
[action contract](cle/harness/README.md) for tool signatures and validation.

New live games use [shared prompt components](cle/harness/suites/shared_v1.yaml)
and fresh context with private notes. Decision and speech reuse authored
definitions without a fixed global order. Requests contain current facts, newly
visible events, and accepted notes, not the entire prior model conversation.
Notes have a 4,000-character ceiling; omitted keeps, empty clears, and a string
replaces them only after admission. Saved games preserve game state, private notes,
and exact historical traces; active inference configuration is independent.

The shared contract now uses reactive **public** table talk: a normal decision
chooses a game action or standalone `say`. Say leaves the board decision pending;
only explicitly named respondents get immediate reaction calls, then the actor
must act. Pass may still update private notes. Routine completed events accumulate
without speech polling. Sevens have a bounded window after discards and before
the robber destination; Knight remains atomic. Conversation budgets and pending
replies survive save/load. See the [speech contract](cle/harness/README.md#reactive-public-speech).

Shared `offer_trade` optionally accepts `confirm_if_accepted_by`: an ordered
player array or `"ANY"`. It preauthorizes one exact exchange after the simultaneous
response batch; any counteroffer, no permitted acceptance, or stale/invalid
state returns control to the model. Omission remains a normal probe. Automatic
confirmation is its own engine-only checkpoint, with an originating-call reference
and no synthetic model response or token usage. See
[trade preauthorization](cle/harness/README.md#one-shot-trade-preauthorization).

Shared decisions also support **deterministic action batches** of up to four
semantic calls: settlement, road, city upgrade, bank/port conversion, and optional
terminal end-turn. Use one setup settlement + attached road per decision, or a
normal road → settlement / conversions → build plan. Each Step commits one action
and checkpoint; queued Steps make no model call. Auto-play drains the queue through
the same Step path. Later invalid actions discard the remainder while preserving
the committed prefix. New events, actor/phase changes, setup-pair completion and
victory stop continuation. Notes and tokens belong to the originating call once.
See [batch syntax and boundaries](cle/harness/README.md#deterministic-action-batches).

### Live history and token usage

The board's compact live strip includes **Previous / Next / Latest** and a checkpoint
picker. Previous/Next select a recorded post-step board and its exact saved
request/reasoning. Browsing is read-only: incoming runtime updates stay separate,
and Step/Auto-play are disabled until **Latest** returns to the active runtime.
Latest never reloads or rewinds the sandbox. Loading a different saved game
remains an explicit **Load latest** action.

The strip shows one status with secondary model settings and elapsed time during
inference. Empty history and token metrics are omitted. Available step/game-average
input and output tokens appear inline; **Token details** expands coverage and
failure-only batches. Controls wrap on narrow screens and errors stay visible.

Token metrics use recorded provider `prompt_tokens`/`completion_tokens` (or
`input_tokens`/`output_tokens`) from canonical call rows. They include rejected
decision retries and accepted/rejected communication; cache and reasoning detail
counts are not added again. **Game avg** divides known totals by completed steps
with usage for each direction separately. Step and call coverage are in Token details;
partial coverage means known tokens only, and missing usage is **unknown**, never
zero. Failure-only batches (including admitted speech) appear separately and are
excluded from completed-step averages. Transport retries without recorded provider
usage cannot be measured; these figures are not billing estimates.

The existing trace endpoint supports `?view=usage`, returning only call identities,
admission status and usage, without requests, boards or checkpoint blobs. The UI
fetches it on game/runtime changes, not on each historical selection. Older server
processes need to pick up this backend change for game averages; selected-step
metrics still use the existing checkpoint endpoint.

### Hand contents on the player chips

Viewer snapshots carry two layers. `all_player_resources` and `all_player_dev_cards`
stay a public projection - a hand total and unplayed development-card count, exactly
what an opponent knows. `player_hands` is a spectator field alongside them with the
exact per-resource and per-card breakdown. It is a viewer artifact only: model
prompts come from the engine's privacy projection and never read a snapshot.

The board's player chips always render that breakdown under the VP/Hand/Dev line:
one pill per held resource, then unplayed development cards (`K2`, `VP1`), with
exact counts in the hover title. A player holding nothing reads `empty`. Stored
step checkpoints carry `player_hands`, so history browsing shows the hand as it
stood at that step; checkpoints recorded before this field existed show no
breakdown, and neither does a server process started before it (the viewer has
no reloader unless `CATAN_VIEWER_RELOAD=1`).

The inspector's **Messages** board renders for any game, before anyone has
spoken: an empty board says so rather than disappearing, so a silent table reads
as silence and not as a missing panel. Rows come from the game log's speech
entries, labelled by recorded game step where one exists and by engine-event
sequence otherwise. Loading a saved game re-projects any speech row the stored
log no longer carries from the checkpoint's public events (checkpoints written
under the old 50-row window kept every event but dropped older rows), so old
games load with their full message board.

### Trace database size

Every live step stores a restorable snapshot, so the trace database grows with
play. Large columns are zlib-packed on write and agent receipts no longer carry
model reasoning (it lives once in `model_calls`), which keeps a 400-step game at
tens of MB rather than nearly a gigabyte. Existing databases keep loading;
`python -m scripts.compact_live_traces --apply --vacuum` packs old rows and
shrinks the file. Details in the [trace store notes](cle/traces/README.md#storage-size).

### Editing prompts during a game

Prompt Studio's **Save active prompts** applies edits to the next decision or
speech batch without starting a fresh game. Each in-flight action or concurrent
trade/speech batch (including retries) finishes under its original contract.
The following boundary atomically stages all agent replacements, preserving
sessions, notes, receipts, events and the pending-decision continuation.

Loading a saved game restores the board, hands, scores, RNG, events and notes,
but uses the current runtime model/settings and active prompt selection. After a
server restart, current defaults/environment apply. Saved source paths and suite
text are historical evidence, never runtime selectors. Every new live request
records exact source text, identity and SHA-256 in `prompt_sources`, alongside
its exact messages/components. Old trace rows and saved configuration stay intact.

The default selection follows the shared/local resolver. Explicit current
`CATAN_SHARED_SUITE`, `CATAN_CONTEXT_SUITE`/`CATAN_COMMUNICATION_SUITE`, or factory
path selections remain authoritative and are reread at boundaries. Studio cannot
shadow environment-selected files. Existing local legacy pairs remain selected
until reset to the shared built-in; reset applies during a live game too.

Changing the selected context mode deliberately migrates session metadata:
legacy-to-fresh preserves notes/history and redelivers visible events to both
channels because the legacy mixed cursor cannot prove separate delivery.
Fresh-to-legacy preserves cursors and retained historical messages; fresh-era
conversations are not reconstructed. Notes exceeding the new limit, unknown
policies, or incompatible action/speech contracts stop rebinding atomically.
Correct the active configuration and step again; notes are never silently cleared.

### Shared fresh request contract

Shared requests show stable, compact tool definitions rather than legal-move
enumerations; the engine still validates every selected action internally.
Setup facts identify the first/second settlement or road, the already-placed
settlement and road anchor, and starting cards from the second settlement only.
Road Building's remaining free placements are separate facts. Private inventory
includes every development-card type even with zero resources, authoritative
current playability, and your actual VP; opponents retain public VP and card totals.

Shared trade calls identify another `player` and `give`/`receive` from the acting
player's perspective. `counter_offer` supplies `player`, `original` and `proposed`
terms. The resolver binds these to one active visible offer internally; stale or
ambiguous terms fail without guessing. Accepting signals willingness; only
confirmation transfers cards. Wildcard proposals and negotiation lifecycle events
carry their terms, so neither decision nor speech needs a trade-window prompt block.

Bare `AgentPlayer` uses the shared contract. Explicit historical suites retain
their indexed/offer-ID parsers and historical rendering; eval defaults were not
migrated. Trace views retain all recorded request messages, including historical
long contexts, and label fresh requests separately. See the
[harness guide](cle/harness/README.md) for exact trade syntax and compatibility.

## Quick Start

```python
import asyncio

from cle.sandbox.factory import LiveSandboxConfig, create_live_sandbox


async def main() -> None:
    sandbox = create_live_sandbox(
        LiveSandboxConfig(mode="random", seed=7, shuffle_players=False)
    )
    while sandbox.game_engine.winning_color() is None:
        await sandbox.step()


asyncio.run(main())
```

## Key Features

- Deterministic, event-driven Catan engine with explicit snapshots
- Asynchronous player inference with single-writer state mutation
- Complete perspective-safe event storage with causal per-channel delivery
- Shared prompt components and bounded private notes without transcript replay
- Semantic action tools with exact engine validation and bounded decision retries
- Barrier-synchronized trading and reactions independent of model latency
- Live viewer, deterministic replay, evaluations, and data pipelines over the
  same engine contracts

## Structure

```text
cle/
  |- game_engine/     # Rules, state, events, privacy, trades, snapshots
  |- sandbox/         # Async composition root and multi-game pool
  |- players/         # Agent, human, scripted, and baseline policies
  |- harness/         # Context suites, parsing, providers, continuity
  |- replay/          # Colonist decoding and deterministic replay runtime
  `- env/             # Shared observation presentation compatibility
evals/
  |- catan_board_bench/    # Benchmark code and frozen evaluation inputs
  `- catan_board_bench_ui/ # Standalone evaluation frontend
playground/game_viewer/   # Flask/WebSocket presentation adapter
```

## Game Viewer UI

Real-time web interface for watching LLM agents play Catan.

### Quick Start

**Backend (Flask + SocketIO):**
```bash
python -m playground.game_viewer.app
# Server runs on http://localhost:5001
```

**Frontend (React + Vite):**
```bash
cd playground/frontend
npm install
npm run dev
# UI runs on http://localhost:5173
```

**Headless speed run (no browser):**
```bash
uv run python scripts/time_live_game.py --seed 1                 # cerebras/qwen-3.8-27b, needs CEREBRAS_API_KEY
uv run python scripts/time_live_game.py --model qwen/qwen3.8-27b  # same game through OpenRouter
```
A `cerebras/<id>` model routes that game to Cerebras; see `cle/harness/README.md`.

### Features
- Real-time hex board rendering with SVG
- Live game state updates via WebSocket
- Player stats (VP, resources, buildings)
- Step-by-step game control
- LLM decision tracking
- Probability dots on number tokens

### Message board labels

Message board rows are labelled with the **game step** they were spoken in: one
Step advance, one `/api/step` call, one recorded trace step. The engine-event
`sequence` carried in each row is a different, larger counter - one step emits
the action plus every speech and trade-response event inside it, so a 349-step
game can reach event #523. The step index only exists once the step is recorded,
so speech rows are logged with their sequence first, stamped afterwards, and the
recorded checkpoint is rewritten so a browsed step shows the same labels the live
board does. Rows with no recorded step - speech logged without a trace store, or
games saved before this change - keep showing `event #<sequence>`.

The socket snapshot and `/api/state` carry the **whole** game log, not the last
50 rows, so rare rows (speech above all) stay reachable in a long game and the
message board keeps the full conversation. Rows average ~500 bytes, so a long
game's log is a few hundred KB per snapshot and per stored checkpoint - small
against the ~2 MB pickled sandbox snapshot every step already stores. Reinstate a
tail in `build_game_state_snapshot` and `_get_state_snapshot` together if a game
ever gets long enough for that to matter.

### Live failure handling

The red Step banner contains validation/retry guidance only. Expand **Rejected
live attempts** in the inspector for final output, provider-native reasoning,
and provider diagnostics; final output is not the native reasoning channel.
Rejected attempts persist in SQLite `live_failures` and the saved-game API's
`failures` field, independently of completed checkpoints. Earlier failures
from before this change were transient and cannot be reconstructed in full.

A blank action response is reported as a missing final answer, not a JSON syntax
error. A provider can return reasoning only even with finish reason `stop` and
no completion cap; this does not by itself establish token exhaustion. Reasoning
is never executed as an action, even when it contains valid action JSON. The
original channels remain in the saved failure; another Step requests a new paid
response under the existing decision-attempt budget, not a recovered answer.

If post-action communication fails, the action and agent history remain
committed and checkpointed; the next Step advances the game rather than
repeating the applied action. The stateful backend intentionally disables
source reload; restart deliberately and load the latest saved game after
backend changes.

**Auto-play retries until the game completes.** A failed step (a rejected
decision, a provider rejection or connection failure, an unhandled sandbox
error, a post-action communication warning, or an unreachable server) does not
end auto-play: the loop waits with exponential backoff (1s doubling to a 30s
cap, reset after a successful step) and requests a fresh Step, which either asks
the model again under a new decision-attempt budget or advances past an applied
action. The controls strip shows the retry count and countdown; **Stop auto**
cancels a pending retry immediately. Checkpointed notices broadcast from other
tabs are shown but do not cancel auto-play. The one hard stop is a failure the
trace store could not record (`retryable: false` with `checkpoint_saved: false`
for a traced game), because continuing before storage is repaired can lose
admitted changes. Retries make a new paid model request each time; a stuck
configuration (an incompatible active prompt, a revoked key) keeps retrying at
the 30s cap until you press Stop or fix it. Manual Step is unchanged.

OpenRouter's `SSLV3_ALERT_BAD_RECORD_MAC` connection failure is retried within
the existing three-attempt transport budget, with 1s/2s backoff and fresh
request-local clients for the default owned transport. Other players' shared
connections are not reset; certificate validation stays enabled and certificate
errors are not retried. Exhaustion returns a retained HTTP 502 notice and saved
failure diagnostics, not an invented model response. Transport retries cannot
duplicate an applied game action, but a lost upstream response can incur repeated
inference charges. Borrowed clients are never closed or cloned by recovery.

OpenRouter HTTP 403 rejections are not retried. The live notice and saved failure
retain a bounded structured `error.message` and request ID when usable, instead
of only HTTPX's status/MDN message. Known credentials and request echoes are
redacted or suppressed; raw bodies and moderation metadata are not published.
The notice distinguishes an unapplied decision from already-committed gameplay
and asks you to resolve the provider rejection before continuing. A successful
key-status check does not establish model access or rule out a guardrail block.

Harness validation preserves opaque trade IDs, preflights concurrent trade
responses before applying them, and prevents overlapping/stale decisions.
Wildcard trade proposals remain negotiable but cannot execute until an exact
named-resource offer is agreed. Invalid explicit private audiences fail closed.

Frontend unit checks run with `npm --prefix playground/frontend test`. The
mounted browser regression (requires local Playwright Chromium) runs with:

```bash
.venv/bin/python -m pytest \
  playground/frontend/tests/test_live_autoplay_browser.py
```

### Correctness Checks

The [audit and remediation report](reports/correctness-audit-2026-09-08.md)
records the rule, replay-history, and harness fixes and their verification.
Reproductions in `tests/audits/` are now ordinary passing regressions. Enable
`CATAN_FULL_GAME_AUDIT=1` for full-game inventory, seat-history, and independent
road/scoring checks. Replay reconstruction and inventory conservation alone
still do not certify every rule or historical training label.

The default shared suite uses semantic action tools and fresh notes. Explicit
historical suites remain available and historical requests remain unchanged.
Resume uses active inference settings and repairs stale derived road caches
without changing placed pieces, resources, or equivalent menu order.

## Eval Frontend

The standalone eval suite exposes bucketed agent-decision spot checks and the
CatanBoardBench board-perception verifier against the same backend:

```bash
cd evals/catan_board_bench_ui
npm install
npm run dev
# Eval UI runs on http://localhost:5174
```

Agent Decisions is the default tab. It shows the complete versioned bucket
catalog, real replay episode/decision counts, human-versus-model choices,
review rubrics, visible rationale, native-reasoning metadata, legal actions, and
retained prompt evidence.

## Player-to-Player Trading

`GameEngine` owns one bounded `TradeWindow` containing canonical `TradeOffer`
objects. A trade offer records its participants, named give/receive resources,
optional root parent, response signals, and lifecycle status.

Negotiation uses deterministic barriers: eligible opponents decide concurrently
from one frozen causal cutoff, then the sandbox applies every response in table
order. Willingness never transfers resources. The turn player may select one
exact internal `TradeCandidate`, ignore all candidates, keep building, or end the
turn. Counteroffers target only root offers and only the turn player; counters
cannot themselves be countered. Both hands are revalidated before the final
atomic exchange.

See [`cle/game_engine/trading.py`](cle/game_engine/trading.py) and the
[sandbox guide](cle/sandbox/README.md).

## Data pipelines

Reusable acquisition and dataset-building code lives under `data_pipeline/`.
Raw payloads and generated run evidence live under `artifacts/`; reviewed
experiment writeups live under `reports/`.

### Colonist acquisition

- Candidate indexes: `artifacts/raw/colonist/indexes/`
- New or rejected captures: `artifacts/staging/colonist/replays/`
- Validated replay payloads: `artifacts/raw/colonist/replays/`
- Split manifests: `artifacts/manifests/colonist/splits/`

See [data_pipeline/bootstrapping/README.md](data_pipeline/bootstrapping/README.md)
for authenticated capture, pacing, validation, and promotion commands.

The old replay observation/action generator was removed because it applied the
target action before formatting the claimed pre-action observation. Training
data must use the authoritative verified replay executor and perspective-safe
decision packets.

### CatanBoardBench

CatanBoardBench measures engine-scored public-board perception and grounded
reasoning across image and text representations. Frozen benchmark inputs remain
under `evals/catan_board_bench/datasets/`.
Provider plans, responses, and summaries live under
`artifacts/runs/catan_board_bench/`; reviewed findings live under
`reports/catan_board_bench/`.

See [evals/catan_board_bench/README.md](evals/catan_board_bench/README.md) for
benchmark ownership and artifact policy.

## V0: VLM Agent with GLM-4.6V-Flash

V0 = "Can a fine-tuned 9B VLM play Catan at above-random level from board renders + text context?"

### Base Model: [GLM-4.6V-Flash](https://huggingface.co/zai-org/GLM-4.6V-Flash) (9B)

Chosen for native agentic capabilities at 9B scale:
- **Native XML tool calling** - trained with `<tool_call>` format during SFT and RL, no prompt hacking
- **128K context window** - board image + game history + strategic notes without truncation
- **Thinking mode toggle** - `<think>` for critical decisions, `/nothink` for fast self-play rollouts
- **Trained on GUI agent tasks** - already understands structured visual environments with action spaces
- **GRPO-trained** - model's reward sensitivity shaped by Group Relative Policy Optimization

Paper: [arxiv:2507.01006](https://arxiv.org/abs/2507.01006)

### Observation Design: Dual-Channel (Image + Text)

The VLM receives two channels per decision point. Spatial understanding is embedded in the visual channel - the model sees adjacency, clustering, expansion paths, opponent proximity directly from pixels. Text handles discrete/hidden state only.

**Visual channel (board render):**
- Rendered game board matching Colonist UI style
- Hex tiles with resource colors + number tokens with pip dots
- Settlements/cities/roads colored by player
- Robber position, ports marked
- Valid build locations highlighted (green dots/lines for current player's legal moves)

**Text channel (non-spatial game state):**
```
YOUR RESOURCES: wood=2, brick=1, wheat=3, sheep=0, ore=0
YOUR VP: 3/10 | DEV CARDS: knight x1
OPPONENTS:
  BLUE: 4 VP, 2 settlements, 1 city, 5 resources
  RED: 3 VP, 2 settlements, 4 resources
  WHITE: 2 VP, 2 settlements, 3 resources
VALID ACTIONS: [0] BUILD_ROAD edge=42, [1] BUILD_SETTLEMENT node=17, [2] MARITIME_TRADE 4 wheat -> 1 ore, [3] END_TURN
```

**Why dual-channel:**
- Image captures spatial relationships (what Catan is fundamentally about) - the model doesn't need "node 42 is adjacent to ore hex with 6 pips" in text because it can see that
- Text captures discrete/hidden state that a 9B VLM would unreliably OCR from a screenshot (resource counts, dev cards, enumerated actions)
- GLM-4.6V-Flash was trained on interleaved image-text; this is on-distribution

### Agent Decision Loop: Recall -> Think -> Act

The agent uses GLM's native XML tool calling in a three-phase loop. Strategic memory is recalled FIRST so that reasoning is grounded in retrieved context, not generated from scratch.

```xml
<!-- 1. RECALL: Load strategic context before reasoning -->
<tool_call>recall_strategy
<arg_key>query</arg_key>
<arg_value>current_plan</arg_value>
</tool_call>
--> "Longest road + dev card VP. Need 2 more roads then pivot."

<!-- 2. THINK: Reason with recalled context + board image + text state -->
<think>
My plan is longest road + dev cards. I see blue expanding toward
my ore port - if they settle there I lose city potential.
Road at edge 42 both extends my longest road AND blocks blue.
Two birds one stone.
</think>

<!-- 3. ACT: Execute decision -->
<tool_call>select_action
<arg_key>action_index</arg_key>
<arg_value>0</arg_value>
</tool_call>
```

**Why Recall -> Think -> Act (not Think -> Recall -> Act):**
- Thinking without context leads to hallucinated plans and wasted reasoning tokens
- Recall first means the `<think>` block integrates retrieved strategy with current board state
- Same pattern as RAG but for the agent's own strategic memory across turns

**Agent tools:**

| Tool | Purpose |
|------|---------|
| `recall_strategy` | Retrieve long-term plan, opponent models, win condition |
| `update_strategy` | Persist strategic notes after key events (placement, trades, pivots) |
| `select_action` | Execute a game action by index from valid action list |

### Training Data Strategy

Two sources, combined via a flywheel:

**Source 1: YouTube videos (small volume, high reasoning quality)**

Expert Catan gameplay with commentary = pre-annotated reasoning data.

```
Raw Video -> Whisper transcription -> segment by decision point
     |                                        |
     v                                        v
Gemini snapshots (board frames            Reasoning chunks
when reasoning detected in audio)               |
     |                                        |
     +----------------+------------------------+
                      |
                      v
         Frontier VLM aligns and cleans:
         "Given this board frame and transcript segment,
          extract the strategic reasoning and action taken"
                      |
                      v
          (board_image, reasoning, action) triple
```

Best sources by signal density:
- Tournament commentary (3rd person expert analysis) - very high signal
- Streamer gameplay on Colonist.io (screen shows exact board state) - high volume
- "How I got to top 100" strategy breakdowns - explicit reasoning
- Initial placement is the highest-signal target: every video covers it, every player explains it, and it's the highest-impact decision in the game

**Source 2: Colonist replays (high volume, no reasoning)**

8.5K expert games with actions but no explanations. Generate synthetic reasoning using YouTube-derived examples as few-shot style templates:

```
Here are examples of how expert Catan players reason about decisions:
[3 YouTube-derived examples with board images]

Now explain this expert's decision:
[Replay board render + expert action + top 3 alternatives not chosen]
```

Contrastive framing ("why X instead of Y and Z") forces board-specific reasoning over generic heuristics.

**The flywheel:**
```
YouTube (500-2K genuine reasoning examples)
     |
     v
Few-shot reasoning style seed
     |
     +-- applied across -->  Replays (50K+ decision points)
                                  |
                                  v
                         Combined SFT dataset
                         (real reasoning style, replay scale)
```

**Critical moment oversampling:** Weight training data 5-10x toward high-variance decisions (initial placement, robber on 7, late-game pivots, build vs save vs trade). "Rolled 8, collected wheat, ended turn" teaches nothing.

### Training Pipeline

Following GLM paper's own recipe, adapted for Catan:

```
Colonist Replays --> Board Renders + Action Labels --> SFT (LoRA)
                                                          |
                                                     Policy v1
                                                          |
                                            Self-Play (4x same policy)
                                                          |
                                                  GRPO with rewards:
                                                  - Win/loss (+5/-1)
                                                  - VP gain (+1.0)
                                                  - Production increase (+0.1)
                                                          |
                                                  LoRA Training v2
                                                          |
                                                       LOOP
```

**Phase 1 - SFT**: Behavioral cloning on expert replays with XML tool-call output format
**Phase 2 - GRPO**: Group Relative Policy Optimization with Catan-specific reward verifiers
**Phase 3 - RLCS**: Curriculum sampling - focus training compute on mid-difficulty decisions

### Key Differences from Previous Design

| Aspect | Old V0 | New V0 |
|--------|--------|--------|
| Model | Llama 3.1 8B (text-only) | GLM-4.6V-Flash 9B (VLM) |
| Observation | Text-only semantic description | Board render + structured text |
| Action format | Free-form rationale + JSON | Exact legal-menu index with typed parameters |
| RL algorithm | PPO | GRPO (no critic network, simpler) |
| Training recipe | Custom | Following GLM paper's SFT -> GRPO -> RLCS pipeline |

The current runtime's `<rationale>` field is ordinary model-authored output. It
is never presented as provider-native reasoning. Native reasoning is retained
only when a supported provider returns a separate reasoning channel or explicit
reasoning-token evidence.

## TODOs

- [x] Training data pipeline from Colonist replays
- [x] Supabase cloud storage integration
- [ ] Scrape full game replays with JWT token
- [ ] Server-side board renderer (engine state -> annotated PNG)
- [ ] Replay-to-image pipeline (render each decision point from engine state)
- [ ] YouTube reasoning pipeline (yt-dlp -> Whisper -> Gemini frame extraction -> alignment)
- [ ] Synthetic reasoning generation (YouTube few-shot style applied to replay decision points)
- [ ] Critical moment classifier (tag initial placement, robber, pivots for oversampling)
- [ ] VLM SFT script (LoRA on GLM-4.6V-Flash with Recall->Think->Act format)
- [ ] Strategic memory tool implementation (recall_strategy, update_strategy, select_action)
- [ ] GRPO training with Catan reward verifiers
- [ ] Self-play runner with `/nothink` mode for throughput
- [ ] RLCS curriculum sampling by decision difficulty
- [ ] Evaluation vs random/rule-based baselines
