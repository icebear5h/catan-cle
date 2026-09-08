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
single-writer orchestrator. Players choose one exact zero-based entry from the
engine's ordered legal-action menu; they never construct arbitrary actions.

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
- Complete perspective-safe game history and bounded table talk
- Exact legal-menu action identity and bounded decision retries
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

### Features
- Real-time hex board rendering with SVG
- Live game state updates via WebSocket
- Player stats (VP, resources, buildings)
- Step-by-step game control
- LLM decision tracking
- Probability dots on number tokens

### Live failure handling

The red Step banner contains validation/retry guidance only. Expand **Rejected
live attempts** in the inspector for final output, provider-native reasoning,
and provider diagnostics; final output is not the native reasoning channel.
Rejected attempts persist in SQLite `live_failures` and the saved-game API's
`failures` field, independently of completed checkpoints. Earlier failures
from before this change were transient and cannot be reconstructed in full.

If post-action communication fails, the action and agent history remain
committed and checkpointed. The warning stops auto-play (including other
connected tabs); the next Step advances the game rather than repeating the
applied action. The stateful backend intentionally disables source reload;
restart deliberately and load the latest saved game after backend changes.

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

### Known correctness findings

The [deeper correctness audit](reports/correctness-audit-2026-09-08.md) records
unresolved rule, replay-history, and harness-boundary defects. Passing inventory
and replay checks do not certify game outcomes. Reproductions in `tests/audits/`
are strict expected failures, not passing correctness gates; use `--runxfail`
to expose them as failures. This audit did not change the production runtime.

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
