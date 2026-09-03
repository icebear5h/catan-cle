# SFT run fixes after catan-qwen38-spatial-sft-b32-20260901

## Findings
- Vision tower and merger were trained as raw bf16 parameters under AdamW with no
  fp32 master copy, so updates at 1e-6 and 1e-5 mostly rounded to zero. The run
  was effectively LoRA r8 plus 154 token rows.
- The 154 atlas rows were seeded from the checkpoint's untrained padding rows
  because the embedding matrix was already 248,320 wide and resize was a no-op.
- Logged loss and token accuracy were diluted 3:1 by the trivial end-of-turn
  and newline tokens; 0.5 loss meant the answer token sat near chance.
- Learning rates were full-fine-tune values on a 515-step LoRA run; warmup was
  16 steps.
- New spatial_localization_v1 curriculum: stage 1 rows are grouped by image
  (2 to 3 images per batch of 32 under the sequential sampler), stage 2 appends
  the marker replay slice at the end, stage 2 leans "no" 4,296 to 3,312, and the
  probe dot is sub-patch (12 px radius vs 32 px merged patch).

## Plan
- [x] Trainer: promote the visual module to fp32 master weights after wrapping,
  after initial-bundle load, and after checkpoint resume; save checkpoints in
  fp32 and the final bundle in bf16; audit dtypes in the trainable scope.
- [x] Trainer: initialize the 154 atlas rows (embed and lm_head) from the base
  vocabulary mean plus small seeded noise before PEFT wrapping; report norms.
- [x] Trainer: log answer-token accuracy and teacher-forced row exact match,
  excluding end-of-turn and newline tokens, via a language-module hidden-state
  hook and the merged trainable-token head.
- [x] Trainer: log pre-clip gradient norm per optimizer group.
- [x] Trainer and launcher: raise default learning rates (tokens 5e-4, LoRA
  1e-4, merger 5e-5, vision 5e-6) and warmup ratio to 0.1.
- [x] Curriculum: deterministic shuffle of stage 1 and stage 2 train rows,
  interleaving the replay slice.
- [x] Curriculum: balance stage 2 yes/no per entity and relationship by adding
  minority-polarity repetitions.
- [x] Curriculum: emit small and marker-sized gray-dot probes.
- [x] Update tests, README, regenerate spatial_localization_v1, run tests.

## Follow-ups
- [ ] Profile eval generation batch on H200 and raise the default above 48
  (memory headroom exists; 96 or higher is likely fine for 16-token answers).
- [ ] Decide whether the regression panel should run inside the training
  container at each checkpoint instead of as a separate launch.

## Result (2026-09-02, Stage 1 marker-only control)
- Profile at batch 16 x accumulation 2: 98.7 GB peak, 34 s/step, fp32 vision
  and merger confirmed by the dtype audit, vision grad norm 150 to 300.
- Full arm reached 93.1% held-out answer-row exact match (diamond markers,
  five unseen boards) at step 256 with eval loss 0.093; chance is 25%.
  Train answer accuracy was 97 to 100% by step 250 and vision grad norm
  had fallen to about 2. Stopped from the CLI at step 293 after checkpoint
  256 persisted. Checkpoint 256 was published privately as
  `icebear5h/catan-qwen3.8-27b-spatial-sft-marker-only-control`.
- Report: `reports/sft/2026-09-02-qwen38-stage1-marker-only-control.json`.
- Still pending: gray-dot probes at both sizes, blank/shuffle/occlusion
  variants, the two patch-loss arms, and the Stage 1 gate check.

## Review
- Trainer changes are confined to `sft/scripts/train_trl_catan_vision.py`
  plus defaults in `sft/modal_catan_vision_sft.py`. Checkpoint visual state is
  now fp32 (about 1.8 GB) so resume keeps master precision; the final bundle
  stays bf16 for the eval and Hub contract.
- Curriculum changes are confined to
  `data_pipeline/board_recognition/spatial_localization.py`; the dataset was
  regenerated with `--overwrite`.
- Local venv pins transformers 4.57.1 / trl 0.24.0 / peft 0.17.1, while the
  Modal image pins 5.16.1 / 1.12.0 / 0.20.0. Unit tests exercise pure
  functions only; the trainer subclass paths run only on Modal.

---

# Strict raw-image resolution beyond 1024px

## Goal
Measure whether Qwen3.8 Max benefits from 1536px or 2048px on the same locked
strict 60-question raw-board cohort, rather than inferring from patch geometry.

## Confirmed design
- [x] User selected Qwen3.8 Max through direct Novita.
- [x] User selected 1024px, 1536px, and 2048px with identical 90% board framing.
- [x] User selected the strict 60 immediately despite typed-JSON compliance being
  a known confound.
- [x] Reuse the completed 1024px Qwen Max run as the control; do not spend on an
  identical rerun unless its locks differ.

## Plan
- [ ] Render locked, ordinary unannotated 1536px and 2048px variants from the
  same 12 engine contracts and preserve identical canonicalized questions.
- [ ] Validate source/question/scorer locks and image dimensions/hashes.
- [ ] Run Qwen3.8 Max sequentially on all 60 questions at each new resolution
  with temperature 0, 256 output tokens, reasoning disabled, and resume enabled.
- [ ] Recompute strict scores and compare exact accuracy, JSON validity, category
  results, visual/prompt tokens, completion tokens, and latency against 1024px.
- [ ] Report whether gains justify the quadratic vision-attention/token cost;
  keep this panel separate from the 110 perception benchmark.
- [ ] Run focused tests, Ruff, artifact integrity, and repository verification.

## Decision rule
Continue beyond 1024px only if higher resolution increases strict exact answers
without merely changing prose/JSON compliance, and if dense node/edge/port
categories improve enough to justify token and latency growth. A hosted
processor cap or unchanged visual-token count is an immediate stop signal.

---

# Board recognition data curriculum

## Goal
Build a source-controlled curriculum and a deterministic pilot-data path for
turning ordinary Catan board renders into explicit tile/node/edge/port symbols.
The primary output is dense engine-labeled board state, not autoregressive QA.

## Pattern audit
- [x] Reuse the deterministic contract/manifest conventions in
  `sft/scripts/build_node_factor_dataset.py`.
- [x] Reuse isolated/local visual primitives from
  `scripts/build_catan_board_bench_piece_visuals.py` without treating them as
  an independent board-state split.
- [x] Reuse non-destructive contract-to-image derivation from
  `sft/scripts/render_contract_images.py`.
- [x] Follow `references/catan_direct_vision_tower_readout.md`: dense labels,
  one-slot counterfactuals, nearby hard negatives, pre-merger readout targets,
  and closed class vocabularies.
- [x] Follow the curriculum lesson: state complexity/game phase is the primary
  axis; crop/scale/translation/style are paired-view robustness axes.

## Confirmed contract
- Source specification: `data/curriculum/board_recognition/`.
- Generated data: `artifacts/generated/board_recognition/curriculum/`.
- Primary stages: empty/setup board, initial placements, sparse midgame, dense
  endgame; earlier stages remain in later mixtures.
- View policy: ordinary raw full-board renders only. No crops, phase shifts,
  labels, boxes, coordinate atlas, or answer state are painted into images.
- Every state stores one dense typed label payload for 19 tiles, 54 nodes,
  72 edges, and 9 ports. Slot/attribute training rows are sampled from that
  payload rather than serializing the whole board as language.
- Counterfactual groups differ in one target slot only and remain in one split.
  All views, queries, and pair members from a source state share the same split.
- Replay-derived data fails closed against the CatanBoardBench held-out game-ID
  ledger; synthetic rows retain generator seed and renderer provenance.

## Plan
- [x] Audit three existing builders and the direct-readout design.
- [x] Confirm top-level folder placement, working-pilot scope, equal entity-type
  sampling, and raw-full-board-only view policy.
- [x] Add versioned curriculum/stage/view specs and sample/label schemas.
- [x] Add a deterministic pilot builder that emits dense labels and true
  one-slot counterfactual groups from ordinary engine renders.
- [x] Add fail-closed validation for entity coverage, class vocabularies,
  one-variable pair deltas, split grouping, provenance, and artifact hashes.
- [x] Generate a small tracked smoke fixture plus ignored scalable output.
- [x] Add focused tests, build documentation, and verification evidence.

## Acceptance gates
- Every dense label has exactly 19 tiles, 54 nodes, 72 edges, and 9 ports.
- No benchmark game ID or benchmark-derived state enters a training split.
- No state, counterfactual pair, or paired view crosses split boundaries.
- Minimal pairs change exactly one declared dynamic entity attribute.
- The same seed reproduces contracts, labels, manifests, and pixels byte for
  byte apart from explicitly excluded timestamps.
- Class counts are reported for every attribute, with stage/split state counts
  alongside them; hierarchical entity → attribute → slot sampling prevents
  sparse `EMPTY` labels and the larger edge/node sets from silently dominating.

## Review
Implemented `data/curriculum/board_recognition/` with an authoritative curriculum
spec, Draft 2020-12 state/label schemas, and direct-slot-classifier documentation.
The deterministic builder at
`scripts/build_catan_board_recognition_curriculum.py` emits ordinary raw board
images, engine contracts, dense labels, grouped split manifests, hashes, class
counts, and equal tile/node/edge/port counterfactual targets.

The tracked 64px infrastructure fixture contains 32 states in 16 true one-label
pairs: eight states per complexity stage and four pair targets per entity type.
It is 64px only to keep source control small; a full 1024px 32-state pilot built
and validated in nine seconds. Production data defaults to 1024px. Synthetic
stage names are documented as piece-density proxies rather than legal-trajectory
claims.

Verification: 25 focused curriculum/SFT/renderer tests pass; Ruff and formatting
pass; both JSON Schemas and all 32 fixture rows validate; 101 generated fixture
files reproduce byte-for-byte except the declared metadata timestamp; engine
contracts, labels, and rendered pixels reproduce one another; and the default
1024px pilot validates. The full suite is 282 passed, one skipped, and the same
pre-existing rename-receipt failure on four dirty UI files plus the changed
hosted evaluator. The receipt remains untouched to avoid blessing unrelated
changes.

---

# Controlled image-architecture comparison

## Goal
Explain why the 1024px end-to-end VLM run did not improve dense Catan board
reading, then run the user-selected hosted VLM family comparison without Modal.

## Current evidence
- [x] Audit the paired 512px and 1024px Qwen3.8 artifacts.
- [x] Confirm that 1024px changed exact accuracy only from 24/110 to 26/110,
  reduced component accuracy from 44/218 to 42/218, and left node occupancy
  and complete road localization at 0/10.
- [x] Confirm that the 1024 run was an unpinned end-to-end OpenRouter test, not
  an isolated or provider-controlled vision-backbone test.
- [x] Confirm that 594 rendered node-factor boards and their dense contracts are
  already available, and that the generator can create independent variants.

## Agreed hosted-model pilot
- [x] User selected a hosted VLM family sweep and explicitly declined Modal.
- [x] Add explicit OpenRouter reasoning-disable and endpoint-pinning controls to
  the existing reproducible benchmark runner.
- [x] Preflight one 1024px board across distinct hosted VLM families, with one
  request at a time where account in-flight limits require it.
- [x] Use direct Novita hosting for the successful full cohort after OpenRouter
  credits were exhausted, eliminating intra-model OpenRouter route variation.
- [x] Keep models that require native reasoning in a separately labeled cohort;
  do not compare them as though reasoning had been disabled.
- [x] Run the same 10 boards and 11 visual categories for the viable leaders,
  then compare them with the existing Qwen3.8 512px/1024px baseline.
- [x] Render and run a 512px/90%-board control so resolution is separated from
  the tighter framing used by the earlier 1024px variant.
- [x] Report exact/component accuracy, dense node/road categories, errors,
  provider, reasoning tokens, latency, prompt/image tokens, and observed spend.

## Decision rule
This sweep compares complete hosted VLM stacks, not isolated image backbones.
Prefer the family that improves dense node/edge localization consistently, not
one that wins only global counts or formatting. If every family remains near
zero on dense localization, return to the controlled direct-readout experiment
rather than spending more on resolution or larger language decoders.

## Review
The current-only 110-question scorecard is Qwen3.8 Max 48/110 exact and 90/218
components, Gemma 4 31B 22/110 and 43/218, GLM-4.6V 18/110 and 53/218, and
DeepSeek V4 Flash Vision Exp 15/110 and 33/218. Old Qwen checkpoints are
preserved only as historical artifacts and excluded from the active scorecard.

The controlled current-Qwen same-framing run moved from 30/110 at 512px to
48/110 at 1024px; image tokens rose from 258 to 1,026 per request. Sixteen of
the net eighteen exact gains came from tile/robber categories. Node occupancy
reached only 1/10 and complete road localization remained 0/10, so resolution
helps local recognition without solving dense binding.

The unified report is
`reports/catan_board_bench/2026-08-25-hosted-vlm-unified-benchmark.md`; the
current-only machine comparison is
`artifacts/runs/catan_board_bench/strict_60_unified_20260825/comparison.json`.

On the strict raw-image 60, Gemma led at 28/60, followed by DeepSeek at 23/60,
GLM at 22/60, and Qwen Max at 11/60. Qwen produced valid JSON only 20/60 times;
the other three did so 60/60. The matched current-Qwen text run scored 60/60,
with 49 text-only wins, no image-only wins, and reference paired p≈3.55e-15.
The image cohort uses ordinary unannotated engine renders; JSON state is only
the rendering/scoring oracle.

Verification: 29 focused tests, Ruff, formatting, diff checks, source locks,
alias inversion, image/state hashes, strict rescoring, 60 unique zero-error rows
per run, direct-Novita identity, zero reasoning tokens, paired IDs, and the
current-only policy all pass. The full suite is 276 passed, one skipped, and the
same rename-receipt failure on four pre-existing dirty UI files plus the changed
eval runner. The receipt was not regenerated because that would bless unrelated
user changes.

## Unified 60-question image/text comparison
- [x] Confirm the latest strict `text_format_optimization_probe` as the shared
  60-question image/text cohort.
- [x] Invert the text-only opaque aliases to canonical engine IDs, retain the
  engine-state JSON as the answer oracle, and render the 12 ordinary unannotated
  engine screenshots at 1024px with the same 90%-board style.
- [x] Add a strict typed-JSON image evaluator that reuses the exact 60 question
  IDs, output schemas, scorer, prompt rules, Novita controls, and no-reasoning
  policy used by the hosted sweep.
- [x] Preflight one board, then run the same four viable Novita VLM families on
  all 60 image questions with resumable sequential calls.
- [x] Run current Qwen3.8 Max on the frozen `indexed_tile_rows` text projection
  of the exact same 60 questions through direct Novita with reasoning disabled.
- [x] Produce one unified artifact and report containing only the current hosted
  model scorecard: the 110-question perception diagnostic, shared 60-question
  image results, and matched Qwen3.8 Max 60-question text result. Preserve but
  exclude every old Qwen checkpoint result; do not add incomparable numerators.
- [x] Verify dataset locks, alias inversion, raw image/state correspondence,
  60 unique responses per model, strict scores, provider/reasoning metadata,
  tests, lint, and artifact hashes.

---

# Direct Catan vision-tower readout research

## Goal
Find a literature-backed way to tune the Catan vision tower for immediate
query-conditioned perception: one forward pass, one class answer, no generated
reasoning or prior answer tokens.

## Plan
- [x] Audit the existing Catan visual benchmark, renderer labels, node-factor
  dataset, and Qwen SFT trainable scope.
- [x] Review primary literature on fixed output queries, dense grounding,
  spatial supervision, board recognition, and selective vision unfreezing.
- [x] Verify the current Qwen3.8 vision configuration and availability of
  pre-merger patch states.
- [x] Design the direct slot-readout architecture, data curriculum, losses,
  trainable-scope ablations, and causal evaluation gates.
- [x] Record the proposal in
  `references/catan_direct_vision_tower_readout.md` without implementing a
  trainer.

## Review
The recommended Catan Slot Readout bypasses autoregressive generation. A learned
atlas-slot plus attribute query cross-attends to Qwen3.8's final pre-merger
vision patches and returns one closed-set class. The required control is a
frozen-tower head; the first actual tower intervention is the last six vision
MLPs at a low BF16 learning rate, followed only if justified by complete-block
and full-tower ablations. Dense engine contracts, pixel-identical one-slot
counterfactuals, nearby-slot hard negatives, target/control occlusion, and
held-out real boards separate genuine localization from class and slot priors.

The existing 4-bit LoRA SFT entrypoint freezes the tower and merger, so it should
remain unchanged. The proposal calls for a separate vision-only trainer. No
training code was added in this research task.

Before implementing that trainer, a paired OpenRouter check tested whether more
visual resolution was already sufficient. Qwen3.8-27B received the same 110
questions over the same 10 boards with native reasoning disabled. Re-rendering
at 1024px with the complete board occupying about 90% of the canvas moved exact
accuracy only from 24/110 to 26/110, while component accuracy fell from 44/218
to 42/218. Node occupancy and full road localization remained 0/10. The larger
images used 149,837 prompt tokens versus 65,357 at 512px and cost $0.064299
versus $0.027169. This rules out resolution alone as the fix; it does not isolate
the vision tower because OpenRouter serves the complete VLM and the 1024 run was
distributed across eight unpinned providers.

Run artifacts:
`artifacts/runs/catan_board_bench/catan_board_bench_100_1024_board90/openrouter/qwen3_8_27b_full_20260825/`.

---

# Minimal shared prompt-suite redesign

## Goal
Apply the supplied RL-derived ideas as general prompt-engineering guidance to the
active shared Catan prompts: remove simulated personas, give the model only the
task and model-visible interface facts, avoid prescribing strategy that the
model should infer, and keep action descriptions attached to the actions they
describe.

## Audit
- The prior shared default was `cle/harness/suites/catan_v4.yaml`; live games,
  interactive replay, and replay action-diff evaluation all load it through
  `load_context_suite()`. Its authored prose is 640 words.
- `catan_v4` begins with an expert-player role, repeats legal-menu facts already
  rendered by the observation formatter, and prescribes detailed opening,
  robber, discard, and main-game strategies.
- `communication_v1.yaml` also begins with a player-role simulation. Its actual
  interface is only silence or a bounded message with audience/intent and an
  optional non-binding commitment.
- The current `<game_plan>` is causal because accepted plans become future
  strategic memory, and `<rationale>` is an intentional user/eval diagnostic.
  Preserve both product-facing fields while simplifying their descriptions.
- The numbered action menu is the model-visible execution interface, not a
  function-calling tool API. Its trade entries already describe willingness,
  confirmation, and named-resource parameters at the action-definition layer.
- `cle/prompts/prompt_suite_v1.py` and `cle/agents/llm_player.py` contain inactive
  legacy prompts with no production callers; changing them would add migration
  scope without affecting the shared harness.
- Reward configuration and trainer design are outside this correction: the
  principles came from RL, but the requested target is the general prompt suite.

## Implementation plan
- [x] Add `catan_v5.yaml` rather than mutating the reproducible v4 baseline.
- [x] Keep only the win objective, player color, decision-specific facts not
  already present in the observation/action menu, and the exact XML contract.
- [x] Remove persona language, probability/VP heuristics, prescribed action
  sequences, opponent-targeting advice, and duplicated menu validation advice.
- [x] Preserve full visible history, current observation, legal actions,
  strategic memory, `<game_plan>`, `<rationale>`, `<action>`, and conditional
  named-resource `<trade_offer>` behavior.
- [x] Add `communication_v2.yaml` with no role/persona and only the communication
  decision plus response-schema semantics.
- [x] Point shared loaders to the new versions and update harness documentation.
- [x] Add regression tests for rendered-prompt minimality, absence of role and
  strategy prescriptions, retained phase/interface facts, and unchanged parsing.
- [x] Run focused harness/player/replay/communication tests, Ruff, the full
  Python suite, `git diff --check`, and inspect exact rendered prompts.

## Review
- `catan_v5.yaml` is now the shared live/replay/eval default. Its complete
  authored task, phase, and response prose is 160 words versus v4's 640 words;
  the rendered system prompt is 75 words and contains no persona or strategy
  heuristics.
- Phase-specific model input now contains only decision facts, in the user
  message rather than the stable system prompt. Full visible history,
  observation, menu identity, strategic memory, and XML output fields are
  unchanged.
- `communication_v2.yaml` reduces the system prompt to 21 words, moves player
  color into the event packet, and removes the simulated-player persona plus
  unsolicited table-talk strategy.
- Default loaders, docs, and prompt-contract tests now target v5/v2. The v4/v1
  files remain available for reproducible comparisons, and inactive legacy
  prompt implementations were left untouched.
- Verification: 33 focused prompt/player/replay/communication tests pass;
  targeted Ruff and `git diff --check` pass; exact rendered prompts were
  inspected; and the built wheel contains both new YAML suites.
- The complete suite ran 262 passing tests and one gated skip. Its sole failure
  is unrelated pre-existing CatanBoardBench rename-receipt drift in four dirty
  `catan-board-bench-ui` files; the same suite passes with that one receipt test
  deselected. Full-repository Ruff also retains unrelated errors in vendored
  references and old playground/scripts, while all touched files pass.

---

# Fixed engine, sandbox, and harness refactor

## Goal
Implement the clarified 2026-08-20 model with the sandbox as the runtime
composition root:

```text
GameViewer --view/step--> CatanSandbox
                            |- deterministic Engine/GameState
                            |- SeatHarness[RED]   -> Policy -> LLM
                            |- SeatHarness[BLUE]  -> Policy -> LLM
                            |- SeatHarness[WHITE] -> Policy -> LLM
                            `- SeatHarness[ORANGE]-> Policy -> LLM
```

The sandbox **contains** both the engine and harness seats. Containment does not
collapse module boundaries: engine rules, seat context assembly, and provider
transport remain independent components owned and coordinated by one sandbox
instance. The game viewer becomes HTTP/WebSocket/frontend I/O only.

Current model input is text-only. The contracts reserve an optional attachment
port so a future engine-state renderer can add images without restoring the
current Playwright/frontend dependency.

## Current dependency problems
- `cle/env/catan_env.py` makes PettingZoo AEC the primary environment, and
  `observe()` mutates per-seat event cursors. That does not match an event-driven
  persistent LLM seat or pure perspective projection.
- `cle/agents/llm_player.py` combines the engine `Player` adapter, prompt policy,
  provider clients, Playwright/frontend capture, parsing, strategic memory, and
  fallback behavior in one class.
- The editable `cle/prompts/` suite is not used by `LLMPlayer`; live prompt text
  and response parsing remain embedded in that 500-line class.
- Live viewer routes choose policies, call `Player.decide()`, execute engine
  actions, fan out events, and assemble decision metadata themselves.
- `Game` seeds Python's process-global RNG, while `Game.copy()`/undo snapshots do
  not preserve an independent RNG stream.

## Prompt/context suite decision
Use a small typed project harness rather than adding a generic orchestration
framework:

- `cle/harness/suites/catan_v1.yaml` owns editable prose, section ordering,
  phase guidance, memory instructions, response tags, and context-policy knobs.
- Pydantic models load and validate YAML. Python owns dynamic facts, privacy,
  legal-action identity, event cursors, compaction, and strict rendering. YAML
  never contains executable game logic or unrestricted expressions.
- `ContextAssembler` is a pure deterministic function from `DecisionRequest +
  SeatSession + ContextSuite` to ordered model messages.
- `Policy` is a narrow provider-independent completion protocol. OpenRouter/Groq
  adapters consume assembled messages and return raw text/usage; parsing produces
  a typed `PolicyDecision` tied to the exact decision ID and legal menu.
- Prime Verifiers' `Harness/State/Trace` split and OpenAI/Pydantic message-history
  APIs validate these concepts, but adopting their complete runtime would add
  provider/training orchestration without solving Catan visibility or causal
  event delivery. The local typed layer stays interoperable at the message and
  trace boundary.

Initial YAML shape:

```yaml
id: catan-agent
version: 1
system: {template: "..."}
context:
  order: [trajectory, strategic_memory, unread_events, observation, legal_actions]
  trajectory: {mode: full, max_messages: null}
sections:
  unread_events: {heading: "RECENT EVENTS", empty: omit}
phase_guidance:
  initial_settlement_1: "..."
  initial_settlement_2: "..."
  initial_road: "..."
  main_game: "..."
response:
  format: xml
  tags: [game_plan, turn_plan, action]
```

## Migration slices

### Slice 1 — contracts and deterministic core
- [x] Freeze the focused baseline: `53 passed, 1 skipped` on game-force,
  replay-response/trading, and action-diff tests.
- [x] Add immutable typed `DecisionRequest`, `PolicyDecision`, `Transition`,
  `SandboxSnapshot`, `SeatSession`, and perspective-safe event contracts.
- [x] Move chance generation to a per-game RNG stream and copy/restore it with
  game snapshots while preserving seeded behavior.

### Slice 2 — sandbox aggregate
- [x] Add `cle/sandbox/catan.py` owning `Game`, the append-only event log, and a
  `SeatHarness` for each color. Expose pure `view()`, `decision_request()`,
  `step()`, `override_step()`, `snapshot()`, `restore()`, and terminal-result methods.
- [x] Keep unread cursors in each `SeatSession`; sandbox observations never
  consume events merely because a viewer or evaluator reads state.
- [x] Keep `cle/env/catan_env.py` only as a compatibility adapter during the
  migration; PettingZoo AEC is not the primary LLM runtime.

### Slice 3 — editable context and provider policy
- [x] Add versioned live and replay YAML suites, strict loaders, deterministic
  assemblers/parsers, receipts, and tests for section order and menu binding.
- [x] Move active prompt/context/memory logic out of `LLMPlayer` and
  `game_viewer`; split provider transport and use per-game/per-seat session IDs.
- [x] Keep a future optional attachment interface, but send text only now.
- [x] Retain `LLMPlayer` only as a temporary compatibility adapter; new sandbox
  control flow never requires the engine to call a model.

### Slice 4 — viewer I/O adapter
- [x] Store/inject the active sandbox in server state.
- [x] Change live start/step/auto-play routes to call sandbox methods and serialize
  sandbox projections; remove direct policy and `Game.execute()` ownership.
- [x] Add `ReplaySandbox`, move Colonist/replay mechanics to `cle/replay`, and
  leave the old viewer modules as compatibility aliases only.
- [x] Change Socket.IO broadcasting to read the active sandbox and seat statuses
  while preserving the existing response JSON.
- [x] Do not redo CatanBoardBench migration; it was outside this slice.

### Slice 5 — verification
- [x] Add same-seed/concurrent-game/copy/restore, pure-view, privacy, exact-menu,
  stale-revision, idempotent-retry, and per-seat continuity tests.
- [x] Run focused tests, the full Python suite, and the consolidated replay corpus.
- [x] Review the diff against the already-dirty worktree and document results.

## Safety rules
- Preserve legal-action ordering, replay semantics, viewer JSON, and provider
  behavior during extraction. The YAML suite initially reproduces current text
  and XML tags; later prompt experiments use a new suite version.
- Do not overwrite or revert unrelated dirty-worktree changes. Prefer new modules
  and targeted compatibility edits.
- Never make provider cache, frontend state, or process-global viewer state the
  source of seat continuity.

## Review
- Added `CatanSandbox` as the live composition root and `ReplaySandbox` as the
  replay facade. Live viewer code now calls `step()`/`view()` and contains no
  `LLMPlayer`, `Player.decide()`, or direct `Game.execute()` control path.
- Moved Colonist decoding and all transactional replay mechanics under
  `cle/replay`; obsolete viewer compatibility modules were subsequently removed.
- Added `cle/harness/suites/catan_v1.yaml` and `replay_v2.yaml`. Strict Pydantic
  loaders keep authored prompts editable while Python retains privacy, cursors,
  exact legal menus, and accepted-response commits.
- Each live seat keeps full accepted conversation history, strategic memory,
  unread-event cursor, stable session ID, and restorable receipts. OpenRouter
  receives the complete active message list and an `x-session-id` affinity key.
- Engine randomness is per-game and cloned by copy/undo/sandbox snapshots; it no
  longer mutates Python's process-global RNG.
- Verification: Ruff and `git diff --check` pass; Python is 191 passed / 1 gated
  skip; the explicit 66-game corpus audit passes 31,506 actions and 15,460 trade
  lifecycle actions; frontend production build passes; wheel inspection confirms
  both YAML suites are packaged.
- Fixed an unrelated stale data-layout test receipt after the manifest verified
  all 352 files at 66,380,376 bytes (+462 over the old assertion).
- `LLMPlayer` and `ServerState.current_game` were initially retained for
  identified callers; viewer re-export modules were removed once repository
  imports moved to the core packages. A later completion audit found the unused
  PettingZoo `CatanEnv` advertised a text space while accepting engine objects,
  so that broken adapter and its dead parser were removed rather than preserved.
  Replay seat persistence beyond the existing replay-response contract and fully
  explicit chance-outcome sources are follow-up migrations, not hidden rewrites.

## Compatibility-shim removal
- [x] Inventory viewer and legacy-package re-export modules and their consumers.
- [x] Migrate repository imports/tests to `cle.harness`, `cle.replay`, and
  `cle.sandbox`, then delete aliases with no required runtime I/O role.
- [x] Verify no stale imports remain; run focused tests, Ruff, the full suite,
  wheel inspection, and import-boundary checks.

Review:
- Deleted the five-module `playground/game_viewer/colonist` alias package and
  seven replay aliases, including `llm_response.py` and `step_executor.py`.
- Reduced replay/live package initializers to viewer-purpose docstrings and
  removed the unused route-level WebSocket re-export.
- Migrated runtime, eval, pipeline, playground utility, and test imports to
  `cle.harness.replay`, `cle.replay.colonist`, or `cle.replay.runtime`.
- Verified old imports fail, core imports succeed, no textual references remain,
  Ruff and diff checks pass, the wheel contains only core runtime modules, and
  the full suite passes at 200 passed / 1 gated skip.

## Lightweight event-driven sandbox implementation
- [x] Hard-rename `engine` to rules-only `game_engine`, with `GameEngine`,
  `GameState`, colors rather than policy objects, strict transitions, per-engine
  RNG, explicit snapshots, and no live per-action state copies.
- [x] Add canonical sequenced events, pure player projections, complete visible
  event context, bounded message windows, and exact private overlays.
- [x] Replace seats with async sandbox players, shared OpenRouter/Groq/vLLM
  transports, bounded choice retries, one-writer async orchestration, whole
  trade barriers, and a 64-game cooperative `SandboxPool`.
- [x] Add bounded multi-proposal `TradeWindow` state, wildcard support,
  counteroffer DAGs, deterministic responses, tunable limits, and exact replay
  mapping for all 66 Colonist replays.
- [x] Add broad communication triggers, fixed-cutoff concurrent reaction rounds,
  `SILENCE`, validated communication YAML, private/public messages, and pinned
  non-binding commitments.
- [x] Migrate live viewer, WebSocket, replay, evals, packaging, and docs without
  engine/seat/sandbox compatibility shims.
- [x] Run complete static, test, corpus, package, and frontend verification.

Review:
- `CatanSandbox` now contains only `game_engine` and `players`; `await step()`
  owns prompt/retry/barrier/application/acknowledgement orchestration. It has no
  locks, duplicate event buffer, stale/pending APIs, override path, or per-step
  snapshots.
- Model requests can run concurrently across games and barrier participants,
  but all engine mutations are deterministic and table ordered. Auto-stop takes
  effect between complete steps.
- `GameEngine` owns mutable state plus the append-only resolved event log,
  privacy projection, trade board, messages, commitments, RNG, and snapshots.
  Policy/search code lives under `cle/players`.
- Trade and communication growth are bounded by snapshotted typed limits;
  recorded replay actions bypass live caps only to preserve authoritative source
  behavior. The live engine uses `TradeWindow`; the replay executor still
  materializes its older color-keyed trade overlays as derived projections while
  replay mechanics are migrated proposal-by-proposal.
- Verification passes: 237 Python tests / 1 gated skip; explicit 66-game corpus
  audit covers 31,506 actions and 15,460 trade lifecycle actions; Ruff, import
  and stale-name boundaries, `git diff --check`, frontend production build, and
  wheel content inspection all pass.

---

# API inference continuity and cache ownership

## Goal
Clarify whether OpenRouter retains reusable model state, how Pi and Codex carry
conversation continuity, and where persistent per-seat Catan context should live.

## Plan
- [x] Verify OpenRouter prompt-cache and routing semantics from official docs.
- [x] Trace Pi and Codex session/context ownership from their docs and source.
- [x] Define the Catan harness source of truth independently of optional provider
  cache hits.

## Review
- OpenRouter API requests are stateless. Upstream prompt caching can reuse the
  prefill for an identical prefix, but it is temporary, provider/model-specific,
  and observable only through cache-read/write usage. `session_id` improves
  provider affinity and log grouping; it is not conversation memory.
- Pi owns continuity in `AgentSession` and `SessionManager`: an in-memory or
  persistent JSONL message tree is rebuilt into the active context on every
  model request, with old context compacted into a summary plus retained tail.
- Codex likewise owns continuity in a local `Thread`; repeated `run()` calls use
  that thread and `resumeThread()` reconstructs persisted sessions from
  `~/.codex/sessions`. Transport-level response IDs and prompt-cache keys are
  optimizations, not the durable source of truth.
- The current replay endpoint is one-shot: `query_text()` sends only a fresh
  system/user pair, and the browser carries only prior goals. It has neither a
  persistent conversation nor an explicit OpenRouter session-affinity key.
- Recommended Catan design: one durable harness session per game and seat,
  backed by a canonical perspective-safe event log and structured strategic
  memory. Each decision appends only unread events plus an authoritative current
  checkpoint and exact legal menu. Provider caching may reduce prefill cost but
  must be safe to lose at any request.

---

# Replay sandbox dependency plan

## Goal
Define a core-first architecture where an agent-facing Catan sandbox extends the
canonical game rules with explicit outcomes, a replay sandbox adds recorded
outcomes and replay navigation, and `game_viewer` remains a presentation adapter.

## Terminology
- **Engine:** internal Catan rules implementation. It should become deterministic
  conditional on a state, player action, and explicit chance outcome.
- **CatanSandbox:** agent-facing environment contract over the engine: reset,
  perspective-safe observation, current actor, legal actions, step, terminal
  result, snapshot, and restore.
- **ReplaySandbox:** extension/wrapper around `CatanSandbox` that translates one
  recorded source event, supplies fixed outcomes, advances a causal cursor, and
  adds undo/goto/audit behavior.
- **Harness:** model-side control loop. It builds decision packets, invokes a
  policy, parses a typed decision, and submits it to a sandbox; it does not own
  Catan transition or replay-navigation logic.
- **Game viewer:** Flask/Socket.IO/React adapter that sends commands to a sandbox
  and renders sandbox projections. It contains no replay mechanics.

## Target dependency direction
- `game_viewer -> replay_sandbox -> catan_sandbox -> engine`
- `agent_harness -> sandbox_protocol` and `provider_adapter -> policy_protocol`
- `evals`, commentary, CatanBoardBench, and future CLI tools consume sandbox APIs
  directly; none imports `game_viewer`.
- `engine`, sandbox modules, and harness contracts never import Flask, Socket.IO,
  React, provider clients, evals, or process-global viewer state.

## Behavior-preserving migration plan
- [ ] Freeze current replay transition, decision-packet, privacy, trade, cursor,
  and audit behavior with fixture fingerprints and the existing corpus suite.
- [ ] Define typed `SandboxProtocol`, `StepResult`, `SandboxSnapshot`,
  `ChanceOutcome`, and `PolicyDecision` contracts without moving behavior.
- [ ] Add `CatanSandbox` as a thin composition wrapper over the existing `Game`;
  keep the engine API stable during this phase.
- [ ] Extract replay-file loading and Colonist translation into core replay
  modules, replacing the duplicate viewer and CatanBoardBench composition roots.
- [ ] Add injectable `RandomOutcomeSource` and `ReplayOutcomeSource`; remove
  process-global RNG and replay `force` knowledge from the eventual engine
  boundary one outcome type at a time.
- [ ] Move checkpoints, exact trade ledger, cursor, causal stepping, audit, and
  `step()` into `ReplaySandbox` while preserving one transactional
  coordinator.
- [ ] Split decision packet construction and output parsing from provider
  transport; make the harness depend only on sandbox and policy protocols.
- [ ] Migrate headless consumers first: CatanBoardBench, action-diff eval, then
  commentary. Each must stop importing `playground.game_viewer` before moving
  the viewer.
- [ ] Reduce `game_viewer` to route/view-model adapters over one injected sandbox;
  unify HTTP and Socket.IO snapshot serialization.
- [ ] Add import-boundary checks forbidding core-to-viewer/provider imports,
  viewer imports from eval/data modules, and cross-package private-symbol imports.
- [ ] Remove compatibility re-exports only after every consumer has migrated and
  the complete replay corpus produces equivalent transitions and decisions.

## External design references
- Prime Intellect verifiers v1 separates taskset (what), harness (how), and
  runtime/sandbox (where); its seeded Kuhn Poker environment is the closest
  host-refereed multi-agent example for Catan.
- OpenSpiel supplies the strongest game-state vocabulary: current player, legal
  actions, explicit chance outcomes, apply action, per-player observation,
  clone, and child-state branching.
- PettingZoo AEC is only a state-machine reference for actor ordering and legal
  masks, not the target LLM harness API. The target is event-driven and invokes
  persistent per-seat LLM interactions only for genuine `DecisionRequest`s.
- Gymnasium/dm_env contribute compact transition-result semantics, not the
  model-control loop.
- Prime uses “sandbox” primarily for execution infrastructure, so project code
  should use qualified names (`CatanSandbox`, `ReplaySandbox`) to avoid ambiguity.

## Acceptance criteria
- No production behavior or prompt/schema changes during extraction.
- `game_viewer` is an outer consumer only.
- Every replay transition is atomic, causal, undoable, and corpus-equivalent.
- Every policy sees only its perspective-safe observation and the exact legal
  menu for that decision.
- Independent sandbox instances can run concurrently without Flask or global
  state, and cloned branches include all outcome/RNG state.
- Static dependency checks report no forbidden edges or production import cycles.

## Review
Plan only; no production implementation started.

---

# Replay harness code map

## Goal
Locate the code that currently serves as the replay/agent harness, trace its
runtime boundaries, and identify focused refactoring opportunities without
changing production behavior.

## Plan
- [x] Locate replay entry points and harness-shaped modules.
- [x] Trace state construction, model invocation, response parsing, and UI/API
  integration.
- [x] Summarize code ownership, duplication, and highest-value improvements.

## Review
- `replay_pipeline/` is currently only a compatibility alias for replay
  acquisition/decoding modules under `data_pipeline.bootstrapping`; it is not
  the runtime replay or policy harness.
- The implemented replay domain is spread across
  `playground/game_viewer/{colonist,replay,routes}`, with policy packets and
  model calls in `replay/llm_response.py`, provider transport in
  `playground/openrouter_client.py`, and offline orchestration in
  `evals/replay_action_diff.py`.
- The offline evaluator loads replays through Flask's test client and the
  process-global `ServerState`; CatanBoardBench duplicates replay construction and
  also imports the UI package. This is the clearest package-boundary problem.
- Recommended first refactor: introduce an isolated `ReplaySession` plus a
  shared replay loader in the real `replay_pipeline` package, then adapt Flask,
  evals, commentary, and CatanBoardBench one consumer at a time. Preserve the
  existing executor and packet behavior initially rather than rewriting both.
- Verification: 46 focused replay harness tests passed; 1 gated corpus test was
  skipped.

---

# Multimodal expert-reasoning data pipeline

## Goal
Build a provenance-preserving pipeline that turns (a) noisy Catan YouTube gameplay commentary plus board frames and (b) authoritative Elo-indexed Colonist replays into high-confidence policy SFT, rationale SFT, value, and same-state preference examples.

## Repository findings
- The noisy example is `artifacts/generated/pretraining/legacy_corpus/transcript_t5RZGAJfKss.md`; phrases such as "this spot", compressed number triples, streamer narration, table talk, profanity, and overlapping speakers are not usable without the corresponding frames and timestamps.
- `data_pipeline/ingestion/youtube_scraper.py` currently saves captions and coarse 30-second chunks only. It does not acquire video/audio, retain a raw asset manifest, run word-timestamp ASR/diarization, extract frames, detect decisions, or align actions.
- `data_pipeline/bootstrapping/generate_training_data.py` is a prototype and must not become the new dataset foundation: it mutates cumulative state before formatting the claimed pre-action observation, can collapse multiple source changes into one action, lacks authoritative legal-action packets, and omits the current replay privacy/leakage guarantees.
- The safer foundations already exist: exact replay parsing/auditing, perspective-safe decision packets in `cle/harness/replay.py`, replay rendering in `data_pipeline/catan_board_bench/`, game-level leakage-safe splits, and multimodal JSONL conventions in `sft/scripts/build_vlm_sft_dataset.py`.
- The top-player index contains 8,495 records with indexed-player ratings from 1,834 to 2,034. That rating is an index-time sampling/quality prior for one player, not a clean per-action reward or historical whole-lobby Elo label.

## Options
1. **Evidence-first two-lane pipeline (recommended):** keep video-derived human rationale seeds and replay-derived authoritative decisions separate; join them only through versioned, validated decision records. More setup, but auditable and scalable.
2. **Fast replay distillation:** skip video alignment, ask a frontier teacher to rationalize replay actions, and filter with a critic. Fastest baseline, but invites polished post-hoc rationalization and loses genuine expert voice.
3. **End-to-end video VLM:** feed long clips directly to a VLM and accept extracted state/action/reasoning. Lowest engineering effort, highest hallucination/alignment risk, and weak reproducibility.

## Proposed artifact layers
1. `source_manifest/v1`: immutable source IDs, URLs/paths, hashes, acquisition method, timestamps, rights/review metadata, channel/player identity, and Elo provenance.
2. `video_timeline/v1`: word-level ASR alternatives, speaker labels, frame references, scene/action change points, and untouched transcript spans.
3. `grounded_video_decision/v1`: pre/action/post frames, public board contract, perspective-visible private state when trustworthy, inferred action, transcript evidence spans, explicit alternatives, and independent confidence fields. Low-confidence fields remain null rather than guessed.
4. `replay_decision/v1`: authoritative pre-action snapshot, perspective-safe recent history, complete indexed legal actions, expert action, trajectory outcome, indexed-player Elo metadata, and canonical board render.
5. `rationale_candidate/v1`: concise evidence-linked goal, relevant facts, beliefs, alternatives, tradeoff, chosen action, expected consequence, generator provenance, and critic findings. It must distinguish quote, normalization, and model inference.
6. `preference_pair/v1`: two actions from the same information state with label source and margin. Labels may come from explicit expert comparison, common-random-number rollouts, or teacher/critic agreement; Elo alone cannot label a pair.
7. Training exports: separate perception grounding, action-only BC, rationale-plus-action SFT, value/outcome, and preference datasets so each contribution can be ablated.
8. `style_exemplar/v1`: curated expert-commentary excerpts and distilled consideration checklists, indexed by decision type, used as ICL context so teachers emulate expert voice; style-only, never a source of board facts.

## Transcript style/terminology exemplar bank (ICL style transfer)
- Idea: expert transcripts are reusable even when a video cannot be board-grounded — they capture *how* experts talk and *what they consider*, separately from any specific board state.
- Build two exemplar forms per decision type (placement, robber, trade, dev timing, blocking, endgame):
  1. Verbatim style snippets: cleaned timestamped quotes showing terminology ("6-9-3", "pop a dev", "smooshed"), framing ("the idea is…", "I'm just concerned…"), and hedging/beliefs.
  2. Consideration templates: distilled checklists of what experts attend to (e.g. robber: leader check, number denial, steal EV, retaliation risk) with source-transcript citations.
- At distillation time, retrieve k exemplars matching the decision type and inject them as few-shot ICL for the teacher (Fable/GPT). Instruct: emulate the style, vocabulary, and consideration coverage; take every board fact from the replay packet only; exemplar board specifics are irrelevant noise.
- Keep a terminology normalization map beside raw quotes (ASR fixes like "693"→"6-9-3", "weed"→"wheat") with provenance; never overwrite the raw span.
- Leakage rule: the critic validates every factual claim against the packet, so exemplars can shape voice but cannot inject facts.
- Eval: hold out transcripts; compare generated traces vs expert style on terminology density and consideration coverage, and vs generic-prompt traces to prove the exemplars change style, not just length.
- Payoff: unpaired playthroughs and strategy videos (the weak tiers) become style/considerations capital for the strong tier, so every transcript we already have contributes to the first-class dataset.

## Paired-game primary lane: transcript injection, no video perception
- For videos paired to an archived replay, the critical path is text-only: replay supplies the board/state/legality from the narrator's own perspective; the transcript supplies the authentic reasoning voice; video pixels are demoted to one-time pairing verification and rare deictic tie-breaks.
- Evidence (bootymunchr 242781000): 737 events carry wall-clock `deltaS` summing to 39.4 min vs 39.9 min recorded duration, so the replay has its own real-time axis; the archived payload is already `playerPerspective=5` (the narrator), giving exactly the player-view supervision state.
- Alignment = piecewise-monotonic map between replay wall-clock and video time, anchored on spoken dice rolls, placements, and dev-card plays; video cuts become piecewise offsets; per-segment confidence with quarantine for low-confidence stretches.
- Per decision export: decide-mode packet (player-view observation, legal actions, recent activity) + pre-action transcript window via the time map; Role B writes the trace with zero video tokens, immune to ASR coordinate errors because the replay names the vertex.
- Follow-along stepping design: build a deterministic merged timeline (transcript segments + replay events sorted by aligned time; see `pilots/_2n5F2DxtPI/merged_timeline.txt`). Two consumption modes: (a) one-shot windowed slices of the merged doc for Role B on uncut videos; (b) a dual-cursor stepping agent that advances the replay with the transcript, keeps running reference annotations and alignment micro-anchors, queries the CLI on demand, and emits decision records exactly when replay decision events arrive — making the pre-action cutoff structural rather than a post-hoc filter. Guard band rule: caption segments starting within ~3s of an action are labeled during-action and excluded from pre-action evidence (caption boundaries straddle actions; observed at 207.1 vs 207.4). This mirrors the event-sourced decision-triggered live harness, so the annotator and the student share packet shape and cursor discipline.
- [x] End-to-end demo on the bootymunchr pair (`pilots/_2n5F2DxtPI/demo_report.md`): 659-segment transcript fetched; 136-action wall-clock timeline decoded; offset ≈ 0 confirmed on three anchors (uncut video); first-settlement packet + gpt-5.6-terra trace at $0.012/16s with 4 real spoken alternatives; gates caught the teacher smoothing a "verbatim" quote (14/15 exact) and confirmed zero pre-action-cutoff violations. Gaps listed in the report: port geometry in packets, engine legal actions, align agent for cut videos, critic pass, frame-level pair verification.

## Spatial reference interface (three layers)
- Measured on the bootymunchr board: 17/18 three-hex corners have unique dice-number triples (one (3,4,8) collision); 6 of this video's 9 spoken refs resolve uniquely by numbers alone, the ambiguous ones carry spoken disambiguators (resource "the ore", direction "up/down"), and one ref ("8 5 10") matches nothing — the resolver fails closed instead of guessing.
- Layer 1 — tool IDs: stable corner/edge indices, machine-facing, never expected from model generation.
- Layer 2 — reasoning language: canonical self-describing descriptors in the expert dialect, `[10WOOD 8BRICK 4SHEEP]` sorted by number, `| coast` tag for 2-hex corners, `+brick-port` tags; packets always print ID + descriptor together.
- Layer 3 — deterministic resolver in the CLI: number triple/pair + optional qualifier (resource, port, screen direction — Colonist renders one fixed orientation) → explicit candidate set or NONE; the model selects among candidates and records the deciding qualifier + confidence; edges named by endpoint descriptors with rotation-invariant "toward" phrasing.
- Student symmetry: decide-time legal actions arrive pre-enumerated as (index, ID, descriptor); no stage of the pipeline generates raw geometry.

## Reference annotation suite (labeling pass over the merged timeline)
- Prototype pass 0 on the bootymunchr game (`pilots/_2n5F2DxtPI/ref_annotations_pass0.json`): 15/17 number-triples bound deterministically via resolver (incl. word-order variants), 1 ambiguous, 1 correctly rejected as nonexistent, 16 pair-refs left as candidate sets, 13 jargon hits, 43/659 segments carry refs.
- Pass 0 — deterministic (free): number patterns → resolver bindings + descriptors (resource gloss comes from the board, not the model); seed lexicon; roll-mention filter.
- Pass 1 — flash-tier typed tagging (~$0.02/game): resource-combo refs ("the ows spot"), player refs ("fourth", "he"), deictic refs, unknown jargon flagged into the lexicon; attaches candidates, never resolves ambiguity.
- Pass 2 — smart RLM on the residue only (~$0.05/game): pair disambiguation from spoken direction/port context, pronoun→seat binding, mishear hypotheses; align-mode CLI tools; every binding carries confidence + deciding evidence.
- Pass 3 — deterministic gates: binding ∈ candidate set; temporal consistency (corner discussed as available must be unoccupied at that time); actor consistency.
- Output `reference_annotations/v1` over the merged timeline; consumed by trace generation, the stepping walk, and the observer lane; also exportable as dialect→board-binding SFT for the student.

## Catan replay query CLI (interactive grounding tool)
- Idea: instead of stuffing one static packet into context, give the aligning/teaching model a CLI to query and advance authoritative replay state until it finds what the commentator is referencing ("this spot", "he blocks our nine", "the 6-9-3").
- v0 commands over the existing authoritative replay executor (read-only, no force paths):
  - `load <replay.json>` / `info`: players, colors, settings, turn count
  - `goto <n>` / `next` / `prev`: authoritative sequential navigation only
  - `state [--perspective COLOR]`: perspective-safe observation at cursor
  - `board`: hexes/numbers/ports plus current occupancy (fingerprint source)
  - `activity [--last k]`: recent public events at cursor
  - `find --color --piece --type`: search events (e.g. first RED settlement, monopoly plays)
  - `legal`: indexed legal actions at cursor
- Output: compact JSON per command for tool use.
- Two enforced access modes (tool-enforced, not prompt-enforced):
  1. `align` mode: full-timeline search allowed; outputs may only be used to build the video-to-replay time map and anchor list.
  2. `decide` mode: hard cursor cap (no future queries), hidden-info redaction by perspective; required when generating reasoning traces so rationales cannot leak future or hidden state.
- Non-mutation invariants: wraps the sequential executor used by the divergence audit; never reuses later-cursor memory; generation never advances or mutates the replay.
- Uses: (1) video-replay pair verification (agent steps replay, matches video frames/log lines); (2) transcript-to-event alignment at scale; (3) teacher distillation with on-demand queries instead of bloated packets; (4) eventually the student's own inference-time "check the board" interface.

### Harness plan (core CLI + pi adapters)
- Layer 0 — Python CLI core (harness-independent): `catan-replay` wraps the authoritative replay executor. Mode enforcement lives HERE (`--mode align|decide`, decide mode carries a hard cursor cap and perspective redaction), so every harness inherits identical guarantees and no prompt or agent config is trusted for leakage safety.
- Layer 1 — pi project extension (interactive/dev): `.pi/extensions/` registers thin tools (`replay_info`, `replay_goto`, `replay_state`, `replay_board`, `replay_find`, `replay_legal`) that shell out to the CLI; project agents in `.pi/agents/` (e.g. `replay-aligner`, `replay-verifier`) get restricted tool lists. Used via pi's subagent suite for ad-hoc runs from a live session.
- Layer 2 — pi SDK batch harness (pipeline): a small TypeScript runner using `createAgentSession` with `tools: []` plus `customTools` per role, in-memory sessions, per-role system prompts, and scripted loops over decision lists. Supports images in `prompt()` for VL follow-along agents (frames + transcript + replay tools). Custom models (Qwen3-VL, teachers) resolve through pi's model runtime/models.json.
- Layer 2-alt — pi RPC mode from Python: `pi --mode rpc` driven by the existing Python pipeline when we want orchestration to stay in Python without rebuilding an agent loop.
- VL follow-along agent shape: input = frame contact sheets + timestamped transcript window; tools = align-mode replay CLI; output = video-time-to-event-cursor map with per-anchor confidence; never asked to name coordinates from pixels when the replay can answer.
- Build order: CLI core first (also unblocks pair verification), then the project extension, then the SDK batch runner once alignment prompts stabilize.

### Model menu by role (live OpenRouter catalog, checked 2026-08-11)
- Role A — lossy video/percept pass (recall over precision; hypotheses only): tested head-to-head on the pilot video (`model_comparison/summary.md`): `google/gemini-3.6-flash` with `reasoning effort minimal` is the new default (11 grounded decisions, $0.217, 36s, finish stop — vs qwen3.8-max's 14 decisions, $0.405, 314s, truncated). `qwen/qwen3.7-flash` FAILED: silently dropped the base64 video (6,383 prompt tokens) and confabulated from captions with finish error — always verify video ingestion via prompt token count. Both working models followed the "6-5-12" ASR trap over pixels and both over-claim `visual_confirmation: confirmed`, so Role A coordinates stay hypothesis-tier regardless of vendor. Untested cheaper/dual tiers: `gemini-3.1-flash-lite`, `kimi-k3`. Resolution test: Gemini tokenizes 480p and 240p identically (110,630 tok) and the 480p run drifted timestamps past video end, so 240p stays the whole-video default and every Role A record must pass a `0 <= t <= duration` gate; high resolution is reserved for microclips/frames.
- Role B — reasoning traces over authoritative replay state via decide-mode tools (no raw video): designated teachers `anthropic/claude-fable-5` ($10/$50) and `openai/gpt-5.6-sol` ($5/$30); mid tier `gpt-5.6-terra` ($1/$6), `gemini-3.1-pro` ($2/$12); volume tier `deepseek/deepseek-v4-pro` ($0.63/$1.26, 1M ctx), `z-ai/glm-5.2` ($0.40/$1.27), `moonshotai/kimi-k2.6`.
- Dual-capability (video + reasoning + tools in one model, for the align-mode follow-along agent): `moonshotai/kimi-k3`, `gemini-3.x` flash/pro, `qwen3.8-max`, `minimax/minimax-m3` (1M ctx, $0.30/M).
- Principles: generator and verifier come from different model families to decorrelate errors; Role A output is always hypothesis-tier regardless of model; frontier spend concentrates on Role B traces and final QA, not on the lossy pass; re-check the live catalog before each batch run since pricing/families move.

## Non-negotiable gates
- Build every replay sample from the state immediately before the action and prove the recorded action is in that exact legal-action set.
- Never expose opponent hidden hands/dev cards or future replay events to a decision, generator, critic, or student target.
- Keep raw quote spans and frame timestamps beside every cleaned rationale; do not overwrite evidence with an LLM paraphrase.
- Treat expert actions as demonstrations, not automatically optimal actions. Permit `unclear` and `questionable` outcomes and discard weak reverse-rationalizations.
- Split by game/video lineage before generation; preserve existing CatanBoardBench exclusions and deduplicate by game ID plus board/action fingerprint.
- Use Elo for stratification, sampling, weighting experiments, and evaluation slices—not as a direct action reward.
- Validate board contracts and action deltas deterministically after VLM extraction; fail closed on unresolved visual ambiguity.
- Actor-attribution gate: verify the narrator's seat from verbal commits matched to replay actions before generating traces; narrator deliberation supervises only the narrator's own actions, and speech about other seats is labeled observer commentary, never actor rationale.

## Pilot plan
- [x] Inventory the existing transcript, replay corpus/index, renderers, decision packets, split tooling, and SFT conventions.
- [ ] Agree on the first decision scope, student input modality, teacher/provider budget, and whether local video download is allowed.
- [ ] Add versioned Pydantic schemas and JSONL/source-manifest helpers for the seven artifact layers.
- [ ] Implement resumable YouTube acquisition with raw captions/audio/video metadata, hashes, and word-timestamp transcript preservation.
- [ ] Implement candidate decision segmentation from transcript cues plus visual change points, initially targeting setup placements only.
- [ ] Ground pre/action/post frames into a public board contract and action delta; optionally match a video game to a replay by board fingerprint and action-sequence alignment.
- [ ] Export authoritative replay decision packets from the existing replay executor rather than the legacy generator.
- [ ] Generate concise contrastive rationale candidates using genuine video seeds, then run an independent grounded critic and deterministic legality/privacy checks.
- [ ] Create same-state alternatives and preference labels only where explicit commentary, rollout evidence, or high-margin critic agreement supports them.
- [ ] Run a small pilot (recommended: the three existing videos plus 20 replay games), manually review a stratified sample, and report keep/reject rates by failure reason before scaling.
- [ ] Train/evaluate action-only versus rationale-augmented baselines before deciding whether synthetic reasoning earns its cost.

## Candidate paired source
- [x] Replay `242781000` captured on first attempt (737 events) and archived to `artifacts/staging/colonist/replays/242781000.json`; sha256 `b07dfe7a…5485eae2e4`; manifest at `data_pipeline/training/reasoning/pilots/_2n5F2DxtPI/pairing_manifest.json`.
- Exact username is `bootymunchr` color 5 (the URL `q=bootymun` was a search prefix); 4 humans, base mode, 10 VP, 85 turns, ~40 min, started 2026-07-13.
- NARRATOR CORRECTION (user-caught): the narrator is `FunDipDevRip` color 2, not bootymunchr — proven by the "I think it's 8 4 10" commit [58.8] matching his wall-94 settlement. The archived payload is the color-5 perspective; narrator-hand supervision needs a playerColor=2 refetch (post rate-limit) or public-only packets. First demo trace invalidated for actor misattribution; corrected demo2 trace passes quote, cutoff, and actor gates.
- Open question: endgame public VP reads 8/7/7/7 with nobody at 10 — verify true winner/end condition against the video.
- [ ] Verify YouTube `_2n5F2DxtPI` against the archived replay: board layout, usernames/colors, opening placements, at least three ordered public-event anchors, winner, and turn count. Keep the replay in staging (not the verified corpus) until this passes.

## Creator-account pairing discovery
- [x] Record the user-discovered link between a YouTube creator and Colonist identity `bootymun`; keep the identity mapping provisional until the first replay/video pair is independently verified.
- Use creator identity plus video publication windows to search only that account's still-available games, archive replay payloads immediately, and then match by board fingerprint and ordered public events.
- This is primarily a prospective acquisition strategy: historical IDs remain metadata-only if Colonist no longer serves their replay payloads.
- Store channel ID, Colonist user ID/username history, mapping evidence, verification status, replay capture time, and raw payload hash in the source manifest.

## Pilot acceptance criteria
- At least 95% of retained replay records reproduce an exact legal pre-action choice; the target is 100%, with any exception quarantined.
- Zero detected hidden-information or future-event leakage in deterministic audits and manual review.
- Every retained video rationale has timestamped transcript evidence and a pre-action frame; every inferred state/action field carries confidence and provenance.
- Human review marks at least 80% of retained rationale records as both board-specific and faithful; otherwise improve extraction rather than scaling generation.
- Dataset manifests report source lineage, schema/prompt/model versions, costs, rejection reasons, and split membership so runs are reproducible.

## Review
Pending user decisions and pilot implementation.

---

# Agent experience harness design

## Goal
Define a human-like, cost-efficient Catan agent experience in which the environment continuously records perspective-safe events but invokes the model only when that agent has a genuine decision.

## Plan
- [x] Inspect the existing replay decision packet, live observation cursor, LLM event queue, memory design, and exact replay trade ledger.
- [x] Separate event delivery from expensive inference cadence.
- [x] Define per-agent unread-event cursors, authoritative state snapshots, fallible strategic memory, decision triggers, and tool-call receipts.
- [x] Show how incoming offers become directed decision packets without deciding trade timeouts or negotiation-round policy.
- [x] Align live packets with replay-derived SFT examples and identify privacy/retry invariants.

## Review
- Recommended an event-sourced, decision-triggered harness: every visible event is retained in order, while belief/plan integration occurs lazily in the same model call that selects the next genuine action.
- Each packet combines the unread perspective-filtered delta with an authoritative current snapshot, externalized prior memory, and a scoped legal-action/tool catalog.
- Decision cursors advance only after a durable accepted response, making retries idempotent and replay annotation causally reproducible.
- The current per-action all-agent LLM update design should become cheap event fan-out; the current turn-window replay context and clear-on-read queues should become stable per-agent cursor ranges.
- Trade timing and round limits remain explicitly out of scope; trade offers are represented as stable-ID events that trigger responder decisions whenever the chosen game policy says a response is available.

---

# Replay payout sub-lines

## Goal
Keep each roll as one logical activity row while placing every resource payout on its own indented line in both the LLM packet and replay UI.

## Plan
- [x] Confirm the existing activity string can carry nested lines without changing source-row accounting.
- [x] Render complete, partial, and empty payouts as indented sub-lines.
- [x] Preserve multiline whitespace in the replay response UI.
- [x] Update formatter and prompt assertions, then run Python and frontend verification.

## Review
- Each roll remains one `recent_activity` entry, so replay source-row counts are unchanged.
- The LLM packet and UI now show one indented line per recipient, with explicit no-payout and partial-data lines.
- The replay card preserves embedded newlines through a dedicated `replay-activity-row` style.
- Verification: 53 Python tests passed with 1 skipped; targeted ESLint and the production frontend build passed; full frontend lint still has pre-existing failures in untouched files tracked separately.

---

# Replay roll payouts in LLM context

## Goal
Expose each dice roll's public per-player resource payouts in replay LLM recent activity without leaking any player's full hidden hand.

## Plan
- [x] Trace replay event parsing, resource snapshots, activity redaction, and prompt construction.
- [x] Compare established before/after resource-delta and redaction patterns.
- [x] Derive positive, event-scoped roll payouts while parsing Colonist state changes; fail closed when no trustworthy pre-roll baseline exists.
- [x] Format those payouts in replay recent activity and bump the decision-context schema version.
- [x] Add parser, privacy, no-production, and end-to-end prompt tests.
- [x] Run focused replay tests, review the diff for hidden-information leakage, and document results.

## Implementation specification
1. Compute payouts from `resources_before_event` versus the cumulative post-event resource snapshot, not from a previous parsed action or the later engine board.
2. Emit only positive roll-scoped deltas in canonical `WOOD, BRICK, SHEEP, WHEAT, ORE` order. Never serialize full `expected_resources` hands.
3. Omit untrustworthy player deltas when the pre-event hand is unknown; suppress the entire payout summary if the roll event contains a negative delta or a seven contains any hand change. Only call the list complete when tracked hands match the replay's authoritative player roster.
4. Preserve an explicit empty payout map for trustworthy no-production rolls so the activity can say that no resources were paid out.
5. Include public payouts for every player regardless of the current observer; continue redacting development cards, discards, and third-party steals.

## Review
- Roll actions now carry public per-player resource deltas derived from Colonist's event-scoped hand changes, while full `expected_resources` snapshots remain private.
- Payout completeness is checked against authoritative `playOrder`; missing baselines are labeled partial, and suspicious negative changes or changes on seven fail closed.
- Replay activity now keeps each roll as one logical row with indented payout lines such as `MYSTIC_BLUE: +1 SHEEP`, `BLUE: +2 ORE`, and `GREEN: +1 SHEEP`.
- The decision packet schema is `replay-decision-v2`.
- Verification: 52 Python tests passed with 1 skipped; Ruff and whitespace checks passed; all 1,067 rolls across 17 roster-bearing local replays had complete payout baselines; independent review found no blocking, high, or medium issues.

---

# Replay LLM responses

## Goal
Add a replay setting that can request an LLM response for a user-selected replay position without accidentally mutating the replay.

## Plan
- [x] Inspect replay navigation, state reconstruction, LLM clients, UI patterns, and existing tests.
- [x] Confirm response behavior, trigger, player perspective, and model configuration with the user.
- [x] Specify the API/result shape and non-mutation guarantees.
- [ ] Implement the backend replay-inference path with mocked tests.
- [ ] Implement the replay controls/settings and a separate response display.
- [ ] Verify focused tests, full Python tests, frontend lint/build, and replay behavior.

## Approved behavior
- The user navigates to any replay position and clicks **Generate response**.
- The model returns updated goals, a legal move, and reasoning but never executes it.
- The packet uses the engine current player's private-information perspective.
- Context is compact: prior completed turn + current partial turn, authoritative current observation, prior safe goals when available, and indexed legal actions. It does not send full replay history.
- Goals may carry forward in the UI session but are cleared on backward jumps, replay changes, or model changes to prevent future leakage.
- The replay setting accepts a custom OpenRouter model ID and persists it in the browser.

## Implementation specification
1. Add a versioned replay decision-context service that snapshots `Game`, redacts hidden replay activity, selects the prior completed turn plus current partial turn, and formats the current player's observation and legal actions.
2. Call the custom OpenRouter model and parse `<goals>`, `<reasoning>`, and an indexed `<action>`, preserving raw output and a parse warning if necessary.
3. Add `POST /api/replay-llm-response` with replay/model/goals validation, a non-blocking request lock, provider error handling, response metadata, and a stale-cursor indicator.
4. Return the action, description, goals, reasoning, observation, recent activity, legal actions, raw response, context version, model, usage, latency, game ID, replay index, and player color.
5. Add a persistent model-ID setting, explicit generate button, loading/error states, safe forward-only goal carryover, and a separate expandable replay-response card.
6. Mock all provider calls in tests and prove that generation does not change the replay cursor, game actions, resources, or history; test turn-window selection and hidden-information redaction.

## Review
Pending implementation and verification.

---

# Replay and trading integrity

## Goal
Prove local Colonist replays remain deterministic across replay-only force paths and make the trade lifecycle—not only resource totals—match the source replay.

## Baseline audit
- [x] Run one sequential divergence pass over all 18 local raw replay files.
- [x] Confirm all 18 complete with zero fatal semantic errors, zero hand-resource divergence, no negative final hands, and 19 cards per resource conserved globally.
- [x] Trace replay-only `force=True` call sites and domestic/maritime trade execution.
- [x] Identify gaps hidden by the resource-only success signal:
  - `replay_goto_fast_logic` skips replay force handlers and diverges after trades; the frontend currently uses sequential navigation, but the endpoint is unsafe.
  - Exact `CONFIRM_TRADE` and forced overlay mutations bypass `Game.history`, so replay undo does not restore trade state/resources.
  - Colonist trade cancellations are dropped by the parser, and simultaneous offers from the same creator are collapsed by color-keyed engine dictionaries.
  - Canonical engine confirm/cancel helpers can clear unrelated concurrent trade state.

## Options
1. **Replay-only exact ledger (recommended):** keep normal environment action contracts stable; track Colonist offers by `trade_id`, parse closures, project only compatibility state into the engine, and make every replay mutation transactional/undoable.
2. **Minimal patch:** make confirmation undoable, route fast navigation through sequential execution, and parse cancellations while retaining color-keyed overlays. Lower impact, but cannot represent multiple offers from one creator exactly.
3. **Engine-wide trade-ID refactor:** change environment trade action/state contracts to identify offers explicitly. Most complete, but unnecessarily invasive for a replay verification fix.

## Approved implementation sequence
The user approved fixing the audit findings one at a time, with focused verification after each issue and no repeated corpus scans between code states.

- [x] Make replay steps transactional so exact `CONFIRM_TRADE` mutations and metadata are fully undoable; add a focused undo/re-step regression test.
- [x] Parse and execute Colonist offer closures without resource mutation or heuristic progression.
- [x] Preserve simultaneous same-creator offers by Colonist `trade_id`, including counter-parent links and responses.
- [x] Make every public navigation path use the authoritative replay executor (or reject non-authoritative jumps).
- [x] Fix canonical confirm/cancel cleanup so one trade cannot erase unrelated trades, with focused engine tests.
- [x] Run the consolidated focused suite and full Python tests, then the final accepted 18-game corpus pass and document results in `docs/DIVERGENCE_PROGRESS.md`.

## Review
- Focused replay/trading regressions cover transactional confirm undo/re-step, standalone and transaction closures, same-creator concurrent offers, counter-parent links, response accept/reject/clear transitions, authoritative navigation, stale-confirm affordability, selective engine cleanup, counter-only broadcast state, parser input immutability, and mixed trade-log ordering.
- Independent review found and then cleared all blockers before the accepted corpus run.
- Static verification: targeted Ruff passes and `git diff --check` passes.
- Python verification: 47 tests pass with the gated corpus audit skipped.
- Corpus verification: 18/18 games, 8,910 actions, and 4,755 trade lifecycle actions pass per-action resource equality, nonnegative-hand, 19-card conservation, exact active-trade-ID/response parity, and zero error-level semantic issue checks.

# Expand the local replay corpus

## Goal
Acquire a larger, deduplicated set of account-accessible Colonist base-game replays and run the consolidated verifier without bypassing access controls or repeatedly rescanning unchanged games.

## Plan
- [x] Inventory local replay files, candidate indexes, scraper paths, authentication hooks, and prior access failures without making network requests.
- [x] Compare acquisition options and choose a bounded first batch that maximizes player/color/game-length diversity.
- [x] Refresh an authenticated browser session, then download the approved batch sequentially with success and attempt caps.
- [x] Validate every downloaded payload before promoting it into `data/raw_replays`; quarantine malformed, unsupported-mode, or duplicate files.
- [x] Run the consolidated audit once on the enlarged corpus and document pass/failure coverage separately from the original 18-game baseline.

## Findings
- The current 18 raw files are unique; the five smoke files are duplicates of five of those games.
- Existing indexes provide 7,309 unused top-player 4-player candidates, plus 83 recent Tournament candidates with at least 20 turns, spanning 15 players and all five standard Colonist colors.
- The preferred persistent-browser Playwright scraper and its profile exist locally. The `.env` JWT expired on 2026-06-12, so direct API access is currently unavailable and the browser profile may require interactive reauthentication.
- The initial recommendation was a deterministic 25-game recent queue, but the user selected a broader 100-game sample from the older indexed four-player corpus and authentication through attached Chrome.

## Approved acquisition
- Build a deterministic 300-candidate queue from `4p_games_top100.json`, excluding the current 18 and games under 20 turns, balanced across 99 indexed players and turn-length quantiles. The initial target was 100 games; the user accepted the 49 four-player downloads already captured when acquisition stopped.
- Attach sequential Playwright capture to the user's authenticated Chrome over CDP. Do not terminate or relaunch their browser without confirmation.
- Acquire in small paced chunks with at least 40 seconds between games. Stop immediately on `429`/`Retry-After` and require a cooldown before resuming; do not retry rate-limited requests automatically.
- Stage downloads outside the verified corpus, validate payload/schema/player count/mode first, then promote only compatible unique games and run one enlarged-corpus audit. The gates proved the source index is mislabeled by exposing a two-player replay and a Cities & Knights replay; both payloads were quarantined.
- Final acquisition checkpoint for this expansion: 50 payloads were captured, 48 compatible four-player base games were promoted, and two incompatible games were quarantined. The first continuous run stopped at its first `429`; no retry was made. The user confirmed this is enough data, so acquisition is closed.

## Review
- Source coverage: 49 distinct indexed top-100 players, ratings 1834–1989 (median 1869, mean 1880); ratings describe the selected player at index time, not historical whole-lobby Elo.
- Scraper safety: hard stop on the first 429, `Retry-After` reporting, no automatic rate-limit retry, 40-second default pacing, and explicit player-count/mode gates.
- Compatibility: one two-player replay and one Cities & Knights replay were preserved outside the supported corpus; no malformed or duplicate promoted payloads.
- Verification: 66/66 base-game replays, 31,506 actions, and 15,460 trade lifecycle actions pass resource equality, nonnegative-hand, 19-card conservation, exact trade-ledger parity, and zero error-level semantic issue checks.

---

# Benchmark Qwen3-VL-32B on CatanBoardBench

## Goal
Measure the disclosed 32B dense vision-language model on the existing engine-scored Catan board-image benchmark.

## Plan
- [x] Locate the current benchmark, frozen dataset, prior runs, and canonical scoring configuration.
- [x] Run `qwen/qwen3-vl-32b-instruct` on the current visual suite at temperature 0 without modifying benchmark prompts or labels.
- [x] Verify response completeness and report exact/component accuracy, category failures, usage, cost, latency, and comparison with the earlier smoke run.

## Review
- The 110-question visual run completed with zero API errors: 17/110 exact (15.45%) and 14.68% component accuracy.
- The model defaulted to `EMPTY`, `NONE`, `NO`, or `GENERIC 3:1` across several categories rather than reliably binding visible board features to atlas tokens.
- Five road-location prompts produced 9,804-token runaway edge enumerations despite a requested 256-token maximum; those failures account for 49,020/49,528 completion tokens and $0.02068/$0.02726 total cost.
- Artifact integrity passed (110 unique questions, 10/category, no empty responses), and the focused CatanBoardBench/token suite passed 7 tests.
- Full report: `reports/catan_board_bench/2026-08-10-qwen3-vl-32b-visual.md`.

---

# Benchmark Claude Fable 5 on CatanBoardBench

## Goal
Measure a frontier-scale proprietary VLM on the same engine-scored board-image suite as the Qwen3-VL-32B run.

## Plan
- [x] Smoke-test exact model `anthropic/claude-fable-5` on one unchanged visual question and confirm image support, scoreability, usage, and cost.
- [x] Run the same 110 questions, 10 boards, 11 categories, atlas prompt, temperature 0, and 256-token requested completion limit.
- [x] Verify artifacts and report exact/component accuracy, category behavior, cost, latency, and comparison with Qwen3-VL-32B.

## Review
- The matched 110-question run completed without API errors at 33/110 exact (30.0%) and 32.11% component accuracy, costing $1.98799 with 7.83 s median request latency.
- Fable scored 10/10 on targeted tile resource/number reading, 7/10 on edge ownership, and 5/10 on node occupancy, but remained weak on global atlas translation, ports, counts, and robber localization.
- Mandatory high-effort reasoning consumed the entire 256-token budget on 61/110 requests; many returned unfinished reasoning or serialized reasoning signatures rather than final answers, so 30% is a matched-budget operational score rather than a clean capability ceiling.
- A one-board 1,024-token diagnostic reached 6/11 exact and 61.11% component, but four spatial questions still hit the cap; a low-effort 512-token road-location probe also ended without final content.
- Artifact integrity passed, and the focused CatanBoardBench/token suite passed 7 tests.
- Full report: `reports/catan_board_bench/2026-08-10-claude-fable-5-visual.md`.

---

# Compare piece perception and text representations

## Goal
Separate Catan failures into isolated visual recognition, dense-board localization, and symbolic text-representation reasoning for Qwen3.8-27B.

## Plan
- [x] Run the existing isolated/local-patch visual suite on Qwen3.8 with macro and per-category scores.
- [x] Normalize each authoritative board contract into one target-neutral public fact set.
- [x] Render identical facts as verbose JSON, compact JSON, graph DSL, and spatial ASCII.
- [x] Run the same 110 questions text-only across all four formats with identical scoring and no image.
- [x] Compare accuracy, prompt tokens, latency, cost, defaults, and failure patterns against full-board vision.

## Invariants
- All text formats derive from the same fact object and contain no question-specific hints.
- Output/scoring stays constant across formats; only the board-state representation changes.
- Visual recognition is reported separately for isolated assets and cluttered local crops.
- Provider/routing and reasoning controls are persisted with the artifacts.

## Review
- Qwen3.8 piece recognition: 74/199 canonical exact (37.19%), 60.00% component. Defensible resource-synonym diagnostic: 111/199 (55.78%); a separate non-canonical BLUE/MYSTIC_BLUE collapse reaches 115/199 and is explicitly labeled diagnostic.
- Primitive strengths: robber presence 22/24 isolated and 2/2 local. Primitive weakness: node occupancy 0/22 isolated and 0/9 local; color is often omitted and cities become settlements.
- Built four lossless projections from one sparse, target-neutral fact schema: verbose JSON, compact JSON, graph DSL, and spatial ASCII. All round-trip to identical digests over all ten contracts.
- Same 110 IDs text-only, pinned AkashML: verbose JSON 84/110, compact JSON 87/110, graph DSL 87/110, spatial ASCII 93/110, versus prior image 24/110. All 440 requests had zero errors/empty outputs/reasoning tokens.
- ASCII’s lead is concentrated in node occupancy (7/10 versus graph 2/10 and JSON 0/10), not universal. All formats still failed settlement/city count composition.
- Verbose JSON used 383,381 prompt tokens; ASCII used 144,219, gained nine exact answers, and cost $0.059382 versus $0.139363.
- Caveats recorded: engine-component exact differs from literal string equality; text/image providers were not matched; ten snapshots come from two games (8+2); input formatting—not structured output—was varied.
- Piece QA/raw/strict/semantic artifacts were relocated from ignored `sft/data` into tracked `artifacts/runs/catan_board_bench/piece_recognition/` with regeneration instructions.
- Verification: tracked hashes/counts, information-equivalent round trips, paired ID match, 10 tests, Ruff, whitespace checks, and independent review passed.
- Report: `reports/catan_board_bench/2026-08-16-qwen3.8-piece-recognition-and-text-formats.md`.

---

# Evaluate hex-direction grounding

## Goal
Measure whether Qwen3.8 can visually resolve relative positions on a pointy-top hex grid without relying on the textual atlas.

## Plan
- [x] Define the six unambiguous directions: left, right, up-left, up-right, down-left, and down-right.
- [x] Generate visibly labeled anchor/candidate images with engine-coordinate ground truth and balanced direction labels.
- [x] Add orientation/invariance checks so fixed label placement or answer priors cannot pass the probe.
- [x] Run a smoke test, then the full Qwen3.8-27B probe with reasoning disabled.
- [x] Verify artifacts and report per-direction accuracy and confusion patterns.

## Invariants
- The answer must come from the image; no atlas text may encode the relation.
- Candidate labels and board locations must be permuted across examples.
- Every direction must have equal representation.
- Hex geometry—not pixel distance heuristics—defines the answer key.

## Review
- Added a 72-question, six-layout isolated probe with engine-derived cube deltas and frontend-compatible pointy-top projection.
- Both inverse tasks are balanced: 36 direction→label and 36 label→direction; every label occupies every direction once; atlas/context contracts are absent.
- Qwen3.8-27B scored 66/72 (91.67%): 30/36 direction→label and 36/36 label→direction, versus 16.67% chance.
- All six errors collapsed a diagonal onto the same-side horizontal neighbor: three `UP-RIGHT→RIGHT`, two `UP-LEFT→LEFT`, and one `DOWN-LEFT→LEFT`.
- Full run: zero errors, 29,280 prompt + 192 completion tokens, $0.011608, 1.44 s median latency, and zero reasoning tokens.
- Provider identity was not persisted in the original run; pricing fingerprints suggest mixed routing and are explicitly labeled inferred. Exact invocation/settings are preserved in `run_metadata.json`, and future runner artifacts now record settings/provider fields.
- Verification: 72 unique balanced rows, 66 exact answers, artifact math/cost checks, 9 tests, Ruff, whitespace checks, and independent review passed.
- Report: `reports/catan_board_bench/2026-08-16-qwen3.8-27b-hex-directions.md`.

---

# Evaluate Qwen 3.8

## Goal
Run the established CatanBoardBench evaluation against the requested Qwen 3.8 model through OpenRouter, preserving comparable settings and verified artifacts.

## Plan
- [x] Resolve the exact OpenRouter model ID, modality, context/output limits, and pricing.
- [x] Confirm the established evaluation command and comparable prior-run settings.
- [x] Run a bounded smoke test before committing to the full paid evaluation.
- [x] Run and score the complete compatible benchmark split.
- [x] Verify artifacts, summarize accuracy/failure modes/cost, and record the report.

## Review
- Interpreted “Qwen 3.8” as the open-weight multimodal `qwen/qwen3.8-27b`, the closest comparison to the prior 32B visual run.
- Added Qwen3.7/3.8 to the runner’s bounded no-reasoning control; the live run reported zero reasoning tokens.
- An 11-question smoke completed without errors before the 110-question full run.
- Full result: 24/110 exact (21.82%), 20.18% component accuracy, zero errors, 65,357 prompt + 578 completion tokens, and $0.027169.
- OpenRouter usage-price fingerprints indicate mixed routing across Chutes FP8 and AkashML BF16, so the artifact represents unpinned OpenRouter service rather than one quantization.
- Verified 110 unique IDs, 10 rows per category, 24 exact answers, 44/218 components, cost/token totals, nonempty outputs, and zero reasoning tokens. CatanBoardBench/token tests passed 7/7; Ruff and whitespace checks passed.
- Report: `reports/catan_board_bench/2026-08-16-qwen3.8-27b-visual.md`.

---

# Render semantic commentary traces

## Goal
Replace the visible timestamp-per-caption transcript list with compact semantic trace cards while retaining time only as secondary provenance.

## Plan
- [x] Derive bounded semantic display traces from causally available commentary.
- [x] Let coherent traces span replay rows and mark selected bridge evidence as overlapping.
- [x] Render semantic role cards rather than timestamp-segmented rows.
- [x] Preserve finite-clock and strict future-commentary containment.
- [x] Verify API output, frontend build, tests, and the live backend response.

## Review
- The panel now renders compact `Table context`, `Board assessment`, `Options and tradeoffs`, `Commitment and plan`, `Opponent read`, `Trade discussion`, and `Plan update` cards.
- Timestamp ranges appear only in the card footer as evidence provenance; individual caption rows are no longer rendered.
- Recent semantic context can persist across a short empty action interval, but stale context is removed after long empty intervals; completion follows the same bounded policy.
- Nonfinite replay clocks fail closed rather than selecting future commentary.
- Verification: live backend returned 5 semantic traces versus 27 raw utterances at cursor 0; 72 tests passed with 1 skipped; Ruff, frontend build, targeted ESLint, whitespace checks, and independent review passed.

---

# Semantic reasoning episodes

## Goal
Group causally available commentary into compact, coherent reasoning episodes rather than treating replay timestamps or engine rows as trace borders. Episodes may span several events and share a small amount of transition evidence.

## Plan
- [x] Add stable commentary evidence references and an episode workspace above `CausalCommentarySession`.
- [x] Support open/extend/close operations, multi-episode evidence membership, and post-reveal event links without imposing an LLM serialization format.
- [x] Keep causal cursor ranges as provenance only; derive episode borders from explicit semantic assignments.
- [x] Validate overlapping opening comparison/commitment/plan episodes and an actor-mismatched opponent-assessment episode.

## Invariants
- An episode can use only commentary already present in a blind context and events already revealed.
- A commentary span may belong to more than one episode; duplicate membership within one episode is rejected.
- Event reveal can confirm, contradict, or contextualize an episode without automatically closing it.
- Raw evidence, grounding, semantic role, and event links remain separate.

## Review
- Added `EpisodeWorkspace` above the causal stepper; semantic roles and explicit evidence assignment define borders, while replay indices remain provenance only.
- Commentary evidence can intentionally belong to multiple episodes, and an episode can remain open across multiple engine reveals.
- Revealed events attach as `confirms`, `contradicts`, `contextualizes`, `motivates`, or `follows` without forcing closure.
- Opening tests form overlapping comparison and port/expansion-plan episodes around the shared `8-4-10` commitment; a longer wood-expansion thread spans events 0–2; `6-9-3` remains an actor-mismatched observer assessment.
- Workspace operations are atomic, game-scoped, and expose deep-copied immutable snapshots so provenance cannot be bypassed by caller mutation.
- Verification: 69 tests passed with 1 skipped; Ruff and whitespace checks passed; independent review approved.

---

# Engine-grounded commentary contextualization

## Goal
Build a format-neutral, causal bridge between human Catan commentary and authoritative replay state so an agent can interpret shorthand against the current board, advance exactly one event, and confirm or reject its provisional interpretation without reverse-rationalizing from future actions.

## Plan
- [x] Mine existing spatial resolver, replay stepper, transcript timing, and actor-attribution patterns before adding new code.
- [x] Extract human number shorthand and resolve it deterministically against current engine topology, returning candidate sets rather than guessing.
- [x] Represent node, port, occupancy, legality, and directional qualifiers as query results independent of the eventual reasoning-trace serialization.
- [x] Build a blind-then-reveal stepper: commentary plus pre-event board tools first, opaque provisional annotation second, one engine event reveal third.
- [x] Validate against the paired replay opening (`8-4-10` → narrator placement), opponent commentary (`6-9-3 smart`), ambiguous/nonexistent references, and strict temporal cutoffs.
- [x] Leave prompt/output formatting behind an adapter boundary for the parallel formatting work.

## Approved invariants
- Transcript is evidence, the engine is the referent oracle, and the contextualizer links them.
- The provisional pass cannot inspect the upcoming parsed action or future commentary.
- Raw quote, normalization, deterministic grounding, model interpretation, and post-action confirmation remain distinguishable.
- Actor mismatch never turns observer commentary into another player’s rationale.
- Ambiguous or nonexistent spatial references remain candidate sets or unresolved.

## Review
- Added deterministic grounding for separated, compact, and common ASR number references; known pilot bindings reproduce `8-4-10` → corner 37/node 7, `6-9-3` → corner 26/node 10, ambiguous `8-4-3`, and nonexistent `8-5-10` without guessing.
- Added a format-neutral `CausalCommentarySession`: blind guarded commentary + public board grounding, opaque commit, then exactly one strict replay-row reveal with public event summary and independent actor/reference confirmation.
- The evidence adapter receives only pre-filtered commentary, and its returned timestamps/provenance are independently validated; it cannot access replay actions or extend the causal cutoff.
- Removed upcoming-action metadata from transcript UI/API state and added monotonic replay revisions plus a shared mutation lock to defeat step/inspect/undo and concurrent route races.
- Narrator legality is exposed only when perspective-safe; opponent commentary keeps topology/occupancy but withholds private affordability. Actor mismatches remain observer evidence.
- Verification: 65 focused/replay tests passed with 1 skipped; Ruff clean; frontend production build and targeted ESLint passed; independent reviewer approved after concurrency hardening.

---

# Paired replay transcript panel

## Goal
Bake the verified-in-progress replay/transcript pair `242781000` / `_2n5F2DxtPI` into the playground as the default example while preserving arbitrary replay loading, and synchronize transcript narration to replay navigation.

## Plan
- [x] Confirm the curated pair, retain the existing manual replay loader, and choose pre-action matching semantics.
- [x] Add a curated replay registry and cursor-scoped transcript payload without moving the still-staged replay into the verified corpus.
- [x] Derive each parsed replay step's wall time from its raw event index and cumulative Colonist `deltaS`; return the transcript since the prior action and before the current upcoming action.
- [ ] Deterministically reflow timestamped rolling-caption chunks into sentence-like utterances while preserving source spans and strict pre-action containment.
- [x] Add a replay-only transcript panel and make `242781000` the input default while keeping custom game IDs editable.
- [x] Verify exact boundary behavior, empty windows, backward navigation, backend tests/Ruff, and frontend build/lint.

## Approved behavior
- Cursor step `N` shows narration available after the prior parsed action and strictly before the upcoming parsed action `N`; no post-action/future transcript is displayed in that window.
- The panel is a playback aid only. Transcript text is not silently injected into replay LLM prompts.
- Narrator attribution is `FunDipDevRip` (Colonist color 2); the staged replay payload remains color-5 perspective, so the panel must not imply narrator-private-state supervision.

## Review
- Replay `242781000` is the editable loader default; arbitrary local replay IDs remain supported.
- Cursor-scoped transcript windows follow replay step, undo, and goto, with whole-caption boundary checks and explicit empty/anomalous states.
- Upcoming action metadata was removed from transcript API/UI state so the display cannot leak the event contextualization is meant to predict.
- Transcript display remains separate from Generate Response context.
- Verification is included in the contextualization review above.

---

# Research Catan board representations

## Goal
Determine an empirically defensible representation strategy for Catan agents rather than assuming images, JSON, or prose are universally best.

## Plan
- [x] Define one perspective-safe information contract and criteria for topology fidelity, exactness, token/visual-token cost, model compatibility, training, validation, and invariance.
- [x] Research primary work on learned board states, symbolic/game encodings, graph serialization, entity-centric RL, VLM spatial grounding, and structured-data token efficiency.
- [x] Encode an equivalent representative Catan state as verbose JSON, compact JSON, graph/atlas DSL, natural-language delta, current observation, and legal-action variants; measure local Qwen tokenizer costs.
- [x] Specify a controlled representation bake-off that separates state decoding from planning and free action construction from menu selection.
- [x] Write a sourced memo with recommendations for black-box teachers, an open-weight student, and a versioned deployment/training contract.
- [x] Independently review the memo against repository code and frozen benchmark reports; correct legal-menu, robber, generic-port-token, reproducibility, and VLM-result qualifications.

## Review
- Recommended a canonical three-lifetime contract: immutable `BoardAtlas/v1`, per-game `BoardSetup/v1`, and observer-relative `PerspectiveObservation/v1`, all derived from the engine.
- Port geometry is fixed but port type is variable: model a port slot by canonical ID, coastal edge, and endpoint corners, then bind its 2:1 resource or generic 3:1 type in the setup.
- Recommended compact sectioned Catan DSL plus complete indexed legal candidates for black-box teachers; use the same format with trained atomic atlas tokens for the open student; derive graph/tensor adapters from the same contract.
- Recommended a heterogeneous tile/corner/road-slot/port/player/global graph plus legal-candidate scorer as the strongest learned-policy structural hypothesis, while retaining brick-tensor and compact-text baselines.
- Measured one 40-action setup state: verbose CatanBoardBench JSON was 23,547 Qwen tokens, atlas-known DSL 446, current observation 419, current legal menu 2,205, and the current full decision prompts 3,145. These are exploratory one-state measurements and need a pinned script/manifest before publication.
- Found that fixed alternating port slots reduce exact base-atlas geometric augmentation from full hex `D6` to six `D3` transforms; this and port/vertex invariants need golden tests before implementation.
- Identified pre-schema hazards: incompatible hard-coded node namespaces, silent unresolved-port-to-generic risk, unversioned atlas IDs, overloaded `None`, formatter action-summary truncation, and base-edge fallback.
- Full research memo: `references/catan_board_representation_research.md`.

---

# Assess synthetic-trace transfer to a board encoder

## Goal
Determine when textual Catan reasoning supervision will causally improve a policy that receives exact board state through learned entity embeddings instead of serialized board text.

## Plan
- [x] Identify interface and supervision paths through which trace loss can or cannot reach the board encoder and action policy.
- [x] Review primary evidence on rationale distillation, rationale faithfulness, and frozen-LLM multimodal/structured-input alignment.
- [x] Define architecture choices and causal ablations that distinguish genuine grounded transfer from trace-style imitation or board-input neglect.

## Review
- Trace loss can train a graph encoder through cross-attention/projector gradients, and rationale-distillation work shows additional rationale supervision can improve sample efficiency and answers; multimodal work shows continuous non-text embeddings can condition a frozen or adapted LLM.
- Neither result guarantees Catan policy transfer. A model can imitate trace style, ignore the board, rationalize a provided action, or route policy logits around the textual trace; faithfulness studies demonstrate all of these broad failure classes.
- Use paired-view training: the identical state is rendered once as canonical DSL and once as graph entities, with matched trace/action targets, board-fact objectives, representation/logit alignment, modality dropout, and minimally changed counterfactual boards.
- Keep factual premises, derived calculations, strategic judgments, and actions separately supervised. Teacher-only chosen-action context may produce useful answer-conditioned rationales but is not evidence of the expert's causal thought process.
- Compare matched action-only, trace, grounded-trace, and shuffled-trace conditions. Separately intervene on board entities and traces; measure exact grounding, action regret/rollout value, generated-trace exposure gaps, and causal patch effects rather than relying on attention or probe decodability.

---

# Operationalize learned board representation

## Goal
Define falsifiable evidence that a Catan model internally represents and causally uses fixed-atlas topology, orientation, and changing board-state bindings.

## Plan
- [x] Distinguish topology, oriented display coordinates, canonical identity, and dynamic state variables.
- [x] Review Anthropic feature/circuit-tracing reports and primary work on Othello-GPT, spatial probes, structural probes, probe controls, causal abstraction, DAS, and attention faithfulness.
- [x] Specify an evidence ladder from behavioral competence through decodability and representation geometry to causal intervention and circuit claims.
- [x] Define Catan-specific metrics, controls, counterfactual interventions, symmetry tests, and safe claim language.

## Review
- The operational threshold is not a visually pleasing attention map or an objective scalar ordering of node IDs. It is a low-complexity alignment between internal states and typed board variables whose interventions reproduce symbolic counterfactual behavior.
- Separate graph topology from screen orientation. The six exact directed edge displacements define `UP`, `UP_LEFT`, `UP_RIGHT`, `DOWN_LEFT`, `DOWN_RIGHT`, and `DOWN`; orientation is conventional and otherwise identifiable only up to atlas symmetries.
- Use exhaustive relation behavior, controlled linear/structural probes, graph-distance RSA/Procrustes/equivariance, then on-manifold activation or distributed interchange interventions. Report attention only as routing evidence until value/output-path ablations validate it causally.
- Compare raw/explicit positional inputs, base and untrained checkpoints, degree and random-label controls, random equal-rank subspaces, wrong layers/entities, and non-target specificity. Distinguish architecturally supplied geometry from learned use and emergent recovery.
- Existing `scripts/probe_catan_board_mech_geometry.py` is an exploratory identity/centroid-cosine diagnostic; by itself it lacks the controls and causal evidence required for a learned-representation claim.
- Full protocol: `references/operationalizing_learned_catan_board_representation.md`.

---

# Estimate Catan vision-alignment compute

## Goal
Bound the dollar cost of reproducing the GLM-5.2 projector-alignment recipe and scale it to the likely Catan student without inventing unreported runtime.

## Plan
- [x] Verify Baseten's published data size, batch, epochs, grok step, frozen modules, projector size, and RL batch evidence.
- [x] Check whether Baseten disclosed training hardware, wall time, or dollar cost.
- [x] Price plausible original runs under explicit hardware/time assumptions and derive a Catan-scale pilot budget.
- [x] Reconcile the estimate with the user's likely Qwen3.8-VL target.

## Review
- Baseten reports 66k short-QA images, batch 64, two SFT epochs (2,070 optimizer steps), grok near step 900, a frozen 744B/40B-active GLM and frozen ~466M MoonViT, and only a ~49.5M projector trained. It does not report training GPU count, wall-clock time, or dollar cost; no exact original-dollar figure is defensible.
- The public checkpoint requires 8×B200 for deployment, but that is an inference fit statement, not proof of the training configuration. At current public rates, every assumed 8×B200 training hour costs roughly $47 on RunPod or $80 on Baseten; 4–8 hours would imply ~$188–$639, but remains a scenario, not a report.
- LLaVA's published projector-only alignment precedent used 558k images and took 3.5 hours on 8×A100 for a 7B LLM, about $42 at current low marketplace A100 rates. Our 66k-example Catan alignment is about 8.5× smaller in examples and can target a few hundred visual tokens, so a native 8–9B VLM pilot should be a one-GPU, tens-of-dollars job rather than GLM-scale hundreds.
- Qwen3.8-Max is a 2.4T/95B-active API model and is not the trainable student. The released `Qwen/Qwen3.8-27B` checkpoint is the selected open-weight student: a dense native vision-language model under Apache 2.0.

---

# Scope Qwen3.8-27B Catan pilot

## Goal
Revise the projector/SFT compute estimate around the user's chosen announced Qwen3.8-27B open checkpoint.

## Plan
- [x] Confirm the chosen target and separate official release facts from third-party extrapolation.
- [x] Build a provisional memory/hardware/cost envelope using the prior native-VL dense 27B model only as a planning proxy.
- [x] Define release-day checks that determine whether projector grafting or native-VL adapter training is appropriate.

## Review
- The selected student is the released `Qwen/Qwen3.8-27B` checkpoint, not Qwen3.8-Max and not Qwen3-VL-8B.
- The official artifacts confirm 27.78B BF16 parameters, Apache 2.0, native image/video input, a 27-layer width-1152 vision encoder, and a dense 64-layer Qwen3.5-family decoder: 48 Gated DeltaNet layers plus 16 full-attention layers with four KV heads of dimension 256. Native context is 262,144 tokens, with documented static-YaRN extension to one million.
- The official BF16 repository occupies 55.58 GB; the official block-FP8 repository occupies 30.88 GB. Transformers, vLLM, SGLang, and TokenSpeed support are documented, but exact training memory and throughput still require measurement on the intended adapter/projector configuration.
- Native vision means the first pilot should adapt the existing vision path rather than replace it with MoonViT. GGUF/MLX/NVFP4 community quants are inference artifacts, not the canonical starting point for projector or RL training.
- Provisional cloud envelope at current low marketplace prices remains: architecture smoke $5–15; 66k-example short-answer alignment $25–75; retries/ablations $75–200. Keep a $250 phase-one cap until a measured training benchmark replaces these allowances.
- Static release inspection is complete; remaining release work is to benchmark visual-token counts, adapter trainability, four-seat KV/GDN cache allocation, latency, and VRAM with the actual serving and training stacks.

---

# Diff model and human actions across one replay

## Goal
Run two models against every exactly comparable decision by the captured human seat in a complete Colonist replay, without executing model actions or leaking future replay events.

## Plan
- [x] Inspect the authoritative replay stepper, perspective-safe decision packet, legal-action matcher, provider client, and available full-game replay.
- [x] Confirm replay, models, seat scope, and comparison policy before making paid provider calls: game `242781000`; `openai/gpt-5.6-sol` versus `qwen/qwen3.8-27b`; archived color-5 seat only; headline includes every exact interface choice, including one-option states.
- [x] Add a resumable stateless batch runner that builds the model packet before revealing each human action, maps choices to the provider call's stored indexed menu, and advances only along the recorded replay.
- [x] Dry-run all 570 replay rows, classify the captured seat's 145 records, smoke-test both providers, and run both models on all 109 exact choices.
- [x] Verify artifact completeness and report agreement by action type, every disagreement, parse/API health, token usage, latency, and cost.

## Enforced invariants
- Use `allow_lookahead=False`; the current human action stays outside the model prompt.
- Use stateless calls because model goals become counterfactual after the first disagreement.
- Include forced one-option states in the user-requested headline, but report the 89 nontrivial choices separately.
- Normalize only random outcomes; mark controlled discard cards, generic trade terms, lifecycle rows, and the collapsed multi-bank trade coarse or unmappable rather than claiming false exact matches.
- Preserve canonical replay row, original replay row, and raw source-event indices; never execute a model-selected action.
- Store each provider's exact menu and prompt. Compare semantic action identity and use an order-invariant hash so cross-process legal-menu reordering cannot corrupt resume or scoring.

## Review
- Full replay: 570/570 rows completed causally with zero semantic errors. Captured BLACK/color-5 records classified as 109 exact, 20 coarse, 15 lifecycle, and 1 unmappable; exact choices contain 20 forced and 89 nontrivial decisions.
- Sol matched 52/109 exact (47.7%) and 32/89 nontrivial (36.0%). Qwen matched 53/109 exact (48.6%) and 33/89 nontrivial (37.1%). Both models selected the same semantic action on 69/109 decisions (63.3%).
- Sol returned 109/109 valid actions with no format warnings. Qwen returned 109/109 recoverable actions with 23 format warnings after explicit no-reasoning control; its default-reasoning smoke had consumed all 8,192 tokens without returning an action and was preserved separately.
- Full-run provider cost was $2.621374; total spend including three diagnostic smoke calls was $2.681468. Full-run median latency was 7.53 s for Sol and 20.30 s for Qwen.
- Artifacts: `data_pipeline/training/reasoning/pilots/_2n5F2DxtPI/action_selection_diff/sol_vs_qwen3_8_27b_20260816/`.
- Verification: 108 tests passed with 1 skipped; targeted Ruff and whitespace checks passed; a second process resumed with zero provider calls; independent review found no blocking, high, or medium issues.

---

# Diagnose spatial-ASCII benchmark gaps

## Goal
Separate format readability, topology understanding, aggregation, and visual perception so an ASCII benchmark does not overclaim spatial competence.

## Plan
- [x] Audit every current spatial-ASCII miss and compare paired answers across formats and vision.
- [x] Identify capabilities absent or confounded in the 110-question suite.
- [x] Define information-equivalent ASCII variations and orthogonal diagnostic tasks.
- [x] Recommend the smallest paid benchmark matrix and explicit success criteria before launching provider calls.

## Review
- The historical 93/110 is component-presence exact, not strict semantic exact. One credited ASCII port answer falsely adds occupied `<N40>`, so the known strict ceiling is 92/110 pending a complete strict rescore.
- Current ASCII misses: 10 single-scalar building-count answers, four wrong road counts, and three partial node tuples. The model nevertheless listed all ten road sets exactly, isolating an aggregation/task-execution weakness.
- Building and node results are confounded by underspecified questions: no settlement/city output grammar was given, and “What building?” was scored for owner color too. ASCII’s node lead mainly reflects copying `COLOR/BUILDING`, not spatial placement.
- The selected robber-flag, node, and edge examples are all positive/occupied; three robber categories duplicate one fact. Only tiles are arranged spatially, while node/edge/port records are flat, and no question tests direction, adjacency, distance, paths, rotation, or topology.
- Eight snapshots come from one replay and two from another. ASCII versus compact JSON is 7 paired wins to 1 loss under the weak scorer (`p≈0.070` two-sided exact McNemar), insufficient for a strong format-ranking claim.
- Next harness must use a full canonical graph, strict typed/set scoring that rejects extras and contradictions, balanced negatives, ID permutations, rotations/reflections, independent boards, and repeated paired calls.
- Recommended format-equivalent cells: flat labeled records, tile-row ASCII, full node/edge board diagram, adjacency blocks, dense explicit tables, and shuffled/wrapped controls. Keep coordinate/topology/default removal as separately labeled information ablations.
- Recommended first paid gate: six formats × 60 balanced diagnostics after deterministic validation; expand to 200 questions and 3 repeats only if the smoke separates cells without scorer failures.
- Corrective caveats were added to `2026-08-16-qwen3.8-piece-recognition-and-text-formats.md`; independent audit supplied the methodological review.

---

# Build and run spatial-ASCII variation smoke

## Goal
Run the user-approved 6-format × 60-question Qwen3.8 smoke only after removing the current scorer, selection, and topology confounds.

## Plan
- [x] Define an isolated `catan_full_public_graph/v1` fact set with all 19 tiles, 54 nodes, 72 edges, nine ports, topology, coordinates, and dynamic public occupancy; exclude player summaries and precomputed answers.
- [x] Deterministically permute opaque entity IDs per board and select density-stratified snapshots from twelve independent games so fixed-atlas ID priors and the prior 8+2 game skew cannot carry the result.
- [x] Render six semantically equivalent ASCII views: sorted records, shuffled records, typed sections, tile rows, local tile blocks, and topology diagram; prove all parse to one digest.
- [x] Generate exactly 60 balanced strict-JSON questions: both six-way direction inverses, occupied/empty node and edge state, connected/disconnected nodes, node→tile sets, occupied/empty port joins, roll production, settlement/city counts, and road count+set inventory.
- [x] Use an exact typed scorer that rejects extra keys, extra set members, broken tuple associations, prose, and contradictory values.
- [x] Add a resumable text-only evaluator pinned to AkashML with reasoning disabled, interleaved format order, complete request provenance, and no image or target leakage.
- [x] Validate locally, run the approved 360-call smoke, verify every artifact and cost, analyze paired/category failures, and obtain independent review before reporting.

## Review
- Built `catan_full_public_graph/v1` over 12 density-stratified boards from 12 games, with per-board opaque ID permutations and complete 19/54/72/9 tile/node/edge/port facts. All 72 renderings round-trip to identical board digests.
- Ran 360 accepted Qwen3.8-27B calls pinned to AkashML: 250/360 strict semantic exact (69.44%), 360/360 valid JSON, zero reasoning, $1.148874. Sixty retained zero-cost local `No route to host` attempts were resumed successfully; a final resume issued zero calls.
- Format exact: tile rows 44/60, topology diagram 44, local blocks 43, shuffled records 41, flat sorted 39, sectioned 39. Tile rows/diagram each beat flat sorted 5–0 paired, but exact McNemar `p=0.0625`; no winner claim.
- Perfect 144/144 on node state, edge state, node connectivity, and node→tile sets. Road inventory reached 33/36 and typed building counts 24/36.
- Direction grounding remained weak at 28/72 across inverse tasks, with 0/12 on `UP-RIGHT` and 0/6 on inverse `DOWN-RIGHT`; failures systematically collapsed diagonals.
- Roll production was the largest genuine reasoning failure: 3/36 exact, 0/18 on positive payouts, tuple precision 41.2% and recall 58.3%. Formats did not repair the multi-hop roll→tile→corner→building→weight→aggregate chain.
- Port occupancy was 18/36: every correct building tuple was retrieved, but every occupied case also emitted the empty second port node. This is a strict filtering/schema-interpretation failure to isolate with an explicit omit-empty instruction.
- All artifacts enforce strict duplicate-free JSON, exact tuples/sets, answer/scorer/prompt hashes, provider/model pinning, zero reasoning, and fail-closed resume admission.
- Verification: 18 tests, Ruff, whitespace, independent answer recomputation, preflight gate review, and final result review passed.
- Report: `reports/catan_board_bench/2026-08-16-qwen3.8-ascii-variations-smoke.md`.

---

# Search full-graph text formats beyond ASCII

## Goal
Treat ASCII as one candidate rather than the optimization target, and compare lower-syntax-OOD full-board serializations on the existing strict 60-question suite.

## Plan
- [x] Freeze the existing ASCII dataset and evaluation artifacts; build a separate format-search dataset from the same canonical facts and questions.
- [x] Render optimized semantic HTML, full-graph JSON, Datalog facts, normalized SQL, and occupation-integrated ASCII; retain a current ASCII baseline and exclude the rejected incident-list candidate.
- [x] Prove every rendering round-trips to the identical 19/54/72/9 public fact digest with no player summaries or precomputed answers.
- [x] Keep one format-neutral prompt, strict typed JSON scorer, opaque IDs, provider/model pinning, zero reasoning, and fail-closed resume admission.
- [x] Measure tokens before paid calls, run one pinned preflight, then run the user-approved paired evaluation.
- [x] Analyze strict accuracy, category saturation, paired disagreements, prompt tokens, cost, and latency; increase difficulty only after the suite saturates.

## Review
- Built a separate six-format dataset over the unchanged 12 boards and 60 strict questions. All 72 representations round-trip to their canonical 19/54/72/9 fact digest; incident lists, player summaries, and answers are absent.
- Completed 360/360 Qwen3.8-27B calls through pinned AkashML with zero errors, 360 valid JSON responses, zero reasoning, and a zero-pending resume. Total usage was 2,078,130 prompt plus 5,664 completion tokens for `$0.91596490`.
- Strict exact ranking was tile rows 44/60, full-graph JSON 41, Datalog 39, integrated ASCII 38, SQL 36, and optimized HTML 35. Cochran `Q=9.970`, `p=0.0761`; no pair survives correction across 15 exploratory McNemar comparisons, so there is no defensible winner claim.
- HTML was the efficiency endpoint: 37.6% fewer prompt tokens and 35.3% lower cost than tile rows, but nine fewer correct answers. The observed accuracy/token/cost/median-latency Pareto set is HTML, integrated ASCII, full-graph JSON, and tile rows.
- The suite is not saturated: occupied ports were 0/18, positive production 0/18, and ten questions defeated all six formats. Keep the broad suite at its current difficulty and use staged direction, production, and port diagnostics before a harder version.
- Integrity and final-result reviews found no blocking or high issues after report corrections; 15 combined old/new tests, Ruff, score recomputation, round-trip verification, and whitespace checks passed.
- Report: `reports/catan_board_bench/2026-08-17-qwen3.8-full-graph-formats.md`.

---

# Guided vLLM blog walkthrough

## Goal
Learn vLLM from its official writing in the order needed to deploy and benchmark Qwen3.8-27B for four persistent Catan seats and later RL rollouts.

## Plan
- [x] Catalog the relevant official posts and separate foundational readings from optional model/vendor announcements.
- [x] Study the original PagedAttention post, the V1 redesign post, and the V1 anatomy deep dive.
- [x] Build a staged path covering the request loop, scheduling and KV blocks, prefix caching, benchmarking, distributed execution, hybrid GDN caveats, multimodal serving, and RL integration.

## Review
- Use the 2025 Anatomy post as the organizing spine, but read the 2023 PagedAttention motivation first and the V1 redesign second.
- Treat historical throughput numbers as demonstrations of the original system, not current Qwen3.8 forecasts.
- Pair blog concepts with small experiments: inspect startup cache capacity, benchmark TTFT/ITL/throughput, test exact-prefix reuse, then test four diverging seat histories.
- Qwen3.8-27B is a hybrid Gated DeltaNet/attention model, so Transformer-only KV explanations are necessary but insufficient; recurrent-state allocation and actual prefix-cache behavior must be measured.
- Read RL posts only after ordinary serving is understood: vLLM performs rollout inference and weight synchronization, while a trainer and the Catan harness retain optimization, authoritative state, and per-seat histories.

---

# Verify vLLM and assess SGLang

## Goal
Verify the prior vLLM lesson against primary sources, then explain SGLang and decide how it should be evaluated for Qwen3.8-27B Catan serving.

## Plan
- [x] Verify the Qwen3.8 cache arithmetic, PagedAttention behavior, continuous-batching distinction, APC semantics, and hybrid-model caveats against official sources.
- [x] Study SGLang's official paper, RadixAttention design, runtime and cache documentation, Qwen3.8 guidance, multimodal path, and RL integration.
- [x] Define a matched vLLM/SGLang four-seat 8K/16K/32K benchmark that measures hybrid recurrent-state reservations, prefix reuse, latency, throughput, correctness, and rebuild behavior.

## Review
- The prior arithmetic is correct: Qwen3.8-27B has 16 full-attention and 48 GDN layers; raw attention KV is 64 KiB/token in BF16 or 32 KiB/token in FP8, so four 16K seats use 4 GiB or 2 GiB respectively before GDN state and overhead.
- The PagedAttention/APC distinction is correct. Current vLLM has merged block-aligned `align`-mode GDN prefix caching, while finer-grained `all` mode remains under active work; pin the mode and measure rather than treating hybrid APC as either absent or complete.
- SGLang is a co-primary candidate, not a proven winner: its dedicated Qwen3.8-27B recipe explicitly serves native vision, models the separate GDN state pool, and offers GDN-aware Radix caching, optional session-aware soft eviction, and RL sleep/refit/pause APIs.
- SGLang `session_id` changes cache eviction priority only; it never appends or reconstructs history. The Catan harness must still send complete client-managed histories and retain authoritative state.
- Benchmark identical checkpoints, KV/state precision, templates, images, request order, and four-seat workloads. Compare engine defaults first; test unified/session-aware caching, cache strategies, FP8 KV, and speculation only as separate ablations.
- Pin container and checkpoint revisions, inspect effective startup cache settings, protect administrative APIs, and restrict remote media before exposing either server.

---

# Place Megatron in the Qwen3.8 stack

## Goal
Explain what Megatron does, how it differs from vLLM/SGLang, and when its distributed-training machinery is justified for Catan adaptation and RL.

## Plan
- [x] Verify current Megatron Core and Megatron Bridge responsibilities, parallelism, checkpointing, and adapter support from official sources.
- [x] Trace the existing Miles Qwen3.8 dense path and its trainer/rollout boundary.
- [x] Recommend a staged stack for projector/LoRA pilots, distributed SFT, and later GRPO without conflating the trainer, serving engine, RL orchestrator, or Catan harness.

## Review
- Megatron Core supplies optimized model, parallelism, optimizer, checkpointing, and inference building blocks; Megatron-LM, Miles, or NeMo owns the actual training/RL loop. TP, PP, CP, DP/FSDP, and EP solve different sharding problems and should be introduced only when the measured bottleneck requires them.
- Megatron now includes sync/async generation and an OpenAI-compatible HTTP surface, but streaming, simple model construction, rollout refit APIs, and a `megatron serve` CLI remain roadmap items. Keep vLLM/SGLang as the primary serving candidates for this project.
- Current Miles pairs Megatron trainer ranks with independent SGLang rollout servers and Ray orchestration. Its Qwen3.8-27B recipe is text/DAPO GRPO: TP4 plus sequence parallelism, optimizer CPU offload, colocation, a TP1 SGLang engine, and preconverted Megatron `torch_dist` weights.
- That recipe reuses Qwen3.5 language-model arguments and language-tower weight mappings. It does not prove image preprocessing, vision/projector training, multimodal masks, or vision-weight synchronization for Qwen3.8 Catan RL.
- A full BF16 Adam update is intrinsically multi-GPU at 27.78B parameters: Megatron documents 18 bytes/parameter before activations without optimizer sharding. LoRA removes most gradient/optimizer state but not the dense forward/backward or frozen base-weight residency.
- Start with a bounded native HF/PEFT image-forward and frozen-vision adapter/LoRA smoke on hardware that actually fits. Adopt Megatron Bridge/Core only after a multimodal HF-to-Megatron round-trip, image-bearing optimizer step, adapter checkpoint reload, and serving update all pass.
- Preserve the Catan harness as owner of authoritative state, legal actions, perspective safety, trajectories, rewards, and evaluation. Miles or NeMo may orchestrate RL; Megatron trains; SGLang/vLLM generates.

---

# Align Qwen replay traces with plain transcript text

## Goal
Show stored Qwen policy reasoning at its causal replay cursor beside the paired YouTube transcript, while replacing semantic-category cards with plain timestamped text between replay actions.

## Findings and decision
- The stored Qwen run has 109 exact decisions for BLACK/color 5. The paired YouTube narrator is FunDipDevRip/BLUE/color 2, so those rows are not valid narrator-comparison traces.
- Actor identity is a hard join key: rerun Qwen on BLUE/color 2 and align only BLUE traces to the BLUE transcript. Keep the archived-capture perspective limitation explicit in artifact provenance rather than substituting BLACK traces.
- Existing transcript windows already contain faithfully reflowed `segments` with source start/end times and strict pre-action containment. `semantic_traces` are a derived display layer and can be removed without changing the raw transcript or contextualization evidence store.
- Policy response rows use a canonical policy order for same-event robber decisions. Original row identity and safe display availability differ: pre-event MOVE reasoning can appear at the event's first cursor, while dependent STEAL reasoning must wait until both reversed source rows are revealed.

## Plan
- [x] Inspect trace artifacts, cursor mappings, transcript payloads, semantic rendering, replay snapshots, and existing reasoning-card patterns.
- [x] Target the narrator's verified BLUE/color-2 seat; do not reuse BLACK traces for the comparison.
- [x] Preflight BLUE decisions, smoke-test one Qwen call, then complete and validate the resumable BLUE trace run without executing model actions.
- [x] Add a versioned, read-only loader for stored Qwen traces; expose only model goals/reasoning/selection and attribution at the earliest causally safe cursor, never the upcoming human action.
- [x] Remove semantic categorization from the replay transcript payload and render only timestamped utterance text within each replay action window.
- [x] Render transcript and Qwen trace as an explicitly labeled aligned view, with long reasoning collapsed and clear empty/unavailable states.
- [x] Verify the completed immutable artifact, causal boundaries, seat attribution, robber-row availability, API payloads, frontend build/lint, focused/full tests, and live rendering.

## Review
- BLUE/color 2 produced 111 exact same-seat policy traces: 22 forced and 89 nontrivial. All 111 latest responses have normalized selections, zero API errors, and zero replay semantic errors; two pathological no-action responses were retried while preserving all 113 attempts.
- Qwen matched the recorded BLUE action on 50/111 choices (45.0%), including 22/22 forced and 28/89 nontrivial choices (31.5%). All-attempt provider cost was `$0.44034495`.
- The UI exposes only model goals, reasoning, message, and selection—not the upcoming human action or agreement. Four BLUE same-event robber pairs use causal availability: MOVE appears pre-event, STEAL is withheld until the reversed source pair is fully revealed, and later cursors support multiple traces.
- The transcript API/UI now contains only faithful reflowed timestamped text. Semantic categories remain available to separate annotation code but are absent from the replay payload and visible panel.
- Attribution is explicitly FunDipDevRip/BLUE for both sides. The panel also warns that the archive is color-5/BLACK perspective and BLUE private state is reconstructed/provisional.
- Artifact: `data_pipeline/training/reasoning/pilots/_2n5F2DxtPI/action_selection_diff/qwen3_8_27b_blue_20260817/`; retry cost provenance is in `retry_audit.json`.
- Verification: 126 tests passed with 1 skipped; targeted Ruff, production frontend build, targeted ESLint, and whitespace checks passed. Live HTTP/UI checks showed 27 plain opening transcript lines, one BLUE Qwen card, no semantic cards, collapsed reasoning, stable refresh state, and 111 ready traces. Independent final review found no blocking, high, or medium issues. Full-repository ESLint retains five unrelated baseline errors in `DecisionLog.tsx`, `HexBoard.tsx`, and pre-existing `types.ts` lines.

---

# Strengthen initial-placement strategy reasoning

## Goal
Make setup decisions reason from a coherent two-placement opening portfolio and provisional 10-VP win condition instead of treating pip totals as the objective.

## Plan
- [x] Add setup-only strategic guidance covering settlement complementarity, number diversity, ports, road expansion, opponent contention, and city/dev-card/Largest Army versus expansion/Longest Road paths.
- [x] Explicitly correct the setup misconception that the first road leads to the independently placed second settlement; require primary and fallback plans rather than premature commitment.
- [x] Add prompt-contract tests and preserve main-game prompt behavior.
- [x] Refresh only the four BLUE initial-placement traces with prompt-version provenance, retaining all prior attempts and cost audit data.
- [x] Verify the new traces discuss paired placements, opponent plans, and endgame routes without collapsing back to pip ranking; rerun full checks and inspect localhost.

## Review
- Setup prompt `setup-strategy-v11` treats both settlements/roads as one portfolio, supplies verified 10-VP route templates and canonical costs, limits setup generation to 2,048 tokens, and leaves the main-game prompt and 8,192-token budget unchanged.
- Setup menus omit pip totals while retaining dice numbers. Road options now state that they grant no resources/VP, cannot determine the free second settlement, and list only current public settlement targets reachable after one additional paid road.
- A public setup-portfolio block exposes every engine-node settlement, road edge, resource/number mix, and port so opponent inferences can be grounded without future replay events or Colonist-corner confusion.
- Four BLUE UI traces are separate from the base 111-decision evaluation. Two selected road rationales use explicitly labeled Qwen self-review with original drafts preserved; the two raw settlement drafts display four exact factual caveats rather than silently editing model output.
- Override and repair rows are exact members of immutable attempt ledgers and are bound to replay/stage, ordered engine menus, selected action identity, base activity history/cutoff, and source/context hashes. Adversarial relabel, reorder, index, menu, and future-activity tests fail closed.
- Provenance: 38 setup action attempts, seven successful rationale-repair attempts, four selected overrides, two selected repairs. Known setup/repair cost was `$0.10406137`; combined known cost with the base run was `$0.54440632`. One failed diagnostic repair lacked persisted usage and is explicitly excluded from the total.
- Verification: 138 tests passed with 1 skipped; targeted Ruff, production build, targeted ESLint, and whitespace checks passed. Live localhost showed `setup-strategy-v10` on the selected first-settlement trace, valid Primary/Fallback routes, no pip totals, two visible caveats at step 0, and a labeled self-reviewed rationale plus preserved draft at step 1. Independent final review found no blocking, high, or medium issues.

---

# Isolate resource descriptions and token wrappers

## Goal
Determine whether Qwen3.8's isolated-tile resource errors come from visual classification, an incomplete output vocabulary, or angle-bracket token syntax.

## Plan
- [x] Freeze the existing 102-image isolated tile set and canonical labels.
- [x] Cross two independent factors: complete label vocabulary with versus without brief visual descriptions, and angle-bracket labels versus plain labels.
- [x] Use the same format-neutral question and strict condition-aware scorer for all four conditions; score resource, number, pair exactness, and protocol separately.
- [x] Pin Qwen3.8-27B to AkashML with fallbacks and reasoning disabled; preserve image, prompt, scorer, provider, model, and response provenance.
- [x] Validate locally, run one request per condition as a preflight, then run all 408 pairs only if admission checks pass.
- [x] Analyze paired effects and per-resource confusion before changing the dense-board or training representation.

## Review
- Completed 408/408 Qwen3.8-27B calls through pinned AkashML with zero errors, explicit zero reasoning, exact submitted-image hashes, and a zero-pending resume. Full-run cost was `$0.06234780`; preflight plus full cost was `$0.06294550`.
- Strict pair exact was angle labels 99/102, plain labels 100/102, angle plus descriptions 101/102, and plain plus descriptions 102/102. All 400 raw numbered responses contained the correct terminal number.
- Every strict failure was a BRICK row. Complete labels alone raised the fixed-grid result to 97–98%; descriptions repaired both repeated BRICK-4 class confusions, leaving one lowercase angle-wrapped protocol miss.
- Observed factorial effects were angle minus plain `-0.98` points, descriptions minus labels `+1.96` points, and zero interaction. Discordance counts were only one or two, so no population-level effect claim is justified.
- Angle wrappers provided no observed accuracy benefit while adding 2–3% prompt tokens, about 30% completion tokens, and about 5.5% cost. Prefer plain canonical class values for this classifier and canonicalize external syntax downstream.
- The historical 32/102 canonical and 67/102 alias-aware scores were prompt-confounded and are not an isolated-vision ceiling, but this run does not separately identify vocabulary enumeration versus framing/provider/service changes.
- Verification: 7 tests, Ruff, whitespace, strict resume admission, independent score/artifact review, and adversarial reasoning-usage checks passed.
- Report: `reports/catan_board_bench/2026-08-17-qwen3.8-isolated-tile-prompt-ablation.md`.

---

# Audit recent harness trajectories

## Goal
Identify reproducibility and runtime inefficiencies in the recent Pi experiment
trajectories and the 111-decision Qwen replay trajectory without changing the
harness.

## Plan
- [x] Inspect recent Pi session trajectories and quantify repeated commands,
  inline analysis, verification, and background-run handling.
- [x] Inspect recent evaluation runners and run artifacts for duplicated
  orchestration, missing provenance, and rerun ergonomics.
- [x] Inspect the full-game Qwen decision trajectory for avoidable model calls,
  token use, latency, output-protocol failures, and low-level decision churn.
- [x] Separate immediate workflow improvements from deeper policy-harness
  redesigns.

## Review
- The experiment layer has strong per-run plans, input hashes, append-only raw
  responses, strict admission, locking, and resume behavior in its newest
  runners. Preserve those properties.
- Five recent CatanBoardBench runners duplicate provider transport, concurrency,
  parsing, plan, response, and summary logic; older runners have weaker resume
  guarantees. The repository exposes no working stable evaluation command, and
  the configured `cle` entry point currently imports a missing `cle.run` module.
- The recent action-diff/replay trajectory used 592 tool calls, including 85
  inline Python analyses, 46 pytest commands, 32 Ruff commands, and 19
  whitespace checks. The special-token ablation used 67 tool calls and seven
  separate review gates. Named plan, run, verify, and report commands would
  remove much of this manual orchestration.
- The 111-decision Qwen replay made 22 forced model calls, generated 113,004
  completion tokens for action selection, produced 35 output-format warnings,
  and ran one model serially across replay decisions. It also combines action
  choice, goals, rationale, and table talk in every response.
- Recommended direction: keep Python as tested library code, move experiment
  parameters into versioned run specifications, expose thin rerunnable shell or
  `just` recipes, materialize immutable decision packets before provider calls,
  and provide network-free verification and reporting stages. For live policy
  inference, skip forced choices, combine compound decisions, use persistent
  per-seat event context, constrain the action schema, and generate expensive
  rationales only when they serve training or review.

---

# Replace setup heuristics with player-specific reasoning

## Goal
Remove canned setup strategy heuristics from the live Qwen prompt and require
claims about other players to be separate, color-specific, and grounded in each
player's visible public setup portfolio.

## Plan
- [x] Inventory every setup-only heuristic in the phase rules, stage guidance,
  XML field instructions, and prompt-contract tests.
- [x] Bump the setup prompt version and retain only authoritative setup mechanics,
  factual grounding constraints, open-ended comparison, and output bounds.
- [x] Require separate color-named reads for each opponent claim; cite that
  color's visible settlement/road evidence and mark absent evidence unknown
  rather than collapsing players into one aggregate opponent forecast.
- [x] Preserve main-game prompt behavior, legal-action menus, road facts, privacy,
  and non-mutation guarantees.
- [x] Run focused prompt tests, inspect a rendered setup prompt, and record review.

## Review
- Bumped the live setup prompt to `setup-strategy-v12` and removed the dice
  ranking, factor checklist, fixed VP compositions, port/resource prescriptions,
  primary/fallback requirement, and stage-specific strategic recommendations.
- Retained only setup mechanics, exact action/road grounding, open-ended legal
  alternative comparison, output bounds, and the existing perspective-safe
  public setup portfolios.
- Every opponent inference must now name one engine color, cite that color's
  visible settlement/road evidence, and remain separate from other colors;
  unsupported color reads must be marked unknown rather than forecast.
- Main-game prompt behavior and token budget remain unchanged. The running Flask
  server auto-reloaded and its health endpoint passed.
- Verification: 34 replay prompt/model-trace tests passed; targeted Ruff and
  whitespace checks passed.

---

# Audit playground transcript transformation

## Goal
Identify where the playground stopped exposing semantic YouTube commentary
traces and distinguish caption reflow from actual engine-grounded rewriting.

## Plan
- [x] Trace transcript loading, reflow, semantic grouping, API payloads, types,
  and frontend rendering.
- [x] Recover the uncommitted removal diff and identify the remaining detached
  commentary modules.
- [x] State exactly what transformation still runs and what never became an
  end-to-end playground feature.

## Review
- On 2026-08-17 the playground transcript builder removed its import and calls
  to `build_semantic_transcript_traces`, the API dropped `semantic_traces`, the
  frontend type dropped `ReplaySemanticTrace`, and the panel switched to plain
  `segments`.
- `parse_transcript_segments()` still joins rolling caption fragments into
  sentence-like utterances without changing words, but it does not resolve
  deictic visual comments or rewrite them into self-contained reasoning.
- `commentary/semantic_display.py` still groups verbatim utterances into labeled
  roles, while `contextualizer.py`, `references.py`, and `episodes.py` retain
  grounding/episode primitives. None is imported by the current replay API/UI.
- The displayed Qwen reasoning comes from independent replay-state policy calls,
  not from a transcript-to-reasoning transformation.

---

# Add grounded narrator-reasoning transcript tab

## Goal
Add Transcript and Reasoning tabs to the YouTube commentary section. The
Reasoning view must contain coherent narrator paragraphs produced by GPT-5.6
from the mumbled captions while using only causally available replay-board
facts and preserving uncertainty.

## Plan
- [x] Reuse existing replay-state, commentary-grounding, provider, and artifact
  patterns; choose an offline persisted schema rather than generating on tab
  clicks.
- [x] Build a versioned generator that gives GPT-5.6 bounded board-inspection
  tools and faithful transcript evidence without future actions or hidden state.
- [x] Generate and validate the curated pilot artifact with source-span and
  replay-state provenance.
- [x] Load the artifact into cursor-safe replay windows and expose typed API
  data without changing the canonical transcript layer.
- [x] Add accessible Transcript/Reasoning tabs and paragraph/evidence rendering
  to the YouTube section while keeping the independent Qwen trace separate.
- [x] Run focused Python tests, frontend lint/build, causal payload checks, and
  final diff review.

## Review
- Added an offline `narrator-reasoning-v2` generator. GPT-5.6 must call the
  public-board tool first, can deterministically inspect captioned number
  locations, receives no next action or hidden hands, and returns paragraphs
  tied to exact source evidence IDs.
- Persisted a resumable run for game `242781000`: 195 transcript intervals,
  147 with substantive reasoning, 48 filler-only intervals, and 167 paragraphs.
  Every selected result passed transcript hash, board-state hash, source-span,
  cursor-boundary, and board-tool checks.
- Added a separate cursor-safe narrator-reasoning payload; raw model/tool logs
  remain in the offline artifact and are not sent to the browser.
- Added accessible Transcript/Reasoning tabs inside YouTube commentary. The
  Qwen policy trace remains visibly separate, and unresolved visual context is
  shown only in expandable caveats.
- Verification: 63 focused Python tests passed; targeted Ruff, frontend build,
  targeted ESLint, artifact verification, live Flask payload/health checks,
  and `git diff --check` passed. Full frontend lint still reports five
  pre-existing errors in `DecisionLog.tsx`, `HexBoard.tsx`, and old `types.ts`
  declarations outside this change.

---

# Restore complete transcript coverage

## Goal
Fix the YouTube commentary view so later-game captions remain discoverable and
source caption chunks are not silently lost at dense replay-action boundaries.

## Plan
- [x] Measure full-video source-index/time coverage and distinguish backend
  selection loss from a cursor-only UI presentation problem.
- [x] Preserve causal availability while assigning every eligible caption to a
  deterministic cursor or presenting bounded prior transcript history.
- [x] Keep GPT reasoning provenance valid; regenerate only if source windows or
  input hashes materially change.
- [x] Add coverage and navigation regressions, rebuild the frontend, inspect the
  live late-game payload, and document any intentionally excluded captions.

## Review
- Root cause: the old whole-chunk rule required a caption to start after the
  previous replay action and end before the next one. Rolling YouTube chunks
  commonly crossed those boundaries, so 249 of 659 source chunks (37.8%) were
  silently dropped; only 71 of 571 cursors displayed any text.
- Alignment `caption-end-availability-v2` assigns each chunk exactly once when
  its end timestamp becomes causally available. The final cursor now accounts
  for all 659/659 source chunks, with exact normalized source text preserved
  across 527 readable transcript blocks and zero intentionally excluded chunks.
- The API now returns cumulative causal transcript history as well as the new
  interval. Transcript and Reasoning lists retain prior content, shrink safely
  on backward navigation, and auto-scroll to the latest available entry instead
  of showing an empty panel at dense action cursors.
- Rebuilt the GPT artifact for all 195 caption-bearing cursors: 147 substantive
  intervals, 48 filler-only intervals, and 167 evidence-linked paragraphs.
  OpenRouter reached the key's weekly limit after 63 jobs, so the remaining 132
  were completed by GPT-5.6 Pi agents using the same persisted public-board and
  location-tool packets; all rows passed source, board, and cursor validation.
- Verification: 63 focused Python tests, targeted Ruff, frontend production
  build, targeted ESLint, artifact verification, live Flask health/payload
  checks at cursors 500 and 570, and `git diff --check` passed. Full frontend
  lint retains the same five pre-existing errors outside this change.

---

# Audit data-pipeline cleanup

## Goal
Identify the highest-value cleanup boundaries in `data_pipeline/` without
moving code or artifacts in the current dirty worktree.

## Plan
- [x] Inventory packages, imports, entry points, tests, and tracked artifacts.
- [x] Identify unsafe prototypes, duplicate ownership, generated-data sprawl,
  stale compatibility layers, and reproducibility hazards.
- [x] Prioritize low-risk cleanup before architectural extraction.

## Review
- Only about 0.41 MiB of the 100.7 MiB tracked under `data_pipeline/` is Python;
  raw replays, benchmark datasets, run logs, and generated split manifests are
  mixed into installable package roots.
- `catan_board_bench`, `ingestion`, `replay_pipeline`, and `training_pipeline` are
  eager `sys.modules` alias shells rather than real packages. Three fail to
  import without optional dependencies, and every new CatanBoardBench module needs a
  manual alias.
- The advertised replay training generator updates action/player state before
  formatting the claimed pre-action observation. It should be quarantined, not
  remain the package's exported default.
- CatanBoardBench depends on viewer/server state for replay execution, while the
  authoritative replay implementation lives under `playground/`; extract a
  pure replay core before moving benchmark modules.
- Recommended order: classify legacy code and artifact policy; move raw/runs
  out of package roots; finish one canonical package boundary at a time; then
  split benchmark/replay monoliths and consolidate geometry/schema authorities.

---

# Clean data-pipeline code and artifacts

## Goal
Remove confirmed-dead pipeline code and separate raw data, generated runs,
manifests, fixtures, and reports from installable source packages. Do not change
the game, policy/LLM, replay-harness behavior, or frontend.

## Approved scope
- Keep the hybrid direction: real `catan_board_bench/` and future
  `replay_pipeline/`; retain acquisition and dataset construction under
  `data_pipeline/`.
- This pass does not extract the authoritative replay core or refactor game,
  LLM, harness, or UI code.
- Preserve frozen benchmark inputs and user-created artifacts; do not silently
  delete unique raw data or accepted run evidence.

## Plan
- [x] Define one root artifact layout and classify every moved path as raw,
  staging, manifest, frozen fixture, generated run, or tracked report.
- [x] Remove only code proven unused, incomplete, incorrect, or an unsafe
  superseded generator; update package exports and documentation.
- [x] Move Colonist raw/index/split artifacts, generated evaluation runs,
  pretraining outputs, and reports out of source packages; defer active
  reasoning pilots owned by concurrent harness/UI work.
- [x] Update defaults, manifests, references, and ignore rules without touching
  game/LLM/harness implementation.
- [x] Verify imports, focused tests, path integrity, duplicate preservation,
  artifact hashes, whitespace, and the final scoped diff.

## Artifact layout
- `artifacts/raw/colonist/replays/`: canonical captured replay payloads.
- `artifacts/raw/colonist/indexes/`: leaderboard/history candidate indexes.
- `artifacts/staging/colonist/replays/`: unverified and rejected captures.
- `artifacts/manifests/colonist/splits/`: deterministic split/queue manifests.
- `artifacts/fixtures/catan_board_bench/smoke5/`: frozen five-game visual fixture.
- `artifacts/generated/pretraining/legacy_corpus/`: historical corpus outputs.
- `artifacts/runs/catan_board_bench/<suite>/`: provider plans, responses, and summaries;
  frozen benchmark inputs remain in `data_pipeline/catan_board_bench/datasets/` for
  this pass to avoid a benchmark/harness refactor.
- `reports/catan_board_bench/`: tracked human-readable benchmark reports.
- Active narrator-reasoning pilot artifacts remain in place because a separate
  harness/UI change currently owns that path; moving them is explicitly
  deferred rather than racing that work.

## Dead-code removals
- Remove the unsafe exported `generate_training_data.py` prototype; retain its
  historical output only as a labeled legacy artifact outside source.
- Remove `bootstrapping/deprecated/`, the incorrect unused manual Colonist
  layout helper/artifacts, obsolete auth and Supabase helpers, the unfinished
  `training/schema.py`, and unused compatibility wrappers.
- Remove eager unused top-level `ingestion`, `training_pipeline`, and
  `replay_pipeline` alias shells; keep the working `catan_board_bench` boundary.
- Retain the standalone replay decoder and current acquisition clients until
  their replacements are independently verified.

## Review
- Established the approved hybrid layout under `artifacts/{raw,staging,
  manifests,fixtures,generated,runs}` and `reports/catan_board_bench/`; installable
  pipeline source no longer contains run, report, cache, or output directories.
- Moved and hash-checked 352 files (66,379,914 current bytes), deleted five
  byte-identical replay copies, and preserved a per-file migration receipt at
  `artifacts/manifests/layout_migrations/data_pipeline_layout_v1.json`.
- Added read-only compatibility links for the two legacy replay directories and
  a network-free verifier at `scripts/verify_data_pipeline_layout.py`.
- Removed the unsafe training-data generator, obsolete schema/layout/auth/DB
  prototypes, dead wrapper packages, and empty continual-learning scaffolding;
  package exports and optional-dependency boundaries now import cleanly.
- Updated acquisition, split, corpus, benchmark, config, report, and
  documentation paths to the canonical layout. Active narrator-reasoning
  pilots were deliberately left untouched because concurrent work owns them.
- Verification passed: migration receipt (15 groups, 352 files, five duplicate
  removals), Ruff on pipeline and touched helpers, shell/compile/import/CLI
  smoke checks, and the full suite at 159 passed with one skipped.
---

# Clean SFT code and artifacts

## Goal
Keep `sft/` as a behavior-preserving training-tool package while removing the
superseded trainer path, separating generated datasets/runs/diagnostics from
source, and making dataset paths portable and leakage checks fail closed.

## Approved scope
- Use the user-approved scoped cleanup: remove only confirmed dead, superseded,
  duplicated, or unsafe SFT code/data; preserve the current Qwen-Series trainer,
  evaluation path, renderer/data builders, and training behavior.
- Do not redesign the game, replay core, CatanBoardBench scoring/harness, policy/LLM,
  or UI. Touch external benchmark scripts only where their SFT-owned path is
  relocated.
- Preserve unique generated evidence. Invalid legacy train/held-out subsets are
  retained only as explicitly quarantined artifacts, not advertised as valid
  evaluation data.

## Audit
- `sft/` has 29 tracked files but also about 265 MiB of ignored generated data
  and diagnostics under installable source. `sft/outputs/` is an empty ignored
  checkpoint shell.
- All 4,187 image-backed JSONL rows use the nonexistent absolute prefix
  `/Users/henry/CascadeProjects/catan-learning`, so current Modal conversion and
  upload commands cannot resolve their images.
- `post_atlas_train_short_100.jsonl` and
  `post_atlas_heldout_short_100.jsonl` share 12 exact image paths; the latter is
  not a valid image-held-out evaluation set.
- The old `modal_train.py`/`train_qwen_vl_sft.py` TRL path is explicitly
  superseded by the pinned Qwen-Series path. Its documented text-only atlas
  smoke is broken because the uploader unconditionally indexes `row["image"]`.
  Both old Qwen3-VL-8B YAML configs have no current consumer.
- `questions/answer_key_with_images.jsonl` is byte-identical to the original
  answer key, wasting 13,389,882 bytes; no code consumes the duplicate.
- Piece-recognition provider results remain mixed into ignored SFT data. The
  full run duplicates the canonical CatanBoardBench run, while the three-file smoke
  run is unique and should be preserved with CatanBoardBench run evidence.
- The held-out game ledger currently fails open when absent, and generated
  training rows store machine-specific absolute paths rather than portable
  dataset- or repository-relative paths.
- `sft` is a real imported package but is omitted from Hatch's wheel package
  list. There are no dedicated SFT tests, and retained modules contain inline
  optional imports plus one Ruff E402 path hack.

## Target layout
- `sft/`: package code, current Modal Qwen-Series entrypoints, experiment design,
  and one concise README only.
- `configs/sft/renderer_style.json`: tracked renderer configuration.
- `artifacts/generated/sft/`: ignored/regenerable atlas and node-factor data;
  invalid historical subsets live under `legacy_pilots/` with a warning.
- `artifacts/generated/catan_board_bench/piece_recognition/`: ignored/regenerable
  isolated-piece probe inputs, no provider responses.
- `artifacts/fixtures/sft/`: tracked renderer contract and self-contained Modal
  smoke fixture.
- `artifacts/diagnostics/sft/`: ignored local renderer/token-grid images.
- `artifacts/runs/sft/`: ignored local checkpoints/logs; accepted lightweight
  run evidence remains eligible for tracking.
- `reports/sft/`: tracked historical training-run reports.

## Plan
- [x] Remove the superseded TRL trainer/launcher, unused Qwen3-VL-8B configs,
  stale pre-Modal checklist, empty source artifact shells, and the exact
  generated answer-key duplicate.
- [x] Hash-preserve and relocate generated SFT datasets, piece-probe inputs,
  diagnostics, fixtures, renderer config, and historical run evidence; preserve
  the unique piece smoke and eliminate canonical run duplicates.
- [x] Make JSONL image references portable, resolve relative assets against the
  dataset then repository root, fail closed when the leakage ledger is missing,
  and quarantine the image-overlapping legacy split.
- [x] Update defaults, Modal mounts, package metadata, scripts, reports,
  documentation, and ignore policies to the canonical layout.
- [x] Add focused SFT tests and a network-free migration verifier, then run Ruff,
  imports/CLI/compile checks, focused tests, and the full suite.

## Review
- Reduced `sft/` from about 265 MiB to code/docs only. Generated atlas and
  node-factor data moved to `artifacts/generated/sft/`, diagnostics to
  `artifacts/diagnostics/sft/`, fixtures to `artifacts/fixtures/sft/`, renderer
  configuration to `configs/sft/`, and run history to `reports/sft/`.
- Removed the superseded custom TRL launcher/trainer, two unconsumed 8B configs,
  stale checklist, source artifact shells, and the 13,389,882-byte exact
  rendered answer-key duplicate. The pinned Qwen-Series train/eval path remains.
- Preserved and verified 1,429 moved/copied files (259,901,197 current bytes)
  across ten groups. Five exact duplicates and one path-superseded copy were
  removed; the unique three-call piece smoke moved to canonical run evidence.
  Receipt: `artifacts/manifests/layout_migrations/sft_layout_v1.json`.
- Replaced 4,187 broken machine-specific image references with portable paths;
  Modal upload, conversion, and eval now resolve dataset-relative assets before
  repository-relative assets. A self-contained 24-row/four-image smoke fixture
  is tracked under `artifacts/fixtures/sft/modal_vlm_smoke/`.
- Made the benchmark leakage ledger fail closed and removed its bypass. The old
  short train/held-out subsets are quarantined because they share 12 images.
- Moved isolated piece-probe generation out of the SFT package, fixed fresh-
  directory and relative-contract execution, normalized run metadata while
  preserving raw provider responses, and added `sft` to the built wheel.
- Verification passed: network-free SFT and data-pipeline migration receipts,
  deterministic builders and leakage rejection, portable asset resolution,
  Modal launcher help, import/compile/wheel checks, Ruff/format/diff checks, 25
  focused tests, and the full suite at 193 passed with one skipped.

---

# Assemble narrator reasoning by decision and observation

## Goal
Replace action-interval rewrites with coherent semantic episodes that can carry
unfinished commentary across replay boundaries and attach it to the public
decision or observation it concerns without exposing future actions.

## Proposed design
- Keep the canonical 659-caption Transcript layer unchanged and lossless.
  Globally reflow captions once so sentence boundaries no longer reset at every
  replay action, then derive stable evidence IDs from source spans.
- Build chronological assembly packets only where new captions become
  available. Each packet contains new evidence, any unresolved carry from the
  previous packet, the public board at the availability cursor, and bounded
  candidate observations already visible at that cursor.
- Have GPT-5.6 partition each bounded packet into coherent semantic paragraphs
  and filler. Paragraphs record prose, speech-act kind, exact evidence IDs, and
  uncertainties; the harness, not the model, owns the packet subject cursor.
- Use strict causal attachment: derive both availability and subject from the
  current packet cursor. Do not point later commentary back to an older move.
  A completed thought that crosses a replay boundary becomes part of the next
  causally available decision/observation packet.
- Join narrator-decision packets to existing decision IDs. Insert bounded
  public-observation packets when commentary between narrator decisions grows
  too large, so opponent reads are not mislabeled as narrator action rationale.
- Validate that every globally reflowed evidence ID is cited once or explicitly
  omitted once; strict subjects equal availability cursors; board hashes,
  decision manifests, and transcript fingerprints match.
- Persist the offline artifact and expose current packets plus subject-grouped
  causal history. The UI shows decision/observation and speech-act badges while
  keeping Qwen reasoning separate.

## Plan
- [x] Confirm strict causal attachment with no retrospective decision links.
- [x] Implement and test global evidence reflow, safe decision anchors, bounded
  public-observation packets, exact partition validation, and versioned runs.
- [x] Load subject/availability-aware episodes into replay API and WebSocket
  payloads; update typed grouped UI rendering.
- [x] Generate the pilot artifact, audit exact evidence coverage and actor
  attribution, inspect early/mid/late decisions, and run full verification.

## Review
- `narrator-observation-assembly-v1` globally reflows all captions before any
  replay partitioning, producing 527 stable evidence units from the lossless
  659-caption Transcript layer. Thoughts can cross engine rows without being
  cut into one-line action fragments.
- The plan preserves all 111 validated narrator decisions at their first safe
  viewer cursor, including same-event reorder cases, and adds 14 bounded public
  observation checkpoints plus replay completion. Of 123 total anchors, 79
  contain new evidence and require model assembly.
- GPT-5.6 produced 189 coherent paragraphs across 77 substantive packets; two
  packets were filler-only. Exactly 419 evidence units are cited and 108 are
  explicitly omitted as filler, repetition, chatter, or unsafe fragments.
  Subject and availability cursors are equal for every paragraph, enforcing the
  selected no-retrospective-links policy.
- The Reasoning UI groups causal history under decision observation, public
  observation, or replay-complete cards and labels decision reasoning,
  opponent assessment, board observation, reaction, and reflection separately.
  Raw Transcript text and Qwen traces remain independent.
- Verification passed: artifact hashes/partitions, 72 focused Python tests,
  Ruff, frontend production build, targeted ESLint, live HTTP/WebSocket payload
  checks at opening, setup, midgame, endgame, and replay completion, and 200
  tests in the full suite. Two unrelated concurrent artifact-layout receipt
  tests fail while the CatanBoardBench/SFT migration is in progress; full
  frontend lint retains the same five pre-existing errors outside this change.
---

# Adopt the CatanBoardBench name

## Goal
Replace the prior ambiguous, externally colliding benchmark brand with
`CatanBoardBench`, which accurately scopes the fixed benchmark to public-board
perception, grounding, topology, counting, production, and representation
reasoning.

## Naming decision
- Public name: `CatanBoardBench`.
- Python/module slug: `catan_board_bench`.
- Dataset: `CatanBoardBench-100` / `catan_board_bench_100`.
- CLI/artifact slug: `catan-board-bench` for URLs and
  `catan_board_bench` for files/directories.
- OpenBench task/entry point: `catan_board_bench`.
- API prefix: `/api/catan-board-bench`.

`CatanPerceptionEval` was rejected as too narrow: the frozen 100-board suite has
3,032 questions across 20 categories, and later text suites test topology,
production, aggregation, and grounded spatial reasoning. A fixed comparable
dataset is conventionally a benchmark, while “eval” better describes the
runner. Recent names follow capability + `Bench` (OCRBench, GSR-Bench,
EmbSpatial-Bench, PerceptionBench). The more specific name also avoids collision
with the existing external strategic-play project named “Catan Bench.”

## Rename policy
- Perform one atomic internal rename; do not retain legacy package aliases or
  duplicate source trees.
- Rename code, tests, scripts, OpenBench metadata, UI/API routes, frozen dataset
  directories, artifact/report roots, and current schema namespaces.
- Preserve raw provider response payloads byte-for-byte even when they contain
  pre-rename provenance. Normalize only lightweight plans, summaries,
  manifests, frozen QA metadata, and documentation.
- Preserve migration receipts' historical source paths where needed, but update
  every current destination/hash and add a dedicated rename receipt.

## Plan
- [x] Hash-inventory every source, dataset, fixture, run, report, and UI path in
  the rename boundary; define exact old-to-new directory/file mappings.
- [x] Move package, dataset, artifact, report, script, test, and UI paths without
  losing dirty-worktree content or rewriting raw provider responses.
- [x] Rename symbols, imports, entry points, schemas, API routes, defaults,
  documentation, and current metadata to the naming matrix above.
- [x] Update data-pipeline and SFT migration receipts, add a benchmark rename
  receipt/verifier, and ensure no non-historical stale brand references remain.
- [ ] Run import/CLI/UI/schema/artifact verification, Ruff/format, focused tests,
  the full suite, wheel inspection, and whitespace checks.

## Review
- Adopted `CatanBoardBench`, `catan_board_bench`, `CatanBoardBench-100`, and
  `/api/catan-board-bench` consistently across packages, datasets, scripts,
  OpenBench metadata, UI routes, artifacts, reports, schemas, and documentation.
  No compatibility source tree or legacy import package remains.
- Hash-preserved 1,023 receipt-tracked files totaling 93,485,244 current bytes,
  including 90 byte-identical raw provider/log payloads. Regenerable UI build,
  dependency, and bytecode files were excluded. Receipt and verifier:
  `artifacts/manifests/layout_migrations/catan_board_bench_rename_v1.json` and
  `scripts/verify_catan_board_bench_rename.py`.
- Updated 842 current text/metadata files and normalized 95 lightweight run
  records to canonical artifact paths. Data-pipeline and SFT migration receipts
  pass under the renamed destinations.
- Verification passed for the renamed package/import boundary, ten renamed
  CLIs, OpenBench metadata, kebab-case API routes, schemas, artifact identities,
  raw-payload immutability, stale-reference/path scans, Ruff/format, wheel
  contents, UI production build, 49 focused tests, and 145 other unaffected
  tests.
- Full-suite completion is blocked by concurrent non-benchmark work: an
  undefined `current_player` in `playground/game_viewer/routes/websocket.py`
  causes live/replay cascades, and active commentary/trading tests call `.color`
  on `Color` enum values. Those files are outside this rename and were not
  modified here; task #18 owns the blocker.

## One-level counteroffer clarification

- [x] Remove configurable counter depth from `TradeLimits`.
- [x] Reject any proposal whose parent is itself a counteroffer.
- [x] Generate counteroffer actions only for root proposals.
- [x] Update contracts, docs, regression tests, and the correction lesson.
- [x] Run focused tests, Ruff, the full suite, and replay-corpus verification.

Review: Counteroffers now always have a root proposal as their parent. Replay
source links that point through another counter are normalized back to an active
root proposal. Verification passed with 238 tests / 1 gated skip, Ruff,
`git diff --check`, and all 66 replays (31,506 actions and 15,460 trade lifecycle
actions).

## Named parameterized trade terms

- [x] Add engine-owned `TradeTerms` and `CounterOfferTerms` contracts.
- [x] Replace the live positional tuple parser with strict named-resource JSON.
- [x] Keep action-index selection authoritative for operation, audience, and root
  proposal identity while allowing model-generated resource terms.
- [x] Decode historical replay tuples at the replay boundary into typed terms.
- [x] Update event projection, serialization, prompts, docs, and regression tests.
- [x] Run focused tests, Ruff, full tests, frontend/wheel checks, and the 66-game
  replay corpus.

Review: `catan_v3.yaml` now asks for named JSON such as
`{"give":{"WOOD":1},"receive":{"ORE":1}}`. The parser produces immutable
`TradeTerms`; counter actions wrap those terms with the exact root ID selected
by the legal-action index. Positional replay data is converted only in
`cle/replay/runtime/action_matcher.py`. Verification passed with 241 tests / 1
gated skip, Ruff, `git diff --check`, frontend build, wheel inspection, and all
66 replays (31,506 actions and 15,460 trade lifecycle actions).

## Catan trade-role clarification

- [x] Represent opponent acceptance as a non-binding willingness signal.
- [x] Allow every eligible opponent to signal willingness independently.
- [x] Allow only the turn player to select one exact `TradeCandidate` for
  execution, while retaining the option not to trade.
- [x] Direct counteroffers only to the turn player and prevent third-party
  acceptance of another opponent's counter.
- [x] Keep `TradeTerms` (what) separate from proposer/counterparty identity
  (who), and use typed candidates for live confirmation.
- [x] Update views, replay projection, docs, lessons, and regression tests.
- [x] Run focused tests, Ruff, full tests, frontend/wheel checks, and the 66-game
  replay corpus.

Review: `ACCEPT_TRADE` is now stored as `willing_by`, never as execution.
Multiple opponents can signal willingness in one deterministic barrier; control
then returns to the turn player, whose normal legal menu contains one typed
`TradeCandidate` per affordable willing partner alongside ordinary actions such
as `END_TURN`. Confirmation identifies proposal, turn player, counterparty, and
terms separately. Counteroffers target only the turn player. Colonist protocol
rows that syntactically counter a counter are normalized to revised root offers.
Verification passed with 247 tests / 1 gated skip, Ruff, `git diff --check`, the
frontend build, wheel inspection, and all 66 replays (31,506 actions and 15,460
trade lifecycle actions).

## Benchmark rename receipt blocker

- [ ] Reconcile the CatanBoardBench rename manifest with four unrelated,
  concurrently modified benchmark files and finish the concurrent
  `routes/bench.py` helper edits before claiming a fully green suite/Ruff run.
  Do not overwrite or bless those changes from the live-viewer task.

## Thin live sandbox viewer

- [x] Keep live-game creation and mutation as a thin wrapper over one
  `CatanSandbox`; remove parallel observation/action execution concepts.
- [x] Make the live UI's gameplay control one `Step` button that invokes one
  complete `sandbox.step()` (including any deterministic barrier).
- [x] Remove the stale live observation/action decision-log presentation.
- [x] Keep model-authored rationale distinct from provider-native reasoning in
  typed diagnostics without presenting either as a second action scaffold.
- [x] Restore the live reasoning panel from accepted typed player receipts only;
  retain the latest 12 HTTP-returned traces and include no observation/legal-menu
  duplication in that diagnostic payload.
- [x] Add a local SQLite live-trace store in WAL mode under `.cle/`.
- [x] Persist each completed step transactionally: game metadata/config, public
  state, restorable sandbox snapshot, contexts/legal menus/events, transitions,
  accepted and rejected attempts, exact model messages, raw/normalized provider
  output, parsed choice, usage, reasoning provenance, and provider IDs.
- [x] Expose read-only local trace inspection endpoints and retention metadata.
- [x] Add schema, atomicity, restore, privacy, and live-route integration tests.
- [ ] Add focused route/frontend tests and run full backend/frontend verification.
  Focused tests, Ruff, compileall, frontend build, boundary scan, and wheel pass;
  the full suite is blocked only by unrelated CatanBoardBench rename-receipt
  drift, and full-repository Ruff by concurrent incomplete `routes/bench.py`
  edits (task #35).

- [x] Fix the reported Step no-op: restart the stale no-reloader backend;
  remove WebSocket emission/dependence from live start/step; normalize every
  public snapshot through `GameEncoder`; validate response content before JSON
  decoding; display non-JSON/non-2xx failures; and keep an outer JSON error
  boundary around `/api/step`.

Review so far: Live start creates one `CatanSandbox`; the only live mutation
endpoint is `/api/step`, and one click awaits one complete `sandbox.step()`.
The live auto-play thread/routes, raw-engine pickle and formatted-observation
endpoints, duplicated decision log, and frontend observation/action card were
removed. Model requests now contain one exact ordered legal menu rather than a
second truncated menu embedded in the state formatter. `<rationale>` remains
ordinary model-authored output in player receipts, while native reasoning comes
only from provider response fields/token evidence; unsupported live transports
reject enabled native-reasoning requests instead of silently dropping them.
The real registered Flask app now passes start -> step without relying on socket
delivery; both responses carry an immediately applicable public state snapshot.
The live reasoning panel now labels `<rationale>` as model-authored and shows
provider-native reasoning only when returned through the provider channel. Its
final contract also retains reasoning-token evidence, normalized/native finish
reasons, OpenRouter generation ID, and provider request ID. A headless frontend
check confirms both provenance-separated sections render. Live runs now write
`.cle/live_traces.sqlite3` in WAL mode: one transaction per completed step with
queryable game/step/event/model-call rows, full JSON request/response/context,
accepted and rejected attempts, and a trusted-local restorable snapshot blob.
The UI shows the active trace ID; read-only list/detail endpoints expose records
without exposing snapshot blobs.
The late Step failure was a raw `Color.BLUE` inside public event 31, which bypassed
`GameEncoder` in the manually assembled event list. A deterministic real-server
run now passes 120 HTTP steps. Headless Chromium passes Start plus 35 Step clicks
against ports 5173/5001: every response is HTTP 200 JSON and no alert appears.

## Saved live-game picker

- [x] Add optional user-facing names to persisted live games.
- [x] Add rename and load-latest-snapshot HTTP routes.
- [x] Add a previous-games bar that shows names and IDs and supports selection,
  rename, and load/continue.
- [x] Verify persistence, restore continuity, route errors, frontend build, and a
  real browser load/continue flow.

Review: The full-width saved-games bar lists the latest 100 local games by
optional name, stable ID, step count, and lifecycle status. A selected game can
be named, renamed, cleared back to ID-only, reloaded, or loaded after another
game. Loading reconstructs the configured player types/transports, restores the
latest trusted-local engine and private player-session checkpoint, and appends
future steps to the original trace ID and next step index. SQLite schema v3
migrates existing databases by adding `display_name`. Verification passed with
267 tests / 1 gated skip, targeted Ruff, compileall, TypeScript/Vite build,
`git diff --check`, migration/route/session-continuity tests, and a real headless
browser name -> replace -> select -> load at step 1 -> continue at step 2 flow.

## Browse-only trace step navigator

- [x] Add a bounded step-detail endpoint that returns one stored public checkpoint
  and every model call for that step without restoring or mutating the sandbox.
- [x] Add previous/next/latest and direct-step controls for the selected saved game.
- [x] Render all selected-step decision and communication traces, including rejected
  attempts, rationale/native reasoning, usage, finish metadata, and raw messages.
- [x] Keep gameplay Step bound to the active latest sandbox only; historical browsing
  must not change the live engine, trace status, or next append index.
- [x] Verify route bounds/privacy/non-mutation, frontend build, and a browser flow
  that browses backward then continues the untouched active game.

Review: The checkpoint navigator exposes Previous, Next, Latest, and a complete
saved-step dropdown while clearly labeling historical state as browse-only.
Selecting a checkpoint applies only its stored public viewer snapshot to the UI;
the server sandbox is neither restored nor mutated, and the gameplay Step button
is disabled until Load latest/Return to live. Every actual model call for the
step is shown in order, including decision/communication kind, actor,
accepted/rejected state, validation errors, model-authored rationale,
provider-native reasoning, finish/provider metadata, exact messages, parsed
choice, usage, and raw provider payloads. Stale list/checkpoint HTTP responses
are ignored client-side. Verification passed with 276 tests / 1 gated skip / 1
unrelated curriculum-validator test deselected, targeted Ruff, compileall,
TypeScript/Vite build, `git diff --check`, bounded endpoint and non-mutation
tests, and a real browser flow: browse step 3 -> step 1, confirm gameplay Step is
disabled, return to live, append step 4, then render eight historical decision
and communication traces from an existing LLM game.

## Explicit discard decisions

- [ ] Replace live `DISCARD` actions that carry `None` and make the engine sample
  cards with a bounded typed player-selected discard contract.
- [ ] Preserve replay-provided card selections, deterministic ordering, privacy,
  snapshots, prompt parsing, and exact legal validation.
- [ ] Add focused and full verification before calling base-game decisions complete.

## Remove obsolete PettingZoo action layer

- [x] Delete the unused free-form action parser and broken PettingZoo adapter.
- [x] Remove adapter-only helpers and dependencies with no repository callers.
- [x] Rewrite active documentation around `CatanSandbox` and exact legal-menu
  indices; remove stale implementation-plan references.
- [x] Verify imports, tests, Ruff, packaging, and placeholder/reference scans.

Review: `cle/env/action_space.py` was an abandoned free-form parser scaffold and
was never imported. The adjacent `CatanEnv` declared a text action space while
its `step()` required engine `Action` objects, so PettingZoo's own bounds wrapper
rejected legal calls. With no repository callers, both files were deleted rather
than completed. Dead node-position and exception prototypes were removed too,
`gymnasium`/`pettingzoo` were removed from project metadata and the lockfile,
and active docs now identify `CatanSandbox` as the sole environment with exact
legal-menu indices. Verification passed with 242 tests / 1 gated skip, Ruff,
compileall, `git diff --check`, stale-reference/placeholder scans, and wheel
inspection proving the deleted modules and dependencies are absent.

## Final trade-model cleanup

- [x] Replace public `TradeProposal`/`TradeTerms`/`CounterOfferTerms` with one
  canonical `TradeOffer` and keep the model-facing input to named resources.
- [x] Rename proposal-board APIs and typed references to offer terminology.
- [x] Migrate replay ingestion and lifecycle projection directly onto
  `TradeWindow`/`TradeOffer`, including incomplete source snapshots.
- [x] Delete `active_trades`, `counter_offers`, `current_trade`, legacy response
  collections, and `sync_legacy_trade_state()` from `GameState`.
- [x] Remove replay/viewer fallbacks and expose semantic who/what offer payloads.
- [x] Update prompts, docs, tests, and lessons without adding compatibility aliases.
- [x] Run focused tests, full tests, Ruff, stale-reference scans, frontend/wheel
  checks, concurrency checks, and the explicit 66-game replay corpus.

Review: The complete player-facing contract is now one `TradeOffer` containing
who offered, audience, named give/receive resources, optional parent offer,
response signals, and lifecycle status. `TradeWindow.offers` is the sole live
and replay engine representation; the replay ledger remains only the exact raw
source record. All legacy GameState trade dictionaries, single-trade fields,
synchronizers, old prompt suites, and proposal/terms contracts were deleted.
The default `catan_v4.yaml` emits a conditional `<trade_offer>` tag and no other
trade tag. Verification passed with 241 tests / 1 gated skip, Ruff,
`git diff --check`, frontend build, wheel inspection, 64-sandbox concurrency,
and all 66 replays (31,506 actions and 15,460 trade lifecycle actions).

## Explicit native reasoning in the playground

- [x] Trace live and replay inference from UI request through OpenRouter response
  parsing, domain contracts, stored decision records, and rendering. Audit found
  that interactive replay still bypasses `AgentPlayer` through the legacy
  `cle/harness/replay.py` prompt path.
- [x] Make native reasoning an explicit request instead of a provider-default
  omission, and preserve returned reasoning content/details plus token metadata
  once at the shared provider and `PlayerChoice` boundaries.
- [x] Rename the shared visible model-authored explanation to `rationale` so it
  cannot be confused with the model's native reasoning channel.
- [x] Replace interactive replay generation with a frozen `PlayerContext` passed
  through the same `AgentPlayer`/`ContextAssembler`/parser used by live games;
  replay retains only cursor, snapshot, and stale-response responsibilities.
- [x] Add controlled playground reasoning settings and record the requested
  configuration with every response.
- [x] Add focused backend and frontend regressions, run static checks, and restart
  the current backend when the unrelated trade-model migration compiles.

Acceptance: a Qwen playground call records the explicit native-reasoning
configuration, preserves the provider-returned native reasoning separately from
the concise visible rationale, and never infers reasoning mode from an XML tag
or an omitted API field.

Review: Native reasoning is now an explicit `off|minimal|low|medium|high|xhigh|max`
condition, with `xhigh` and `exclude=false` as the exploratory default. The
OpenRouter transport preserves raw reasoning, structured details, the exact
request, and token usage in `ModelResponse`/`PlayerChoice`; the UI renders those
separately from `<rationale>`. Interactive replay and the causal action-diff eval
now use the same `catan_v4.yaml`, `PlayerContext`, `AgentPlayer`, context
assembler, and parser as live games. The obsolete replay prompt module, loader,
and YAML suite were deleted; replay-only activity projection moved under
`cle/replay/`.

Verification passed with 242 tests / 1 gated skip, focused Ruff,
`git diff --check`, frontend TypeScript/Vite production build, a 78-decision
action-diff dry run, and a real `qwen/qwen3.8-27b` opening decision. That call
returned action 7 with no parse error, 3,526 native reasoning tokens, 13,999
reasoning characters, one structured reasoning detail, a separate 43-character
rationale, and the recorded request `{"effort":"xhigh","exclude":false}`.
The current-worktree backend is healthy on port 5001 with reload disabled so
concurrent file edits cannot interrupt an in-flight model call.

## Decision spot-check buckets in the eval frontend

- [x] Author one strict, versioned bucket suite with deterministic stage rules,
  multi-label scenario categories, reviewer-assigned failure labels, rubrics,
  episode units, and sample targets.
- [x] Add state-derived decision features and a pure classifier to the replay
  action-diff path; keep classification separate from policy prompting.
- [x] Build a read-only eval artifact adapter that joins manifests, latest model
  responses, comparisons, and bucket tags without mutating replay state.
- [x] Expose bucket catalog, run summaries, filtered decision lists, and decision
  details from the eval backend API.
- [x] Turn the standalone verifier into a two-suite frontend with Board Perception
  and Agent Decisions tabs; render bucket counts, filters, tags, human/model
  choices, rationale, reasoning metadata, legal actions, and prompt context.
- [x] Add strict loader/classifier/API regressions, run a representative replay
  extraction, full Python tests, focused Ruff, frontend build, and backend smoke.

Acceptance: the eval frontend shows the authored decision-category catalog and
real bucketed replay decisions, uses explicit stage evidence rather than replay
position guesses, supports multi-label filtering, and preserves the distinction
between visible rationale and native provider reasoning.

Review: `decision-bucket-suite-v1` defines setup/early/mid/late from engine turn
and public-VP evidence, plus critical pressure, 24 scenario buckets, stable
setup/turn/trade/robber episode IDs, 12 reviewer labels, four quality verdicts,
and balanced sampling targets. The action-diff manifest now records only the
state features needed to reproduce those assignments. A provider-free build
indexed all 164 BLUE-seat decisions in replay 242781000: 133 are in at least one
bucket, 111 have archived Qwen responses, and 60 differ from the recorded human.

The eval UI defaults to Agent Decisions and keeps Board Perception as a second
tab. It shows all categories including zero-count coverage gaps, episode/decision
counts, filters, authored detection/rubric text, human/model choices, local
review controls, legal menus, prompt evidence, and separate visible-rationale
and native-provider-reasoning panels. The read-only API does not load or mutate
the shared viewer replay. Verification passed with 258 tests / 1 gated skip,
focused Ruff, both frontend production builds, the 1,023-file rename receipt,
`git diff --check`, API smoke checks, and headless Chromium checks covering 25
catalog rows, the two early-robber decisions, both frontend tabs, and zero
browser errors. Backend and eval frontend are healthy at ports 5001 and 5174.

## Enable OpenRouter native reasoning by default

- [x] Normalize every shared OpenRouter transport request to an explicit
  reasoning object, defaulting to `xhigh` with returned reasoning included.
- [x] Preserve the explicit `enabled=false` condition in the existing playground
  control and action-diff CLI for user choice and controlled comparisons.
- [x] Retain OpenRouter generation/request identifiers and native finish metadata
  through shared response, player-choice, and replay/eval artifact boundaries.
- [x] Add regressions for default-on, explicit-off, provider metadata, and both
  playground request conditions.
- [x] Run focused tests, Ruff, frontend build, diff checks, and one live Qwen
  default-on smoke request proving native reasoning evidence is returned.

Acceptance: ordinary OpenRouter agent calls cannot silently omit the reasoning
parameter, while an explicit user-selected Off condition remains possible and is
recorded exactly as `{"enabled": false}`.

## Review
The shared OpenRouter transport now normalizes omitted reasoning to
`{"effort":"xhigh","exclude":false}` before constructing the payload; an
explicit Off selection remains `{"enabled":false}` end to end. Provider
response/generation ID, request ID when supplied, and native finish reason now
flow through `ModelResponse`, `PlayerChoice`, replay/action-diff artifacts, and
both reasoning UIs.

A live default-config `qwen/qwen3.8-27b` smoke call returned 51 reasoning tokens,
195 native-reasoning characters, one structured detail, a normal final answer,
and generation ID
`gen-1787638050-JDQGckYQr8ZZYAo5nXfv`. Focused tests passed 40/40; the suite
passes 262 tests with 1 gated skip when the unrelated rename-receipt test is
deselected. Focused first-party Ruff, both frontend builds, and `git diff
--check` pass. Final task completion remains blocked by concurrent
CatanBoardBench receipt drift in `.gitignore`, `index.html`, `src/App.tsx`, and
generated `tsconfig.tsbuildinfo`; full-repository Ruff also reports only
pre-existing vendored/reference findings.

# Single-piece v4: adjacent and far hard negatives (2026-09-03)

## Findings
- Regression panel on the v3 final adapter (single-piece v2 validation, original
  images, first 590 rows): localization 291/295, occupancy/owner misses were
  25 false positives that all named the real piece on an empty spot plus 8
  false negatives. Adjacent empties fail 6/8, far empties 19/139.
- The exporter drew the one empty token uniformly from every other token of
  the same type, so only 5.3% of v3 training negatives touch the piece.

## Plan
- [x] Exporter: derive node and edge neighbors from the contract; emit two
  negatives per image, `occupancy_negative_adjacent` from the piece's
  neighbors and `occupancy_negative_far` from non-neighbors.
- [x] Update tests for the extra row and adjacency classification.
- [x] Export `spatial_localization_v4` with tile rows and 40 placements.
- [x] Add the v4 validation set to the regression panel; document in README.
- [x] Launch a continuation run from the v3 final bundle on v4.
- [x] User asked for more negatives: add cross-type touching negatives and a
  `--negatives` knob (default adjacent=2,cross=1,far=2), export v5, stop the
  v4 run, relaunch on v5.
- [x] User then asked for about 20% "empty" rows: default back to one
  touching and one far negative (25%), export v7, stop the v6 run, relaunch.
- [x] (Superseded) User asked for no cross negatives and a 35/35/30 near/far/rest mix:
  drop cross, rank near negatives touching-first out to three hops, default
  adjacent=7,far=7, export v6 with 10 eval images per board per entity, stop
  the v5 run, relaunch on v6 at 256 steps as a loss and forgetting check.

Stopped within minutes, superseded: run `catan-qwen38-sl-v6-single-piece-near7far7-s256-20260903`,
data identity `cac36091f53f`, app ap-3zD033fPGb5DwN9VCjjiJm, call
fc-01M1MD633VTGD5EBH8T1AQG8QQ, from the v3 final bundle, batch 16 x 2, 256
steps, eval and save every 64. Train 68,800 rows (35% near, 35% far, 5% each
of the six other row types); near negatives are 44% touching, 51% two hops,
5% three hops. Output under
`/runs/catan-vision-sft/catan-qwen38-sl-v6-single-piece-near7far7-s256-20260903/cac36091f53f`.
Gate: `occupancy_positive`, `tile_*`, and `piece_to_token` on v6 validation
must hold near the v3 panel (94.2% / 100% / 100%) while the negatives climb
from 89.0%.

Relaunched 2026-09-03: run `catan-qwen38-sl-v5-single-piece-neg5-s256-20260903`,
data identity `7f1bf048fa32`, app ap-AuE4sQgeHstcUPKdBuGRLW, call
fc-01M1MCQ44YD5HMCRQMKYFJ3V0J, from the v3 final bundle, batch 16 x 2, 256
steps, eval and save every 64. Output under
`/runs/catan-vision-sft/catan-qwen38-sl-v5-single-piece-neg5-s256-20260903/7f1bf048fa32`.
Gate: `occupancy_positive` on v5 validation must hold near the v3 panel's
94.2% while the three negative kinds climb from 89.0%.

Launched 2026-09-03: run `catan-qwen38-sl-v4-single-piece-adjneg-s256-20260903`,
data identity `9394f6a3c699`, call fc-01M1MBCX9M74DY46AB2MM2P27R, batch 16 x 2,
capped at 256 steps, eval and save every 64 steps. Output under
`/runs/catan-vision-sft/catan-qwen38-sl-v4-single-piece-adjneg-s256-20260903/9394f6a3c699`.
A first launch of the same data at one full epoch (app ap-SmkyFaQNTmi5UKJQd46ZJm)
was stopped within minutes; ignore its output directory.
Score checkpoints with the regression panel; the v4 validation set is now in
the panel so adjacent versus far negatives report separately.

# Adjacent-pair curriculum, then sparse, then dense (approved plan 2026-09-03)

Plan file: ~/.claude/plans/async-wandering-phoenix.md. Dense zero-shot table
for the v3 adapter recorded in reports/sft/2026-09-03-qwen38-single-piece-v2.json.

- [x] Step 0: commit today's exporter work and the zero-shot record.
- [ ] Step 1: sft/scripts/analyze_occupancy_misses.py with a fixture test.
- [ ] Step 2: adjacent_pair_localization.py exporter, render_contract and
  _row refactor, tests, evaluator metadata and neighbor_confusion block,
  panel entry, README; export pairs_v1; launch 256 steps from v3 final.
- [ ] Step 3: after the pair gate (positives and negatives >= 98%, no hop-1
  false positives), run production_curriculum_v1 empty/setup + sparse from
  the pair bundle, then dense.
