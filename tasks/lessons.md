# Lessons

- [scope] A request to catch up on denser self-play RL reward signal is research, not authorization to implement decision filtering. Read the reward proposals and later RL lessons first; distinguish unimplemented credit-assignment ideas from board-recognition training and existing runtime inference shortcuts.

- [gotcha] Modal resolves `/runs/...` to `/__modal/volumes/vo-.../...` in trainer
  evidence. Offline identity checks must bind the shared mount through a verified
  dataset source and compare exact relative paths plus content hashes; raw alias
  string equality can reject the correct immutable initialization.

- [sft] Additional SFT steps past the first cosine horizon show diminishing greedy
  returns even as teacher-forced loss keeps falling (extension: loss 0.49 → 0.36,
  held-out +9, review −1 with 25 churned IDs). Report paired improved/regressed
  counts alongside headline accuracy; a flat headline can hide real movement.
- [sft] A cosine schedule restarted with warmup starts at exactly LR 0.0 on the
  first update. Offline schedule-shape checks must allow zero at update 1 only
  and require strictly positive rates after; blanket positivity rejects the
  correct schedule (seen in the 256-step extension audit).
- [sft] LR peak/ramp/decay shape assertions belong on the complete update
  history only. Earlier checkpoints hold truncated warmup prefixes whose partial
  maxima sit at the truncation point; history-prefix continuity already binds
  them to the verified full schedule (seen in the 512-step extension audit).

- [ux] Board controls should be a compact strip: one busy status, stable action
  labels, inline history, and usage only when recorded. Put coverage diagnostics
  behind disclosure rather than stacking empty metrics and nested cards.

- [sft] Frozen tensor invariance is not serialized-file byte identity. A real
  two-step text run preserved all333 FP32 visual tensors bitwise while the
  safetensors file hash changed. Compare canonical names/dtypes/shapes/tensor
  bytes, and retain raw artifact hashes separately for provenance.

- [harness] Deterministic plans need full syntax admission but incremental live
  legality: a road or conversion can unlock a later action absent from the first
  menu. Persist next-action consumption separately from model receipts; own-input
  cursors and notes advance once, not on automatic continuations. Setup pair
  boundaries are explicit even when snake reversal preserves the actor. Detach
  returned provenance as well as snapshots so callers cannot rewrite queued calls.

- [trade] Keep premoves narrow: one admitted exact-offer proposer authorization,
  not responder willingness or a general conditional plan. Wait for the whole
  barrier; use explicit priority/engine seat order, and pause on every arriving
  counteroffer even alongside acceptance. Persist consumption and causal call
  identity separately from model receipts so retries cannot exchange twice or
  duplicate notes/tokens. Normal offers remain probes.

- [communication] Public event delivery is not a reason for inference. Shared
  decisions choose an action OR standalone speech; only explicit respondents and
  the post-discard seven window trigger reactions. Do not reintroduce routine
  actor/after-build polls or speculative setup windows. Skipped polls acknowledge
  nothing; accepted pass can commit notes. Keep one checkpointed conversation
  budget across replies/retries, and never split the atomic Knight bundle.

- [sft] Rank-expansion parity must use the actual text-inference precision:
  BF16 base plus PEFT FP32 adapter arithmetic, without an added autocast context.
  BF16-autocasting LoRA matmuls can produce rank-dependent differences. The real
  r8/r16 probe matched exactly after aligning precision. Temporarily bypass
  Accelerate's AMP forward wrapper for parity probes, then restore it for training.

- [gotcha] Modal FunctionCall.get(timeout=...) raises built-in TimeoutError when
  polling expires; modal.exception.TimeoutError is a different remote-failure
  class. Catch the polling case without treating it as a failed training stage.
  Give heavy-import coordinators enough CPU to avoid slow cold starts.

- [gotcha] Normalize launch plans to JSON-native types before both Modal RPC and
  writing receipts. Integer-keyed histograms become string-keyed on disk and
  otherwise fail immutable-plan equality. Use the selected interpreter's Modal
  CLI flags; the project's CLI lacks the global executable's --yes stop flag.

- [tracing] Keep historical board presentation separate from authoritative runtime
  state and inference settings. Returning live is a client view change, never a
  checkpoint load. Wait for the initial runtime identity before auto-selecting
  saved history; otherwise startup request ordering can silently enter browse mode.
- [tokens] Aggregate canonical recorded call identities, not duplicated result
  attempts. Keep failure-only batches outside completed-step averages, expose
  direction-specific coverage, and never add reasoning/cache subsets to totals.

- [context] Active prompts are runtime configuration, not saved-game continuity.
  Save exact requests/sources as historical evidence, but never restore those
  sources as active policy. Apply edits at safe inference boundaries, retaining
  notes and events and validating mode migrations rather than silently resetting.

- [prompt] Shared fresh requests expose current facts, new visible events and notes,
  not legality-derived answer lists. Strategy is guidance, never labeled facts.
  Keep trade syntax and Knight semantics in stable tool definitions rather than
  burying them in phase-specific prose. Preserve explicit historical requests.
- [testing] When removing visible legal menus, test transports must obtain their
  scripted replies from independent engine fixtures, not scrape stable tool
  definitions. Typed legacy agents must explicitly select their historical suite;
  bypassing model requests cannot satisfy fresh-context provenance checks.
- [events] Fresh channel delivery needs self-contained trade lifecycle terms.
  Capture live terms before mutation and join replay terms only by exact recorded
  source ID; never choose a latest/similar offer to explain a negotiation.

- [sft] Transformers 5.16.1 tokenizer.apply_chat_template returns BatchEncoding
  by default. Native text helpers expecting list[int] must explicitly request
  return_dict=False; list(BatchEncoding) yields field names and silently destroys
  prompt/completion boundaries. Check the actual saved tokenizer in the pinned
  runtime rather than relying on an older local library's defaults.

- [gotcha] Symbolic scorer imports currently traverse board_recognition's package
  initializer into benchmark source helpers and playground.game_viewer.state.
  Remote eval images need the transitive playground Python package and jsonschema
  dependency as well as data_pipeline. Modal retries=0 does not prevent container-import restart loops;
  stop a failed-start app before relaunching a corrected CPU preflight.

- [curriculum] Structural queries and deterministic game calculations belong to
  one board-fluency class: derive facts from explicit state using atlas relations,
  rules, and weights. Organize datasets by operation/composition (joins, sets,
  weighted aggregation, connectivity, constraints), not an artificial spatial
  versus game-calculation split. Strategic judgment belongs to self-play RL.

- [arch] Group atlas fluency, state/topology composition, and configuration
  reasoning into board fluency alongside deterministic game calculations. The
  user targets immediate answers for rule-constrained computations;
  distinguish reasoning complexity from inference latency and do not assume
  these computations must involve explicit deliberation or generated search.

- [arch] For the symbolic board-token capability hierarchy, current ownership and
  occupation are explicitly supplied as atlas-token/state records. Treat these
  facts as the input contract, not a perception or retrieval capability to learn.
  Separate any implicit atlas-topology knowledge from composition over supplied
  state; prioritize this symbolic direction rather than reintroducing vision.

- [workflow] When the user chooses human verification, stop automated test runs
  and test expansion immediately. Report the implementation and known limitations
  briefly; do not spend more turns on verification machinery.

- [workflow] For dataset exploration, deliver inspectable questions and answers
  before expanding trainer/test infrastructure. The user explicitly stopped
  further pytest work on the symbolic atlas task and requested the generated set.

- [testing] Prefer a small set of high-value integration and contract tests over
  large overlapping helper/validation matrices. The user explicitly requested
  pruning the prompt/notes test expansion; keep unrelated tests untouched.

- [context] A speech acknowledgment is not action-context delivery. Keep separate
  channel cutoffs, bind updates to the exact input context before mutation, and
  preserve accepted silence/notes in failure snapshots even at the same revision.
- [tracing] Resuming after accepted speech must preserve both its canonical event
  index and the accepted action's model call on post-action cancellation. Preserve
  asyncio cancellation in the core while carrying the committed result across
  the synchronous viewer bridge explicitly.

- [eval] For the symbolic atlas experiment, the user selected settlement and
  Longest Road as transfer tests, not SFT targets. Distinguish rule recall from
  applying spatial constraints; state rules in transfer prompts and exclude
  equivalent full-rule training examples disguised as explicit predicates.
  Road-state inputs must omit computed lengths, award answers, and legal lists.

- [prompt] Build shared authored prompt components first. Reuse definitions by
  reference across decision and speech compositions; neither one global section
  order nor a provider message count is part of the component contract.
- [context] The selected first implementation is fresh context plus explicit
  private notes and new visible events. Defer the long-context alternative;
  do not carry forward an accidental option selection or replay raw reasoning.

- [arch] For the September 12 spatial-SFT direction, retain the user's existing
  checkpoint and atlas token IDs while moving to symbolic board-state inputs.
  Prioritize text/graph supervision resources over visual grounding recipes.
  An out-of-the-box longest-road failure does not isolate perception from graph
  reasoning; do not treat it as proof that vision is intrinsically useless.

- [workflow] Resolve a stop request against the active experiment's receipt and
  app ID, including a user's RL-to-SFT correction. Stop the coordinator as well
  as the training worker so queued evaluation cannot start; verify zero tasks.

- [context] A large request-message count is not by itself the architecture
  defect: identify whether prior cumulative decision packets duplicate the
  current authoritative event history. Do not prescribe an arbitrary last-N
  transcript window or switch to snapshot-only inference before confirming the
  intended bounded versus non-duplicating long-context path.

- [debugging] HTTPX's status exception omits the provider response body. A 403
  cannot establish which permission, guardrail, or moderation check failed.
  A successful non-inference key check does not prove inference authorization;
  retain a bounded credential-redacted structured reason without retrying 403s.
- [restore] Public/checkpoint equality does not prove all failure evidence is
  durable. Barrier siblings can exist only in decision_trace, which is not in
  sandbox snapshots. If no resident-process export exists, obtain consent before
  restarting and potentially discarding those unpersisted calls.

- [debugging] A reasoning-only OpenRouter response can finish with `stop` and no
  completion cap. Inspect raw content/reasoning and both finish reasons before
  blaming token exhaustion, invalid action arguments, or the earlier TLS error.
  Report missing final output distinctly; never execute action-shaped reasoning.
- [restore] Recheck the current checkpoint's suite at each live investigation.
  A previously restored legacy game can later reload using current defaults if
  its old config has no suite source pin; yesterday's verified version is not
  evidence of today's active contract.
- [restore] A rejected decision can follow successful pre-action SILENCE that
  acknowledges events without adding gameplay or history. Compare live player
  cursors to the last checkpoint before restarting, and preserve only verified
  acknowledgments rather than silently treating the DB as the full live state.

- [scoring] Touching-tile answers are unordered exact sets; shortest paths are
  ordered routes and every equally short valid route must pass. Dispatch by task
  before generic readout parsing so long node lists cannot lose their order.
- [curriculum] Allocate the approved spatial/readout mix by interleaved optimizer
  steps and report token exposure separately. Completion-token share is not
  gradient share or a guarantee against forgetting; use matched retention evals.

- [debugging] Inspect the active process log before prioritizing hypothetical
  autoplay failures. A null last_live_step_error and empty live_failures do not
  exclude raw provider exceptions that only reached the generic server log.
- [network] Retry a classified TLS record failure without weakening certificate
  verification or closing a shared client underneath other players. Bound the
  retries and distinguish exactly-once gameplay from possibly duplicated billed
  inference after an upstream response is lost.

- [eval] Whole-answer exact matching needs an explicit output contract. Before
  interpreting low spatial accuracy, check whether yes/no and two-token-choice
  prompts actually request bare answers. Preserve the original result and use
  a matched format-only control rather than retroactively awarding prefix credit.
- [sft] FP32 training checkpoints must be restored into FP32 visual parameters
  before evaluation, not copied into BF16 and promoted afterward. Match the
  validated generation autocast recipe and record source/runtime tensor dtypes.

- [api] Semantic action handlers and provider-native function calling are
  different layers. When the user asks for executable tools, clarify whether
  they expect native tools/tool_calls and tool-result messages or a parsed
  text-response protocol before choosing the transport. Do not present JSON
  emitted as ordinary assistant text as native function calling.

- [data] In two-choice spatial QA, inverse relations do not remove answer-position
  leakage if each prompt still lists its answer first. Balance displayed choices
  explicitly and test the rendered prompt, answer, and sampled blocks. Regenerate
  derived data under new names so historical eval prompts remain reproducible.

- [tokens] In this project, the user's angle-bracket spatial notation refers to
  the trained atlas vocabulary (<N00>, <E00_01>, <T00>), not XML control tags.
  Preserve those exact strings in tool arguments; do not replace them with raw
  integer IDs, sequential edge IDs, or escaped token spellings.
- [api] When expanding the model tool surface, use semantic operations across
  action families rather than assuming a three-tool exception is the goal.
  A Knight destination belongs in the same model decision when requested;
  preserve the engine's canonical transitions and victory boundary underneath.
- [harness] Separate model-facing action syntax from engine legality. For
  parameterized resource tools, resolve named bundles against the authoritative
  menu rather than exposing every combination or weakening live validation.
- [review] Check implemented behavior before proposing contract migrations:
  v10 already parameterizes discards, and semantic tool calls do not require
  changing replay actions or old datasets when canonical internal choices stay
  intact. Do not infer retry causes or training behavior without measurements.

- [harness] The sandbox must own a detached canonical context and give each
  callback its own copy; detach returned attempts before awaiting siblings.
  Stage complete trade batches against strict engine validation, retain
  completed-but-withheld replies, and commit before invoking player callbacks.
- [parsing] Validate the complete control-field XML structure before extracting
  actions or speech. Comments and inert plans cannot supply controls, and trusted
  schema-echo normalization must receive only the exact authored schema template.
- [restore] Repair stale derived caches in concrete old snapshots, but retain
  indexed menu ordering when regenerated action identities are the same multiset.
  A saved incumbent cannot reveal historically incorrect tie ownership; do not
  claim to reconstruct history from current material state alone.
- [replay] Source reconstruction must not invent cancellation or turn actions
  while recovering an unmatched recorded build. Publish source response snapshots
  consistently with the reconstructed offer board, and keep source completion
  distinct from live score-based termination even in read-only previews.

- [harness] Barrier failure must retain completed sibling calls as withheld,
  not just the failing actor's rejected attempts. Remove a pending attempt on
  rejection and replace it after retry so no call is lost or recorded twice.
- [privacy] Trusted schema-echo parsing must receive authored response-schema
  text, never a rendered prompt containing player messages or other dynamic data.

- [verification] Conservation and replay reconstruction are not independent
  Catan rule oracles. Check blocked-road legality, award minimum/tie ownership,
  own-turn victory, shortages, and player choice separately before trusting
  full-game outcomes as training labels.
- [testing] Known-defect audit xfails must catch a dedicated exception raised
  only after normal fixture/precondition assertions. Strict xfail alone can
  hide a broken setup. Count failing cases separately from unique bugs and
  proposed extension-boundary policy, and label reduced versus reachable states.
- [arch] Checkpoint completeness includes causal events, RNG binding, and
  commitments, not just material GameState. Freeze or detach nested mutable
  observations and event payloads; frozen outer dataclasses do not isolate them.

- [harness] Treat menu IDs as opaque and parse only explicit whole action
  indices, never numbers in plans or historical rationale. Validate authored
  template references before substitution so model text remains inert data.
- [trade] Independently legal concurrent replies need batch admission against
  shared trade capacity before any live mutation. Unresolved wildcard bundles
  are negotiable proposals, not executable resource exchanges.
- [harness] Commit accepted agent history immediately after engine application,
  before optional speech awaits. Persist rejected/withheld communication as
  such, distinguish post-action warnings from unapplied failures, and stop
  autoplay on runtime WebSocket warnings as well as HTTP errors.
- [privacy] An explicit but malformed audience tag is not a missing audience.
  Whitespace, self-closing/duplicate tags, unknown colors, and self-only private
  recipients must never silently become public broadcasts.

- [gotcha] Live public development-card payloads intentionally omit `in_hand` and expose only `total_in_hand` plus public `played` counts. Frontend rendering must treat the exact breakdown as optional, show only the total when redacted, and test the first nonzero hidden-card case; zero totals can otherwise mask an `Object.entries(undefined)` crash.

- [workflow] A stateful live-game server must not enable Werkzeug's source reloader by default. Any watched frontend, test, or documentation edit can restart the process, erase the in-memory sandbox, and surface as a transient browser `Failed to fetch`. Make reload explicitly opt-in, restart intentionally, then restore the latest persisted checkpoint before handing control back.

- [workflow] Treat a Catan supply audit as a complete finite-inventory audit: resource cards, development cards, roads, settlements, and cities. If the user names only part of that set, explicitly report the unmentioned finite pools rather than silently limiting verification to the examples.

- [data] Do not treat `artifacts/` as a coherent canonical data root merely because its README assigns categories. Audit the live tree first: it mixes raw inputs, generated corpora, eval derivatives, fixtures, diagnostics, migration receipts, visualizations, staging, and provider runs. Design canonical dataset ownership separately from ephemeral experiment evidence.

- [arch] When consolidating package ownership, remove compatibility wrapper packages and import aliases unless an identified external caller explicitly requires them. A clean hard move should leave one canonical namespace, not forwarding shells at the old boundaries.

- [sft] A pinned community finetuning repository is not an official model-family trainer merely because it supports the architecture. Name its provenance explicitly and revalidate chat-role conversion, loss masking, trainable scope, and checkpoint restoration against the official implementation before a paid run.
- [vision] For full vision-tower adaptation, repeated atomic questions over one rendered board increase supervision but not pixel diversity. Budget unique images first, then choose the smallest queries-per-image ratio that still covers every head, class, polarity, and slot; do not treat 16 rows from one image as equivalent to 16 visual examples.
- [data] When admitting replay sources for a visual board dataset, gate on exact rendered board reconstruction rather than the raw count of replay diagnostics. Explicitly allowlist only nonvisual trade-bookkeeping and final score/award sync records; reject any error or diagnostic that can change tiles, node occupancy, edge ownership, ports, or robber position.
- [curriculum] When the engine and archived replays can generate legal board states, use those states as the primary recognition corpus instead of handcrafted phase proxies. Keep density bins as sampling/audit metadata, not model input or an unimplemented stage schedule.

- [process] When discussing a new vision curriculum, explicitly distinguish the source specification, generator, generated training corpus, dataloader, SFT projection, and frozen-model evaluation before spending on hosted runs. A benchmark rerender is not a training dataset; confirm the requested artifact and trainer interface first.
- [curriculum] For direct board recognition, store one dense engine-labeled payload per raw board image, split counterfactual groups before sampling queries, and sample hierarchically by entity type → attribute → slot so 72 edges do not silently dominate 9 ports. Validate that engine contract, dense labels, and rendered pixels reproduce each other and that one-slot pairs change exactly one declared label.
- [eval] The simple text-to-vision projection is: load the engine public-state contract, invert any text-only alias permutation back to canonical engine IDs, render the ordinary engine board screenshot, and retain JSON only as the answer oracle. Do not add a second coordinate-annotation representation.
- [eval] Do not invent a new annotated/coordinate-atlas image projection when unifying an existing text benchmark with vision. Reuse the established raw board images and identity contract; if opaque permuted IDs make the full question set impossible from raw pixels, surface that incompatibility before generating artifacts or spending on API calls.
- [eval] A unified current-model benchmark should not carry forward old Qwen checkpoints as headline comparators. When replacing a model generation, rerun the shared image/text cohort with the current checkpoint and keep historical artifacts out of the active scorecard unless explicitly requested.
- [eval] Before launching a hosted vision sweep, identify every existing benchmark cohort the user expects to compare—especially the text-format cohort—and run the same model set on that shared question set. Publish one unified report rather than treating visual-only and text-only evaluations as separate conclusions.
- [process] When the user chooses a hosted VLM image-family sweep, do not translate the budget question into Modal/GPU training. Use the configured hosted APIs, label the result as a complete-stack comparison rather than a vision-backbone isolation, and ask separately before introducing local or Modal compute.
- [eval] When the user asks whether a hosted model can already perceive a new render, run the existing no-reasoning OpenRouter benchmark before building a direct vision-feature extraction stack. Treat the hosted result as an end-to-end capability check, not an isolated vision-tower measurement.
- [vision] When the target is immediate Catan perception, do not turn vision-tower adaptation into a reasoning or autoregressive board-derendering task. Train image-plus-query to a first-position class or one atomic answer token, so success requires the queried visual fact to be directly readable without generated scratch tokens.
- [ux] Do not equate one replay-action interval with a complete transcript view. Before calling transcript display complete, audit source-caption coverage across the full video and provide enough history/navigation that dense action boundaries do not make later commentary appear missing.
- [eval] Do not infer an internal spatial heuristic from an end-to-end direction miss. A full-board question conflates ID lookup, coordinate extraction, vector subtraction, convention mapping, and layout reading. Localize the failure with staged probes before calling it a genuine orientation-binding defect.
- [eval] A spatial sidecar does not test spatial piece grounding when player buildings and roads remain only in appended flat records. Distinguish semantic presence from visual integration: render dynamic occupation directly at node/edge positions, then compare against a length-matched flat encoding so the model cannot ignore the diagram.
- [eval] Do not call a representation benchmark spatial merely because one section is arranged like a board. Require balanced direction/topology questions, strict tuple/set scoring that rejects extras, and output schemas that request every scored field. Separate information-equivalent renderings from information ablations; otherwise syntax copying, positive-only selection, and permissive component presence can masquerade as spatial competence.
- [research] When asked for comparable Catan tools in a learning-agent project, do not lead with static pip calculators as though they were meaningful AI competitors. First classify deterministic calculators, search/simulation agents, learned policies, and user-facing products, then surface systems at the requested technical depth.
- [arch] For replay-agent prompts, make the harness/model boundary a versioned decision packet. Keep harness-derived observation, visible history, and legal actions authoritative; keep model-authored goals explicitly fallible and bounded.
- [gotcha] Arbitrary replay jumps must not reuse strategic memory produced at a later cursor, or future information leaks into the decision.
- [pattern] Measure canonical replay token lengths before designing summarization or choosing a long-context model; raw replay JSON size is not representative of a compact event stream.
- [cost] Compare models using actual per-decision token counts, not only per-million-token sticker prices. The compact replay packets measured roughly 0.8k–5k input tokens, so even a premium open model can cost well under one cent per move.
- [research] Before calling a model comparison current, verify the session date and inspect the live provider catalog for newly released families; distinguish a representative historical size chart from an exhaustive up-to-date catalog.
- [research] Optimize model choices for the user's learning and experimental control, not only dollar cost or benchmark strength. Use frontier hosted models as teachers/baselines while keeping a tractable open student for hands-on training, probing, and serving.
- [research] Do not assume Qwen or any other student model before the model-selection experiment is complete. Choose the base model first, then retain its native chat template unless evidence justifies template retraining.
- [arch] Do not transfer CICERO's planning setup to Catan without accounting for information structure. Diplomacy's board state is public, while Catan has persistent hidden hands/dev cards and chance; Catan policy execution must consume player-view observations plus beliefs, never omniscient engine state.
- [ui] When a user asks to change visible reasoning borders, implementing only a backend abstraction is incomplete. Trace the feature through API state and frontend rendering, then verify the displayed behavior—not merely core tests.
- [arch] Replay timestamps and engine rows enforce causal availability, but they do not define human reasoning boundaries. Reflow captions globally, then let a model assemble coherent decision/public-observation packets that may span several engine events. In strict-causal mode, keep subject and availability cursors equal, insert bounded observation checkpoints between narrator decisions, and use event reveals only to confirm, split, or close episodes.
- [process] When a user casually says to "use/run a parser" or another tool, confirm whether they mean an external one-off tool or repository implementation before writing code. Do not convert an exploratory suggestion into a feature build without checking.
- [process] In a dirty repository, never use `git checkout -- <file>` to undo my draft unless I captured the file's exact pre-edit contents or baseline diff first. Session summaries distinguish assistant-touched files, not necessarily all pre-existing user changes. Undo only my exact hunks; if a broad revert already happened, restore the saved full patch and subtract only those hunks.
- [process] Observation/formatter schemas are eval and training-data contracts shared across pipelines. Propose and agree on schema changes before editing; never hot-edit a shared formatter mid-discussion, especially with an auto-reloading server attached.
- [arch] Keep the Catan engine primarily as referee and ground-truth environment, not a strategic copilot. The learned agent should internalize topology, production, beliefs, valuation, planning, and negotiation; engine validation prevents illegal execution without outsourcing the Catan intelligence being studied.
- [workflow] Run the local replay corpus once per meaningful code state. Put additional invariants (trade lifecycle, undo, alternate navigation) into one consolidated regression harness instead of repeatedly launching ad-hoc Python scans over the same files.
- [workflow] Present the requested audit status and evidence before asking the user to choose an implementation scope; do not jump from investigation straight to a fix-plan questionnaire.
- [pattern] Replay decision history must state public dice-roll payouts explicitly. A post-action hand snapshot or opponent card total preserves state but loses the causal public information needed for resource tracking; derive only event-scoped positive deltas and never expose full hidden hands.
- [ux] Keep compound replay actions as one logical history row, but render public sub-events such as roll payouts on indented lines. Preserve those newlines in both the model packet and UI so readability does not change source-row accounting.
- [gotcha] Colonist trade events are delta-encoded and nested response dictionaries must be deep-copied before merging; a shallow copy silently mutates the raw replay oracle and invalidates lifecycle audits.
- [pattern] Keep an exact replay trade ledger keyed by source `trade_id` and treat color-keyed engine trade dictionaries as compatibility projections only. Rebuild projection response sets from the exact ledger rather than applying deltas additively.
- [gotcha] A synthetic lifecycle row inside an atomic source event has no authoritative intermediate hand snapshot when the same event also buys/builds/trades. Validate resources on the actual resource-changing action, not the synthetic closure row.
- [workflow] For authenticated third-party replay acquisition, prioritize account safety over throughput: use small paced chunks with at least 40 seconds between Colonist replay attempts, stop immediately on the first 429/Retry-After, cool down before any retry, and never turn an access/rate-limit response into a tight automated retry loop.
- [arch] When designing the Catan agent harness, separate the environment's continuous perspective-filtered event stream from the cadence of expensive model inference. Do not reduce the question to trade-window limits; define how events accumulate, become decision packets, update memory, and trigger actions or reactions.
- [ux] Keep decision and event data canonical and typed inside the harness, but render the policy-facing experience as compact natural Catan language. Cursor IDs, schema metadata, and JSON transport fields are for replay/auditing, not necessarily model tokens; retain structure only at the validated tool-call boundary.
- [research] When auditing an external environment, distinguish each seat's actual source-code path rather than generalizing a README summary. In Antim's public Catan setup, only the LLM seat uses `random.choice` for initial placement; the opponent advances through its configured bot via `game.play_tick()`.
- [verification] Recheck time-sensitive model-release claims at response time against the official repository/API, not a prior-day article or cached search result. Record the exact check time and distinguish a private placeholder from publicly downloadable weights.
- [research] Do not equate fine-tuning broadly with instruction SFT. First identify the adaptation goal: use continued/domain-adaptive pretraining for internalizing corpus knowledge, SFT for behavior and response format, and retrieval for precise or frequently changing facts.
- [research] For the Catan specialist, do not optimize for broad world-knowledge retention when the user explicitly accepts forgetting. Preserve and evaluate only task-instrumental capabilities such as language, arithmetic, planning, rule application, and the action interface.
- [research] Do not recommend specialized-pretraining-from-scratch results as if they were guidance for adapting an already pretrained checkpoint. For this project, distinguish early pretraining/SPT from post-hoc CPT and from knowledge-focused SFT.
- [research] The Catan adaptation target is not merely factual knowledge injection. It is a domain-specific reasoning and policy prior: teach Catan terminology, strategic abstractions, and usable reasoning trajectories well enough to bootstrap informative self-play.
- [research] Default to specializing a strong open-weight post-trained reasoning/instruct VLM, not rebuilding reasoning and instruction behavior from a raw base checkpoint. Treat raw-base training as an ablation unless there is evidence that post-training has made the chosen model too rigid or poorly matched to the Catan interface.
- [research] For Catan self-play, optimize model choice for adapted policy quality per rollout compute, not parameter count alone. Start with a dense 9B student and use larger models as offline teachers unless a matched Catan reasoning ablation demonstrates a qualitative 9B capacity ceiling.
- [research] Compare MoE models by both total and active parameters. A 35B-A3B checkpoint may have modest inference compute but still carries roughly 35B parameters in memory and adds expert-routing complexity during adaptation; it is not equivalent to a dense 30B policy.
- [research] The intended Catan teacher is a frontier proprietary model such as Claude Fable 5 or GPT-5.6, not a 27B/35B open-weight model. Use the frontier model to generate and critique concise reasoning traces, while the engine and perspective-safe replay state remain the factual and legality oracle.
- [research] Distinguish the cheapest model for validating the training loop from the highest-ceiling final policy. Qwen3.8-27B is the intended learned Catan policy because nuanced belief tracking and long-horizon reasoning are part of the target behavior; do not casually replace its on-policy rollouts with a compact student, which would optimize a different policy and alter the rollout distribution. A smaller model remains only an explicit ablation or engineering smoke-test option.
- [research] Treat SDFT as a serious cold-start alternative to vanilla reasoning SFT: put the engine-verified expert action and frontier-generated Catan explanation in teacher-only privileged context, then distill corrections on the student's own rollouts. The external frontier model supplies demonstrations, not logits; SDFT's actual teacher is the student/EMA checkpoint conditioned on those demonstrations.
- [research] Classify GRPO by where its actions come from, not where prompts live: freshly sampling the current/frozen-old policy against the engine is online near-on-policy RL even when anchor states or queries come from a fixed dataset. Logged trajectories reused without fresh policy sampling are offline.
- [arch] Do not apply TL-GRPO's name or max-over-turn objective directly to Catan. TL-GRPO assumes a fixed single-state evaluator and independently scored proposals; Catan is a stochastic partially observed multi-agent Markov game. Same-state Catan branching with continuation returns is a counterfactual decision-state or tree/grouped GRPO variant.
- [gotcha] The engine now owns per-game RNG and copies it with state; ordinary
  snapshot continuation tests pass. Replay undo/goto and shared mutable event
  payloads still break full branch equivalence. Verify complete causal state
  and event-keyed common-random-number semantics, not only RNG ownership.
- [rl] A huge continuation-outcome space is not itself the statistical problem; Monte Carlo estimates expectations without enumerating outcomes. The real problem is high return variance relative to small action-value gaps. In Catan, one or a few terminal branch rollouts can mostly label dice/opponent luck, so use multiple scenario seeds or uncertainty-aware filtering rather than assuming `K=1` terminal branches are informative; do not assume a learned value bootstrap until it wins a held-out ranking/calibration ablation.
- [research] “Value model” covers materially different mechanisms. Keep separate: a training-only action-independent critic baseline, a search leaf evaluator, a learned reward/shaping model, and a deployed action-ranking oracle. Evidence that one role works does not validate the others. Catan’s closest precedents show value usefulness only inside specific systems and restricted rules, not a trustworthy universal scalar oracle.
- [process] When the user points to Qwen3-VL video understanding, do not collapse the idea into an assumed 8B checkpoint. Separate the family capability from checkpoint size and inspect the supplied capability source before evaluating the design.
- [verification] Never call an assistant or vision-model board-state reading "confirmed." Model agreement and resource-animation interpretation are corroboration, not proof; keep coordinates training-ineligible until verified by an authoritative replay/log or an independently reviewed deterministic pixel-to-board mapping.
- [data] A historical Colonist game ID or index row does not imply that its replay payload remains retrievable. Never design retrospective video matching around fetching expired games; first-class pairing requires an already archived raw replay or prospective replay capture while the game is still available.
- [data] A verified YouTube-creator-to-Colonist-account mapping converts replay/video pairing from global search into bounded account-anchored capture. Preserve identity evidence and username history, archive still-available payloads immediately, and require board/event verification before promoting any pair.
- [verification] Establish the narrator's SEAT before building any supervision from a paired video: the username used to find a replay is not necessarily the narrator, and a trace can pass quote and cutoff gates while attributing one player's deliberation to another player's action. Require an actor-attribution gate (narrator deliberation supervises only the narrator's seat; other-seat events become observer commentary) and confirm the seat from verbal commits matched to that seat's replay actions before generating traces.
- [arch] When evaluating a vision-head analogy, separate the modality from the adaptation recipe. A frozen tower plus small projector into a frozen reasoning LLM is equally applicable to exact structured board entities; compare literal vision and board towers through one shared soft-token interface rather than framing the choice as VLM versus standalone GNN policy.
- [curriculum] For engine-generated Catan vision alignment, organize the primary curriculum by board-state complexity and game phase (setup board → initial placements → sparse midgame → dense endgame), not arbitrary renderer-mixture percentages. Treat format variation as a separate paired-view robustness axis, retain earlier stages during later training, and split evaluation by whole board state before expanding it into questions.
- [gotcha] When linking a paired replay for the narrator, set `playerColor` to the narrator's verified seat rather than the perspective used by an archived capture. Preserve the archived capture URL separately as provenance when those perspectives differ.
- [data] Never silently repair a commentator's possibly mistaken board reference. Preserve the verbatim utterance, bind engine-verifiable facts separately, and store any proposed interpretation as a provenance-bearing hypothesis with confidence and alternatives. Exclude unresolved spans from positive reasoning supervision or use them as explicit error-detection examples.
- [data] Do not replace continuous commentary with mutually exclusive heuristic categories. Keep one faithful, non-overlapping cleaned transcript as the canonical readable layer; attach overlapping speech-act, decision-link, temporal, and grounding annotations by span ID. Categories are optional retrieval/UI projections, and shared evidence should be referenced rather than duplicated as a “shared bridge.”
- [arch] Multi-agent Catan requires three explicit channels: private strategy memory, public timestamped communication, and authoritative structured engine actions/offers. YouTube narration can supervise private explanation or criticism, but it is not opponent-facing dialogue. Train whether/when to speak separately from action choice, wake recipients for actionable messages, and evaluate cross-model communication without exposing private chain-of-thought or hidden state.
- [arch] Human-like table talk is an asynchronous event-reaction process, not a byproduct emitted only at action turns. Every public event should update each agent and pass through a low-cost reaction gate that usually returns silence but can schedule a causally timed public utterance; dialogue events can trigger further bounded reactions. Preserve trigger event, visibility cutoff, latency, audience, and interrupted/cancelled status in training data.
- [data] Normalize player references to per-game canonical color identities in model-facing transcripts, events, prompts, and table talk (`SELF`, `BLUE`, `BLACK`, etc.), while retaining usernames only in private provenance. Resolve aliases such as usernames, UI seat numbers, “he,” and creator-specific nicknames to colors with confidence; do not replace uncertain mentions. Split by game so a username/color pairing cannot become a cross-game shortcut.
- [perf] For human-like persistent Catan agents, do not force one evolving public decoder-KV trunk with ephemeral private branches: branch non-mergeability discards each seat's continuous private interpretation and memory. Use one shared authoritative world state and, where supported, one cached board/vision encoding, but maintain persistent per-seat decoder sessions after the initial common prefix. Append each public delta to all seat caches in a batch; weights and source state are shared, while post-divergence KV and delta compute are necessarily per seat.
- [perf] Treat NVFP4 as a Blackwell-native rollout/deployment format, not the canonical trainable checkpoint. Add and train atlas embedding/LM-head rows against a high-precision or supported QLoRA base, merge them, then quantize with those rows plus vision and sensitive attention/SSM paths protected; admit the quant only after Catan-specific long-history, visual, action-logit, and fixed-league comparisons against FP8.
- [arch] Distinguish trajectory conditioning from its KV implementation: retaining the full per-seat action-observation-dialogue history creates a game-tracking agent; KV caching merely makes that exact long-context prompt efficient. Repeated snapshot + goals prompts instead approximate a spot policy with a lossy belief summary. For hidden-information/table-talk Catan, retain the game-long trajectory, append authoritative current-state checkpoints at decisions, and use explicit goals as attention anchors—not as replacements for history or as mandatory reset boundaries.
- [perf] For hybrid Gated DeltaNet serving, do not estimate concurrency from per-token attention KV alone. Qwen3.8-27B has 65.5 KB/token BF16 or 32.8 KB/token FP8 attention KV, but SGLang reports a 153.9 MB FP32 recurrent/conv state slot and may reserve five slots per request by default (four with lazy radix caching, one with radix caching disabled). Multiply both token KV and state-slot policy by four persistent seats before selecting hardware.
- [rl] TEMPO's implicit BPE-prefix tree is a fixed-prompt, single-completion credit estimator, not an agentic rollout scheduler. Do not transfer it directly to stochastic multi-turn Catan: exact lexical forks need not be strategic forks, unique children reduce the empirical branch value to one noisy terminal return, and environment observations change the information state. Any Catan adaptation must key semantic decision nodes by perspective-safe history and legal action, require multiple supported descendants, and beat episode GRPO/RLOO in an equal-rollout ablation.
- [rl] Pausing an environment at an update boundary and resuming it under the new policy is not inherently invalid; fixed-horizon PPO routinely continues environments across updates. The failure appears when terminal-return trajectories, PPO denominators, or model recurrent/KV state silently span policy versions: record the exact behavior version/logprobs per action, never reuse KV or DeltaNet state after a weight change, and either pin terminal-only Catan episodes to one adapter version or use explicitly bounded off-policy correction. Also stagger game phases so synchronized short rollout chunks do not create cyclical early/mid/late state-distribution waves.
- [rl] Frame off-policy bias and variance in Catan terms. `R(game)` is one terminal score; `J(pi)` is the expected score of learner policy `pi` against a specified opponent league, seat mix, and board distribution. A replay generated by behavior policy `mu` directly estimates `J(mu)`, not `J(pi)`; treating it as current-policy data creates systematic bias. Exact trajectory importance ratios can correct this in principle, but multiplying ratios across a long sparse-reward game has extreme variance; clipping, truncation, bootstrapping, and bounded lag intentionally exchange some bias for stability. Classical Q-learning tolerates large replay buffers mainly because stationary one-step `(state, action, reward, next_state)` transitions remain environment facts and Bellman targets are recomputed, not because stale whole-episode returns are harmless. Our LLM policy-gradient learner is unusually sensitive because long token/action trajectories, terminal rewards, mixed-policy continuations, and trainer-rollouter logprob differences directly perturb PPO ratios. Therefore pin terminal-only games to one policy adapter when practical; otherwise record exact behavior versions/logprobs and tightly bound staleness.
- [rl] Preserve Beren Millidge's 2026-07-26 essay, “How Can LLM RL Work Despite Information-Theoretic Inefficiency?”, as a useful speculative frame rather than a theorem. Its project-relevant thesis is that pretraining/SFT provides many token-level bits about a broad proxy objective, whereas terminal RL provides few but highly task-aligned bits; RL can therefore make rapid narrow gains only when the pretrained/mid-trained policy is already close enough to succeed at nontrivial pass@k. For Catan, use board grounding plus verified SFT/SDFT to put legal strategic behavior in support, then use engine-outcome RL to select and sharpen it rather than expecting sparse reward to invent rules, representations, or planning. Prefer variance reduction that preserves the objective—verified rewards, stratified boards/seats, fixed opponent leagues, grouped rollouts, larger effective batches, and conservative learning rates—and require evidence before adding biased critics, PRMs, replay, or shaping. Keep the caveats explicit: “bits,” SNR, and nested-loss-valley language are heuristic; next-token training is unbiased only for its own objective; and massive replay is chiefly a property of suitable Bellman/value methods, not classical RL in general. Source: https://www.beren.io/2026-07-26-How-Can-LLM-RL-Work-Despite-Information-Theoretic-Inefficiency/
- [gotcha] Qwen3.8's `reasoning_effort` is a chat-template elicitation control, not a hard token budget: `xhigh` and `low` inject different system instructions, `medium` injects none, and actual compute still depends on generation limits or a two-call early-stop scheme. Because `xhigh` is the default, explicitly set effort per Catan event instead of assuming omitted parameters are cheap.
- [gotcha] Do not infer playground reasoning settings from an eval-only wrapper. `evals/replay_action_diff.py` explicitly disables native Qwen reasoning, while live and interactive replay playground requests omit the `reasoning` field unless the caller supplies it, which delegates behavior to the provider default. Keep native reasoning controls distinct from the visible XML rationale requested by the prompt.
- [arch] Native reasoning is a shared provider/response concern, and the visible game plan plus rationale is a shared `AgentPlayer` contract. Do not introduce replay-specific reasoning semantics or prompt changes: replay may freeze and project state, but it must invoke the same general player harness as a live game.
- [arch] Qwen3.8 reverses older Qwen3's default reasoning-history policy: `preserve_thinking=True` retains all prior private reasoning, whereas false strips completed reasoning before the latest user query while keeping final answers. Compare preserved per-seat cache against structured strategic-memory compaction; never assume longer preserved reasoning is current truth, and append authoritative engine checkpoints either way.
- [arch] Use vLLM as the primary Qwen3.8 serving and rollout API, not as the trainer. Keep conversation history and authoritative state in the Catan harness and communicate through the OpenAI-compatible boundary. Do not hard-couple game logic to vLLM internals: Gated DeltaNet prefix caching is still evolving and must pass a real hit-rate/rebuild benchmark before persistent-session efficiency is assumed.
- [eval] Full-game policy diffs must score each response against the exact legal menu stored with that provider call, not a menu regenerated in another process. Engine actions sourced from sets can reorder under a new hash seed; preserve raw menus/prompts, compare semantic action identities, and use an order-invariant manifest hash for resume safety.
- [gotcha] An OpenRouter Qwen3.8 call can spend the full completion budget on hidden reasoning and return no action even when the prompt requires XML. Smoke-test the exact policy packet and set an explicit reasoning control; retain robust parsing for echoed schema tags and explicit named-index fallbacks without resampling until the model agrees.
- [pattern] Colonist can encode a robber move and steal in one raw event while the parser emits `STEAL` before `MOVE_ROBBER`. For causal policy evaluation only, canonicalize adjacent same-player, same-event pairs to move then steal; preserve both original replay-row and raw-event provenance.
- [format] The user is reading this project in a terminal. Use fenced plain-text equations with ASCII operators, keep authored lines near 80 columns, and avoid LaTeX/display-math delimiters unless the user explicitly requests them. Do not misread a reported “ASCII 94%” result as terminal width; it refers to the ASCII board-state representation's benchmark score.
- [arch] Reconfirm the active input modality before recommending hardware or training infrastructure. If the project drops images in favor of symbolic/text state, remove vision-projector, VLM-collation, and image-serving assumptions and revisit whether carrying a multimodal checkpoint is still justified.
- [data] When the user asks to compare model reasoning against a narrator transcript, actor identity is a hard join key, not an optional label. Target the narrator's verified seat for model inference; do not recommend reusing another seat's traces merely because they are already available.
- [gotcha] A policy decision's original replay-row identity is not necessarily its safe UI availability. When policy evaluation causally reorders same-event actions (for example MOVE_ROBBER before Colonist's emitted STEAL), show the prerequisite trace at the pre-event cursor but withhold any dependent post-move trace until the entire original event is revealed; support multiple traces at the later cursor rather than leaking future context.
- [prompt] Initial-placement policy guidance must frame the two settlements and roads as one opening portfolio tied to provisional 10-VP routes, not as a pip-count maximization task. Make the model compare city/dev-card/Largest Army, expansion/Longest Road, ports and balanced fallbacks; account for opponent picks and resource/number complementarity; and state in every active suite that the second settlement is independently placed at any legal vertex while the first road only seeds post-setup expansion. Lock this rule with a rendered-packet test so a concise suite cannot accidentally omit it.
- [eval] In paired format studies, nominal item-level Cochran Q and McNemar tests can overstate inferential independence when category-selected questions share a small set of boards. Report the clustering and asymptotic-reference caveats, prefer repeated boards/runs or cluster-aware inference for winner claims, and correct exploratory pairwise comparisons.
- [eval] Distinguish serialization parsing from solver-backed question answering. Executing SQL in an in-memory database to validate a lossless round trip does not make the model condition solver-assisted; state exactly whether any query result or logic-engine output was exposed at inference.
- [eval] Do not raise benchmark difficulty because one direct-lookup category reaches ceiling while procedural categories remain near floor. Freeze the broad suite, localize current failures with staged probes, and require repeated overall and per-category saturation before creating a harder version.
- [eval] A strong incumbent format should not be discarded because end-to-end operator chains fail. Keep tile rows as the baseline, feed gold inputs independently at each diagnostic stage, and identify the first broken operation before testing another serialization or harder questions.
- [eval] Decompose composite vision scores before labeling perception broadly. A resource-plus-number exact score can hide near-perfect number OCR behind resource-class or vocabulary errors; report each component and show canonical, alias-rescued, and genuinely confused examples.
- [eval] Closed-class visual probes must enumerate every allowed output class. Showing only `<WOOD>` and `<ORE>` as token examples can create default-label bias and turn vocabulary/task induction into apparent perception failure. Isolate closed-vocabulary guidance from visual class descriptions before estimating the vision ceiling.
- [eval] Literal angle brackets are serialization punctuation, not automatically tokenizer special tokens. In the isolated-tile ablation they added token cost without an accuracy benefit; prefer plain canonical class values at the model boundary and add engine/display wrappers deterministically unless a separate benchmark shows value.
- [workflow] Do not leave experiment analysis or verification as inline Python heredocs that exist only in a Pi trajectory. Put stable logic in tested library modules, record parameters in a versioned run specification, and expose short rerunnable plan/run/verify/report scripts or task-runner commands.
- [arch] Treat `game_viewer` as an outer adapter over the replay domain. Replay loading, sessions, stepping, decision packets, and policy interfaces must not depend on Flask, Socket.IO, React, or process-global viewer state; evals and data pipelines should consume the same core API directly.
- [arch] Distinguish the deterministic replay engine from the agent decision harness. The replay engine extends or wraps the canonical engine with fixed outcomes and replay-specific step/undo/goto behavior; the viewer wraps that replay engine, while the harness should consume the shared engine interface rather than own replay mechanics.
- [arch] Use qualified sandbox terminology for the agent-facing Catan runtime: `CatanSandbox` wraps internal game rules, while `ReplaySandbox` adds fixed recorded outcomes and replay controls. Do not erase the internal engine distinction or confuse this with Prime Intellect's infrastructure sandbox/runtime.
- [arch] Do not present PettingZoo AEC as an LLM harness design. It is useful only as turn-state vocabulary; Catan LLM agents need event-triggered decision requests, persistent per-seat context, and model calls only at genuine choices, closer to Prime Verifiers' host-refereed `Env.run(..., agents)` interactions.
- [perf] Treat provider KV/prompt caching as a losable prefill optimization, never as agent continuity. The harness must persist and rebuild each seat's causal transcript and structured memory; routing/session IDs can improve cache affinity and observability but do not replace conversation state.
- [arch] Separate module dependencies from runtime aggregate ownership. For this project, `CatanSandbox` is the composition root that contains the engine and per-seat harnesses, while engine rules, context assembly, and provider policies remain independently testable components; do not force the harness to be an externally owned sibling merely because the dependency arrows are decoupled.
- [process] Before carrying a migration checklist forward, recheck the live worktree and the user's stated scope. Do not schedule already-completed CatanBoardBench migration when the requested slice is the editable prompt/context suite and viewer-to-sandbox I/O boundary.
- [pattern] Keep prompt suites declarative but not authoritative: versioned YAML may own prose, section order, phase guidance, output tags, and bounded window settings, while typed Python must retain hidden-information filtering, causal cursors, exact legal-action identity, accepted-response commits, and snapshots.
- [gotcha] A seat must not append model messages or advance its event cursor when inference merely returns. Hold a pending decision keyed by decision ID and commit its transcript/memory only after the sandbox accepts the exact indexed action; otherwise stale concurrent responses corrupt continuity.
- [arch] For a dirty-worktree domain extraction, copy current behavior into a core package first and use aliases only as a short, explicitly scheduled migration stage. Once repository consumers use the core path, delete internal-only aliases rather than preserving hypothetical downstream compatibility; compatibility needs an identified caller or user requirement.
- [api] Name the primary sandbox operation after the domain result: `step()` advances one action for both live and replay. Keep scheduler mechanics subordinate (`decide()`, `discard()`) and call a caller-supplied policy bypass `override_step()`; do not expose transaction jargon such as prepare/commit as the main API.
- [simplicity] Do not add future-facing contracts before an implementation consumes them. Remove speculative types such as an unused chance-outcome DTO, and keep boundary annotations matched to the data actually stored.
- [process] A green test suite does not make an environment complete while dead TODO or `NotImplementedError` scaffolds remain. Before declaring completion, scan first-party runtime code for placeholders, identify whether each path is active, and either implement the real contract or delete the unused scaffold and its documentation.
- [context] Do not replace observable game history with a lossy public-state summary. Agent calls receive the complete privacy-projected game events available to that player; only table-talk message history uses a small explicit window. Event cursors schedule reactions and acknowledgements, not hidden compaction.
- [reasoning] Never label model-authored XML rationale as provider-native reasoning. Persist and render the prompted `<rationale>` as an explanation, keep native reasoning only when the provider returns a distinct reasoning channel, and show unavailable native reasoning honestly rather than synthesizing it from ordinary output.
- [reasoning] Removing a stale observation/action panel must not erase useful model diagnostics. Preserve an accepted-decision trace as a read-only presentation of typed player receipts—model-authored rationale and provider-native reasoning with explicit provenance—without turning it into a second runtime or action contract.
- [viewer] Keep the live viewer a thin adapter over `CatanSandbox`: configure/start one sandbox, render its projected state, and let one Step control invoke one complete `sandbox.step()`. Do not recreate a second observation/action runtime or legacy decision protocol in frontend state.
- [verification] A route unit test plus a frontend build does not prove an interactive control works. For viewer control changes, exercise the real registered Flask app and WebSocket state transition from start through the clicked endpoint, inspect non-2xx response bodies in the UI, and add an end-to-end regression for the exact button flow before reporting success.
- [ops] After changing server code, verify the process actually listening on the configured port was restarted; a no-reloader Flask process can serve hours-old modules while source tests pass. API clients must inspect status/content type before decoding JSON, and mutation routes must return structured JSON even for unexpected failures.
- [serialization] Do not assume early-game snapshots prove a public API is JSON-safe. Normalize the complete projected payload through the canonical encoder before any HTTP or WebSocket boundary, and run enough deterministic steps to reach events whose payloads contain enums such as `Color`.
- [tracing] For a single-process local LLM runtime, use SQLite/WAL as the durable trace index: normalized game/step/call rows for queries, JSON payloads for evolving schemas, and an optional trusted local snapshot blob for exact restore. Commit one completed sandbox step per transaction, store accepted and rejected attempts, and never persist authorization headers or API secrets.
- [data] Colonist replay logs provide exact causal order through state changes, trade-response updates, and recipient-specific events, but the multi-agent simulator must not reproduce wall-clock races from `deltaS`. Resolve trade, reaction, and negotiation windows with barriers: freeze one causal cutoff, await all required players, then apply outputs in deterministic table order. Model latency never grants gameplay priority.
- [trade] Do not collapse standard Catan negotiation into one `ActiveTrade`. A trade window can hold multiple simultaneous root proposals, acceptances, and counters, and the active player must compare them; only the final bilateral resource exchange is singular. Model this as a proposal board/graph scoped to the turn's negotiation window.
- [trade] A counteroffer can respond only to a root proposal; it cannot itself be countered. Encode this as an unconditional contract invariant rather than a configurable graph-depth limit, and never generate a counter action for a proposal that already has a parent.
- [api] Keep trades parameterized, but never expose positional 10/12-value resource tuples as the player contract. Parse named `give`/`receive` resource maps into one engine-owned `TradeOffer`; retain tuple decoding only at historical replay ingestion boundaries where the source format requires it.
- [trade] Separate willingness from execution in Catan domestic trades. Every eligible opponent may signal willingness to a turn-player offer, but only the turn player selects one exact counterparty and may ignore all willingness signals. Expose one simple `TradeOffer` surface—who, give, receive, optional parent, responses, and status—instead of making players reason about a public Proposal/Terms type split; keep execution candidates internal and exact.
- [trade] Do not represent an optional embargo as a permanent legal-action menu item. Keep embargoes disabled by default and separate from trade offers; if enabled later, activate them through bounded communication/commitment state and surface them only when relevant to a targeted trade.
- [rl] Treat each dice event as the canonical luck unit. Record its per-seat conditional production innovation and variance at the exact pre-roll state; derive cumulative or phase views only as optional analysis rather than imposing segments.
- [rl] Do not smuggle a state-value function into a requested dice-distribution metric. For value-free Catan luck, retain exact per-player, per-resource production residuals. When the intended use is policy credit, treat future zero-mean dice residuals as event-ordered control variates and trace their corrections backward only to decisions that preceded each roll; estimate stopped-gradient coefficients across games rather than inventing per-state values. Raw opponent targeting is endogenous strategic response, not an unbiased luck correction.
- [rl] A plan-conditioned reasoning critic is distinct from a value critic. Freeze each model-authored goal, target, resource deficit, timing claim, assumption, and replan trigger before future events; let a small semantic agent bind those claims to deterministic engine calculations, then compare claimed timing, engine-expected progress, and realized per-resource residuals. Keep dice/random-event luck, opponent intervention, own conversion, and reasoning miscalibration separate, retain raw event-level vectors, and use the resulting attribution primarily as process/auxiliary supervision rather than a replacement scalar reward.
- [eval] Qualitative Catan scenario buckets should be multi-label projections over explicit state evidence, not mutually exclusive action-name bins. Store stage inputs, stable whole-turn/trade/robber/setup episode IDs, and the authored detection rule; keep reviewer failure labels separate from deterministic scenario tags.
- [gotcha] Archived action-diff artifacts may call a prompted visible explanation `reasoning`. Treat that field as a legacy rationale, never native provider reasoning; only display native reasoning when a distinct provider field and request metadata were retained.
- [process] Do not infer that prompting guidance derived from RL is a request for a separate RL-only suite. Distinguish where a heuristic came from from where the user wants it applied; by default, apply general prompt-engineering principles to the active shared suite and preserve product-facing contracts unless asked to redesign them.
- [reasoning] Turning native reasoning on means making the enabled condition explicit and observable by default, not removing the user's explicit on/off control. Preserve a true disabled condition for comparisons and user choice.
- [ux] Persisting historical model calls and drawing a checkpoint navigator does not make reasoning history usable by itself. Automatically fetch detail for the selected/default saved game and every just-completed live step, keep the active latest checkpoint playable, and verify the visible trace rather than only the SQLite/API payload.
- [ux] Historical reasoning should reuse the normal live reasoning location, not turn navigation controls into a second content surface. Keep checkpoint selection in the toolbar and render the selected step’s calls in the right column beneath the game log, preserving the full accepted/rejected and communication detail.
- [tracing] Legacy trace databases may contain deterministic baseline choices in `model_calls` even though no inference occurred. A canned `rationale` is not evidence of a model call; require a recorded model request or response before labeling a row as model reasoning, preserve the legacy row as metadata, and state clearly that no provider reasoning exists.
- [prompt] Production prompts should pair exact dice numbers with pip probability rather than choosing one representation. Show the per-tile mapping (for example `WOOD dice=6(5pip)`) and the aggregate node total, so pips summarize frequency without hiding the actual roll outcomes.
- [communication] Silence is the default unless speech changes another player's strategic choice. Do not spend model calls on compliments, thanks, acknowledgements, or narration. A required intent enum (TRADE/QUESTION/WARNING/BRIBE) did not police this: every hotline message passed as WARNING, nothing branched on the value, and rendering it to listeners stapled the speaker's frame to the text and made bluffing impossible. Never make the model classify its own speech; categorize post-hoc offline if needed. Removed 2026-09-16.
- [config] Never let a user-facing live sandbox silently inherit an obsolete provider fallback. Give live inference its own visible model selector, send the model explicitly, persist and return the realized ID, and keep its default aligned with the current policy baseline.
- [communication] Bounded reaction rounds still create noisy message cascades when `MESSAGE_SENT` itself is a wakeup cause. Messages may remain in the recent-talk context, but do not recursively schedule another communication barrier; let players answer at the next normal pre-action, trade, robber, or strategic event opportunity.
- [communication] Answer own-turn speech questions with the concrete sequence: normal main-game steps call communicate (message or SILENCE) before the gameplay decision; setup and forced rolls skip that pre-action call. This is part of the turn despite using separate model calls. Do not substitute an explanation of message-triggered wakeups for whether the agent can speak on its own turn.
- [ux] Setup controls should reflect lifecycle state, not merely become disabled. Once a live game exists, remove the start-game configuration and start buttons from the visible controls; restore them only after the game is cleared.
- [prompt] Keep the communication system message limited to stable role identity: the model is playing Catan as a specific color. Put opportunity-specific speech policy, visible events, recent talk, commitments, and the response schema in the user message rather than overloading the system prompt.
- [process] Treat a reported model-quality observation as evidence for architectural discussion, not an implicit request to change defaults. Confirm intent before changing model selection, and distinguish the strongest interactive teacher from the intended student/evaluation policy.
- [prompt-ui] Do not make one concatenated provider packet the primary prompt-editing abstraction when the user asks for observation strings. Preserve typed system/environment components, expose each authored template, engine value, rendered string, and provenance separately, and map environment to the wire-level `user` role only at the provider boundary.
- [prompt] Phase guidance must do more than name the phase. For each real prompt key, include engine-accurate, bounded decision considerations and useful follow-up consequences; keep dynamic state in observation components and never restore generic strategy claims that the visible context cannot support.
- [prompt] Model-facing strategy guidance must speak from the player's decision boundary, not explain the harness. Never expose seeded RNG, internal menu limitations, prompt routing, implementation status, or instructions about what to claim in a rationale. If a phase offers no meaningful choice, keep its guidance minimal instead of narrating runtime internals.
- [prompt-routing] When two real game states need different guidance, give them separate typed prompt keys. Do not keep one shared key and make the LLM interpret conditional wording that the runtime can resolve deterministically, such as the first and second setup roads.
- [prompt] First-settlement guidance should evaluate complete opening pairs: projected second-settlement production, the exact starting cards that placement would grant, the early options those cards enable, and remaining resource gaps—not just the first node in isolation.
- [arch] Treat the public board as an immutable canonical fact snapshot plus a typed presentation, not as a raw provider payload. Every text/image projection must carry state/content hashes, renderer version, and an explicit identity space so opaque eval aliases can never be confused with canonical engine IDs.
- [tracing] Multimodal requests must keep image bytes outside durable messages, sessions, and trace JSON. Attach one bounded local image only to the current provider turn, persist content-addressed metadata, and sanitize every inline data URL before retaining provider payloads.
- [context] A large lossless board projection should be attached only to the current model request rather than copied into durable conversation history. Keep the compact semantic component in history and compose the current board presentation at the transport boundary.
- [sft] Confirm whether learned atlas tokens are input-only perception pointers or a bidirectional agent communication vocabulary before freezing the output side. If agents should emit locations, train only the corresponding 154 output rows and include explicit atlas-token targets; unfreezing rows without target occurrences supplies no learning signal.
- [research] Do not reduce a report that a post-trained model's language is broadly degraded to verbosity, format compliance, or hidden-reasoning policy. Re-evaluate the underlying language prior and the mostly-frozen-language training assumption separately; distinguish base, SFT-only, preference-optimized, and online-RL checkpoints before selecting a VLM.
- [research] Do not classify a checkpoint as text-only from an LLM-oriented model card or text-only post-training recipe. Inspect its config, processor, repository file inventory, and tensor index: Cogito v2 109B retains Llama-4's full vision tower/projector and image processor even though its IDA stage used no image examples.
- [scope] Backbone exploration must not displace the locked infrastructure deliverable. Keep the semantic data and selective-token trainer model-independent through object-identity module discovery and explicit architecture contracts; treat any concrete VLM as a later adapter and evaluation choice.
- [process] Do not equate the surrounding Pi harness with the requested worker runtime. When the user asks to use Codex and authenticated `codex exec` is available, launch constrained Codex workers directly; do not route the work through Pi's extension-isolated subagent broker.
- [process] When a concrete datagen or trainer plan is already locked, execute it instead of turning a brief mention of another framework into an unsolicited framework survey. Point to the plan and artifacts first; teach frameworks only when the user explicitly asks.
- [model-selection] “RL-fried” behavior can be evidence for or against downstream adaptation. Do not translate it directly into a weak-prior verdict: collect the exact checkpoint, prompts, inference controls, and outputs, then distinguish distorted post-training behavior from missing underlying language, vision, or planning capability before selecting a trainer.
- [scope] Keep semantic-v2 datagen independent from its transport filename. `ms_swift_semantic_v2` is a JSON projection contract, not permission to implement ms-swift after the user has paused that trainer decision.
- [arch] The dedicated eval/data-visualization frontend is not the playground game viewer. Do not add dataset-explorer APIs to `playground/game_viewer` merely because legacy benchmark routes exist there; trace and preserve the eval app's own launch/server boundary first.
- [process] A request for existing LLM data-visualization frameworks is research, not authorization to build a custom dataset explorer. Compare off-the-shelf tools against the actual image/input/target/oracle workflow first, then implement only a small import adapter after the user chooses one.
- [tooling] Do not recommend a visualization framework merely because its low-level primitives can represent the data. Product/domain fit matters: a physical-AI telemetry viewer is not automatically the right LLM dataset and game-inspection workbench, even if it supports timelines, images, and text.
- [tooling] Product positioning is a filter, not a hard technical constraint. When a generic open-source viewer has unusually strong primitives, validate it on one representative Catan slice: use Observable Plot for aggregate web data QA and consider Rerun for synchronized per-game timelines, while keeping the canonical data contract independent of either viewer.
- [tooling] Do not describe Rerun as offline-only. It supports live gRPC streaming and an embeddable web viewer; distinguish live observability from bidirectional game control. Try an observer stream, embedded viewer, and supported extension points before forking, because custom viewer APIs and web builds can impose upstream/version-maintenance costs.
- [ux] Do not answer an aesthetic complaint about the existing playground with a new observability stack. If the domain-specific controls, replay, board, and traces already work, improve visual hierarchy and styling in place; introduce Rerun only for demonstrated timeline, scale, synchronization, or recording problems.
- [ux] Collapsing fixed rails is insufficient when long transcripts and dense reasoning are permanently confined to a narrow side column. A serious playground overhaul should provide board, analysis, and focus workspace modes; an expandable readable-width inspector; compact drawers/docks for controls and players; and reduced-motion-safe layout transitions rather than merely hiding existing panels.
- [research-ux] Do not hide a substantial model audit behind a terse final summary and a file path. Provide a readable detailed analysis in the conversation, then make first-party Catan benchmark evidence discoverable in the project's existing evaluation system; keep literature/vendor claims visually separate from locally reproduced results.
- [eval] Never say a model “saturated vision” without naming the exact modality, cohort, category, and scoring contract. Qwen3.8 Max reached 60/60 only on the authoritative indexed-text projection and 10/10 on one narrow image tile/resource category; its complete raw-image scores were 48/110 and 11/60, so neither establishes saturated board vision.
- [ux] When the existing Catan Eval Suite is already hard to read, do not bolt a model-evidence tab onto it unchanged. Audit and simplify the suite-level information architecture, visual hierarchy, navigation, comparison semantics, and responsive behavior first; then add research evidence into the redesigned shell.
- [tooling] Before choosing a custom Catan Eval redesign, investigate what working LLM-evaluation researchers actually use for run comparison, sample triage, traces, human review, and reporting. Prefer adapting the repository's existing Inspect/OpenBench path or a proven tool over recreating an eval platform from screenshots.
- [ux] Treat workspace placement as semantic choreography, not just resizable columns. Session/history navigation belongs in a hideable left rail, all setup and gameplay controls belong in an expandable board-bottom dock, compact public player state can occupy unused board-ocean space, and long model output/native reasoning belongs in a separately toggleable right drawer.
- [ux] “Toggleable inspector” does not imply collapsed by default. When the user wants analysis visible on arrival, initialize the right drawer open and version the persisted panel-layout key so an obsolete collapsed default cannot silently win; afterward, persist the user's explicit close/resize choice.
- [ux] Preserve the axis and silhouette of a supplied scoreboard reference. A requested vertical player strip should remain one column on the board edge at every pane width; do not reinterpret it as horizontal chips or a 2×2 top overlay. Prefer a narrow overlay over unused ocean when reserving canvas width would shrink the actual board materially.
- [ux] Keep deterministic activity and model reasoning visually distinct even when both live in the right inspector. A game log can remain directly above reasoning while collapsed by default; communication/speech inference cards should also start collapsed with speaker, status, and message preview visible, while decision reasoning stays expanded for immediate inspection.
- [reasoning] When the product chooses provider-native reasoning only, remove authored rationale from a new versioned prompt suite, parser projections, APIs, and UI rather than relabeling it. Preserve immutable historical suites and an empty legacy dataclass slot only where required to unpickle trusted local checkpoints; never render or populate that slot in new decisions. Do not redact authentic old provider reasoning merely because it quotes an obsolete prompt—label the trace as historical and require a new game for the new prompt contract.
- [model-selection] When selecting an open VLM while diagnosing “RL-fried” behavior, use two matched lanes rather than one blended score: strict raw-board reading with reasoning explicitly off, and fresh matched game initializations with provider-native reasoning on. The qualitative lane is not a replay benchmark: do not attach replay history, human-action agreement, or action-diff scoring when the user wants raw reasoning traces to inspect. Preserve the exact prompt, distinct native-reasoning channel, final response, and provider metadata. Omit `max_tokens`; a forced `finish_reason=length` cannot distinguish pathological output policy from an otherwise valid answer that would have appeared later. Keep any capped replay run only as a labeled superseded artifact.
- [prompting] A first-settlement strategy probe should not leave “maximize pips and diversify resources” as the easiest implied objective. Version an eval-only guidance variant that asks for a coherent two-settlement/build-order engine, complementary backups after opponent placements, resource ratios, road direction, scarcity, and realistic port timing. Keep pips as production evidence rather than the target, preserve the baseline for A/B inspection, and do not silently mutate the live prompt suite.
- [cost] Hosted `max_tokens` may not even bound reported provider-native reasoning usage. In the 27B–72B placement preflight, requests capped at 4,096 returned 5,904, 11,280, and 30,001 completion tokens. For intentionally uncapped reasoning, enforce the dollar boundary with a catalog-price worst-case audit, sequential scheduling, and recorded-usage checks between requests—not by calling a token setting a hard spend cap.
- [context] Do not use a free-form, model-rewritten `<game_plan>` as the context-management mechanism. Keep authoritative events/current state separate from bounded private agent memory, and do not retain prior cumulative decision packets that re-embed complete history; define ownership and compaction explicitly before changing the shared response schema.
- [research] Before locking a long-trajectory agent context design, inspect the requested lab's actual papers, repositories, and social posts, then compare adjacent work. Keep semantic context policy (history, memory, summaries, masks) separate from systems optimization (KV reuse, scheduling, offloading, context parallelism), because one does not answer the other.
- [context] When the user asks for coding-agent-style continuity, do not translate it into independently reconstructed one-shot decision packets. Preserve a non-duplicating logical conversation so the model sees its prior model-visible tokens and event deltas, then compact that conversation into a new authoritative epoch; prompt/KV caching accelerates the same transcript but never supplies omitted semantic history.
- [terminology] Name the primary context comparison by semantics: **long-context path** versus **bounded path**. Do not call the long-context arm the “coding-agent/PPO path,” because a coding-agent scaffold and PPO-trained compaction are optional implementation and training choices within that arm.
- [rl] Do not default the bounded-memory agent to terminal-reward GRPO. The requested direction is a structured process reward model with rubric dropout, dice-luck variance accounting, and adversarial scoring; keep the PRM/scorer separate from the eventual policy optimizer, preserve raw game outcomes alongside adjusted scores, and validate privileged-information boundaries and reward-hacking resistance before training against it.
- [ux] A board-bottom gameplay bar must operate the current game or replay, not hold setup and configuration forms. Put setup, model, replay-loading, and diagnostic controls in the hideable left sidebar; reserve the persistent bar for true previous/step and explicit `0..N` cursor navigation, and never present browse-only trace selection as runtime rewind.
- [prompt] Treat setup order as public typed state, not static guidance: derive round one from the engine's realized player order and round two by reversing it, render both only during setup, and regression-test shuffled seats plus the complete model packet. Version-gate the new dynamic context policy so historical suites keep their prior rendering.
- [diagnostics] Do not infer provider timeouts merely because latency is close to a retry multiple. Inspect the terminal sandbox error first: several slow invalid model decisions can look like transport retries. Interactive Step must surface each rejected final response and validation error, and should not silently launch multiple expensive retries.
- [sync] Do not call a sandbox “ready” merely because `/api/state` reports an active backend. If a mutation is issued outside the viewer that initiated the current WebSocket session, broadcast its authoritative snapshot; verify both server state and connected-client delivery before claiming the UI is synchronized.
- [config] When matching a validated model run, copy the complete inference condition rather than one field: `reasoning.effort=high`, `exclude=false`, and omission of `max_tokens` are jointly part of the baseline/strategy-guided setup. Do not describe an unreliable hosted completion limit as the reference configuration.
- [config] Groq is no longer a live sandbox backend. Keep the active live transport selection to explicit vLLM or OpenRouter paths; do not extend or silently retain an obsolete Groq environment fallback while changing inference contracts.
- [prompt] Diff versioned suites before describing their semantic changes. `catan-agent@8.0.0` differs from v7 only by version and authoritative `initial_placement_order: both_rounds`; the strategy overlay is a separate prompt variant and must remain conceptually independent.
- [prompt] Never annotate legal actions with topology inferred from numeric node-ID distance. Canonical IDs are labels, not a graph metric; either compute an exact graph relation and name it precisely or omit the annotation so the model cannot reinterpret `Near:` as adjacency or legality.
- [model-selection] Keep fluent prompt-induced strategic errors separate from “RL-fry” language pathology. Borderline neuralese, broken grammar, malformed wording, code-like fragments, and loss of ordinary language coherence are model/output-policy symptoms; a misleading prompt can trigger or expose them but does not by itself explain that linguistic degradation.
- [frontend] Type and test privacy-redacted payloads in their actual wire shape. Live resource visibility is `{TOTAL: n}`, not a named-resource map; counters must prefer `TOTAL` while authorized exact-resource payloads may sum `WOOD`/`BRICK`/`SHEEP`/`WHEAT`/`ORE`.
- [presentation] Never expose `str(Action)` for polymorphic structured contracts. Trade actions already have named semantic payloads; game logs and API display fields must project those payloads into concise deterministic text while trace storage retains the structured action itself.
- [sandbox] Do not equate one sandbox Step with one engine transition. Trade barriers batch one response transition per non-turn player; downstream logs and projections must iterate `result.transitions` in engine order rather than selecting only `transitions[-1]`.
- [inference] Bypass inference only from the authoritative legal menu, not hidden-hand guesses. A sole legal `ROLL` is deterministic and should mutate neither provider/session state nor pre-action speech history; if Knight or any other development-card play is legal, keep the normal model decision path.
- [sft] Never hand raw bf16 parameters to AdamW at learning rates near 1e-6 to 1e-5; the per-element step is below half a bf16 ulp and rounds to zero, so the group is frozen while the logs look healthy. Keep fp32 master weights for any fully trained module and verify with a per-group dtype audit before a paid run.
- [sft] `resize_token_embeddings` is a no-op when the checkpoint's embedding matrix is already padded past the tokenizer, so new token rows inherit untrained padding values. Initialize added rows explicitly and record their norms.
- [sft] A short fixed completion suffix (end-of-turn plus newline) dilutes `loss` and `mean_token_accuracy`; a 0.5 loss with 0.83 accuracy on three-token completions means the answer token is near chance. Log answer-only accuracy and compare every eval against the majority-class baseline of that split.
- [curriculum] When the trainer uses a sequential sampler, the dataset file order is the batch composition. Write train rows in a deterministic shuffled order and check unique boards per batch before launching.
- [modal] `train_h200.spawn(...)` from a `modal run` local entrypoint dies when the ephemeral app stops; always launch paid training with `modal run --detach` and persist the returned function call id locally.
- [eval] Eval throughput is batch-bound, not load-bound. Generation at batch 8 left an H200 mostly idle (12,000 rows in 80 minutes); batch 48 is the current default and the H200 has headroom for more aggressive batching (96 or higher) once the 27B eval path is profiled. Always batch every eval set and image variant for a checkpoint into one model load, and size the batch to the GPU, not to the L40S default.
- [modal] `.spawn()` from a `modal run` entrypoint needs `modal run --detach` regardless of what the entrypoint does afterwards; the ephemeral app stops when the entrypoint returns and Modal cancels every spawned call ("Function call was cancelled by user or a failure"). This bit the first training launch and the first spawned regression panel.
- [gotcha] A Modal eval container sees the runs Volume as of its own start, so spawning `modal_qwen_series_eval.py` the moment the trainer logs a step-N eval loads a half-written `checkpoint-N` and PEFT fails with "Repo id must be in the form" on the adapter path. Confirm `adapter_config.json`, `adapter_model.safetensors`, and `visual_model.safetensors` are all listed under the checkpoint with `modal volume ls` before launching; single-set evals write `summary.json` flat under `--output-dir`, only multi-set panels nest `<set>/<variant>`.
- [gotcha] Aggregate eval loss on the board-reader sets cannot see a broken head: tile, localization, and far-empty rows are near-free and dilute it four to one, so occupancy positives at 73% and 95% both sit inside 0.02 to 0.06. Gate on per-task exact match and the neighbor_confusion block, never on loss.
- [gotcha] Every stage so far trains the 154 atlas-token rows at the first stage's peak rate, and they keep moving: after two pair stages six rows (T10, N16, N26, E08_27, E37_38, E45_46) share a direction at 4 to 6 sigma above the 0.03 baseline and bleed behaviour into each other. Check pairwise row cosines between bundles before blaming data; anchor or freeze settled rows in later stages.
- [data] Slanted roads fail four times as often as vertical ones on synthetic boards (12.8% vs 3.2%), independent of patch-boundary distance and training frequency. Real boards are almost all slanted roads, so read road recall by orientation before adding resolution.

- [gotcha] Full-coverage board evals are ~70% `empty` rows. Exact accuracy
  and teacher-forced token accuracy reward an all-empty model (stage-3 ck128:
  85% accuracy, 4% road recall). Read blindness and occupied recall by piece
  and density first; accuracy last.
- [gotcha] A single-set panel run writes `summary.json` at the label root
  (`regression-panel/<label>/summary.json`); multi-set runs nest
  `<set>/<variant>/`. Waiters must check the label root for single-set jobs.
- [pattern] Sequential rungs on one family forget the others through the
  shared LoRA, vision tower and merger, not the token rows (rows stay put
  when absent from prompts). Every rung after the first carries rehearsal
  rows of every earlier head; `mix_rung_data.py` with a JSON recipe.
- [gotcha] Readouts are 60 to 100x longer than short rows, so a mix with two
  readouts per image is >90% readout tokens; balance by estimated completion
  tokens (the mixer reports shares), keep readouts in the hundreds.
- [gotcha] `get_peft_model` mutates the module tree in place: targets become `<name>.base_layer` and gain `lora_A/lora_B` children, so any helper that lists targets by module name (`vision_linear_targets`, `language_linear_targets`) must run before wrapping and its result reused; recounting afterwards either returns a different number or raises. Same trap for state files: a `visual_model.safetensors` saved from a wrapped model only loads into an equally wrapped model, so restore before `merge_and_unload`.
- [gotcha] Match the local peft to the Modal image pin before trusting a local PEFT test. peft 0.17 refuses `trainable_token_indices` on an untied `lm_head` that 0.20 accepts, so a test that passes or fails locally may say nothing about the H200 container.
- [rl] Do not recommend a learned Catan value model as an easy answer to sparse
  rewards: values depend on hidden-information beliefs, negotiation history,
  and the joint policy/opponent population. Distinguish a training-only baseline
  from a reward oracle. Potential shaping preserves returns under its endpoint
  conditions without an accurate critic, but supplies no automatic strategic
  credit assignment; narrow process scores remain proxies, not win guarantees.
- [rl] Read classic-net Catan RL as the wrong prior for LLM training.
  DQN/PPO-from-scratch, Settlers-RL, Deep Catan, fastCatan, and catanrl learn
  representation plus policy from zero with dense shaping, critics, MCTS, and
  10M-plus steps. For LLMs the base model already supplies language,
  arithmetic, rules, and reasoning priors; board grounding plus verified
  SFT/SDFT puts legal strategic behavior in support, and outcome RL only
  selects and sharpens it. Do not expect sparse reward to invent rules,
  topology, production, beliefs, or planning.
- [rl] Treat process reward models as denser auxiliary signal, not the
  objective. Use turn/action-level scores first for search rerank, Best-of-N,
  and data filtering, where PRMs reliably help. Before training a policy
  against a PRM, require a held-out ranking/calibration win, preserve raw
  game outcomes alongside adjusted scores, keep scorer/optimizer separate
  with privileged-info boundaries, and bound aggregation (min-form or
  Clip plus Delta, outcome-linked, weakest-link) because sum-form step
  rewards invite verbosity and thinking-only hacks.
- [rl] Default to straight games over GRPO-style branched rollouts for Catan
  policy learning. Branched LLM continuations cost full games each yet with
  small K mostly label dice/opponent luck, so they are pricier and noisier
  per gradient bit than complete games. GRPO/RLOO/DAPO are built for RLVR
  verifiable single-turn math/code with cheap checkers; Catan is a
  stochastic multi-agent partial-information game, and the same-prompt
  group assumption breaks across seats, turns, and dice seeds. Prefer
  grouped scenario seeds with scenario-conditioned baselines, and treat
  RLOO as undecided and DAPO-style clip-higher/dynamic-sampling/token-loss
  as suspect until each beats straight games in an equal-rollout ablation.
- [gotcha] Live games carry two independent counters and UI labels must say
  which one they show. Engine events (action plus every speech and trade
  response inside one step) run well ahead of recorded trace steps, so a
  349-step game reaches event #523. Reusing a "step N" label for a sequence
  reads as a bug in the engine. When a label needs the step index, remember it
  only exists after `record_step` commits: log the row first, stamp it after,
  rebuild the snapshot so the response, the broadcast and the stored public
  state stay identical, and fall back to the sequence when no step was
  recorded rather than inventing one.
- [gotcha] Two dev servers, one reload story. Vite hot-reloads the playground
  frontend on :5173, but the viewer's Flask process on :5001 has no reloader
  unless `CATAN_VIEWER_RELOAD=1`. A backend change made after the server started
  is invisible while the frontend half of the same feature is live, so the UI
  renders the new markup against the old payload and the work looks unapplied.
  Check the process start time against the source mtime before debugging the
  frontend, and prefer `CATAN_VIEWER_RELOAD=1` when iterating across both.
- [pattern] Do not put a spectator view behind a toggle by default. The
  playground exists to watch agents play; hand contents belong on the chip
  where the totals already are. A toggle adds a click, a persisted preference
  and a second state to test for information the operator always wants.
- [pattern] A panel that disappears when empty reads as a broken panel. Gating
  the inspector's message board on `entries.length > 0` meant a silent table and
  a missing feature looked identical, which cost a debugging round. Render the
  container whenever its subject exists and let it say it is empty.
- [perf] Per-step snapshots must be O(state), never O(history). The sandbox
  snapshot pickled every agent's full receipt map, and each receipt deep-copied
  the model's reasoning trace, so a 400-step game re-serialized 4 MB per step
  and cost 0.85 GB. History that is already appended once (`model_calls`,
  `game_events`) must not ride along inside the thing that gets stored every
  step. When a blob grows with the step index, find the list inside it before
  reaching for compression; compress as well, but second.
- [gotcha] Before compressing a JSON column, grep the SQL for `json_extract` and
  `json_each` on it. `get_usage` reads `response_json` and `payload_json` inside
  SQLite, so those stay plain text while their siblings are packed.
- [gotcha] Restarting a server does not un-truncate data it already stored. The
  viewer rebuilds its in-memory log from the last checkpoint on load, so a
  window removed in code still came back windowed from the checkpoint the old
  process wrote. When a stored projection can be lossy, rebuild it from the
  durable record (events) on load rather than trusting the stored copy.
- [arch] The viewer's game log is a derived narration of the engine's event
  stream but lives in memory and is stored per checkpoint, which is why it
  needed windowing, post-hoc step stamping and event backfill. The clean shape
  is a pure `events -> rows` renderer with no stored log; `game_events` already
  carries sequence and step index. Deferred deliberately (2026-09-16): the
  current patches are small and tested. Revisit if the log grows another
  special case.

- [gotcha] Socket echo-dedupe keys must cover every field a notice-only broadcast
  can change (`last_live_step_error`, `live_inference`, `player_types`), not just
  game progress. A key built from state index/log length silently drops failure
  notices sent to other tabs, and the browser tests that emit such notices fail
  for a reason unrelated to the feature under test.
- [pattern] "Unlimited retries" for live auto-play belongs in the frontend loop,
  not the per-request decision budget: each retry is a fresh `/api/step` under a
  new attempt budget with its failure checkpointed, so the backend and its
  in-memory game never need a restart. Keep one hard stop for failures the trace
  store could not record.
