# Multimodal expert-reasoning data pipeline

## Goal
Build a provenance-preserving pipeline that turns (a) noisy Catan YouTube gameplay commentary plus board frames and (b) authoritative Elo-indexed Colonist replays into high-confidence policy SFT, rationale SFT, value, and same-state preference examples.

## Repository findings
- The noisy example is `data_pipeline/training/pretraining/output/transcript_t5RZGAJfKss.md`; phrases such as "this spot", compressed number triples, streamer narration, table talk, profanity, and overlapping speakers are not usable without the corresponding frames and timestamps.
- `data_pipeline/ingestion/youtube_scraper.py` currently saves captions and coarse 30-second chunks only. It does not acquire video/audio, retain a raw asset manifest, run word-timestamp ASR/diarization, extract frames, detect decisions, or align actions.
- `data_pipeline/bootstrapping/generate_training_data.py` is a prototype and must not become the new dataset foundation: it mutates cumulative state before formatting the claimed pre-action observation, can collapse multiple source changes into one action, lacks authoritative legal-action packets, and omits the current replay privacy/leakage guarantees.
- The safer foundations already exist: exact replay parsing/auditing, perspective-safe decision packets in `playground/game_viewer/replay/llm_response.py`, replay rendering in `data_pipeline/catanbench/`, game-level leakage-safe splits, and multimodal JSONL conventions in `sft/scripts/build_vlm_sft_dataset.py`.
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
- Split by game/video lineage before generation; preserve existing CatanBench exclusions and deduplicate by game ID plus board/action fingerprint.
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
- [x] Replay `242781000` captured on first attempt (737 events) and archived to `data_pipeline/bootstrapping/data/replay_staging/242781000.json`; sha256 `b07dfe7a…5485eae2e4`; manifest at `data_pipeline/training/reasoning/pilots/_2n5F2DxtPI/pairing_manifest.json`.
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

# Benchmark Qwen3-VL-32B on CatanBench

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
- Artifact integrity passed (110 unique questions, 10/category, no empty responses), and the focused CatanBench/token suite passed 7 tests.
- Full report: `data_pipeline/catanbench/datasets/catanbench_100/reports/2026-08-10-qwen3-vl-32b-visual.md`.

---

# Benchmark Claude Fable 5 on CatanBench

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
- Artifact integrity passed, and the focused CatanBench/token suite passed 7 tests.
- Full report: `data_pipeline/catanbench/datasets/catanbench_100/reports/2026-08-10-claude-fable-5-visual.md`.

---

# Research Catan board representations

## Goal
Determine an empirically defensible representation strategy for Catan agents rather than assuming images, JSON, or prose are universally best.

## Plan
- [ ] Define one perspective-safe information contract and criteria for topology fidelity, exactness, token/visual-token cost, model compatibility, training, validation, and invariance.
- [ ] Research primary work on learned board states, symbolic/game encodings, graph serialization, entity-centric RL, VLM spatial grounding, and structured-data token efficiency.
- [ ] Encode equivalent representative Catan states as verbose JSON, compact JSON, graph/atlas DSL, event deltas, ASCII, image, and hybrid forms; measure actual tokenizer and image costs where possible.
- [ ] Specify a controlled representation bake-off that separates state decoding from planning and free action construction from menu selection.
- [ ] Write a sourced memo with recommendations for black-box teachers, an open-weight student, and a versioned deployment/training contract.

## Review
Pending research and measurements.
