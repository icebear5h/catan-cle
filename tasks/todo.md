# Resume Miles SFT smoke (2026-09-23)

- [x] Confirm the user authorized resuming the stopped corrected build.
- [x] Reuse the completed r03 merge/data and select fresh `training-r03` output.
- [ ] Complete two audited optimizer updates and validate the full serving export.
- [ ] Run the 32-case SGLang reload panel, collect evidence, and stop smoke resources.
- [ ] Record results and run the required quality gate for any repair edits.

Plan: use the compatible pinned Bridge/Core image and the existing immutable
`miles-topology-smoke-20260923-r03` inputs in tetracorp. Keep one H200, the existing
two-update limit and 30-minute function cap; preserve all earlier attempts.
Review: resumed smoke pending.

# Focused complete Miles export repair (2026-09-23)

- [x] Inspect merge provenance and runtime save/finalization contracts.
- [x] Compose fresh complete HF shards from admitted Bridge updates and exact frozen base tensors/assets.
- [x] Bind post-save/final receipts to raw Bridge and final model paths plus composition hashes.
- [x] Verify real safetensors preservation/tamper cases, focused runtime tests, and scoped lint/typing.

Design: manifest-authorized trained keys only; one base shard at a time; preserve
raw Bridge output; seal a distinct composition manifest last. Full artifact hashes
are export/finalization work, never per-step work.

Review: focused export/runtime tests **22 passed, 1 skipped** (Miles unavailable
locally); scoped Ruff and strict mypy pass. Repository-required quality check
passes structure/mypy, with unrelated import-order failures in the three
`tests/traces_journal/test_{calls,commands,transactions}.py` files. API and
raw/final receipt fields are documented in `sft/miles_sft/runtime/README.md`.

# Miles/Megatron LoRA SFT port (2026-09-22)

- [x] Inspect the existing SFT bundle, Miles hooks, and exact Qwen3.8 architecture.
- [x] Resolve initialization: user selected merged r04 weights plus fresh
      Megatron LoRA, with historical token rows preserved in the frozen base.
- [x] Implement a separate primitive-topology SFT data projection with native
      no-thinking tokens, completion masks, provenance, and no truncation.
- [x] Implement the pinned single-H200 Miles/Megatron training and checkpoint path.
- [ ] Verify local contracts and prepare a bounded training/save/reload/eval smoke.
- [x] Review changes, run the full quality gate, and record results.

Plan: keep SFT primitives separate from symbolic-RL tasks while sharing model and
evaluation infrastructure. Ray orchestrates, Megatron trains LoRA, and SGLang runs
generation only for evaluations/GRPO. Preserve existing work and historical receipts.
The first run must explicitly name its initialization and trainable scope.

Discovery: exact Qwen3.8-27B Bridge support exists, but Miles cannot directly import
our HF PEFT checkpoint. Independent Q/K/V, gate/up, and GDN adapter factors cannot
be losslessly collapsed into the same-rank fused adapters. Its stock adapter saver
also omits our selected embedding/output rows. A merged-weight warm-start avoids
the factor-import problem but intentionally starts a new adapter parameterization;
exact continuation requires custom split adapters and import/export support.

Review: source implementation and local verification complete: 90 tests pass,
one real-Miles integration test is skipped, and full quality passes. Review found
Bridge MTP/FP32-vision/tokenizer export gaps; final export composition now restores
all frozen tensors/assets exactly. CLI and bounded Modal CPU/GPU launcher added.
Historical r04 assets were copied from icebear5h to tetracorp with matching saved
hashes. Real CPU merge/preflight passed for miles-topology-smoke-20260923-r03:
all 18 shards merged, FP32 visual bytes preserved, 1,600 native-tokenized examples
(82,386 tokens, 7,084 supervised, max length 65). Earlier CPU attempts exposed and
fixed resolved-volume aliases and Qwen's integral-float index byte count.
The first GPU attempt (ap-rbt0zz8JAddcFLoVkTyqa5) found an upstream Bridge/Core
mismatch before training. Pins now use the documented compatible pair
40b93089/73b54618; native TE imports require libcuda and are checked before weight
scans on the GPU worker. The corrected attempt ap-vla1cKW0SyrnwkN6x2aJnu reached
Qwen35VL provider/fresh LoRA creation, then received SIGTERM and Modal reported
"function is stopped". Only initialize-rank-0.json exists under training-r02;
no optimizer updates or successful reload are verified. No Catan Miles app remains
active. Confirm whether termination was intentional before another paid attempt.

# Per-sandbox ordered durable execution (2026-09-22)

- [x] Audit current step, barrier, trace transaction, and checkpoint boundaries.
- [x] Add an append-ordered SQLite command/call journal with checkpoint fencing.
- [x] Add an opt-in durable sandbox runner and replayable completion transport.
- [x] Verify duplicate delivery, interrupted acquisition, cancellation, stale writers,
      same-revision state, and atomic persistence with real sandbox integration tests.
- [x] Document usage, recovery limits, and review; run the full quality gate.

Design: keep canonical engine event order and current viewer trace projections.
A separate journal orders command starts, model invocations/results, recovery, and
settlement across all seats of one sandbox. Stable caller command IDs return saved
outcomes; transactionally committed snapshots are authoritative. Recovery fences
old writers and re-drives uncommitted local work from the command checkpoint using
saved provider responses. Unknown remote outcomes require explicit retry consent;
this does not claim exactly-once remote inference or arbitrary external callbacks.
Existing viewer restore stays compatible; durable execution is an explicit runner
for rollout callers, with no live-server restart or paid run during implementation.

Options considered: after-the-fact log-only ordering cannot prevent duplicate
execution; modifying every engine action risks recording staged validation. The
selected command wrapper + transport receipt boundary preserves engine semantics
and supports headless games without depending on Flask.

Review: added `cle/sandbox/durable/` and `cle/traces/journal/`, with 46 focused
journal/runtime cases including a spawned worker killed without Python cleanup.
Independent review fixes cover admission write uncertainty, resolved provider
identity, policy changes on settled delivery, image/exception evidence redaction,
and idle context-policy migration. Related sandbox/trace regression run:
**375 passed**. Scoped Ruff and strict mypy pass on all 30 authored Python files.
Required `uv run --no-sync python -m scripts.quality` ran and remains red outside
this feature: `playground/frontend/src/App.tsx` is 1,550 lines, plus Ruff/mypy
failures elsewhere in the ongoing repository cleanup. No suppressions added.
Docs/usage: `cle/sandbox/durable/README.md`. No live server restart or paid run.

# Stricter linter cleanup

The active checklist, completed work, pending review repairs, and verification
commands are in [stricter-linter.md](stricter-linter.md). Update that file for
ongoing quality cleanup. Earlier execution records remain below.

# Board-recognition structure cleanup (2026-09-21)

- [x] Inspect ownership, physical-source consumers, and relocated test baselines.
- [x] Split eight eligible oversized modules into same-name packages.
- [x] Check byte/order parity on existing fixtures, public identities, and paths.
- [x] Run scoped tests/lint/typing; record remaining debt and structural counts.

Plan: cohesive implementation modules with explicit public re-exports. Preserve
source-hashed replay/source/robber/piece modules because their SFT callers are
outside scope. Selected: inverse_grounding, spatial_localization,
adjacent_pair_localization, query_schedule, replay_ms_swift,
production_curriculum, dataset, density_curriculum. Baseline: 109 tests pass.
No source.py or semantic_v2.py exists. Historical manifests and hashes stay pinned.

Review: root 23 -> 15 Python files. Eight oversized files (4,563 lines) became
28 source files (4,704 lines including explicit exports/imports), each <=300
physical lines; package sizes range from 2 to 6 direct source files. Largest
file: 856 -> 300. Original owned exports and historical pickle lookups pass;
dataset class identity/project root and both module CLIs are preserved.

Final explicit-interpreter run: 177 passed, 11 pre-existing setup errors in
tests/data_pipeline/recognition/export/test_reweight_node_edge.py (obsolete
full_coverage keyword; original node readout signature already uses coverage).
All 19 new contract tests pass, including 37 complete export files compared
byte-for-byte with revision 2a27dbc, 12 frozen output/order witnesses including
292 marker/probe PNGs, and 16 pair images/rows under serial and spawned workers.
Scoped Ruff and diff checks pass. Scoped strict mypy retains 12 diagnostics:
eight pre-existing JsonDict/Any aliases and four missing jsonschema/networkx
stub sites; no added Any or ignores. The global quality gate is reserved for
the coordinating main's final run while concurrent work continues.

Use uv run --no-sync python -m pytest: .venv/bin/pytest has a stale shebang into
another checkout and different Pillow byte output. Details and test commands
are in tests/recognition_contracts/README.md. Source-hashed replay_dataset,
sources, spatial_robber, single_piece_localization, and node_edge_readout remain
at their existing paths; their external SFT callers were not changed.

# Format module structure cleanup (2026-09-21)

- [x] Capture the four-test baseline and immutable scorer source fingerprints.
- [x] Pin byte/order witnesses and split only ASCII variations/full-graph formats.
- [x] Verify exports, focused tests, file/folder counts, and strict typing debt.

Plan: same-name package facades over cohesive graph, codec, rendering, question,
data, and scoring modules; preserve all original owned public/private symbols.
Review: 24 baseline tests and 29 final tests pass. All 192 rendered-output/order
witnesses, minimal-graph projections, 99 generated file bytes, and six scoring
source fingerprints match the pre-edit capture. Scorer version remains v2 and
SHA256 remains b80971f02ef525a56f5e84bd97801276f953620d82211cabc4a716baf9b9ed3e.
All 92 original owned functions/classes and their constants remain exported.
Benchmark root: 17 -> 15 direct files; both new packages: 11 files, maxima 279
and 216 physical lines. Strict scoped mypy: 29 -> 5 errors (one existing JSON
alias plus four frozen-scoring diagnostics); Ruff: only three guarded ANN401s.
No suppressions added. Required full quality run: 209 repository structural
violations and existing Ruff/mypy failures; scoped structure and diff checks pass.

# Repository quality cleanup (2026-09-21)

Continue from the strict local quality gate. Starting measurement: 213 files
over 300 physical lines, 10 source folders over 15 direct files, plus Ruff/mypy
debt. Preserve behavior, saved snapshot identities, canonical board topology,
and existing concurrent work. No baselines or weakened rules.

- [x] Inspect current worktree and prioritize cohesive clean areas.
- [x] Extract typed trade data contracts from lifecycle behavior; preserve
      public imports, duplicate-deal identity, ordering, and pickle restoration.
- [x] Split map types/templates/generation from assembly while preserving
      canonical entity IDs, RNG consumption, and supplied-list behavior.
- [x] Clean player type contracts and extract cohesive oversized methods;
      preserve accepted-action/history and communication behavior.
- [x] Confirm relocation scope: user explicitly authorized moves and same-name
      package replacements while preserving behavior and active work.
- [x] Group 44 scripts and 124 tests into canonical ownership subfolders.
- [x] Split two benchmark format modules and eight recognition generators
      into same-name packages; retain scorer/output/pickle identity.
- [x] Organize frontend components and split nine oversized components/styles
      across both passes while preserving complete production CSS bytes.
- [x] Verify focused behavior tests, inspect the combined changes, run the
      complete quality gate, and record remaining debt.

Plan choice: extract cohesive typed implementations behind existing public
module boundaries first. A wholesale namespace rewrite would unnecessarily
affect imports and persisted snapshots; new engine model files fit the folder
limit. Source/test relocations use the user-authorized canonical ownership layout.

## Review

Implementation and independent reviews complete. The full quality run now
reports 194 structural violations (190 oversized files, 4 oversized folders),
down from 223 (213/10), plus 5,750 Ruff errors and strict mypy failures. Remaining
overfull folders are cle/harness, sft, sft/launchers, and sft/scripts. Existing
JSON aliases and frozen-source scorer typing remain explicit debt; no rules or
baselines were relaxed. `git diff --check` passes.

Focused verification completed by area:
- Player suites: 318 before/after plus 4 new contracts; map: 43 before/46 after;
  trade: 100 before/106 after with the same 2 existing payload assertions and
  1 skip. Concrete local type checks pass for player and map/trade modules.
- Benchmark formats: 29 pass; 192 rendered outputs and 99 generated files
  match pre-edit witnesses; strict scorer SHA remains
  b80971f02ef525a56f5e84bd97801276f953620d82211cabc4a716baf9b9ed3e.
- Script migration: 96 before/after; final builders rename: 37 pass. Modal
  probe's unsupported `required_hint` argument reproduces on HEAD.
- Recognition: 177 pass with 11 existing obsolete-full_coverage setup errors;
  witnesses include 37 export files, 292 PNGs, and 16 serial/spawned pair images.
  Review found a lost public sampler override seam; repaired and verified by
  14 focused tests, including all six helper call sequences against HEAD.
- Frontends: 61 tests and both production builds pass. Both full CSS output
  hashes match the original baseline exactly.
- Relocation identities match under the original launcher, but its shebang was
  stale. Correct explicit-interpreter collection succeeds for all 3,429 tests;
  no missing-peft errors occur in the intended environment. Commands corrected.

Final combined explicit-interpreter regression hit the 180-second budget while
repeating full-dataset export byte checks. Those four checks passed separately.
The remaining combined selection completed in 63 seconds: **862 passed,
1 skipped, 1 failed, 4 deselected**. The failure is
`tests/evals/reasoning/test_initial_settlement_reasoning.py::test_capture_trace_omits_max_tokens_and_separates_native_reasoning`:
the request now sends `max_tokens` despite the probe's uncapped transport config.
It reproduces alone. During this work, concurrent changes added per-request
token overrides in `cle/harness/models.py` and provider transports, including
OpenRouter's request-over-config precedence. Those active changes are outside
the cleanup; the integration conflict is reported and left for their owner.
Independent review verified two-way engine pickle interchange, live call sites,
scorer source identity, board draw order, and exact stylesheet output.

# Local code-quality gates and OpenCode (2026-09-21)

Scope: deterministic local lint/typing/structure checks and project OpenCode
integration. The user clarified that established CI is absent in this repo.

- [x] Inspect existing tooling, instructions, OpenCode hooks, and structural debt.
- [x] Confirm rollout policy for existing violations before implementation:
      user chose strict repo-wide, with no existing-debt exemptions.
- [x] Define source scope, 300 physical lines/file, 15 direct files/folder,
      Python signature/type checks, and uv-managed dependency edits.
- [x] Implement a shared local check command and OpenCode integration.
- [x] Verify passing/failing cases and document commands, enforcement limits,
      existing debt, and the required OpenCode restart.

Initial audit: 213 of 534 source files exceeded 300 physical lines; 9 source
folders exceeded 15 direct files. Annotation rules/type checking and project
OpenCode integration were absent. The concurrent SFT launcher migration adds
another oversized source folder; the current gate reports 223 structural failures.

## Review

Implemented `scripts.quality`, strict Ruff/mypy configuration, a project-local
OpenCode plugin and `/quality`. Package changes used uv. The plugin reports
structure violations after edits/shell calls, compares manifest dependency fields
before edits, and blocks ordinary standalone commits unless the entire worktree
is staged and the full check passes. Checks remain advisory for repair edits.

Independent review identified and resolved host edit/patch semantic mismatches
and overly broad exclusion names. Manifest previews now match pinned OpenCode
1.18.31 behavior; exact-context configuration patches work for GPT sessions.
Authored nested `data`/`cache` folders remain in scope.

Final verification: 38 Python quality tests and 26 Node plugin tests pass. Ruff
and strict mypy pass for all 11 quality Python source/test files. `uv lock --check`
and `git diff --check` pass. The required full `uv run --no-sync python -m
scripts.quality` run correctly exits 1: 213 files exceed 300 lines, 10 folders
exceed 15 direct files, and existing Ruff/mypy failures remain. No baseline or
exemptions were added for that debt. Restart OpenCode to activate the plugin;
its commit gate will stay red until the repository-wide failures are resolved.

# Organize SFT modules under the folder cap (2026-09-21)

Scope: bring `sft/` (21 direct files), `sft/launchers/` (22), and `sft/scripts/`
(28) under the 15-file structural cap by relocating modules into ownership
subpackages, with no behavior change and no re-export shims at the old paths.

- [x] Baseline the focused SFT selection: 644 passed, 10 failed, 33 errors
      (the errors and failures are pre-existing engine-relocation fallout under
      `cle/game_engine/models/board.py`, not SFT); 3463 tests collected.
- [x] Move root board modules into `sft/board/` and diagnostics into `sft/analysis/`.
      `sft/diagnostics` is a forbidden path in `scripts/verify_sft_layout.py`
      (it was the migrated artifact directory), so the diagnostics package is
      named `analysis`. `board_state_readout.py` stays at the root because
      `data_pipeline/board_recognition/full_board_readout.py` imports it and
      pipeline source was out of scope.
- [x] Group launchers into `board_fluency/`, `full_board/`, `qwen_series/`, and
      `spatial/`; the seven singletons stay at `sft/launchers/`.
- [x] Group scripts into `builders/`, `eval/`, `render/`, `report/`, and `train/`.
- [x] Rewrite every live import, `-m` module string, source-file list, relative
      `__file__` depth, test, and README command; fix import order only in files
      that were isort-clean before the move.

## Review

53 modules moved (38 tracked via `git mv`, 15 untracked from the in-flight
launcher/pack drafts via plain rename). Content changes are limited to import
lines, module-path strings, `parents[n]` depth, and docstring commands. Source-hash
manifests keyed by repository path (`SOURCE_FILES`, `SCORER_FILES`, `code_sha256`,
`source_sha256`) now record the new paths and new content hashes; historical
receipts keep theirs, and the existing revision checks continue to reject them.
Structure gate: no `max-files` findings remain. Focused selection and collection
counts match the baseline.

# Organize SFT launchers (2026-09-21)

Scope: move the 21 flat `sft/modal_*.py` entrypoints into `sft/launchers/`,
retaining descriptive basenames. This groups the largest source of root clutter
without a broader trainer/dataset ownership migration.

- [x] Inspect module groups, current imports/commands, packaging, and source-hash guards.
- [x] Establish the focused launcher-test baseline: 281 passed, 10 existing
      retained-summary failures (`by_family` / `by_operation` / `by_representation`).
- [x] Move launchers into a minimal subpackage and update current code references.
- [x] Update affected tests and current README/CLI documentation.
- [x] Verify the moved module imports, help entrypoints, focused tests, and stale
      live references; document historical source-revision requirements.

Historical receipts/manifests/reports keep their recorded paths and hashes;
existing checks must continue to reject mismatched source revisions. The dirty
atlas/pack drafts retain their current files and contents. No training or remote
evaluation is part of this source-organization task.

## Review

Moved 21 launchers into `sft/launchers/` with a minimal initializer. Updated live
Python imports, worker/stop commands, source-file lists, two local scripts, seven
test modules, and current SFT/evaluation README commands. Reviewed every move
against HEAD: only relocation references and module descriptions changed.

Verification: all 21 launchers cold-import in separate network-blocked processes;
the board-fluency evaluator's Modal CLI help and r04 extension's Python CLI help
work under their new names. The original 291-test selection remains 281 passed /
10 identical pre-existing historical-summary failures; adding the seven coordinate
comparison tests gives 288 passed / 10 failed. No new failures. Remaining old
launcher paths are intentional historical receipt keys. No remote jobs launched.

# Matched atlas-token / integer-coordinate comparison (2026-09-21)

User approved a small matched representation comparison after discussing replacing
atomic atlas tokens with coordinates. Initial experiment is inference-only on the
same latest r04 checkpoint; it measures usability by the current atlas-trained
model, not equal-budget retraining or tokenization alone.

- [x] Inspect reusable symbolic cases, geometry, scoring, and bounded Modal evaluation.
- [x] Build 200 paired canonical cases (400 generations): direction 64, neighbors 32,
      incidence 16, piece owner 32, owned nodes 24, owned roads 16, owned incident
      roads 16. Use original test rows plus two validation port-incidence rows.
- [x] Render atomic atlas IDs versus ordinary typed integer coordinates. Double the
      existing render lattice for integer edge/port midpoints; retain exact state,
      query, record order, source split/provenance, and canonical oracle per pair.
      Supply equivalent entity inventories for static queries in both arms.
- [x] Validate coordinate identity round trips, gold scoring, strict malformed-output
      rejection, source/mapping hashes, and complete pair membership before inference.
- [ ] Run checkpoint board-fluency-extension-20260915-r04/checkpoint-512 in text mode,
      greedy/no thinking/no candidate scoring, shared 4096 context and 512 output
      budget; CPU token preflight followed by one bounded H200 evaluation.
- [ ] Report paired wins/losses, per-operation and area accuracy, format failures,
      token lengths, compute estimate, and limitations; retain raw outputs.

Design choice: midpoint coordinates rather than endpoint-pair road names avoid
making edge-to-node queries literal copying. Coordinates expose geometry by
design. Static atlas tasks are fixed-topology diagnostics; original directional
pair and dynamic source holdouts retain their documented exposure limitations.
This work does not edit the existing atlas/pack drafts or source dataset.

## Review

Generated `artifacts/generated/sft/coordinate_comparison_v1/` with 200 pairs,
154 reversible typed positions, and 61 dynamic source states. Seven comparison
tests pass. Independent review found no critical/medium issues. Two existing
text-loader tests reproduce on HEAD due to local Transformers lazy-module
replacement; this does not affect the comparison scoring tests. Checkpoint-512
exists on the Modal volume. GPU execution and remote token preflight are pending
the user's requested code review; no inference job has been launched.

# Perception-only mix, coarse-to-fine, with NONE down-weighting (2026-09-18)

Constraint: board perception only. No composition, no counterfactuals, no reasoning.
Ordered the way a person reads the board - tiles first, then spatial relations, then down
to nodes and edges - and answers are atlas tokens wherever possible, because those are what
train `atlas_output_rows` (788,480 params, currently fed only incidentally).

## Tiers (1,318 static facts)

    0  identity        tile_coordinate 19, node_status 54, port_direction 9,
                       port_coordinate 9                                        (non-token)
    1  tiles           tile_nodes 19, tile_edges 19, tile_neighbors 19
    2  spatial L/R     tile_step 114 (LEFT/RIGHT/UP-LEFT/UP-RIGHT/DOWN-LEFT/DOWN-RIGHT),
                       node_step 324
    3  nodes           node_tiles 54, node_neighbors 54, node_port 54, node_edges 54
    4  edges           edge_endpoints 72, edge_tiles 72
    5  ports           port_nodes 9
    6  local distance  node_distance where d<=3, 363 pairs                      (non-token)

1,318 facts -> 5,272 examples at 4x exposure -> ~659 packed sequences at k=8.
r04 by comparison: 4,096 rows at 1.28 exposures per unique example.

Held OUT of the perception mix: node_path (1,431) and node_distance d>3 (1,068). Those are
traversal results, not perception - nobody *sees* that N02 is 7 edges from N46. Keeping
d<=3 gives local neighbourhood awareness; the long tail is substrate for the reasoning
track and would otherwise be 2,499 of 3,521 facts.

Dynamic layer, single-hop readouts only, rebalanced by convergence rather than uniform 205:
down-weight local_node_tiles and resource_pip_totals hard (both 10/10 since r02), hold
port_access / owned_incident_roads / owned_buildings_touching_resource, add raw tile,
node-occupancy and edge-road readouts.

## NONE down-weighting (implemented)

`generate_examples(..., none_weight=0.2)` in `sft/board_atlas.py`. Empty-answer keys are
drawn at `none_weight` relative to non-empty ones via Efraimidis-Spirakis weighted sampling
without replacement (`weighted_sample`), so no key repeats inside an example.

Why it is needed: most nodes have 2-3 neighbours across 6 directions, so uniform sampling
makes node_step 56% NONE and node_port 67% NONE. r04 trained at 35% NONE against 9% in its
eval, and 10 of its 72 review failures were "answered NONE when the answer was non-empty".
That prior came straight from the sampler.

Measured, exhaustive pass + 3,000 sampled:

    none_weight   overall   node_step   node_port
       1.00        10.7%      54.3%       67.7%
       0.50         8.1%      39.5%       53.3%
       0.20         5.2%      23.8%       36.2%
       0.10         3.6%      16.5%       25.5%
       0.05         2.6%      12.0%       18.0%

0.5 matches the eval's ~9% NONE; 0.2 deliberately undershoots it to counteract the existing
over-prediction. Coverage is unaffected at every setting (3,521/3,521) because the
exhaustive pass still emits each empty fact once - only repetition is reduced.

`answer_shape_report(tables, examples)` reports NONE share per table and overall, so the
sampler's bias is visible before a run rather than after one.

65 tests passing across board_atlas / board_readouts / board_packs.

- [ ] Add tile_step (114 directional tile relations) - currently only in the live renderer
      as QI|TILE_NEIGHBORS, not in any generator.
- [ ] Per-tier exposure multipliers feeding generate_examples table_weights.
- [ ] Wire the dynamic single-hop readouts into the same pack format.

# Orientation was dropped, not missing; node_step added (2026-09-17)

## The board-fluency corpus dropped capabilities the symbolic lineage had

`sft/symbolic_board_tasks.py` STATIC_TASKS already contains:
  symbolic_direction        "Is <N12> strictly left of <N30>? Compare that axis
                             independently; equality means no."
  symbolic_direction_choice "Which is farther left: A or B?"
  symbolic_oriented_step    "From <N12>, take exactly one node-edge step NORTHEAST."
  symbolic_neighbors        node adjacency
  symbolic_incidence

None of the 20 board-fluency operations mention left, right, above, below or any compass
direction, and none ask for bare adjacency. So left/right grounding and node adjacency are
not unbuilt - they existed in the earlier lineage and did not carry over into the corpus
r04 trained on. Check whether that lineage still feeds anything before building more.

Left/right is positional, not topological: `_relation` compares `_atlas()["positions"]` x/y
in the render frame. It therefore belongs with coordinates (cube/axial exist in
`coordinate_system.py`, and `tile_coordinate` atlas facts already emit Q R S), not with the
graph.

## node_step: oriented adjacency, added to sft/board_atlas.py

Measured first: every node-to-node edge lies along one of the six `OFFSETS` directions,
exactly 24 edges per direction, and ZERO nodes have two neighbours in the same direction.
So `node_step(node, direction) -> neighbour | NONE` is total and unambiguous.

  54 nodes x 6 directions = 324 facts; 144 filled (= 2 x 72 edges), 180 NONE (the coastline).
  Union over the six directions reproduces `node_neighbors` exactly for all 54 nodes.

It subsumes node_neighbors and is the only table that grounds orientation. Tests assert
totality, at-most-one-neighbour-per-direction, union equivalence, filled-slot count, and
step-back symmetry (stepping NORTH then SOUTH returns the origin).

Atlas is now 3,521 facts / 11 tables. 37 tests passing.

CAVEAT: node_step is 56% NONE (180/324). The r04 failure profile already includes 10 cases
of answering NONE when the answer was non-empty, and 88.9% accuracy on genuinely-empty
answers, i.e. a NONE bias already exists. Watch NONE-precision when this table is mixed in,
and consider downweighting the empty slots rather than sampling them uniformly.

Also fixed: multi-token keys ("<N19> NORTH", "<N00> <N01>") were space-joined in the prompt,
leaving the query list unsegmentable. Keys are now semicolon-separated, with a test.

## VP: visible VP needs knights in the serialization

Decision taken: visible VP, not board-only VP. Visible VP includes Largest Army, and knight
counts appear nowhere in the board text (only tiles/nodes/edges/ports/robber), so
`get_visible_victory_points` (`state_functions.py:76`, reads P{i}_VICTORY_POINTS) is not
derivable from the prompt.

- [ ] Add knight counts (and army holder) to `cle/env/observation_formatter.py` so visible
      VP becomes computable from the prompt. Without this, any VP task trained on board
      state alone is systematically 2 points low on exactly the players who are winning.
- [ ] Longest road is currently hard (6/10 component_roads, 1/10 reachable_nodes upstream).
      Its VP rung depends on the component partition, so `connected_roads` lands first.
- [ ] symbolic_longest_{lengths,leaders,award} are TRANSFER_TASKS, admitted only under
      transfer_validation/transfer_test. Training VP trains their composition and
      contaminates that holdout. Decide deliberately before building the longest-road rung.

# Connected-roads readout + atlas reconciliation (2026-09-17)

## Already covered, do not rebuild

node->tile and tile->node adjacency exist in BOTH `sft/scripts/build_atlas_topology_dataset.py`
(as node_tiles / tile_nodes) and `sft/board_atlas.py`. Cross-checked fact by fact: 0
mismatches on node_tiles, node_edges, tile_nodes, edge_endpoints.

The cross-check caught a bug in `sft/board_atlas.py`, not in the existing builder: port_nodes
disagreed on 6 of 9 ports because intersecting a port hex's six node refs with the land set
yields three nodes for several ports. A port grants access through the two corners of the hex
edge it occupies - `PORT_DIRECTION_TO_NODEREFS`. Fixed; all overlapping tables now agree and
node_port is a clean inverse (18 nodes = 9 ports x 2). Keeping both sources so they continue
to check each other.

## New: `sft/board_readouts.py` + `tests/test_board_readouts.py` (13 tests, passing)

`connected_roads(facts, color)` states a colour's road partition outright, as perception
rather than traversal. Component semantics are NOT redefined - it calls `Facts.components`
from `build_board_fluency_review`, so the readout and the graded ops cannot drift.

Verified: the readout implies the existing `component_count` and `component_roads` golds
exactly, 0 inconsistencies across those rows in the r04 review panel.

Scoring is partition-aware, because exact match is useless for grading here - one misplaced
edge among twenty reads the same as answering nothing. `score_readout` reports component
recall, edge recall/missing/spurious, and a Rand-style `pair_agreement` over edge pairs that
degrades smoothly. Tested: merging all components keeps edge_recall 1.0 but drops component
recall to 0; splitting one component keeps every edge but lowers pair agreement; one misplaced
edge scores strictly better than total collapse.

## Load-bearing measurement: the pack generator can ride on existing semantics

`decode_state` + `Facts` + `answer` reproduced **200/200 review golds across all 20 operations**.
So board packs can be generated against the live implementation with no reimplementation and
no risk of semantic drift.

## Honest limit: connected_roads is a scaffold, not a density anchor

Measured over 800 colour-boards: readout answers average 12.9 tokens (median 10.2, max 35.8).
Boards carry ~21 roads split across 4 colours, so ~5 roads per colour. At 4 colours that is
~51 answer tokens against ~800 prompt tokens = 6% loss-bearing from the anchor alone.

Its value is as substrate for the four connectivity ops (reachable_nodes, shortest_distance,
component_count, component_roads), which become lookups into an answer already in the
sequence. Density still has to come from `local_node_tiles` (30 tok x 54) and
`settlement_upgrade_production` (43 tok).

Board density itself is healthy and is NOT the problem: density_bin setup/dense/sparse =
58/83/59 over the review rows, roads mean 20.9 median 18 max 57, buildings mean 8.5 max 19.

- [ ] Compose the pack: connected_roads as scaffold + local_node_tiles as density anchor +
      traversal queries riding along on the colour whose readout is already present.
- [ ] Consolidate the two atlas builders, or keep both and wire only the connectivity tables
      (node_neighbors, node_distance, node_path, tile_neighbors, node_port) into the mix -
      the existing builder is pure incidence and has no connectivity at all.

# Board packs: compose queries per board, do not batch same-task (2026-09-17)

Measured against r04 review records. Query space per board, from the `target.query`
parameterization (4 colors, 54 nodes, 72 edges, 19 tiles, 11 rolls, 5 resources):

    robber_move_production        15,048      settlement_upgrade_production  2,376
    shortest_distance              5,724      component_roads                  288
    reachable_nodes / owned_incident_roads 216
    local_node_tiles / node_pip_sum / distance_rule_witnesses  54
    coverage_* / port_access / component_count    4-12
    resource_pip_totals / resource_pip_argmax        1

The corpus uses ONE query per board and then discards the board. 3,200 boards against
~24k available queries each is roughly 0.004% utilisation.

## Three regimes, not one

A. Same-task batching wins: long answers, large query space.
   settlement_upgrade_production (43 ans tok) hits 30% loss-bearing at N=8, 93% at cap.
   local_node_tiles (30 tok) 30% at N=12, 63% at all 54. component_roads N=22.
   reachable_nodes N=68.
B. Same-task is impossible: resource_pip_totals and resource_pip_argmax admit exactly ONE
   query per board; port_access / component_count / coverage_missing admit four. Asking
   every query that exists still leaves 1-4% loss-bearing. The query space binds, not the
   sampling.
C. Answer smaller than the query: node_pip_sum answers average 0.4 tokens against ~3 to
   name the node; all 54 queries reach 2%. Same for component_count (0.2) and
   shortest_distance (1.2).

## Design

Pack by BOARD, composing operations by answer length - same-task is a special case that
works for A and cannot work for B.

- Anchor each pack with regime-A ops to supply loss-bearing tokens.
- Ride regime-C ops along on the SAME arguments. node_pip_sum(<N23>) costs 0.4 tokens when
  local_node_tiles(<N23>) is already in the sequence.
- Sweep regime B wholesale - only ~36 global/color-indexed queries exist per board.

The ride-along is the main prize: local_node_tiles is at 100% and node_pip_sum at 40%,
and pip sum is exactly "take those tiles, map number to pip, add". Co-locating them on the
same node puts the retrieval in context when the arithmetic happens - the compositional
scaffold, delivered as a prior answer rather than as reasoning tokens, at ~0.4 tokens
marginal cost. Satisfies the no-CoT-for-perception constraint.

- [ ] Build the board-pack generator over `sft/symbolic_board_tasks.py`, reusing the
      multi-entry KEY: VALUE format and per-entry scorer from `sft/board_atlas.py`.
- [ ] Pack composition policy: anchor/ride-along/sweep proportions per pack.
- [ ] Argument-sharing policy: how often ride-along ops reuse the anchor's node, vs
      independent draws. Full sharing maximises scaffold, zero sharing maximises coverage.
- [ ] Re-measure loss-bearing fraction with the real tokenizer, not the 4-chars/token proxy
      used for the table above.

# Atlas generator built (2026-09-17)

`sft/board_atlas.py` + `tests/test_board_atlas.py` (18 tests, passing). Generates the
static-topology curriculum described in the section below. Nothing existing was touched.

3,197 facts across 10 tables, all derived from `STATIC_GRAPH` / `base_map` rather than
restated: node_neighbors 54, node_edges 54, edge_endpoints 72, node_tiles 54, tile_nodes 19,
tile_neighbors 19, port_nodes 9, node_port 54, node_distance 1431, node_path 1431.
Verified against corpus gold - `node_tiles <N09>` = T01 T02 T08, `node_neighbors <N19>` =
N20 N21 N46.

Design decisions worth keeping:

- Keys travel in the prompt and examples draw a random subset in random order. A model
  trained on full ordered dumps learns the sequence, not the facts, and then cannot answer
  about one node in isolation - which is the form traversal consumes.
- Cardinality is sampled log-uniform (octave-uniform) over [1, max_k]: ~20% of examples are
  k=1, ~45% are k<=4. Uniform-over-k would drown single-key queries.
- Per-entry scoring, not exact match. A 24-entry dump with 23 right scores 0 under
  `score_board_fluency`, making progress invisible; `score_atlas` reports entry accuracy
  plus atom precision/recall, set answers order-free and sequence answers order-sensitive.
- `coverage_report` asserts every fact was emitted. Closed world, so an uncovered key is a
  generation bug, not a sampling outcome - and there is no held-out split because there is
  nothing to generalize to.

Measured: one exhaustive pass = 406 examples covering all 3,197 facts. A 2,436-example
corpus (exhaustive pass + 2,000 sampled) is 45.1% loss-bearing tokens against ~5% for the
current board-fluency corpus.

- [ ] Decide max_k and the exhaustive:sampled ratio against a real tokenizer count rather
      than the whitespace proxy used above.
- [ ] Wire into a prepare stage and pick the curriculum schedule (atlas to saturation
      first, vs mixed in at high weight and annealed).
- [ ] Keep incident-encoding as a control arm. If the atlas takes single-hop adjacency to
      ~100% and traversal still does not move, the remaining failure is frontier
      termination, not missing structure - worth knowing which.
- [ ] Rotations/reflections rejected: the land graph is static with fixed labels, so a
      rotation is an automorphism that relabels nodes and would teach node identity as
      relative, fighting the fixed atlas. Resource/number/piece placement already varies
      per generated board, so a rotated assignment adds no information.

# Board topology is absent from the prompt (2026-09-17)

Supersedes the ordering in the section below: test this BEFORE deleting the engine-track
ops or building any CoT/RL machinery. It is the cheapest hypothesis and it may explain
most of the failure.

## Finding

The serialized board (`metadata.target.state.board`, ~2373 chars) is four flat lists plus
the robber:

    <T00> brick 9; ... <T18> sheep 4;          19 tiles: resource + number
    <N00> empty; ... <N53> empty;              54 nodes: occupancy only
    <E00_01> empty; ... <E52_53> empty;        72 edges: road owner only
    <P00> brick port; ... <P08> 3:1 port;      9 ports
    robber <T03>

It says what sits ON each element. It never says how elements CONNECT.

- Node -> tile incidence: absent (verified, zero occurrences of a tile token in any node
  clause). "Which tiles touch <N09>" is answerable only from memorized topology.
- Node -> node adjacency: only implicit in edge token NAMES.
- Node -> port incidence: absent.

And `trainable_tokens.json` shows all 154 atlas tokens (54 N, 72 E, 19 T, 9 P) are
`regular_added_tokens`, trainable on input and output. So `<E19_21>` is a SINGLE ATOMIC
token - the model cannot lexically read "19" or "21" out of it. The endpoints are
recoverable only from what the embedding learned.

Therefore the entire board graph lives in 154 learned embeddings, not in context. A
reachability query forces alternation between embedding-recall (which edges touch N19?
what is the far endpoint?) and prompt-read (is that edge blue?), once per hop, with no
place to write intermediate state.

This predicts the observed profile exactly:
- local_node_tiles 100%: one embedding recall + one prompt read. Single hop.
- reachable_nodes 10%: unbounded alternation of recall and read.
- Decay with answer cardinality: each additional element needs another full recall/read
  cycle.

Note the topology is STATIC (`board.py:23-36`, `STATIC_GRAPH` + lru_cached
floyd_warshall) - only resources, numbers, buildings, roads and robber vary per game. So
`<N39>` has a stable referent; the tokens are not ill-defined, they are simply being asked
to carry structure that belongs in the context window.

## Action

- [ ] Re-serialize incident-style: for each node, explicitly list its neighbouring nodes,
      its touching tiles, and its port if any. Talk Like a Graph (Fatemi et al., ICLR 2024,
      arXiv:2310.04560) measures Incident encoding 53.8% vs Friendship 4.0% zero-shot on
      connected-nodes, and Incident 25.0% vs Adjacency 12.4% on node degree; encoding
      choice alone swings accuracy 4.8-61.8 points across tasks. We are currently below
      even their worst encoding, since we supply no adjacency at all.
      Cost: roughly doubles the prompt (~650 extra tokens). The block is identical across
      boards, so it prefix-caches, and the batched multi-QA plan amortizes it further.
      Requires a short retrain - the current checkpoint is trained on the present format,
      so a prompt change alone is out of distribution.
- [ ] Re-test reachable_nodes / shortest_distance / component_* under incident encoding
      BEFORE committing to the engine-offload track. If they move substantially, the
      ceiling argument was partly a serialization artifact and the offload is optional
      rather than necessary.
- [ ] Consider replacing opaque `<N39>` with axial hex coordinates in a parallel arm.
      Talk Like a Graph found integer IDs help integer-output tasks while semantic IDs win
      on relational-output tasks like ours; a fused `<N39>` token discards even the integer
      39 the model could do arithmetic on, and added tokens carry no pretraining prior plus
      documented under-training pathologies (arXiv:2608.03494 - subword-composition init
      beats random init by >6x fewer steps to equivalent loss). Audit per-token occurrence
      counts in the SFT corpus for under-trained embeddings.
      No paper runs the exact ablation (opaque per-entity token vs axial coordinate, same
      task/model) - this is a real literature gap and a cheap, well-scoped experiment.
- [ ] Longer shot: the field's actual answer to "give an LLM graph structure" is
      structure-derived tokens, not per-entity symbols - GraphToken (arXiv:2402.05862,
      GNN-derived soft prompts) reports up to 73 point gains over text encodings; <SOG_k>
      (arXiv:2602.01771) compresses a whole graph into one VQ codebook token. Both train
      the token via a topology-aware encoder. Our embeddings get no structural training
      signal at all.

## Also from the spatial sweep

- Serialization friction is independently documented: text serialization of 2D structure
  collapsed 92.7% -> 0.8% from 12x12 to 20x20 grids while 2D-native stayed >90%, and
  rendering the serialized text as an image did NOT recover it (arXiv:2604.27272). The
  loss happens at serialization time, not at the input modality - so the vision track will
  not rescue a bad serialization.
- Our cardinality decay matches documented VLM counting decay (Gemini 2.5 Pro 60.3% at 1-5
  objects -> 13.9% at 50+). Diagnosed mechanism is diffuse attention failing to keep
  instances distinct ("feature conflation"), NOT language-side arithmetic - consistent with
  a representation problem rather than a reasoning-step problem. No canonical name for the
  pattern; "subitizing cliff" is the borrowed term.
- SFT memorizes / RL generalizes was tested on GeneralPoints (a card game) and V-IRL in
  both text and visual variants (arXiv:2501.17161): SFT failed OOD, RL generalized.
  Reason-RFT (arXiv:2503.20752) on counting specifically: RL beat SFT +12% (2B) / +17%
  (7B), and OOD SFT got worse while RL kept generalizing.
- Steal SVQA-R1's consistency reward (arXiv:2506.01371): they perturb the scene
  (mirror-flip) and require consistent answers. The hex board has rotational and
  reflective symmetry, so we can generate equivalent boards from the engine and require
  answers to transform correspondingly - a free grounding reward with no extra labels,
  and a direct check against memorization shortcuts on a Qwen base.
- Data scale reality check: working synthetic-spatial pipelines run 3.4M (SpaRE), 8.5M
  (GRAID) and 2B (SpatialVLM) QA pairs. We run 3,200 unique examples. Boards are cheap to
  generate; this may be a plain data-volume problem on top of the 5% loss-density problem.
- Biggest evidence gap: no study isolates generalization to novel structural
  configurations (held-out layouts) as opposed to novel phrasing or novel rendering.
  Confirm our eval splits hold out LAYOUTS, not just questions.

# Board fluency: split perception / engine / thinking tracks (2026-09-17)

Context: `board-fluency-extension-20260915-r04` completed 2026-09-16 17:37Z, status
completed, 512 updates (1536 cumulative), train_loss 0.128 / eval_loss 0.201 at corpus
epoch 1.0. Review panel 64.0% (128/200), validation_eval 71.6% (136/190). Four extension
rounds moved review 29% -> 64%, but the gain is concentrated in operations that were
already working.

## Diagnosis (recomputed from r04 posteval records)

- Per-operation, review panel (/10): reachable_nodes 1, settlement_upgrade_production 3,
  coverage_intersection 4, node_pip_sum 4, coverage_difference 5, coverage_missing 5,
  ... local_node_tiles 10, resource_pip_totals 10.
- Accuracy decays monotonically with expected-answer cardinality:
  size 0 = 88.9%, 1 = 66.7%, 2 = 52.2%, 3 = 40.0%, 4 = 20.0%.
- Of 72 failures: 31 set answers off by 1-2 elements, 6 numerics off by +/-1, 10 answered
  NONE when the answer was non-empty. Boundary errors, not confusion.
- r02/r03/r04 per-op trend: reachable_nodes 0/0/1, node_pip_sum 4/4/4,
  settlement_upgrade_production 1/3/3, vs resource_pip_totals 4/6/10 and
  distance_rule_witnesses 1/2/7. Every multi-hop op is pinned; lookups saturate.
- Training allocation is uniform ~205 examples/op regardless of 100% or 10% accuracy.
- Every training row uses a unique board state (epoch3 2304 rows / 2304 states; epoch4
  1792/1792). Loss-bearing tokens are ~5% of each sequence (~900 prompt : ~52 answer).
- Prompts end in "No explanation."; targets average 1.9 space-separated tokens;
  `reasoning_enabled: false`. No reasoning/derivation/witness field exists in
  `sft/symbolic_board_tasks.py`.

Conclusion: not undertrained. Direct-answer decoding caps data-dependent-depth
computation. Supporting: no-CoT transformers are TC0-bounded (Merrill & Sabharwal, TACL
2022); CoT lifts toward NC1 (arXiv:2402.12875, ICLR 2024); BAPO proves an Omega(n)
reasoning-token lower bound specifically for graph reachability (arXiv:2602.02909);
compositional error compounds with scale (Faith and Fate, arXiv:2305.18654);
graph-task difficulty tracks minimum BFS iterations required (arXiv:2602.06319).
Scope the claim to fixed-depth single-pass decoding - looped transformers can simulate
BFS/Dijkstra exactly (arXiv:2402.01107).

Note: `summary.json` already carries `by_operation` and `by_family`.
`by_task_type` / `by_task_family` / `categories` are generic aliases reading metadata
keys board-fluency rows never set (`eval_qwen_vl_adapter.py:451-457`, fallback
`or "unknown"`). Read the former; `modal_board_fluency_extension4.py:301-307` already does.

## Track assignment

1. Perception (bare answer, keep current format): local_node_tiles, resource_pip_totals,
   port_access, owned_incident_roads, owned_buildings_touching_resource, coverage_union
2. Engine (delete from training, surface in observation): reachable_nodes,
   shortest_distance, component_count, component_roads, road_removal_connectivity,
   distance_rule_witnesses
3. Thinking (separate corpus + panel, RL target): settlement_upgrade_production,
   robber_move_production, coverage_intersection, coverage_difference, coverage_missing,
   resource_pip_argmax, roll_production, node_pip_sum

The existing `FAMILIES` map (`board_fluency_scoring.py:16-35`) groups by topic, which
cuts across this boundary (aggregation_comparison holds both resource_pip_totals at 100%
and node_pip_sum at 40%). Track is a new axis, not a regrouping of families.

## Work items, ordered by cost

- [ ] Engine track: surface reachability, node distances and components in
      `cle/env/observation_formatter.py` from the existing exact implementations -
      `board.py:30 get_node_distances()` (floyd_warshall, lru_cached),
      `board.py:269 find_connected_components()`, `board.py:279 continuous_roads_by_player()`,
      `features.py:329 reachability_features()`. Drop those 6 ops from the training mix
      (~30% of corpus). OPEN: push into every observation vs expose as a tool call.
- [ ] Prompt-token loss weighting on the existing corpus. Cheapest possible test of the
      loss-starvation hypothesis: config change, no data regeneration. Low-moderate weight
      on prompt tokens, moderate-high on answers (WIT, TACL 2025, arXiv:2507.07817;
      Instruction Tuning With Loss Over Instructions, NeurIPS 2024, arXiv:2405.14394 -
      advantage is driven by low answer-token density and shrinks as dataset grows;
      our ~17:1 prompt:answer ratio is squarely in their regime).
- [ ] Assert and log loss-bearing token fraction per example in prepare. Would have
      surfaced the 5% problem four runs ago.
- [ ] Ablate `orthogonal_lambda=0` for one round. Our 0.5 matches O-LoRA's paper default
      (arXiv:2310.14152), but their evidence is distinct sequential tasks; we apply it
      across rounds of the same objective, where overlapping subspaces may be desirable.
      Not evidenced either way in the literature.
- [ ] Audit LoRA target modules against the full weight set. Attention-only rank-256
      underperforms MLP-only rank-128 at equal parameter count (LoRA Without Regret,
      Thinking Machines, Sept 2025 - non-peer-reviewed, single source).
- [ ] Perception track: batched multi-QA SFT. 8-16 questions per board, board encoded
      once, loss on answer spans only. Raises loss-bearing fraction ~5% -> ~30%.
      Sequence budget is fine (max_sequence_length 4096, observed max 971).
      - Pack board + its Q&A as one atomic unit; never split the board across a pack
        boundary (Best-fit Packing, ICML 2024, arXiv:2404.10830).
      - Block-diagonal masking isolates boards from each other, NOT questions within a
        board - intra-board attention is the scaffold (arXiv:2107.02027; naive
        separator-only packing costs ~0.35% F1).
      - Packing related items beats random packing (TFP, NAACL 2025, arXiv:2408.09327).
      - Shuffle question order in ~50% of sequences, not 100% (arXiv:2311.09198;
        position bias is architectural and full shuffling does not remove it).
      - Do NOT scale effective batch naively: LoRA tolerates large batches worse than
        full FT and the gap grows with batch size independent of rank. Prefer gradient
        accumulation at smaller effective batch.
      - RISK: multi-turn training has documented single-turn degradation
        (arXiv:2510.21339, arXiv:2606.00135). No paper measures our exact setup (frozen
        shared context + many independent short probes). Run BOTH eval panels:
        single-question (comparable to r01-r04) and batched (deployment condition).
- [ ] Thinking track: cold start before any RL. Rejection-sample traces with the verifier
      as filter.
      - k>=8 per problem, temp 0.7 (ReST-EM uses k=32-64, top-k 40, caps 10 kept per
        problem to avoid imbalance, arXiv:2312.06585; RFT sweeps k to 100, arXiv:2308.01825).
      - Dedup by reasoning-path signature, not final answer (RFT's canonicalized
        equation-list dedup) or we keep k copies of one lucky trace.
      - Keep the SHORTEST correct trace per problem (Kimi k1.5, arXiv:2501.12599) - doubles
        as a false-positive filter and as anti-verbosity pressure.
      - Consider STaR rationalization for problems never solved: re-prompt with the answer
        as a hint, generate the backward rationale, strip the hint (arXiv:2203.14465).
        Far cheaper than 10x resampling.
      - False positives: "medium sampling-frequency" answers (correct on 40-60% of
        resamples) are the dominant spurious-signal source (arXiv:2604.21327). Require
        self-consistency across samples. Hand-audit a sample - no paper quantifies the
        false-positive rate for game-state tasks.
      - ReST-EM resets to the base model each iteration rather than continuing; 2-3
        iterations before returns go negative (test accuracy regressed at iteration 2 on
        APPS while train accuracy kept rising).
      - Start cold start from a FRESH adapter on the frozen base, not a continuation of
        checkpoint-512, and ablate against the continuation. OOD capability peaks early in
        SFT then degrades, and RL only recovers that peak rather than exceeding it, with
        recovery failing if SFT drifted too far (arXiv:2509.12235). 1536 updates of
        direct-answer training is a live drift risk. Nobody has published this comparison.
- [ ] Thinking track: GRPO with correctness-gated length reward.
      - reward = 0 if wrong; 1 + alpha*(1 - len/max_len) if correct, alpha ~0.1-0.2.
        Well precedented: Kimi k1.5 clamps its length term to min(0, lambda) so a short
        wrong answer never scores positive; "Shorten After You're Right" (arXiv:2505.12284)
        names the same gates - RightGate (correct-only) and StableSwitch (activate once
        accuracy is stable). Reported Logic-RL inference length 2632 -> 535 tokens with
        accuracy 79% -> 93%.
      - ADD SlackBand from that paper: do not penalize correct answers only marginally
        longer than the minimum. We did not have this.
      - Phase alpha in from 0 after correctness stabilizes (Kimi ran RL with no length
        penalty first). Correctness-only reward does NOT yield brevity on its own -
        R1 length grew 1k -> 14k emergently.
      - Zero-gradient groups: use DAPO Dynamic Sampling, filter/oversample until
        0 < correct < G (arXiv:2503.14476). Degeneracy is worse than (1-p)^G predicts
        because correctness is correlated within a batch - measured rate 0.69 at G=4
        (arXiv:2605.07689). That paper's Sign advantage (A = 2r-1) is a cheap complementary
        fix, reported GSM8K 73.8% vs 28.4% at G=4; single unreplicated result.
      - Do NOT rely on the length term to restore gradient in all-correct groups. That was
        my assumption and it is unevidenced; treat it as an ablation, not a mechanism.
      - `scale_rewards` is an active hyperparameter, not a default to accept. Vanilla GRPO
        already has length bias from 1/|o_i| token normalization and group-std division
        (Dr. GRPO, arXiv:2503.20783); its own unbiasedness claim is disputed
        (arXiv:2607.23364). Tune jointly with alpha.
      - Reward function drops straight onto `score_board_fluency`
        (`board_fluency_scoring.py:188`): pure, and malformed predictions score False
        rather than raising, so missing-delimiter rollouts fail closed.
      - Monitor NONE-precision as a first-class metric. 18/200 review rows have NONE as
        the correct answer and we already have 10 "said NONE but non-empty" failures.
      - We are on a Qwen base: spurious rewards produced real gains on Qwen2.5 by
        activating memorization shortcuts (arXiv:2601.11061). Treat headline RL gains
        with suspicion until they hold on held-out layouts.
- [ ] Split the eval into three panels (perception / engine-as-observation / thinking).
      A blended 64% averages incommensurable things and is why four runs looked like
      progress. Also raise per-op row count: 10 rows/op gives roughly +/-15pt error bars,
      fine for spotting reachable_nodes, useless for judging a 3-point delta.

## Open questions

- Inference contract for the perception track: does the agent ask many questions per
  board per turn (batched matches deployment) or one ad hoc (relying on transfer)?
  Determines whether the batched panel or the single panel is the real metric.
- Engine track delivery: observation push vs tool call.
- Is rank 16 a bottleneck for the perception SFT? Likely fine for GRPO (policy gradient
  carries ~O(1) bits/episode) but LoRA underperforms full FT on knowledge injection, with
  real update ranks 10-100x higher than typical LoRA (Biderman et al., TMLR 2024,
  arXiv:2405.09673). Split verdict, worth a rank sweep on the SFT track only.
- Budget: r04 was ~$40 / 88 min against a $41 approved ceiling and `actual_billed_usd` is
  still null in the receipt. Reconcile against Modal billing before committing to GRPO,
  which is 8-15x SFT cost per update because it is generation-bound.

# Colonist top-player replay capture for table-talk data (2026-09-17)

Goal: Speak-or-Stay-Silent style "when to speak" dataset from human Catan chat.
Existing 66 replays already hold 3,245 event-aligned human utterances that no
code reads (`event_parser.py` ignores `gameChatState`). Scaling for positives
and player diversity.

- [x] Stale sessions: `.env` JWT expired 2026-06-12; `.colonist-playwright-profile`
      returns 401. `.colonist-cdp-profile` (Chrome 151, cf_clearance valid to 2027)
      only needed a re-login. Playwright's Chromium cannot read Chrome-encrypted
      cookies, so use real Chrome + `--cdp-url`, per bootstrapping README.
- [x] Chrome launched on the cdp profile with `--remote-debugging-port=9222`;
      one-game CDP test captured 192299640 (601 events, 62 human chat lines).
- [x] First batch: 50 captured, then 429 at request ~52 (~35 min, no Retry-After).
      Loop retuned: 45/batch, 1h cooldown, 429 -> 90 min + one retry, second 429 stops.
      `pull_replays_loop.py` running detached with a 1h initial delay.
- [x] Retry probes at 02:46, 04:16 and 13:42 all 429 on the first request: the
      window is at least 12h, likely a daily quota (~50), and rejected probes may
      extend it. Scraper now logs 429 headers+body. Loop restarted to probe once at
      ~24h after the last 429, then 45/day (24h cooldown, 6h after a 429, stop on 2).
- [x] Original batch plan: 150 games from `4p_games_training_candidates.json` minus the
      69 already on disk (raw + staging + rejected), 40s pacing, stop on 429.
      Log: `logs/colonist_scrape_*.log`. Filtered index lives in the session
      scratchpad; regenerate from the candidates index for the next run.
- [x] Diversity fix: candidates index was grouped by player (first 51 captures = 2
      accounts). Loop now round-robins across players; first batch = 45 distinct.
- [x] Seat ratings: only source is the Classic4P leaderboard (30k+ deep). Daily
      snapshot + `seat_ratings.json` manifest; loop runs it after each batch.
      First snapshot 2026-09-17: 39,804 rated players, 237s. Coverage 250/462 human
      seats (54%); unmatched seats are likely renamed or inactive accounts. New
      top-100-indexed games: opponent rating p10/p50/p90 = 1627/1832/1968.
- [ ] Validate + promote staged captures to `artifacts/raw/colonist/replays/`.
- [ ] Repeat daily (~150/run) until ~500 games; then build the chat extractor and
      per-player decision-point labeler (ADDRESSED / TARGETED / AFFECTED /
      BYSTANDER), rendered through the harness's own observation components.

# Player reasoning filter (2026-09-17)

- [x] Add an All players / color filter for saved, live, and rejected reasoning; preserve within a game and reset across games.
- [x] Make Previous/Next find matching inference checkpoints, including origin-call continuations, with stale-request protection and lightweight actor caching. Keep direct checkpoint selection and Latest available.
- [x] Verify filtering, navigation boundaries, continuation precedence, and run frontend tests, lint, and TypeScript build.

Review: 61 node unit tests and 23 mounted browser tests pass, `tsc -b`
and production build clean; remaining lint errors pre-exist in
HexBoard.tsx/types.ts. Origin continuation applies only when the current
step has no inference, matching the shared renderer.

# Remove the speech intent enum (2026-09-16)

Traces: 12 WARNING / 3 TRADE across all recorded messages; all 12 WARNINGs were
the robber-lobby spam. No runtime code branched on intent. Listeners saw
`'intent': 'BRIBE'` in the dict-repr render, which forecloses bluffing.

- [x] `intent` removed from `CommunicationChoice`, `append_message`, the
      `MESSAGE_SENT` payload, trace `choice_json`, and the game-log `[INTENT]` prefix.
- [x] Reactive parser: say requires `mode, text, respondents`; `intent` is now an
      unknown key and is rejected like any other. Legacy XML parser ignores `<intent>`;
      legacy fresh-JSON parser tolerates and ignores the key (recorded responses).
- [x] Prompts: `shared_v1` v11, `shared_rl_v1` v4, `communication_v5` schema. Also
      dropped the word "non-binding" from commitment wording.
- [x] Tests updated; lesson rewritten in `tasks/lessons.md`.
- [ ] Next: render table talk to the model as `COLOR: "text"`, not a dict repr.

# Prompt suite lifecycle status (2026-09-16)

Only `shared_v1.yaml` is live; the other 14 files under `cle/harness/suites/`
were indistinguishable from it by name. Renaming would touch ~65 test path
references for no runtime gain, so status is a field.

- [x] `status: active | legacy | deprecated` on `ContextSuite`,
      `CommunicationSuite`, `SharedPromptSuite` (shared `SuiteStatus` Literal in
      `components.py`); derived decision/speech suites inherit the bundle's status.
- [x] Every suite YAML declares status; deprecated/legacy files carry a banner
      comment naming the successor. `shared_rl_v1.yaml` is deprecated: no code,
      test, or doc references it.
- [x] `load_*` by file path warns `DeprecationWarning` on deprecated files;
      `parse_*` (embedded trace sources, replay) stays silent.
- [x] Status surfaced in `PromptSuiteDocument` and `/api/prompt-suite`.
- [x] `tests/test_suite_status.py`: exactly one active and it is the default,
      legacy pair marked legacy, banners present, warnings fire.
- [ ] Prompt Studio badge for status (frontend `SuiteMetadata` type has no
      `status` yet; payload already carries it).

# "Return one JSON object with tool, arguments..." rejections (2026-09-16)

All 10 occurrences in game 7af0253f were `{"tool":"end_turn","notes":...}`:
a valid object missing `arguments` on a tool that takes none. The error did
not say so, so retries could not learn from it. 10 wasted calls.

- [x] `decision_response` now shows the argument-less form explicitly and
      names the tools it applies to (end_turn, roll_dice, buy_development_card,
      play_road_building, cancel_trade). Suite v9.
- [x] Parser (`cle/harness/context.py`) names the defect: `Missing
      "arguments"` with the exact fix, or `Unexpected top-level keys: ...`.
      Generic message kept for everything else. Pinned in
      `test_envelope_errors_name_the_actual_defect`.
- [x] User call: `"arguments":{}` on a no-parameter tool is a formality. A
      bare `{"tool":"end_turn"}` now parses as `{}` (single calls and batch
      entries); tools with parameters still fail on their own missing fields.
      Prompt keeps the canonical form and says the bare form is accepted.
      Suite v10. Backend restarted so the parser change is live.
- [x] Game 7af0253f completed: BLUE won at 431 steps (8 VP; others 3/4/4).
      Backend restarted on the new formatter/tool/parser code, game reloaded.

Known gap: `GET /api/prompt-suite` returns 500 on a terminal game because the
preview builds a decision context; it should fall back to a static render.

# Bank trades and discard exposure in the model's view (2026-09-16)

Game 7af0253f: 79 player offers, 11 forced discards, 2 bank trades, while
maritime_trade was legal in all 503 decisions. The only mention of the bank
in a request was the tool line "exact port/bank rate for one different card";
no request mentioned 4:1, ports owned, or that no partner is needed.

- [x] Resources block (shared path) now lists the player's exact rates:
      "Bank trade (maritime_trade, no partner needed): ... WOOD 4, SHEEP 2 (2:1
      port), ..." from owned port nodes, plus "DISCARD EXPOSURE: 9 cards held;
      any 7 rolled costs you 4 cards" when over the limit.
- [x] Tool signature and the failed-call hint state the rule (4, 3 with 3:1,
      2 with matching 2:1; any time after rolling; bank must hold the card).
- [x] `main_game` guidance: use the bank to finish a build or get under the
      limit when partners will not trade fairly. Suite bumped to v7.
- [x] Test pins rates, port discounts and the exposure flag.

The YAML half is live (server rereads it); the formatter/tool-text half is
Python and needs a backend restart plus Load latest. Not restarted: auto-play
was running.

# Richer private notes (2026-09-16)

Measured first: 519 accepted notes updates in game 7af0253f, median 487 chars,
p90 701, max 1,374, zero oversized rejections. The 4,000 ceiling never binds,
so the fix is the policy text, not the limit. `memory_policy` now asks for
substantial notes (1,500-3,000 chars normal) with a fixed shape: plan and its
needs, production and gaps, each opponent's position/needs/trade behaviour,
open offers and promises, discard exposure and spend intent. Update, don't
append; delete stale entries. Suite bumped to `catan-shared` v6.

If the ceiling ever needs raising, it is enforced in three places:
`cle/players/notes.py:MAX_NOTES_CHARS`, `shared_suite.py` (`le=4000`) and the
suite's `max_notes_chars`.

# Discard-risk guidance in the shared prompt suite (2026-09-16)

- [x] `shared_v1.yaml` bumped to `catan-shared` v5. `main_game` guidance now
      states the rule exactly as the engine applies it (`state.py:700-767`):
      8 or more resource cards when anyone rolls a 7 loses half, rounded down;
      a 7 is 1 in 6, so roughly even odds across a round of opponents' turns;
      spend or trade down to 7 or fewer before `end_turn`. `robber` guidance
      notes that large post-discard hands are the richest steal targets.
- [x] Version pins updated (`test_shared_prompt_components`, the default-suite
      assertion in `test_communication`); the legacy `communication_v4.yaml`
      pin stays 4. 140 prompt-suite tests pass.
- [x] The live server already serves v5 through the default resolver.

# Full game log and message board on load (2026-09-16)

- [x] Backend already emits the whole log (`list(state.game_log)`); pinned by
      `test_snapshots_carry_the_whole_game_log_not_a_window`.
- [x] Load latest rebuilt `state.game_log` from the last checkpoint's stored
      slice, so old games came back with 50 rows and no speech. Speech rows are
      now re-projected from the checkpoint's public events in
      `normalize_public_state_game_log` (`backfill_message_log_entries`),
      the same path that already re-projects trade rows.
- [x] Backend restarted with `CATAN_VIEWER_RELOAD=1` (PID 71801) at the user's
      request; game 7af0253f reloaded: 63 log rows, 12 messages, hands present.

# Trace DB: stop re-snapshotting model reasoning every step (2026-09-16)

`.cle/live_traces.sqlite3` hit 1.49 GB from 598 steps. Breakdown: `sandbox_snapshot`
1031 MB, `result_json` 202 MB, `public_state_json` 84 MB, model calls 150 MB.
Snapshots grow 43 KB -> 4.4 MB over a 400-step game because every agent's
`session.receipts` deep-copies the full `PlayerChoice` (native reasoning and
details, ~20 KB each) and the whole map is re-pickled on every step: O(n^2).
`model_calls` already holds every request, response and reasoning for training;
receipts exist only for idempotent redelivery, so they need the decision, not
the trace.

- [x] `receipt_choice()` in `cle/harness/models.py`: drop `raw_response`,
      `native_reasoning`, `native_reasoning_details`, `reasoning_request`,
      `usage` when minting a `ChoiceReceipt`. Decision fields unchanged.
- [x] zlib pack/unpack in `cle/traces/sqlite.py` with a magic prefix and legacy
      passthrough for: `initial_snapshot`, `sandbox_snapshot` (steps, failures),
      `result_json`, `public_state_json` (steps, failures), `request_json`.
      `response_json` and `payload_json` stay plain text: `get_usage` runs
      `json_extract` / `json_each` on them in SQL.
- [x] `scripts/compact_live_traces.py`: dry-run by default, `--apply` packs
      legacy rows in batches, `--vacuum` reclaims the file. Idempotent.
- [x] Tests: receipt slimming and redelivery, packed round-trips, legacy rows,
      compaction; update the three assertions that inspected raw columns.
- [x] Docs: `cle/traces/README.md`, README, lessons.

Full suite: 3140 passed; remaining failures are pre-existing (Modal/SFT
artifact tests, replay trading payloads from the engine work, two route tests
on new validation strings). Two route tests asserted raw responses and
reasoning off session receipts (`..._tls_recovery...`, `..._pinned_v11_agent...`);
both now check the accepted model call in the trace store, where that record
lives. A new route test pins that snapshots carry the whole game log, not a
trailing window.

Dry run on the real database: 1,438 MB of packed-column bytes -> 204 MB
(1,235 MB saved across 3,171 rows) before VACUUM. Not applied: the user runs
`--apply --vacuum` with the viewer stopped.

# Playground hand contents: resources + dev cards (2026-09-16)

Live viewer snapshots collapsed every hand to a total, so the frontend's
existing breakdown branches never received data. Added a spectator layer rather
than widening the public projection.

- [x] `get_player_hands` in `live/game_logging.py`; `player_hands` added to the
      websocket broadcast, `/api/state`, the inject payload and `serialize.py`.
      Public `all_player_resources` / `all_player_dev_cards` unchanged.
- [x] `src/playerHands.ts` with canonical ordering and unknown-card tolerance;
      overlay chips always render the breakdown. A dock toggle was built first
      and removed at the user's request: contents belong on the chip, not behind
      a button.
- [x] Verified: node unit tests (54), a playwright render test asserting the
      chips, backend route test, and an end-to-end run proving the stored
      checkpoint hands match the live snapshot.

- [x] Messages board always renders for a game (was gated on having entries),
      with a "No messages yet" state and a 0-message count in the summary.

Gotcha found in use: the viewer's Flask process has no reloader unless
`CATAN_VIEWER_RELOAD=1`, so a server started before a backend change keeps
emitting the old snapshot shape while Vite hot-reloads the frontend. The chips
then render totals with no contents and the change looks unapplied.

Note for later: `snapshotKey` echo-dedupe in `App.tsx` ignores socket updates
that change only `last_live_step_error` or `live_inference`, which already fails
`test_socket_notice_during_autoplay_pause` and
`test_compact_empty_and_busy_controls` on the current working tree. Pre-existing
and untouched here.

# Fourth board-fluency extension: 512 more steps (2026-09-16)

Learning is not saturating (+27 review / +10 held-out in r03). User approved
512 more steps after a ceiling correction (my ~$37 estimate was short; honest
caps need ~$41). Suffix is epoch-3 rows 897–3200 (2,304) plus epoch-4 rows
1–1792 (1,792): 4,096 presentations at batch 8. Parent is the r03 trained
checkpoint-512 (cumulative 1,024); target is cumulative 1,536 updates (12,288
presentations, 3,200 unique). Baselines are r03 posteval: review 103/200,
validation 126/190.

New ceiling is $41 (explicit user approval). Carry-forward is $26.09012499
(pinned r06/r01/r02 allowances + r03 recorded). New caps: prepare 300 s, train
6,200 s, eval 900 s, startup 300 s, coordinator 8,000 s, absolute 7,500 s, plus
$0.75 reserve. New envelope is about $14.46; cumulative bound is about $40.55,
under $41.

- [x] Specify repeat data order, checkpoint plan and raised budget.
- [x] Implement CPU admission, pinned suffix/baselines and bounded train/eval stages.
- [x] Verify dry plan, source/checkpoint/data identity and independent code review.
- [ ] Execute the extension; retain 16 checkpoints and both 390 predictions.
- [ ] Rescore paired results, reconcile cumulative compute, verify stop, and report.

Prelaunch r04 review: new sibling launcher leaves r06/r01/r02/r03 pins intact.
Suffix 4,096 rows (epoch-3 finish + epoch-4 start), full 390 baseline rescoring,
JSON roundtrip and plan/config checks passed; two independent reviews clear.
Cumulative bound $40.55011099 under the user-approved $41. Run name:
`board-fluency-extension-20260915-r04`; no Catan apps running.

# Third board-fluency extension: 512 more steps (2026-09-16)

User approved 512 more repeat steps with a ~$31 ceiling (no fresh rows remain).
Suffix is the full 3,200-row corpus (second epoch) plus rows 1–896 (partial
third epoch): 4,096 presentations at batch 8. Parent is the r02 trained
checkpoint-256 (cumulative 512); target is cumulative 1,024 updates. Baselines
are r02 posteval: review 76/200, validation 116/190.

New ceiling is $31 (explicit user approval). Carry-forward is $16.43294677
(pinned r06 allowances + r01/r02 recorded). New caps: prepare 300 s, train
6,200 s, eval 900 s, startup 300 s, coordinator 8,000 s, absolute 7,500 s, plus
$0.75 reserve. New envelope is about $14.46; cumulative bound is about $30.89,
under $31.

- [x] Specify repeat data order, checkpoint plan and raised budget.
- [x] Implement CPU admission, pinned suffix/baselines and bounded train/eval stages.
- [x] Verify dry plan, source/checkpoint/data identity and independent code review.
- [x] Execute the extension; retain 16 checkpoints and both 390 predictions.

Prelaunch r03 review: new sibling launcher leaves r06/r01/r02 pins intact.
Suffix 4,096 rows (full second epoch + rows 1–896 third), full 390 baseline
rescoring, JSON roundtrip and plan/config checks passed; two independent
reviews clear. Cumulative bound $30.89293277 under the user-approved $31. Run
name: `board-fluency-extension-20260915-r03`; prior app stopped, zero tasks.
- [x] Rescore paired results, reconcile cumulative compute, verify stop, and report.

## Completed third extension result

`board-fluency-extension-20260915-r03` completed at 08:01:20 UTC on 2026-09-16:
prepare, 512 training updates and full 390-row post-evaluation. Offline audit
passed all retained/new predictions, hashes, stage receipts and checkpoint
histories (one verifier fix: LR peak/ramp/decay shape checks apply to the full
512-update history only; earlier checkpoints hold truncated warmup prefixes).
Review moved 76/200 → 103/200 (37 improved, 10 regressed); held-out190 moved
116/190 → 126/190 (27 improved, 17 regressed). Combined 192/390 → 229/390
(49.2% → 58.7%); malformed answers fell to 2. Teacher120 loss 0.299 → 0.190,
row exact 60.8% → 71.7%. Frozen visual digest unchanged. App
`ap-etvhT392SCzo9rPo5PSdjL` stopped with zero tasks. Recorded-window extension
cost is $9.6572; cumulative with pinned prior allowances is $26.0901, under
the user-approved $31 (provisioned-rate estimate, not a provider bill).

# Second board-fluency extension: 256 more steps (2026-09-16)

User chose 256 more steps with a raised ceiling over 144 steps within $15.
Only 1,152 fresh rows remain (2049–3200); the run wraps to rows 1–896 for the
rest (896 second-epoch repeats; cumulative unique reaches all 3,200 = 1.0 epoch).
Parent is the ext-r01 trained checkpoint (cumulative 256); target is cumulative
512 updates. Baselines are ext-r01 posteval: review 57/200, validation 96/190.

New ceiling is $21 (explicit user approval via the raise-the-ceiling choice).
Carry-forward is $11.25575718 (pinned r06 $7.47126099 allowances + pinned ext-r01
$3.78449619 recorded). New caps: prepare 300 s, train 3,300 s, eval 900 s,
startup 300 s, coordinator 6,000 s, absolute 5,400 s, plus $0.75 reserve. New
envelope is about $9.34; cumulative bound is about $20.59, under $21.

- [x] Specify wrap-around data order, checkpoint plan and raised budget.
- [x] Implement CPU admission, pinned suffix/baselines and bounded train/eval stages.
- [x] Verify dry plan, source/checkpoint/data identity and independent code review.
- [x] Execute the extension; retain 8 checkpoints and both 390 predictions.

Prelaunch r02 review: new sibling launcher leaves r06/r01 pins intact. Suffix
2048 rows (1152 fresh + 896 exact repeats), full 390 baseline rescoring, JSON
roundtrip and plan/config checks passed; two independent reviews clear.
Cumulative bound $20.59185918 under the user-approved $21. Run name:
`board-fluency-extension-20260915-r02`; prior extension app stopped, zero tasks.
- [x] Rescore paired results, reconcile cumulative compute, verify stop, and report.

## Completed second extension result

`board-fluency-extension-20260915-r02` completed at 05:59:20 UTC on 2026-09-16:
prepare, 256 training updates and full 390-row post-evaluation. Offline audit
passed all retained/new predictions, hashes, stage receipts and checkpoint
histories (one verifier assumption fixed: warmup starts at exactly LR 0.0 on
update 1, as in r06 itself). Review moved 57/200 → 76/200 (28 improved, 9
regressed); held-out190 moved 96/190 → 116/190 (28 improved, 8 regressed).
Combined 153/390 → 192/390 (39.2% → 49.2%); malformed answers fell 28 → 3.
Teacher120 loss 0.360 → 0.299, row exact 47.5% → 60.8%. Frozen visual digest
unchanged. App `ap-dTpI6ngcgd2AgGuACY13Db` stopped with zero tasks.
Recorded-window extension cost is $5.1772; cumulative with pinned prior
allowances is $16.4329, under the user-approved $21 (provisioned-rate estimate,
not a provider bill).

# Symbolic board-fluency extension (2026-09-15)

User approved more steps after the completed r06 pilot. Continue its trained
checkpoint-128 with another128 steps, preserving rank16/alpha32, text-only scope,
atlas rows, learning rates and effective batch8. Use the existing fresh-optimizer
initial_bundle path and original rows1025–2048 via an immutable suffix file;
cumulative main training becomes256 updates /2048 unique examples /0.64 epoch.

Options considered:128 steps provides comfortable startup/train/eval margins;
256 would require tighter deadlines to fit the existing$15 budget. Choose128.
Carry the complete$7.47126099194740 previous allowance-inclusive estimate. New
caps: preparation300s, training1800s, evaluation1050s, startup300s, coordinator
4500s plus an additional$0.75 reserve. Modeled cumulative upper is about$14.41.
Reuse verified r06 review200(58correct) and validation190(87correct) as the full
baseline. One fresh sibling launcher; preserve the pinned pilot source/results.

- [x] Inspect supported weight-continuation/data-order semantics and budget options.
- [x] Implement CPU admission, pinned suffix/baselines and bounded train/eval stages.
- [x] Verify dry plan, source/checkpoint/data identity and independent code review.
- [x] Execute the extension; retain checkpoints32/64/96/128 and both390 predictions.
- [x] Rescore paired results, reconcile cumulative compute, verify stop, and report.

## Completed extension result

`board-fluency-extension-20260915-r01` completed at 04:50:36 UTC on 2026-09-16:
prepare, 128 training updates and full 390-row post-evaluation. Offline audit
passed all retained/new predictions, hashes, stage receipts and checkpoint
histories. Review moved 58/200 → 57/200 (12 improved, 13 regressed); held-out190
moved 87/190 → 96/190 (18 improved, 9 regressed). Combined 145/390 → 153/390.
Teacher120 loss 0.487 → 0.360. Frozen visual digest unchanged. App
`ap-2cb1HYO4C07Rxhuk3WcF3T` stopped with zero tasks. Recorded-window extension
cost is $3.7845; cumulative with pinned prior allowances is $11.2558, under the
approved $15 (provisioned-rate estimate, not a provider bill).

Prelaunch review: new sibling launcher leaves all r06 source pins intact. Raw
suffix count/hash, disjoint next1024 IDs/states, full390 baseline rescoring,
JSON roundtrip and plan/config checks passed. Review tightened the one-time
budget-lineage claim, local coordinator-startup supervision and bounded cleanup.
Cumulative envelope is$14.41212699194740. Run name:
`board-fluency-extension-20260915-r01`; existing Catan apps have no active tasks.

# Live stale prompt-suite backend recovery (2026-09-15)

- [x] Confirm listener, launch command/workspace, and reproduce stale validation.
- [x] Verify idle inference and saved checkpoints; gracefully restart only backend.
- [x] Verify health, shared flags over HTTP, and checkpoint availability; report restore status.

Review: confirmed workspace listener PID 19121 and uv launch parent; read-only
state returned No game before shutdown (no live sandbox for inference), with no
outbound provider TCP connection. SIGINT was ignored by the nohup-launched process;
after rechecking idle state, SIGTERM exited it. Same uv command now serves as PID
48530. Health, GET prompt-suite (shared v4, both flags true), and non-saving POST
prompt-suite/validate all return HTTP 200. No application source fix was needed.
Logs: `logs/game-viewer-backend-20260915-stale-suite.log`.
Saved trace metadata is unchanged; latest game
`02926b63-505a-483d-8813-f220c8f040af` has 22 steps, checkpoint index 21 with a
376193-byte snapshot verified through read-only SQLite. State remains No game;
resuming saved play requires an explicit load. No game load/step/reset was issued.

# Deterministic action batches (2026-09-15)

Approved build: optional shared/fresh `actions` envelope, 1–4 semantic calls,
one optional notes update. Existing single calls and historical contracts remain.
Validate all syntax before admission, then resolve each action against live state.
Commit prefixes incrementally; consume invalid remainders with private feedback.
Use one engine action/checkpoint per Step and zero inference for continuations.
Stop at setup-pair boundaries (including snake reversal), actor/phase changes,
new information, speech/trade barriers, end-turn and victory. No random tools.

- [x] Implement strict opt-in parsing and typed, durable continuation state.
- [x] Integrate live validation, boundary consumption and exact call provenance.
- [x] Verify real setup/build/conversion, prefix failure, persistence and legacy paths.
- [x] Review trace/UI accounting, document behavior and run targeted checks.

No subagent tool is available; parallel inspection and direct review are used.

## Results and review

- Built-in shared v4 / RL v3 opt in with `deterministic_batches: true`. Strict
  shared/fresh parsing validates the whole envelope and notes first, then binds
  only the first action. Existing single calls and explicit historical sources
  retain their contracts. Batch prompts expose stable signatures, not legal lists.
- One action commits per Step; pending continuations regenerate legality from
  updated state before resolving semantic arguments. Setup pairs, including
  snake reversal and second-settlement resource grants, are handled explicitly.
  A failed later action consumes the remainder with private feedback and fresh
  model control; earlier actions and costs remain, with no replay or fallback.
- Durable queue consumption survives SQLite/pickle restore and cancellation
  before inference, after admission and midway through continuation. Prompt/player
  rebinding preserves accepted plans. Notes and delivered-input cursors advance
  once; later engine-only steps acknowledge no unseen observations.
- Each continuation has an ordinary checkpoint and exact context/provider/event
  provenance with no model-call row or invented usage. Live/saved UI distinguishes
  requested batches from committed actions and displays automatic provenance.
- Direct review tightened first-action fresh legality, literal-token admission,
  terminal handling and defensive copies of returned provenance. Tests confirm
  that callers cannot mutate the remaining queue through results or snapshots.

Verification: **1,217 backend tests passed, 32 existing skips** across 21 focused
modules (new batches, shared/legacy contracts, trade preauthorization, speech,
Knight, live routes/traces, fresh notes, replay, engine/harness/checkpoint audits).
Frontend: **45 unit tests passed**, TypeScript and Vite build passed. Scoped Ruff
and `git diff --check` passed. Real engine fixtures cover all eight setup pairs,
newly opened settlement sites, 2:1/3:1/4:1 conversions into cities, stale cached
menus, actor/phase/speech interruption, paid-prefix failures and immediate wins.

Limitations: max four deterministic actions; dev cards/randomness, speech and
player trades (including preauthorization) remain standalone. Old/local authored
sources must explicitly opt in. Read-only replay previews validate the first
action without executing a live queue. Mixed legacy speech policies still apply;
emitted speech invalidates a pending queue before the next continuation.
No full-repository test run, hosted inference, live-game mutation or server restart.

# Narrow trade preauthorization (2026-09-15)

Approved implementation: shared/fresh offer_trade optionally accepts an ordered
nonempty player list or ANY. Authorization belongs to one admitted exact root
offer and its first complete response barrier. Any counteroffer pauses; no
permitted acceptance pauses. ANY uses engine seat order, never completion order.
Revalidate current window, offer, willingness, both hands and legality; consume
once on resolution with no fallback. Normal offers remain probes.

- [x] Inspect admission, barriers, persistence and actual-call trace accounting.
- [x] Implement typed shared-only parsing, durable authorization and engine steps.
- [x] Verify priority, pause/stale paths, restore/retry and compatibility offline.
- [x] Review changes and document interface, provenance and verification.

No delegation tool is exposed; parallel inspection and direct review instead.

## Results and review

- Shared-only optional `confirm_if_accepted_by` accepts a distinct ordered audience
  list or literal ANY. Typed validation rejects non-root/non-proposer, wildcard,
  duplicate, self/outside-audience and malformed authorizations. Normal probes and
  historical indexed/v11 parsers retain their behavior.
- Immutable sandbox authorization binds original engine window/offer, exact terms,
  audience, turn/round, deterministic priority and accepted-call provenance. The
  first response barrier commits unchanged; the following engine-only step checks
  fresh legality, both hands and exact admitted response events before transfer.
  Any counteroffer pauses; missing acceptance, changed willingness, cancellation,
  withdrawal, expiry or other stale state consumes with proposer-private feedback.
- No lower-priority fallback after funding failure, no later-round continuation,
  no repeated transfer. Barrier failure/cancellation retains pending authorization;
  successful or paused resolution clears it before further inference. Save/load
  validates original event identity and preserves pending/consumed state. Safe
  actor rebinding cannot reinterpret the accepted instruction.
- Confirmation is an ordinary engine action/event and separate live checkpoint,
  with empty model contexts/attempts and an `automatic_action` source reference in
  result JSON/live API. No fabricated reasoning card/call, token duplication,
  note update or input-cursor acknowledgment. Ordinary trade visibility remains.
- Direct review tightened current willingness checks against exact committed
  response events, checked old-slot defaults, checkpoint restoration, atomic
  transfer and cancellation boundaries, and preserved existing reactive speech.

Verification: broad focused backend suite **1,052 passed, 32 existing skips**
(18 modules covering semantic/shared contracts, sandbox, reactive speech, Knight,
players, live routes/traces, trade/events/RNG and engine/harness/checkpoint audits).
Final regression after review: **147 passed**, including all **42** new trade
preauthorization cases plus compatibility, trace-store, reactive and sandbox tests.
Scoped Ruff and `git diff --check` passed. SQLite tests use temporary stores and
offline transports; four actual calls are recorded for offer + three responses,
and zero calls/usage are duplicated on the confirmation checkpoint.

Limitations: no general condition/rejection language, wildcard execution or
multi-round authorization. The UI shows the normal trade action; causal provenance
is exposed through saved results/live API. No frontend changes/build required.
No full repository run, hosted inference, live-game mutation or server restart.

# Reactive public speech (2026-09-15)

Approved build; preserve the existing dirty tree and resident game.

- [x] Add an explicit versioned reactive contract: action OR public say, separate
  respondent metadata, and accepted pass/speech notes.
- [x] Implement bounded pending-decision conversations and the once-only
  post-discard/pre-robber window with checkpoint continuity.
- [x] Preserve trade barriers, atomic Knight bundles, historical contracts,
  safe prompt rebinding, exact requests and navigable speech traces.
- [x] Verify isolated engine/parser/runtime/persistence and focused UI contracts;
  document files, checks, and limitations.

Implementation choice: one actor-initiated speech per pending game decision;
bounded addressed replies use the existing communication limits. Speech is a
typed choice, never an engine Action or synthetic turn. Routine observations
accumulate without inference or cursor acknowledgment. Required trade decisions
handle negotiation offers directly. No automatic setup polling. Subagent tools
are unavailable in this harness; use parallel reads and a separate direct review.

## Results and review

- Shared default v3 and RL v2 explicitly opt into the reactive contract. Omitted
  `reactive_speech` preserves old parsers/scheduling for authored historical sources.
  Normal sandbox decisions can return CommunicationChoice or PlayerChoice; the
  engine receives only validated gameplay actions. Say never consumes an action,
  turn, placement or synthetic step. The continuing action sees accepted notes.
- Public audience and explicit respondents are separate. Pass (or the accepted
  silence alias) updates notes exactly once. Skipped routine polls update no
  cursor. Other players receive public speech at their next actual observation.
- Persistent pending-reaction queue, shared call budget and initiating-speech
  marker prevent reply/retry loops. Default cap is 12 calls including initiating
  say, with two reaction rounds; failed reaction calls consume slots. Reactions
  use deterministic queue order. Trade barriers remain simultaneous and private.
- Seven window opens after all discards, before destination, once per roll event.
  Both zero-discard and multi-discard retry/restore paths pass. Knight applies its
  bundle with no intermediate await. Placement can initiate speech; no automatic
  setup, ordinary roll/build/end-turn/trade/theft or actor speech polling.
- Traces retain selected standalone speech as communication with its exact
  action-channel request/response and usage. Trigger reason/respondents are saved
  and exposed in live provenance. SQLite continuation and cancellation retain
  accepted speech without a fake completed step or duplicate message.
- Active prompt rebinding still occurs at safe actual inference boundaries. Tests
  verify old in-flight admission, next-boundary edits/incompatibilities, and
  unchanged historical rows. Read-only action-comparison previews remain action-only.
- Direct review checked cursor separation, stale-response admission, public
  visibility, context-ID uniqueness, restore defaults, budgets and trade isolation.
  No subagent review was possible because the harness exposes no delegation tool.

Verification (offline transports, temporary stores only):
- 1,292 selected backend tests passed across reactive/fresh/shared contracts,
  sandbox, Knight, historical compatibility, live routes/traces, prompt stores,
  tools, engine/trade boundaries and harness/checkpoint/replay audits.
- 166 additional communication, replay action-diff, setup reasoning and RNG tests
  passed. Final focused rerun after typed-intent validation and the cancellation
  regression: 140 passed (includes the added cancellation test).
- 44 frontend Node tests passed. TypeScript/Vite build passed to the approved
  temporary directory; scoped Ruff, frontend ESLint and `git diff --check` passed.

No live restart, game advance, hosted model inference, reasoning-setting change,
style redesign, file deletion or commit. Full repository and mounted browser
suites were not run; no model-quality, latency or token-savings claim is made.

# Symbolic board-fluency r16 SFT pilot (2026-09-15)

## Approved scope
Start from `/runs/catan-vision-sft/spatial-continuation-20260912-r01/checkpoints/checkpoint-128`.
Expand standard LoRA r8/alpha16 to r16/alpha32 with preserved effective updates:
retain old A/B blocks, initialize added A normally and added B to zero. Preserve
the 154 input/output atlas rows. Train language LoRA and atlas rows only; keep
base language and visual/merger weights frozen. Use a fresh optimizer/scheduler.

Generate 3,200 admitted symbolic training rows (160 per reviewed operation),
excluding the 200 inspected review states. Preserve original validation/test
source splits and separate full settlement/Longest Road transfer tasks. Run 128
optimizer steps with effective batch8, LoRA LR5e-5 and atlas LR1e-4; save every32.
Verify expansion parity, actual backward/update, checkpoint save/reload, then
evaluate held-out queries and the unchanged review200 baseline (24/200).

## Plan
- [x] Implement rank-preserving conversion and rank-aware trainer/evaluator checks.
- [x] Generate and validate new train/validation/test corpora and evaluation panel.
- [x] Build CPU preparation and bounded GPU gate/train/eval orchestration.
- [x] Pass real conversion/backward/save/reload checks and run the pilot.
- [x] Verify predictions, checkpoint identities and budget receipts; report results.

## Budget
Approved ceiling: $15. Current Modal standard rates checked at modal.com/pricing:
H200 $0.001261/s, CPU $0.0000131/core/s, RAM $0.00000222/GiB/s. Plan capped stages
with one GPU worker at a time, no automatic retries, explicit cancellation, and
reserved evaluation time. Keep the sum of execution/startup resource bounds below
$15 before requesting any paid GPU call; preserve source checkpoints and prior runs.

## Implementation review
Generated `symbolic_board_fluency_sft_v1`: train3200 on3200 distinct states/332maps,
validation370, test370, validation_eval190; all200 reviewed state IDs/content
hashes excluded. Training has160 examples peroperation and640 perfamily; first
1024 presentations cover everyoperation (51–52 each). Gold/provenance/admission
checks passed, including the oldreview scorer compatibility path.

Function-preserving expansion and supportedranks8/16 are implemented. Targeted
existing scope/config checks and algebraic conversion checks passed. Static review
found and resolved the oldreview schema alias, base metadata hashing, nonzero
atlas-gradient evidence, and cancellation handling. Stop uses the dedicated
Modal app ID so coordinator and all children stop together without a spawn race.
Local dryrun bounds compute at $13.572673 using current standard rates. No GPU
call has been made yet. Launch target: `board-fluency-sft-20260915-r01`.

R01 stopped before prepare/GPU because audit histogram integer keys differed
between the RPC object and saved JSON. Normalized the launch plan before both
transports and verified the project interpreter's app-stop command. Stopped app
`ap-hY5V0lZXsn78Es8AXpamhh`; corrected launch will be r02. CPU-only failed startup
remains within the existing termination/control reserve, not a new $15 budget.

R02 reached the coordinator but treated a normal 30-second FunctionCall polling
timeout as a failed stage (built-in TimeoutError differs from Modal's exception).
Corrected the catch and raised control-plane CPU from0.125 to1 core to avoid
slow heavy-library cold starts; capped compute remains below$15. The coordinator
cancelled its CPU prepare call; no GPU stage ran. Stopped the dedicated r02 app;
next launch r03 includes these corrections under the original budget.

R03 CPU preparation/real r16 conversion passed. Its first GPU gate stopped on
different greedy IDs before training; factor blocks/scaling/token and visual
bytes passed conversion checks. The probe was autocasting LoRA arithmetic while
the actual text evaluator does not. Align probe precision with text inference
and persist complete logits/metadata before asserting parity. R03 gate occupied
about114 seconds including startup; prior attempts used CPU only. A corrected
full-run envelope plus this consumed work remains below the original$15 ceiling.

R04 now passes exact GPU logit parity (`max_abs=0.0`) in matched text-inference
precision. Its gate is collecting pre-validation190 and will then perform the
two-step gradient/save/reload check. Budget includes a conservative$0.50 allowance
for earlier attempts; prospective total compute bound is$14.179274 under$15.

R04's generation-heavy baseline hit the gate deadline before any training step.
Saved144/190 complete predictions (21correct), SHA256
`2fc6ce78b4f1620d1b186a8e4052cf38d3cad4841250d1522703c2e9edccdd00`.
Preserve these as an explicitly partial baseline and compare only the same IDs
after training; full post-validation190 and unchangedreview200 remain requested.
Next gate skips baseline regeneration, retaining parity/gradient/save checks with
a450-second cap. Prior-attempt allowance rises to$2.00; total prospective bound
is still under$15. No source checkpoint or training dataset is changed.

R05 passed real two-step backward/update and complete checkpoint-save checks:
nonzero gradients/updates in language LoRA and both atlas-row groups, nonzero
new-rank B gradients, all1184 frozen parameter versions unchanged, visual model
tensor digest unchanged. It stopped at an overly strict saved-file SHA comparison.
An independent CPU check proved all333 saved FP32 visual tensors are bitwise
identical to the parent; only serialization differed. Switched frozen checks to
canonical tensor digests. Concurrent palette imports now require pydantic in the
remote image; included the validated local2.12.3 version without changing model
library pins. Prior-attempt allowance is$2.50; main training cap reduced to3300s
(still128 steps) to keep the prospective all-attempt compute bound below$15.

R06 prelaunch: local dataset admission, JSON roundtrip, plan/config/source checks
and scoped diff checks passed. Independent gate/save/reload review found no
blocker. Original evaluator/trainer/scorer hashes still match retainedr04.
All earlier SFT/diagnostic apps are stopped withzero tasks. Modeled compute upper
is$14.863204 including$2.50 prior allowance, not a provider billing total. Recorded
r03-r05 stage/control windows model about$2.12, with remaining prior allowance for
CPU-only starts/diagnostics and termination. Monitor the dedicated app through
startup and completion; stage timeout is terminal, never an automatic extension.

R06 app `ap-AyeiKbmYnRGN9mBMLtkBD0`: CPU preparation and fullGPU gate completed.
Parent/expanded, trainer initialization and trained-checkpoint reload all match
exactly (`max_abs=0.0`). Frozen visual canonical digest is unchanged; real nonzero
gradients/updates passed. Main128-step training began at23:12:40UTC from pristine
expanded-r16 with a fresh optimizer; isolated gate checkpoint2 is not its parent.

R06 main training completed at23:36:03UTC with all32/64/96/128 checkpoints
committed and frozen visual tensors preserved. Finalgreedy review200 then
validation190 started at23:36:07UTC; quality results remain pending.

## Completed result

R06 completed at23:48:05UTC. Independent offline audit passed all390 final raw
predictions, baseline IDs/scores, hashes, complete stage receipts and checkpoint
histories. Review improved24/200→58/200(12%→29%); matched held-out144 improved
21/144→69/144(14.6%→47.9%). Full postvalidation is87/190(45.8%);46 baseline
answers remain unavailable. Review improved42/regressed8; matchedvalidation
improved50/regressed2. Review malformed answers fell48→13. Reachable-node sets
and resource pip totals remain atzero on bothfull panels.

All32/64/96/128 checkpoints are saved under
`/runs/catan-vision-sft/board-fluency-sft-20260915-r06/training/checkpoints/`.
App `ap-AyeiKbmYnRGN9mBMLtkBD0` explicitly stopped; Modal reports stopped/zero
tasks at23:48:20UTC. Recordedr06 resource-window estimate is$4.1852; diagnostics
plus fullprior allowance/control reserve yield$7.47, not a provider billing total.
Report: `reports/sft/2026-09-15-symbolic-board-fluency-sft.md`; run artifacts
include `analysis.json`, reproducible `analyze.py`, and `cost_estimate.json`.

---

# Active prompts independent of saved games (2026-09-15)

User-approved implementation plan:
- [x] Inspect runtime, restore, prompt storage, traces and editor contracts.
- [x] Resolve active selections at safe inference boundaries; stage all player
  replacements before publishing, preserve historical requests and session data.
- [x] Load saved game state with active prompts; migrate context modes explicitly
  in code with validated notes and conservative channel event delivery.
- [x] Enable live prompt editing, update labels/docs and correction lesson.
- [x] Run meaningful runtime/route regressions, frontend checks and diff review.

Design: active source selection belongs to the runtime, never the checkpoint.
Each concurrent inference batch keeps one immutable contract through admission.
Legacy-to-fresh migration redelivers visible history because legacy acknowledgments
cannot establish per-channel delivery; fresh-to-legacy retains history and cursors.
Invalid notes/contract pairs reject rebinding atomically without clearing state.

## Implementation review

- Live factory installs a non-snapshotted runtime binding; the sandbox refreshes
  before decisions and concurrent speech/trade batches. Retries in an acquired
  batch preserve its original parser/notes contract. Rebinding publishes only
  after every replacement session validates; no existing requests are mutated.
- Studio source writes use the prompt store's own lock rather than waiting for
  provider I/O. Preview capture still uses the state lock. Explicit current path
  selections are visible and cannot be silently shadowed by local editor saves.
- Load reconstructs gameplay composition from saved metadata and uses current
  runtime inference settings and prompt selection. Historical source metadata is
  ignored as configuration and retained as evidence. Each new live request stores
  exact source text/identity/hash; no migration rewrites old trace rows.
- Notes, receipts, historical messages and pending-decision state survive rebinding.
  Legacy-to-fresh replays visible events to both channels because mixed legacy
  acknowledgments cannot establish per-channel delivery. Invalid source/schema or
  notes-limit changes pause safely with configuration guidance.
- Updated historical fixture expectations for active restore/editing. Three audit
  fixtures still scripted legacy XML/game_plan with bare shared-default agents;
  they now explicitly select their intended legacy suite.
- Frontend production build and 41 Node tests passed. Six Prompt Studio browser
  flows passed on desktop/mobile. Broader browser run: desktop New Game passed;
  mobile New Game is blocked by existing workspace separator pointer interception
  (outside prompt editing; no style redesign). Backend server was not restarted.
- Verification: 334 targeted Python regressions passed across fresh/live routes,
  prompt routes/store/components, players, sandbox, trace persistence, compatibility,
  and harness/checkpoint audits. TypeScript build and focused editor ESLint passed;
  `git diff --check` passed. Final review removed a redundant source reread after
  publishing restored state: metadata now uses the actual bound source snapshot.
- Final live/fresh route rerun: 80 passed, including a tightened in-flight regression
  where the new notes limit would reject the old response. The old action/notes
  still commit exactly once; only the following speech boundary pauses with clear
  configuration guidance. Historical requests and saved source evidence survive.

# Shared fresh request contract (2026-09-15)

User approved implementation. Preserve existing work and explicit historical suites.

- [x] Implement shared-only compact tools, semantic trade resolution, and prompt composition.
- [x] Supply exact setup/free-road facts, complete private inventory and actual VP,
  and self-contained visible negotiation events.
- [x] Inspect exact-request frontend rendering and invalid-call atomicity.
- [x] Run focused integration/compatibility tests; document results and limitations.

Design: keep historical tool rendering/parser defaults; opt shared compositions into
stable definitions and acting-perspective trade terms. Resolve against current
observed offers before internal engine admission; ambiguity is an error.

## Results and review

- Shared tools have stable signatures, no legal enumeration or opaque trade IDs.
  Historical v11 tool rendering/IDs and indexed parsers remain explicit paths;
  no eval-default migration. Counter terms are original/proposed from the actor's
  perspective, and ambiguous active offers fail before legality filtering.
- Setup facts distinguish all four decisions and the placed settlement anchor.
  Starting cards are verified against the real second settlement's adjacent tiles.
  Free Road Building placements, zero-resource dev inventory, authoritative
  playability, private actual VP and public opponent VP are supplied separately.
- Decision/speech compositions omit trade_window; validation no longer requires
  it. Strategy is labeled guidance; the main-game manual is reduced to strategy.
  Live lifecycle events carry terms. Replay responses/closures use exact recorded
  offer IDs to carry source terms, including nonexecutable source-only proposals;
  replay action matching and original source rows are not rewritten.
- Repeated real-engine shared requests carry current facts/new channel events and
  accepted notes only. Invalid calls preserve gameplay, notes and pending events;
  retries include specific feedback without a legal-menu dump.
- Live traces now expose all recorded request messages plus board presentation.
  Saved traces retain their exact message mapping. Fresh and historical requests
  are labeled separately, with no truncation of transmitted historical messages.
- Kept existing dirty-tree changes. Updated historical typed-agent test fixtures
  to pin their contract; shared route fixtures now select replies from independent
  engine contexts rather than scraping a removed legal menu.

## Verification

- **1,304 passed**: shared fresh contract/components, semantic tools, context,
  players, sandbox, Knight, compatibility, fresh/shared routes, trace store,
  initial placement, prompt stores/routes, v11, engine events/trades/boundaries/RNG,
  replay core/LLM response/action-diff tests. Command: `uv run python -m pytest`
  with the 22 corresponding test modules (latest run: 29.91 seconds).
- **41 passed**: `npm test` in `playground/frontend`.
- **Build passed**: `npm run build` (TypeScript and Vite), also run by browser fixture.
- **9 passed**: `uv run python -m pytest playground/frontend/tests/test_live_autoplay_browser.py -q`.
  Includes exact two-message fresh and 82-message historical display plus the board
  attachment; every recorded message is compared with mounted browser text.
- **7 passed, 1 failed**: separate `test_shared_prompt_browser.py` run. The mobile
  session/New Game click is intercepted by the workspace separator/board. The
  unchanged session panel has minSize 260px but maxSize 28%, incompatible at 390px.
  Left this unrelated responsive-layout issue explicit; no style redesign.
- `git diff --check` passed. No hosted model inference, paid eval, running-game
  restart, or full repository-wide suite was performed. Archived request payloads
  remain exact; missing historical source terms are never invented.

# Published September 2 checkpoint comparison (2026-09-15)

## Scope
User requested `icebear5h/catan-qwen3.8-27b-spatial-sft`, pinned to
`8030a960ee73e758994c3938be347f2fc4abb38e`. Evaluate the same 200 symbolic review
examples and compare against the completed September 12 checkpoint-128 run.

## Plan
- [x] Verify the HF bundle in CPU preflight, then run one capped H200 eval.
- [x] Retrieve all 200 predictions and verify matched inputs/settings/scoring.
- [x] Save overall/family/operation comparison and representative changed answers.

Use the existing corrected launcher: text-only greedy generation, thinking off,
batch16, 512-token completion cap, context4096, no candidate scoring, the same
pinned base revision and scorer. Run name: `hf-sept02-fluency200-20260915-r01`.

## Result
Completed: **16/200 (8.0%)**, versus September 12 checkpoint-128's 24/200 (12.0%).
Older/newer family counts: joins 2/6, coverage 5/6, aggregation 1/3, connectivity
6/6, consequences 2/3 (40 rows each). Both miss all 20 board pip totals/argmax
questions. Older model: 146 well-formed, 130 wrong-but-valid, 54 malformed,
zero small-vocabulary atlas loops under the prior definition. Paired outcomes:
nine both correct, seven older-only, 15 newer-only, 169 neither.

One H200 stage took 335.863 seconds. All 200 predictions, expected answers,
metadata, scores, summaries, and receipt hashes verified in offline comparison.
Same data, base, generation settings and chat template; saved tokenizer file
hashes differ, while atlas IDs and every prompt/gold token length match.
App `ap-YeAQBoYvs7cAwegBaP5dNK` completed; container listing is empty.
Saved `comparison.json` and `compare.py` in the run directory. Report:
`reports/sft/2026-09-15-hf-spatial-board-fluency-comparison.md`.

---

# Quick board-fluency checkpoint evaluation (2026-09-15)

### Offline r06 artifact analysis
- [x] Inspect saved artifacts and the current strict scorer/evaluator contract.
- [x] Re-score all 200 saved responses; verify IDs, source targets, and summary counts.
- [x] Persist format/family/operation/gold-type counts, repetition, and exact failures
  under the run directory. Verification is offline aggregation only; no tests/inference.
Review: all identity/metadata/gold/score/summary checks passed. Saved reproducible
`analyze.py` and `analysis.json` in the r06 run directory: 24 correct, 128 wrong but
format-valid, 48 malformed; final report/documentation handled by the main task.

## Scope
Evaluate the exact 200 review examples with the latest saved spatial checkpoint:
`/runs/catan-vision-sft/spatial-continuation-20260912-r01/checkpoints/checkpoint-128`
in the `icebear5h` Modal workspace. The user accepted this continuation candidate
after reviewing the Gaussian-init and later checkpoints.
Use text-only direct greedy generation and report exact task-aware scores by
family/operation, preserving raw predictions. This is a review-set diagnostic,
not an untouched held-out generalization claim.

## Plan
- [x] Resolve checkpoint choice and locate its saved bundle/tokenizer.
- [x] Add the minimal strict review-scoring dispatch and bounded remote eval path.
- [x] Run one capped H200 evaluation after CPU-only checkpoint/tokenizer prep.
- [x] Retrieve all predictions, verify row coverage, and summarize actual results.

## Findings
Local HF auth is absent; the existing Modal `catan-hf` secret can be used for
CPU-only repository discovery. Existing evaluator supports text-only generation,
but the review's set/typed-JSON/integer answers need explicit scoring dispatch.

Authenticated discovery confirmed two published HF repos: the spatial-SFT bundle
at `icebear5h/catan-qwen3.8-27b-spatial-sft@8030a960ee73e758994c3938be347f2fc4abb38e`
and marker-only control at commit `b366613bb835fee10ccbc84d3af99ac8d456ab7b`.
Neither is labeled v2. User requested the other available checkpoints before
selecting. Remote listings confirm single-piece-v2 checkpoint-256 and terrain-v2
checkpoint-384 in the `tetracorp` workspace; full-board-new-layouts checkpoint-128,
September 9 mixed continuation checkpoint-128, and September 12 saved checkpoints
32/64/96/128 in `icebear5h`. The last extension has no final generated evaluation.

Added strict review scoring and family/operation summaries; direct checks passed
for 200 gold rows and malformed-answer cases. Added a bounded HF text-eval launcher,
but checkpoint preparation/model loading/inference have not run. Await checkpoint
selection; no GPU evaluation was launched. Workspace-specific HF secrets are
`catan-hf` in tetracorp and `huggingface-secret-2` in icebear5h.

Selection resolved to the latest saved September 12 checkpoint-128. The launcher
now accepts a saved Modal bundle as well as immutable HF snapshots. Planned run:
`spatial-ck128-fluency200-20260915-r01`; one H200, 900-second execution cap,
840-second inner deadline, no automatic retries, batch16, 512-token completion
cap, context4096. CPU preparation validates tokenizer/tensor compatibility and
source hashes before GPU allocation.

Run r01 hit a CPU-container import error: the symbolic scorer transitively needs
`playground.game_viewer.state`. App `ap-TXcmtaZew2g6d1uUj7SpDd` was stopped;
orchestration recorded zero GPU calls. Added the missing Python source mount and
will launch corrected r02; no predictions exist from r01.

R02 exposed the same package initializer's additional jsonschema dependency;
stopped app `ap-EWJL1WJrQ2QsW301dinbNE` before any GPU call. The eval image now
includes the locally validated jsonschema 4.25.1. Corrected launch will use r03.

R03 failed at image construction because pip installation followed runtime source
mounts. Split the existing pinned training base image from its source mounts so
the eval dependency is installed first. R04 then reached CPU preflight and caught
the native text helper converting Transformers 5.16.1 BatchEncoding to its field
names instead of token IDs. A bounded CPU inspection of the saved tokenizer
confirmed return_dict=False yields the correct answer/EOT boundary. Applied the
explicit return type; r01-r04 made zero GPU calls. The checkpoint's saved base
identifier is its exact immutable cache snapshot, now retained in evaluator args
after matching it against the pinned repo/revision. Corrected run is r05.

R05 passed tokenizer/context checks but the header-based LoRA preflight mistook
seven training-only `mtp.*` tensors for inference modules. CPU inspection showed
496 intended language modules plus those seven false positives. Preflight now
uses the actual inference architecture on the meta device (no weight allocation),
matching the evaluator's own module discovery. No GPU call occurred; next run r06.

## Completed Result
R06 completed all 200 examples: **24/200 correct (12.0%)**. Family scores:
relations/joins 6/40, sets/coverage 6/40, aggregation/comparison 3/40,
connectivity/structure 6/40, constraints/consequences 3/40. Of 152 format-valid
answers, 128 were wrong; another 48 were malformed. Pip totals and tied argmax
were 0/20 despite all 20 being well-formed. Twenty-four responses met the recorded
small-vocabulary atlas-loop criterion.

One H200 call lasted 615.544 seconds. App `ap-BajhKL3lk4VpZgpq9mTwTr` is stopped
with zero tasks. All 200 IDs, expected answers, metadata, score dictionaries,
grouped totals, and saved record/summary hashes passed offline verification.
Run root: `artifacts/runs/sft/spatial-ck128-fluency200-20260915-r06/`.
Report: `reports/sft/2026-09-15-board-fluency-text-eval.md`. This is a text-only,
train-source-derived review diagnostic; no initialization comparison was run.

---

# Board fluency review on port 5174 (2026-09-14)

## Approved Scope
Generate an inspectable 200-example review batch from existing symbolic training
states, with 40 examples per approved operator family, and expose it in the eval
viewer on port 5174. Model input is explicit symbolic state; board renderings are
review aids. Full settlement composition and Longest Road remain transfer tests.

## Plan
- [x] Identify port 5174 and existing viewer/data interfaces.
- [x] Build a bounded-source generator with exact answers and provenance, and
  generate 200 examples covering joins, sets, aggregation, connectivity, and
  constraints/consequences.
- [x] Add a scoped viewer tab with family/operation filters, exact input/gold,
  and matching board renderings using the current UI patterns.
- [x] Verify generated counts/labels and viewer integration, start the eval
  frontend on 5174, and provide the direct review URL.

## Design
Use a standalone review builder and a local generated artifact bundle served by
Vite. Source inspection is bounded to a few hundred v2 training rows. Keep the
existing canonical atlas and source facts, recompute answers, and preserve source
contract hashes. The interface consumes a compact preview JSON plus read-only
board render states derived from matching contracts. No training is part of this
preview task.

## Result
Available at `http://localhost:5174/?tab=board-fluency`. Generated 200 rows from
200 distinct states, 195 maps/trajectories, and 20 operations (10 each), with 40
rows per approved family. Sources are 167 engine rollouts and 33 replay states;
only the first 400 symbolic-v2 source lines were examined. Bundle path:
`artifacts/generated/sft/symbolic_board_fluency_review_v1/`.

Verification: all 200 source-contract, answer, and render-state checks passed;
rebuilding under another Python hash seed was byte-identical. Targeted code
review found no blockers. TypeScript and Vite production build passed (output
under the approved temporary directory). Browser verification covered all five
family filters, 20 operation filters, matching board rendering, exact question/
gold/input text, row navigation/permalink reload, search/reset, and downloads;
zero page errors or failed requests. Eval frontend is listening on 127.0.0.1:5174.

---

# Symbolic board dataset review (2026-09-14)

## Scope
Review existing v1/v2 and related datasets using at most 300 sampled corpus rows.
Treat structural queries and fast deterministic game calculations as one
board-fluency class; strategic judgment remains for self-play RL. Recommend reuse and generation
gaps while preserving settlement composition and Longest Road as transfer tests.

## Plan
- [x] Locate dataset versions and inspect manifests, source mix, and task definitions.
- [x] Inspect up to 280 rows across symbolic v1/v2 and related atlas/spatial data.
- [x] Summarize existing coverage, sample findings, and prioritized generation work.

## Review
Inspected exactly 280 unique corpus rows: 100 training + 20 transfer-validation
from each symbolic version, 20 atlas-topology rows, and 20 spatial-continuation
rows. Counts below come from manifests/metadata, not full corpus scans. Stored
golds were inspected, not independently revalidated. No builds/tests/training.

- `artifacts/generated/sft/symbolic_board_v1/`: 3,200 train, 480/480 component
  validation/test, 2,240/2,240 transfer validation/test. Preserve its concise
  explicit-state question style; sampled incident-road labels exhibit the
  documented roster-position shortcut.
- `artifacts/generated/sft/symbolic_board_v2/`: same component counts, 548/535
  transfer rows; corrects roster sampling and raises distinct dynamic training
  states from 306 to 1,600. Prefer its generator/sampling over raw v1 artifacts.
- `artifacts/generated/sft/atlas_topology/catan_atlas_topology.jsonl`: documented
  390 text-only atlas facts; reusable relation foundation.
- `artifacts/generated/board_recognition/spatial_continuation_v1/`: 1,024
  image-conditioned training examples. Reuse local-tile/production oracles after
  rendering explicit symbolic source state and recomputing labels.
- Both symbolic versions allocate 1,600 rows to fixed-atlas tasks; another 560
  select supplied owners/pieces/resources. Several near-tile/port questions also
  ignore current state. Existing composition and traversal tasks are useful but
  configuration coverage and symbolic production tasks need expansion.

Recommendation: retain v1-style prompts and v2 sampling safeguards; replace most
supplied-fact selection with state/topology joins, coverage/overlap, blocker-aware
components/reachable sets, exact production/probability, port access, and paired
local-change consequences. A proposed first 3,200-row pilot can allocate roughly
800 each to atlas, state/topology joins, configurations, and immediate calculations;
this is a starting design, not an established optimal mixture. Full settlement
composition and Longest Road stay transfer-only. Expand held-out engine-state
coverage for cycles, effective blockers, and qualifying ties; existing replay
transfer boards miss these. Source availability is 5,120 training states across
333 resource/number layouts, not a reason to bulk-generate repetitive questions.

Scope refinement: balance query operators/compositions rather than separate
structural/game categories. Add board-wide resource pip totals and tied maxima
as weighted-aggregation questions; distinguish printed tile-pip totals from
ownership/city/robber-conditioned production.

User approved the qualitative mix: relations/joins, sets/coverage,
aggregation/comparison, connectivity/structure, and constraints/consequences.
Use explicit symbolic state and short exact answers. Balance operator coverage
and composition depth; the earlier numerical quotas remain provisional.
Next proposed artifact is a small inspectable generation preview before scaling.

Sampling windows (1-based, inclusive):
- V1 train: 1–25, 801–825, 1601–1625, 2401–2425. Transfer validation:
  4–5, 33–35, 564–565, 593–595, 1124–1125, 1153–1155, 1684–1685, 1713–1715.
- V2 train: 1–40, 511–525, 1021–1035, 1544–1558, 2093–2107. Transfer validation:
  1–8, 182–185, 363–366, 495–498.
- Atlas: 1–3, 28–30, 58–60, 217–219, 220–221, 362–363, 364–367.
- Prior spatial train: 1–6, 17–18, 33–42, 49, 57.

---

# Playground new game and test cleanup (2026-09-13)

## Scope
Add a discoverable New Game action to Session controls. Reuse reset/start APIs,
preserve saved traces and model preferences, and return to empty setup without
automatically reopening an old checkpoint. Prune tests added for the shared
prompt/notes work, not unrelated engine or training tests.

## Plan
- [x] Inspect existing session controls, reset API, saved-checkpoint selection,
  and task-specific tests. Subagents unavailable due usage limit; direct review.
- [x] Add New Game, guard concurrent session changes, and keep cleared setup empty.
- [x] Remove redundant task-specific test modules and browser parameter matrices.
- Cancelled: further automated verification at the user's request; human verification.

## Result
New Game returns to setup through the existing reset API without deleting saved
traces or resetting model preferences. Old checkpoints no longer reopen
automatically after clearing. Removed six redundant Python test modules, two
frontend test modules, and the delayed browser-test matrix.
Before the stop request, retained Python/unit tests and desktop browser flow
passed. The existing narrow-screen session-panel layout obstructed the mobile
button; left as a known limitation for human verification. No further tests or
live backend/game operations after the user's stop request.

---

# Symbolic atlas and road transfer experiment (2026-09-13)

## Agreed Scope
Reuse the existing atlas vocabulary and matching language adapters. Compare the
pre-spatial `full-board-new-layouts-20260907/checkpoints/checkpoint-128` with
`spatial-continuation-20260912-r01/checkpoints/checkpoint-128`, the last saved
checkpoint of the cancelled extension (confirmed by a prior remote listing and
trainer-state read). Its interrupted run status does not invalidate a saved
checkpoint; full tensor/hash admission remains required before training.

Learn geometry through text-only SFT. Train node/tile directions, neighbors,
incidence, road ownership/connectivity, and explicit spatial components.
The user selected settlement placement and Longest Road as downstream transfer
tests, with rules stated. Reserve complete settlement-rule combinations and
longest-trail optimization from SFT; do not hide equivalent training targets
behind different wording. Settlement tests use board placement under an
explicit setup/normal mode, assuming resources and pieces are available.

## Plan
- [x] Inspect current trainer, evaluator, datasets, rule oracles, and provenance.
- [x] Confirm learned geometry and training-versus-transfer scope with the user.
- [ ] Add checkpoint-compatible text-only training/evaluation with frozen vision,
  preserved atlas input/output rows, correct completion masking, and scope audits.
- [x] Add canonical geometry/road task oracles and strict task-aware scoring;
  validate transfer labels independently against engine rules.
- [x] Build a new deterministic symbolic projection from existing training-safe
  contracts, preserving source splits and explicit fact/pair holdouts.
- [ ] Prepare identical two-anchor experiment configs and local dry-run checks.
- [ ] Run focused tests, independent review, and document actual readiness.
- [ ] Confirm step/time budget and pinned-runtime preflight before paid training.

## User Review Handoff
The user stopped further pytest work and requested the generated dataset for
inspection. Further trainer integration, paired-launch work, and GPU execution
are deferred. No new training was launched. Text-mode changes are present;
the last shared-scorer integration has not received its final verification.

Generated `artifacts/generated/sft/symbolic_board_v2/`: 3,200 training examples,
480 component validation, 480 component test, 548 transfer validation, and
535 transfer test. Inputs and gold answers are the two `messages` entries.
V2 fixes the reviewed roster-position shortcut and uses 1,600 distinct training
states for its 1,600 dynamic component examples. Transfer settlement positives
and negatives are balanced within supported groups. Existing held-out replay
boards lack cycles/effective blockers/tied qualifying road lengths; metadata
records this limited coverage. The original v1 artifacts remain preserved.

## Findings
The diverse source pool contains 5,120 train and 64 each validation/test/color
diagnostic states, including the original replay_v1 corpus. Saved contracts
contain answer-leaking road counters/awards and explicit geometry; model inputs
must use a factual allowlist, with geometry kept in the label oracle. The
original trainer required image tensors even with zero auxiliary loss.
Road traversal, maximum trail, award ties, and settlement legality are distinct
contracts. Board-only award questions are admissible only when the answer is
determined without unknown incumbent history. Current state sources lack hands
and complete action-phase state, so they do not support full executable-action
legality labels. Existing benchmark games remain excluded from training.

---

# Shared components and fresh notes (2026-09-13)

## Approved Scope
Shared authored prompt components come first; composition order and provider
message packing are not global component invariants. Implement fresh context
with accepted private notes and new visible events. Long context is deferred.
Preserve historical contracts and the resident live game; no hosted calls,
backend restart, source reset, or saved-game conversion in this task.

## Implementation Plan
- [x] Inspect current source boundaries, lessons, dirty worktree, and tests.
- [x] Add one versioned authored component bundle and a reusable renderer;
  decision/speech compositions reference shared definitions, with validated
  membership and inputs rather than one fixed order.
- [x] Add fresh-notes action/speech contracts and per-player accepted memory,
  separate channel event coverage, strict omission/clear/replace semantics,
  and historical snapshot compatibility.
- [x] Integrate bundle source pinning/editor ownership, sandbox admission,
  failure-boundary snapshots, replay preview isolation, and trace presentation.
- [x] Verify component reuse, flexible composition, causal delivery, privacy,
  acceptance and failure restoration; run focused tests and independent review.
- [x] Update READMEs and this review with measured verification and limitations.

## Implementation Boundaries
- A self-contained authored bundle pins the entire shared-definition closure.
  Existing two-source suites remain self-contained historical contracts.
- New policy is `fresh_notes`; the default notes ceiling is 4,000 characters.
  Notes are plain text: omitted keeps, empty clears, nonempty replaces.
  Validation is atomic with the action/speech response; no automatic repair call.
- Shared notes are private per seat. Action and speech have separate delivery
  cursors; only admitted responses advance the matching input cutoff. The
  existing reaction cursor is not repurposed as a model-delivery cursor.
- Current state and all not-yet-delivered perspective-safe events are supplied;
  no prior conversation or native-reasoning replay. Historical traces stay intact.
- New source/config behavior must not silently change an existing saved game.
  All tests use temporary stores and offline transports, not the resident DB.

## Review
Implemented the shared `suites/shared_v1.yaml` bundle, typed component definitions,
independent decision/speech compositions, and the accepted fresh-notes runtime.
New live games and cold replay previews use the shared resolver; explicit legacy
sources and saved source pins retain their historical contracts. Unpinned saved
LLM games require original sources rather than silently adopting new defaults.

The existing editor now edits shared definitions once, validates both consumers,
and handles reordered references. Coherent previews and save/reset permission
checks share the state lock. Whole-bundle optimistic writes preserve overrides
on validation/reset failure; busy editors cannot lose newly entered drafts.
Notes inspection uses typed variable bindings, not hardcoded component names.

Acceptance binds player/context/channel/input cutoff/memory revision before any
action mutation. Independent action/talk cursors preserve unseen messages. Failed
decisions checkpoint accepted pre-action notes/speech and resume without speaking
twice. Schema-v5 failure snapshots and event backfill preserve exact state/evidence.
Post-action cancellation carries the committed result without changing headless
CancelledError semantics; the viewer persists its accepted model call and warning.

Final verification:
- 2,412 engine/harness/provider/live/replay/component regressions passed; 33
  existing opt-in cases skipped. No task-focused failures remain.
- 53 frontend unit tests passed. 18 mounted browser cases passed on desktop and
  mobile, including delayed save/refresh/reset responses; no JS errors or
  unexpected network requests. Fixed toolbar/banner overlap without redesign.
- TypeScript and production Vite build passed into the approved temp directory;
  existing dist was not replaced. Scoped Ruff/ESLint and git diff checks passed
  (the three existing types.ts no-explicit-any lint findings remain unrelated).
- Independent component/store/editor and lifecycle/persistence reviews cleared
  after regression-tested fixes, including source pins, atomic restore, and
  cancellation evidence.

The full repository run timed out; a later full collection was blocked by
unrelated text-training tests importing unavailable torchvision. Replay tests
whose old monkeypatch targeted a removed loader were updated and pass. No full
repository-green or model-quality/cost claim is made.

No hosted inference, gameplay on the resident server, real trace-store migration,
backend restart, saved-game conversion, dependency installation, or commit was
performed. All fixtures/stores were temporary. Concurrent unrelated changes,
including the symbolic-SFT work, remain untouched.

---

# Action context duplication investigation (2026-09-12)

## Plan
- [x] Trace displayed request counts to provider messages, session accumulation,
  authoritative decision packets, design notes, tests, and persisted suites.
- [x] Record the correction: replaying cumulative packets duplicates complete
  visible game history; an arbitrary transcript cap is not an agreed design.
- [x] Confirm the context policy: fresh context plus notes, shared components
  first; preserve historical games. Implementation is tracked above.

## Findings
`ContextAssembler.assemble()` includes all retained session messages before the
current environment. `AgentPlayer.accept()` retains prior cumulative environment
packets, while the current packet already includes complete visible game events.
The default v11 trajectory is uncapped. This mixes cumulative snapshot context
with transcript replay. Design notes explicitly reject this duplication but also
distinguish bounded context from intentional long-context continuity; the latter
cannot be replaced by snapshot-only calls without a policy decision.

## Review
Read-only investigation complete; the user subsequently approved implementation.
No hosted calls, game advances, database writes, or restarts in the investigation.

---

# Longer spatial/readout continuation (2026-09-12)

## Plan
- [x] Inspect existing continuation controls, saved checkpoint results, and
  complete-board rehearsal data.
- [x] User selected the unchanged mixed curriculum and 256 additional optimizer
  steps (384 cumulative mixed steps).
- [x] Prepare a bounded continuation from the completed spatial run's FP32
  checkpoint-128; verify configuration and matched evaluation before launch.
- [x] Launch the agreed detached run, pass remote CPU preflight, and confirm
  real training progress on the H200.
- [x] Stop the SFT worker and coordinator at the user's request; verify the
  Modal app is stopped with zero tasks.
- Cancelled: final generated evaluation and post-run scoring, because the user
  terminated the extension during training.

## Findings
The existing mixture already uses complete-board readouts for 25% of optimizer
steps. The generic trainer supports longer runs; the historical coordinator and
pilot worker are fixed to 128 steps. Local receipts do not contain numerical
retention results for checkpoints 32/64/96. Repeating the mixture increases
exposure, not the number of unique images.

## Approved Run
Continue from `spatial-continuation-20260909-r01/checkpoints/checkpoint-128` for
256 additional updates: two passes over the same ordered 1,024 examples, with
64 full-board rehearsal updates and 32 updates for each of six spatial families.
Keep inherited learning rates, rank-8 language LoRA, 154 atlas input/output rows,
FP32 visual masters and BF16 computation; fresh optimizer/schedule as in the prior
continuation. Save and teacher-force-evaluate every 32 steps; retain all eight
new checkpoints. Reuse the six saved parent panels after independent rescoring,
then generate the same six final panels. Estimated mixed-training time is about
65 minutes plus 27 minutes for final evaluation. Use bounded sequential H200
workers with a detached CPU coordinator and durable receipts.

## Verification
350 focused tests passed, including the longer-run coordinator, independent
receipt verifier, spatial scorer, and visual-precision contracts. Scoped Ruff
and diff checks passed. Independent launch review found no blockers. All 430
saved parent responses passed offline rescoring; remote checkpoint/input checks
remain a required CPU preflight. The first combined test invocation hit its
two-minute shell limit; rerunning with a sufficient limit passed in 169 seconds.

## Cancelled receipt
`spatial-continuation-20260912-r01` launched successfully. Remote CPU preflight
reproduced all six baselines and verified checkpoint/data identities. Training
logs earlier confirmed step 16/256 at 19:32:34 UTC; the final step reached before
cancellation has not been audited. User-requested stop completed at 20:04:36 UTC,
with the Modal app stopped and zero tasks. No final generated evaluation ran.
App: `ap-a4jolUBznIMMAdwGQFJsrs`; coordinator:
`fc-01M2BGP3QRPN8R3AD5FAJ7ZB1Y`; training: `fc-01M2BH8VSZB14T9QBC5S0BQY6W`.
Both training and the coordinator were cancelled; the planned final evaluation
will not be launched by this run. The remote receipt's failed/RemoteError status
reflects cancellation during training. Local `cancellation.json` records it.
Status and download/verification commands are documented in
`reports/sft/2026-09-12-mixed-spatial-extension.md`.

---

# OpenRouter forbidden-request diagnostics (2026-09-09)

## Plan
- [x] Inspect the actual 403 log and saved checkpoints without replaying inference.
- [x] Check configured key access with a non-inference request; distinguish
  evidence from possible permission, guardrail, and moderation causes.
- [x] Retain a bounded, credential-safe structured 403 reason through the existing
  live provider-failure path, with no retries, invented response, or action fallback.
- [x] Test provider rejection, off-turn attribution, persistence, and applied-action
  safety offline; preserve the current paused game if deploying the diagnostic.

## Findings
Upstream HTTP 403 at 2026-09-09 18:48:32 UTC, in the trade-response decision
barrier after checkpoint 125/revision 163. The generic HTTP exception discarded
the response body from logs/storage and returned local HTTP 500. No failed
session or rejection reason is recoverable from that log. Stored model calls:
561; the only saved failure remains the earlier reasoning-only response.

## Review
- Non-inference GET `/api/v1/key` returned 200 for the currently configured key,
  with no key-level spending limit. This does not prove model authorization or
  identify the original rejection. OpenRouter documents 403 as insufficient
  permissions, guardrail block, or moderation flag. No inference was retried;
  no keys, permissions, provider routing, or moderation settings were changed.
- Added bounded `OpenRouterHTTPFailure` for 403 only, preserving the original
  non-retry policy and other HTTP/TLS behavior. Structured error.message/request
  ID are retained when usable; known credentials/echoes are scrubbed or suppressed.
  No raw error bodies, arbitrary metadata, or fabricated model responses published.
- Reused the existing provider-failure route: retained/broadcast 502 includes the
  upstream status and forbids automatic retry; safe summary/request ID persist
  through existing failure JSON with no schema change. Off-turn attribution,
  withheld siblings, storage failure, and applied-action guards remain intact.
  Post-action 403 is a checkpointed 200 warning with safe cause, not a rollback.
- Verification: 642 focused Python tests, 20 frontend unit tests, scoped Ruff,
  and whitespace checks passed. Independent review found no remaining issues.
  Real-provider rejection behavior remains untested; mounted browser tests were
  not rerun (the separate known Knight-fixture timing failure is documented below).
- Public runtime state matched checkpoint 125 exactly, but resident decision
  traces could contain unpersisted siblings. The user explicitly selected
  "Restart from checkpoint" after disclosure of potential loss of that evidence.
  No existing export API could prove or preserve those trace-only objects.
- WAL-aware backup, isolated restore preflight, final guard, and approved restart
  completed: PID 80046 replaced by 4654. Full public HTTP/WebSocket state, private
  hands/cards, v11 suite, inference, all player cursors, DB config/checkpoint hash,
  561 stored calls and one saved failure matched afterward. No gameplay advances.
  GOLD remains at cursor 162, exactly as captured; no old cursor/error overlays.
- Temp artifacts: `catan-http403-state.json`, `catan-http403-backup.sqlite3`,
  `catan_http403_verify.py`, `catan_http403_serve.py`, and active log
  `catan-backend-http403-5001.log` in the approved OpenCode temp directory.
- The original 403 reason remains unavailable. The diagnostic fix is deployed,
  not a claim that the upstream rejection is resolved. Next evidence is the
  provider's error message from its activity/support record, if present, or a
  future captured rejection; do not use blind automatic retries to obtain it.

---

# Missing final action response (2026-09-09)

## Plan
- [x] Inspect the actual saved failure and distinguish provider channels,
  completion limits, and the active response contract without paid calls.
- [x] Classify blank final responses before JSON/XML action parsing; retain
  native reasoning as diagnostics, never executable output or a fallback action.
- [x] Verify real transport-to-route rejection, persisted diagnostics, unchanged
  gameplay/history, and existing strict response contracts with offline tests.
- [x] Deploy only after a fresh backup and checkpoint/suite verification; do not
  advance gameplay, add automatic resampling, or change inference settings.

## Findings
Failure `785646fa-17e0-481e-8eab-4845eb6a36c5`, GOLD at revision 129, followed
checkpoint 100. OpenRouter/Reka returned null final content, no native tool calls,
and action-shaped JSON only in reasoning. Both finish reasons were `stop`; the
request had no completion cap. The active contract at this failure was v11.
This is not the previous TLS failure. The trace cannot distinguish a model
channel error from upstream reasoning-parser/normalization behavior.

## Review
- The shared action parser now reports a missing final answer before JSON/XML
  decoding. Reasoning-only responses remain rejected, including legal action
  JSON/XML; completion-limit guidance appears only for an explicit limit finish.
  This fixes misleading diagnostics, not the upstream missing-answer behavior.
- 522 focused Python tests passed, including both suite formats, actual
  OpenRouter-to-live-route admission through a local HTTP transport, retained
  HTTP/WebSocket/saved diagnostics, and one application on a later manual Step.
  Scoped Ruff and whitespace checks passed. Independent review has no remaining
  findings after correcting a test's parser attribute reference.
- Browser verification: 6 passed, 1 failed, reproduced once on review. The
  existing immediate-victory Knight fixture's live panel disappears when its
  background saved-checkpoint fetch selects an empty saved trace. That fixture
  intercepts Step and never invokes this parser change. No frontend code/test
  changes made for this separate race; this is not a green browser-suite claim.
- Captured and backed up checkpoint 100/revision 129, 446 stored model calls,
  one saved failure, and active v11. Preflight reproduced the actual saved
  response through the new parser offline. GOLD's pre-action SILENCE had advanced
  its live event cursor from checkpoint 128 to 129; the restore preserved that
  observed acknowledgment plus the original retained failure, not just the DB.
- Restarted PID 72357 as 80046. HTTP/WebSocket snapshots, player statuses,
  inference settings, private hands/cards, DB config, checkpoint hash, call and
  failure counts matched after restore. No model calls or gameplay advances.
  Capture/backup/verifier/bootstrap and `catan-backend-missing-final-5001.log`
  are in the approved OpenCode temp directory. Original failure evidence remains
  unchanged; future missing answers receive the new diagnostic.

---

# OpenRouter TLS autoplay recovery (2026-09-08)

## Plan
- [x] Confirm recurring SSLV3_ALERT_BAD_RECORD_MAC in the active backend log;
  inspect retry ownership, shared-player concurrency, and failure persistence.
- [x] Retry only this TLS alert within the existing transport budget, using a
  fresh request-local client for owned transports and bounded backoff. Preserve
  shared clients, cancellation, and certificate/hostname verification.
- [x] Surface exhausted TLS calls as a safe, retained live error and saved
  failure without inventing model output or claiming post-commit rollback.
- [x] Test recovery, exhaustion, concurrent calls, no duplicate gameplay,
  failure persistence, and existing post-action warning semantics offline.
- [x] Verify live state against a consistent backup; intentionally restart and
  restore the same game/suite only after tests, without advancing it.

## Scope
Target the observed temporary blocker, not a general provider/retry framework.
No paid probes, extra games, forced actions, or weakened TLS verification.
Retries can duplicate upstream inference billing if a response was lost.

## Review
- Classified raw/wrapped bad-record alerts now recover within the existing
  retry budget. Owned clients use fresh request-local clients; shared/injected
  clients are never replaced or closed during recovery. TLS validation is intact.
- Exhaustion produces safe HTTP 502 diagnostics, retained state/WebSocket notice,
  and a failure row. No invented model response; off-turn actors are matched by
  session identity. Pre-action speech and post-application errors are distinguished;
  persistence failure is visible without exposing raw storage/provider secrets.
- Verification: 291 focused Python tests and seven mounted browser regressions
  passed. Scoped Ruff and whitespace checks passed. Independent review confirmed
  the installed HTTPX/httpcore/AnyIO wrapping and concurrency behavior.
- Backed up and verified game 27886234-8ae9-4529-8801-6399e42ac412 at checkpoint
  95/revision 124, 424 stored model calls. Restarted PID 9916 as 72357 and restored
  the same v10 suite using a restore-only pin, leaving new-game defaults unchanged.
- HTTP and WebSocket snapshots, private resources/cards, player sessions, saved
  configuration, model-call count, and checkpoint blob hash remained identical.
  No gameplay advances or paid calls. Backup/verifier/bootstrap and active log
  `catan-backend-tls-recovery-5001.log` are in the approved OpenCode temp directory.

---

# Mixed spatial continuation, 128 steps (2026-09-08)

## Approved Plan
- [x] Inspect actual dataset contracts, parent checkpoint, trainer, and frozen
  evaluation identities; confirm existing images suffice without rerendering.
- [x] Add typed task scoring: unordered exact touching-tile sets, ordered valid
  shortest paths (accept ties), strict local-neighborhood and production JSON.
- [x] Build 1,024 engine-labeled rows in eight-example task-specific batches:
  16 steps each directions, adjacency/connectivity, node tiles, shortest paths,
  local tile/resource/number, and dice production; 32 full-board readout steps.
- [x] Freeze new-task validation panels with board-layout exclusion and path
  endpoint-pair holdouts, preserving existing spatial120 and full-board64 bytes.
- [x] Verify exact labels, output contracts, quotas, source/image hashes,
  token lengths/shares, checkpoint restoration, and baseline scorer equivalence.
- [x] Launch one bounded detached cycle: parent new-task baselines, 128-step
  continuation, and matched final evaluation of new and existing panels.
- [x] Retrieve results, independently rescore, compare forgetting, and record
  final outcome or precise running status with resumable call IDs.

## Contract And Scope
Parent: `full-board-new-layouts-20260907/checkpoints/checkpoint-128`.
Keep rank-8 language LoRA, all 154 atlas rows, full FP32 visual masters, BF16
computation, existing learning rates, microbatch 4 and accumulation 2. Fresh
optimizer/schedule, token initialization kept, checkpoint every 32 steps.
Paths ignore pieces/ownership; production respects cities/robber but ignores
bank shortages. No additional brainstormed families enter this first training
mix. Static atlas recall is not unseen-graph or image-dependent reasoning.
Reuse matched 57/120 spatial and 64/64 board-readout baselines; baseline only
new task panels before training. One H200 stage at a time, no retries, bounded
stage timeouts, no automatic additional training, no historical overwrites.

## Review
242 focused tests passed; scoped Ruff/diff checks passed. Independent review
verified real engine production labels, shortest-path ties/holdouts, mixture
quotas, and bounded orchestration. A task-metadata scoring bypass was fixed
before launch. Dataset has 1,024 distinct training images over 333 layouts.

Launched `spatial-continuation-20260909-r01`, coordinator
`fc-01M22ZWJGBAV7R05K19DYKJEK0`, app `ap-5DfPPuxoiAEKDyNt8R50bo`.
Remote CPU preflight passed: parent FP32 checkpoint/token IDs, data/pixel
identities, and both reused baselines verified. Measured readout target-token
share 94.38%, explicitly not gradient/optimizer-step share. New-task parent
evaluation completed and all 246 responses independently rescored: node tiles
0/54, paths 0/64, local neighborhoods 0/64, production 16/64. All 128 training
steps completed in 32.1 worker minutes (33.5 including startup). Six-panel
post-evaluation completed in 26.7 worker minutes, call
`fc-01M232GXRCGP3GR9YPKY8EXFSE`. Coordinator completed successfully.
All 676 pre/post responses independently rescored with exact receipt matches.
After: spatial 53/120, readouts 53/64, touching tiles 19/54, paths 1/64,
local neighborhoods 18/64, production 46/64. Occupied readout accuracy remains
1,193/1,204, with failures concentrated in node/building facts on two layouts.
Parent preserved; no automatic extra training or checkpoint promotion.
Report: `reports/sft/2026-09-09-mixed-spatial-continuation.md`.

---

# Corrected spatial evaluation (2026-09-08)

## Plan
- [x] Verify the latest completed full-board checkpoint and corrected held-out
  supplement, including live Modal checkpoint availability.
- [x] Extend the existing eval builder to select the corrected supplement and
  only its 120 spatial validation questions, with distinct IDs and provenance.
- [x] Preserve saved FP32 visual weights in the general evaluator, matching the
  completed full-board evaluator; verify builder/loading changes offline.
- [x] Launch one detached H200 evaluation, greedy/no-thinking, batch 48,
  16-token answers, original images only, no candidate scoring or training.
- [x] Run one separately named, matched answer-format control: the original
  result is 1/120 exact, but none of the 100 binary responses are bare yes/no
  and the source prompts omit an explicit answer-only instruction. Preserve
  the original result; change only output-format guidance and versioned IDs.
- [x] Retrieve all 120 responses, independently rescore, verify checkpoint/input
  identity, and publish per-relation/task/entity results and limitations.

## Scope
Checkpoint: `full-board-new-layouts-20260907/checkpoints/checkpoint-128`.
Use new dataset/run paths. Preserve historical prompts/results and keep the
corrected report out of legacy fingerprint/first-token diagnostic aggregation.

## Review
User explicitly requested this model eval. Input SHA256:
`930c9ce829566f40f9c7de0dc0bcf2b1f1c11dfdc7f3165f094767baef0bfa1e`.
All 120 rows / five images uploaded; detached call
`fc-01M21SZH04RS53CJ4JCBNQYBWP` in app `ap-F8zKIl6B9UYda6tR6FT4DE`.
Local receipt: `artifacts/runs/sft/full-board-new-layouts-ck128-spatial-choice-order-v1-20260908-r01/launch.json`.
Original run completed: 1/120 exact. Raw responses independently rescored with
no mismatches; FP32 restoration confirmed. Stopped its Modal app after retrieval.
Replanned one matched format control to avoid treating verbosity as a spatial
error. This is not training or a new task/image/checkpoint sweep.

Answer-only control completed: 57/120 (47.5%); binary 55/100, node choices 0/10,
tile choices 2/10. All binary answers now follow the requested format; only
6/20 token answers name an offered choice. Independent audits pass every check
for both runs, including the prior full-board visual-file hash. Both GPU apps
are stopped with zero tasks. Final focused suite: 82 passed; scoped Ruff and
diff whitespace checks passed. Report:
`reports/sft/2026-09-08-corrected-spatial-eval.md`.

---

# Spatial QA choice-order fix (2026-09-08)

## Plan
- [x] Trace the shared direction-question bank, sampling, stage-2 projection,
  saved artifacts, and existing test patterns.
- [x] Keep displayed candidates fixed across inverse questions while preserving
  answers, relation labels, bank size/order, and IDs. Prefer exact paired balance
  over independent random swaps, which can leave sampled blocks biased.
- [x] Add regressions for node/tile choice positions, sampled blocks, and
  renderer-aligned answer correctness through the stage-2 projection.
- [x] Run focused tests and independent review; export a corrected supplement
  to a new directory without overwriting historical data or model results.
- [x] Document verification, corrected-data location, and legacy-data caveats.

## Scope
Fix the answer-position shortcut only. No training, hosted model calls,
curriculum redesign, or changes to historical eval inputs/results.

## Review
- Seven regression cases failed before the fix; all 28 spatial/supplement/
  production tests passed after it. Targeted Ruff passed. Renderer-aligned
  centers verify both node and tile labels through repeated/shuffled stage 2.
- Exported and validated 5,304 rows (4,104 train) under
  `artifacts/generated/board_recognition/replay_v1/spatial_robber_choice_order_v1/`.
  Actual prompt/answer counts: train first/second 86/86, validation 10/10,
  test 10/10, color diagnostic 32/32. Historical inputs/results were untouched.
- Independent review found no generator/test blocker. Before promoting new
  evals, account for existing diagnostics: behavior fingerprints omit prompt
  text, and failure-scorecard subjects default to the first prompt token.
  Neither path was used to publish corrected eval results here. Existing
  production/eval builders still select the historical `spatial_robber_v1`.

---

# Semantic action tools v11 (2026-09-08)

## Plan
- [x] Inspect v10 parsing, prompt assembly, engine legality, and saved-suite paths.
- [x] Confirm expanded scope: semantic tools for every action, literal trained
  board tokens for spatial arguments, and Knight plus robber destination.
- [x] Add the v11 JSON tool-call contract (game_plan, tool, arguments), preserving
  literal <Nxx>, <Exx_yy>, <Txx> tokens rather than treating them as XML markup.
- [x] Resolve semantic calls against the frozen engine menu; preserve strict
  legality, bounded retries, old indexed suites, and domestic trade lifecycle.
- [x] Preflight and apply Knight plus movement in one sandbox decision, retaining
  canonical events, immediate-victory boundaries, and separate victim selection.
- [x] Cover valid calls, illegal bundles, unavailable tools, conflicting controls,
  unchanged rejection state, and recorded v10 compatibility. Update docs.
- [x] Run focused tests and independent review; record verification below.

## Scope
Keep canonical engine actions unchanged; append only optional Knight destination
metadata to PlayerChoice with actual saved-receipt compatibility. No reward
penalties, silent fallback policy, renderer refactor, live restart, or hosted
model calls. New default is v11; recorded suite sources remain authoritative.

## Review
- Implemented all 20 action families using semantic JSON and literal trained
  atlas tokens. No model-facing action indices or resource-combination menus.
  Internal exact menus remain the legality authority; v10 source bytes and
  15-/16-slot saved PlayerChoice receipts remain compatible.
- Knight plus destination is preflighted, committed synchronously as canonical
  events, and accepted once. Immediate victory suppresses movement; subsequent
  victim selection and stolen-resource RNG remain engine-owned. Live traces show
  committed actions separately from requested destination metadata.
- Replay evals preserve parsed semantic receipts and historical indexed records.
  Knight agreement is explicitly primary-action/coarse, with followup unscored;
  no future replay action is used to construct a decision or invent a score.
- Final Python verification: 2,233 passed, 34 skipped, excluding the pre-existing
  `tests/test_reweight_node_edge.py` fixture error (`full_coverage` argument).
  A full run reproduced that error after 1,967 passes; unrelated tests unchanged.
- All six mounted browser tests and 20 frontend unit tests passed. Browser setup
  built the production frontend. Scoped Ruff, component ESLint, and whitespace
  checks passed. `types.ts` ESLint still reports its three pre-existing `any`
  annotations; the new optional trace fields pass TypeScript compilation.
- Independent reviews cleared after fixing replay normalization, concrete/mixed
  trade selection, exact cancellation IDs, and raw-token escaping checks.
  Initial broad/browser runs timed out; isolated reruns completed successfully.
- No hosted calls, live-game advances, or backend restarts. One review import
  invoked configured trace-schema initialization without reading/writing game
  rows; subsequent broad/review/browser runs explicitly used temporary databases.

---

# Harness boundary review follow-up (2026-09-08)

- [x] Inspect acquisition/staged failure tracing, communication schema metadata,
  and the terminal observation/menu boundary.
- [x] Retain pending barrier replies without changing the failing actor's API;
  wire authored communication instructions and explicit terminal preview opt-in.
- [x] Verify capacity/retry/cancellation evidence, private schema echoes, and
  terminal preview menu isolation with offline tests and Ruff.

Review: 363 targeted offline tests passed, including all boundary-audit cases.
Ruff and targeted diff whitespace checks passed. No hosted models, live-game
operations, corpus runs, or production database writes were performed.

# Correctness remediation (2026-09-08)

## Approved scope and implementation plan
User approved implementing the deeper audit findings. Keep unrelated changes
and the live game intact; verify with local transports and temporary databases.

- [x] Review failing audit cases and coordinate engine/replay/harness ownership.
- [x] Correct road connectivity/longest-road awards, own-turn victory, terminal
  boundaries, shortage payouts, discard limits/exact bundles, and trade IDs.
- [x] Restore complete replay checkpoints and publish canonical replay lifecycle
  events; index speech and preserve compatible saved inference settings.
- [x] Structurally parse control fields, detach mutable player inputs/results,
  validate typed outputs/commitments, and preflight whole trade batches.
- [x] Version action-visible talk/commitments and named discard parameters in a
  new suite, retaining actual persisted-suite and pickle compatibility.
- [x] Convert repaired audit xfails into ordinary regressions and run combined
  tests, full-game checks, then one consolidated local corpus audit.
- [x] Obtain independent review, resolve findings, document evidence, and
  intentionally restart/restore the backend only after verification.

## Design boundaries
Use the existing typed engine action/event contracts, not a parallel runtime.
Ordinary live actions fail closed after victory; replay retains explicit forced
outcomes. Preserve historical prompt source bytes and resolved discard-card
serialization. A new suite owns added social context and exact-discard output;
do not silently rewrite saved prompts or infer hidden state from table talk.

## Verification and deployment
- 1,076 targeted Python/browser tests and 20 frontend tests passed; build,
  targeted Ruff, and whitespace checks passed. All prior audit xfails removed.
- 32 full games / 17,304 transitions passed inventory, independent road/award,
  own-turn victory and seat-continuity checks; timed-out batch seed 31 was
  completed separately. Full local corpus: 66 games, 31,506 actions, 15,460
  trade-lifecycle actions, no asserted resource mismatches or semantic errors.
- Independent rule/replay/harness reviews cleared after targeted fixes.
  Legacy restore repairs derived caches without reordering equivalent menus.
- User chose v10 for the legacy unpinned game. Restored checkpoint 83/revision
  108 with unchanged material state, snapshot blob, and 368 stored model calls.
  Confirmed WebSocket state and rendered v10 prompt components. Corrected
  P3_LONGEST_ROAD_LENGTH from 2 to 3; saved xhigh reasoning now honored.
- No hosted calls or game advances for verification; backups are in the
  approved OpenCode temp directory. Historical stored labels are not recertified.
- Architecture clarification: ReplaySandbox and CatanSandbox remain separate
  classes, sharing engine/context/player machinery but not one step contract.
  No inheritance refactor was requested or performed during this clarification.

---

# Deeper correctness checks (2026-09-08)

## Scope
Verification and executable reproductions only. Do not silently change game
rules, prompt/action contracts, or the running live game during this audit.

## Plan
- [x] Run isolated full-game inventory/continuity probes and adversarial
  parser/typed-player/provider checks.
- [x] Run the full existing local replay corpus audit once for this code state.
- [x] Preserve confirmed defects as strict expected-failure audit tests, with
  positive controls and opt-in full-game sweeps.
- [x] Independently review the reproductions and run the combined audit suite.
- [x] Record evidence, rule references, coverage limitations, and fix priority.

## Initial findings
- Inventory conservation passed across 288 engine games plus six sandbox
  games, but independent checks found road-legality, Longest Road scoring,
  turn-owned victory, payout, and discard-contract errors. Conservation is
  not a rule-correctness oracle.
- Full local corpus: 66/66 replays, 31,506 actions, 15,460 trade-lifecycle
  actions, no fatal issues or asserted resource/trade mismatches. That audit
  allows replay force paths and final-score sync; it does not certify undo,
  causal event history, RNG continuation, or live game rules.
- Further confirmed defects concern replay event restoration/publication,
  shared mutable observations/events, malformed control tags/typed outputs,
  decision-invisible table talk, vLLM resume, and speech indexing.

## Review
- Added guarded audit tests only; no production fixes, live actions, hosted
  model calls, restart, or real `.cle` database writes in this follow-up.
- Existing targeted suite: 395 passed, one opt-in corpus skip (run separately
  in full above). Combined new audit with four full-game seeds: 10 passed,
  35 strict expected failures. Of those, 31 reproduce correctness cases and
  four explicitly propose custom-player robustness policy, not 35 unique bugs.
- Independent review verified each failure at its intended comparison. Tightened
  xfail exception types so fixture/inventory/I/O errors cannot satisfy them;
  reduced unit positions are distinguished from natural full-game evidence.
- Audit Ruff and independent re-review passed. Detailed findings, sources,
  rerun commands, caveats, and priority order:
  `reports/correctness-audit-2026-09-08.md`.
- Follow-up implementation needs separate scope for game-rule fixes versus
  shared context/action contract changes. Existing-award revocation below five
  remains an additional coverage gap, not a verified extra failing case.

---

# Live harness correctness audit (2026-09-07)

## Scope and plan
- [x] Trace the reported rejected counteroffer and red response dump; audit
  parsing, engine trade admission, decision lifecycle, and viewer diagnostics.
- [x] Preserve opaque counteroffer IDs; reject ambiguous indices and duplicate
  JSON keys; keep literal template-like model text as data; fail closed on
  invalid private recipients; identify singleton Year of Plenty choices.
- [x] Prevent unresolved wildcard trade execution and share non-mutating trade
  admission checks with action validation.
- [x] Protect step ownership/freshness, preflight trade barriers, commit applied
  decisions before communication, cancel sibling inference on failure, and
  refresh round-dependent menus and reaction cutoffs.
- [x] Keep concise live errors and show rejected attempts in collapsed inspector
  diagnostics with correct responder attribution and final/native separation.
- [x] Run focused regressions, frontend checks, and independent code review.
- [x] Preserve and restore the live checkpoint for a verified backend restart;
  do not advance the user's game or call hosted models during verification.

## Review
Implementation and safe restart complete.
Existing unrelated worktree changes are outside this audit. Exhausted attempts
now persist in independent schema-v4 failure rows without changing checkpoints.
Communication admission records distinguish accepted, rejected, and withheld
responses; applied-action speech failures still persist the committed step and
stop auto-play rather than inviting a duplicate retry.

- Combined verification: 375 Python/browser tests passed, one opt-in local
  replay-corpus audit skipped; 20 frontend tests passed; production build,
  targeted Ruff/ESLint, and git diff whitespace checks passed.
- Two independent reviews were rerun after fixing audience-tag whitespace,
  inert legacy rationale, broadcast-triggered autoplay stopping, and rejected
  communication attribution. No remaining blockers in the reviewed changes.
- Actual localhost UI: concise banner, collapsed rejected final-output
  diagnostics, no page errors or mutations at desktop; mobile loads without
  horizontal overflow. Existing 390px panel sizing prevents opening the
  inspector; follow up separately without silently redesigning styles.
- Backed up SQLite and the pre-fix API error under the approved OpenCode temp
  directory. Restarted listener 53580 as 69015; schema migrated to v4. Restored
  game `27886234-8ae9-4529-8801-6399e42ac412`, checkpoint 43, revision 50.
  GET state game/resources/dev cards/inference match exactly; WebSocket restore
  delivery verified. Stored calls remain 183; no model calls or actions added.
- Limits: no full local replay-corpus run, no rollback for arbitrary engine or
  custom player.accept exceptions, and no guarantee that cancelling a local
  provider task stops already-running remote inference/billing. Pre-fix native
  reasoning omitted by the old error API cannot be recovered from its preview.

---

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
- [x] Step 1: sft/scripts/analyze_occupancy_misses.py with a fixture test.
- [x] Step 2a: adjacent_pair_localization.py exporter, render_contract and
  _row refactor, 9 tests, panel entry, README.
- [x] Step 2b: evaluator metadata and neighbor_confusion block.
- [x] Step 2c: exported pairs_v1 (51,564 train rows, 20% empty, 2,159
  validation rows, audit clean); launched 256 steps from v3 final.

Launched 2026-09-03 13:55 PDT: run `catan-qwen38-sl-pairs-v1-s256-20260903`,
data identity `82b9664c0c46`, app ap-GJWUfVfWDMWGJ6CGc0nADW, call
fc-01M1MH3P4TKCWJGQAHPBK8ZJ4N, batch 16 x 2, 256 steps, eval and save every
64. Output under
`/runs/catan-vision-sft/catan-qwen38-sl-pairs-v1-s256-20260903/82b9664c0c46`.
Gate before sparse: occupancy positives and negatives >= 98% on pair
validation, no hop-1 false positives in the neighbor_confusion block, tile
and localization rows unchanged. Then run the regression panel on the final
bundle and `sft/scripts/analyze_occupancy_misses.py` on its replay set.
Checkpoint-64 on pair validation (app ap-FxCTlNirGsvjLXJWmHx0w6, output
/runs/qwen-series-eval/pairs-v1/ck64): exact 0.826, candidate 0.950.
Negatives learned (adjacent 0.849, far 0.982; adjacent misses name the
target 22 and the partner 10). Positives collapsed: occupancy_positive
0.336 with 248 of 269 misses answering "empty", uniform across pair kinds
(30-39%), colors (19-51%), and pieces. Tiles and localization held (0.93 to
1.00). Same shape as the v2 step-128 collapse that recovered by step 256;
decision deferred to the step-128 eval.

Checkpoint-128 on pair validation (/runs/qwen-series-eval/pairs-v1/ck128):
exact 0.883, candidate 0.992; positives 0.728 (64 "empty", 44 name the
partner), adjacent negatives fell to 0.529 (58 name the target, 39 the
partner), far 0.938, localization 0.99, tiles 0.94 to 1.00. The prior
swung from "empty" to "nearby piece"; piece-to-token is solved, token-to-
piece next to a second piece is the residual.

In-run eval: step 64 loss 0.176 / row exact 0.829; 128 0.089 / 0.885;
192 0.068 / 0.918; 256 0.059 / 0.930. Still improving at a quarter of peak
LR, so the next stage launch uses 512 steps. Final bundle at
`.../82b9664c0c46/final`; regression panel (7 sets, label pairs-v1-final,
app ap-spno9s5TpSQ668zloZT3FB) launched on it.

Panel (pairs-v1-final) single-piece v3 set: negatives 0.950 (up from
0.890), positives 0.825 (down from 0.942, all 42 misses "empty", roads
77.5% / settlements 84% / cities 91%), localization 1.00, tiles 0.99.
User asked for more road data and a longer run: exporter gained per-kind
image counts; pairs_v2 = node_node=30, edge_edge=80, node_edge=50 per
board (eval 10/20/15), 1 adjacent + 1 far negative, tiles. Panel entry
swapped to pairs_v2. Next launch: 512 steps from the pairs_v1 final bundle.

Launched 2026-09-03 ~16:05 PDT: run `catan-qwen38-sl-pairs-v2-roads-s512-20260903`,
data identity `d430f310de8a`, app ap-5628OGxpUM5CLarPop9j7r, call
fc-01M1N012MCXBQQ8E763Z0YGNP1, from the pairs_v1 final bundle, batch 16 x 2,
512 steps, eval and save every 128. pairs_v2: 67,478 train rows, roads 66%
of positives, 20.4% empty. Output under
`/runs/catan-vision-sft/catan-qwen38-sl-pairs-v2-roads-s512-20260903/d430f310de8a`.
Local disk was full (404 MB free); with permission the abandoned v4, v5, v6
single-piece exports were deleted (4.7 GB).

Panel pairs-v1-final complete (recorded under pairs_v1_run in
reports/sft/2026-09-03-qwen38-single-piece-v2.json): pair validation
positives 0.788 / adjacent 0.747 / far 0.982; real boards node.occupancy
0.75 (from 0.625), edge.owner 0.65 (from 0.602), 77 misses (from 99),
neighbor confusions 27 (from 32), cross-type 0 (from 9), false negatives
42 unchanged, roads 52% recall, bronze 0/8. Far-pair control set exported
(`spatial_localization_pairs_control_v1`, validation/test only) and added
to the panel; occlusion variants launched on the pairs_v1 bundle (app
ap-MjEP6EuStM6eDmuqKzlA7E). Occlusion controls on the pairs_v1 bundle: positives 0.788 original,
0.012 with the queried spot masked (274/405 become "empty"), 0.662 with an
unrelated spot masked (empties rise 66 -> 110). Grounding is real; the
positive deficit is clutter sensitivity. Open decisions: single-piece kinds
in the pair mix, patch loss weight 1 versus LoRA rank 16 arms.

pairs_v2 checkpoint-128 (/runs/qwen-series-eval/pairs-v2/ck128): single-
piece v3 positives 0.908 (from 0.825 after pairs_v1; roads 0.89 from 0.775,
settlements 0.97, cities 0.88), negatives 0.853 (from 0.950; 36 of 44 name
the piece), tiles 0.96. pairs_v2 validation positives 0.805, adjacent
0.680, far 0.951, localization 0.985, in-run loss 0.116 / row exact 0.902.
Road-heavy mix is recovering lone-piece roads; run continues to 512.
Checkpoint-256: pairs_v2 validation positives 0.872 (roads 0.90), adjacent
0.849 (26 of 34 name the target), far 0.956, localization 0.99, tiles 0.92
to 1.00; in-run loss 0.055 / row exact 0.945. Single-piece v3 positives
0.892 (roads 0.85), negatives 0.930, tiles 0.96 to 1.00. Both heads rising
together for the first time; pair kinds edge_edge 0.927, node_edge 0.951,
node_node 0.902.
Checkpoint-256 failure geometry (both eval sets, 126 tokens with n>=4):
error vs distance to a 32 px patch boundary r=0.09, vs training frequency
r=-0.01 (every token gets 140-277 forward queries). Slanted edges fail 4x
more than vertical ones (0.128 vs 0.032); coastal tokens slightly worse.
Worst tokens fail on several validation boards, so it is geometry not one
background. Token glitches: <T10> number? answered with a resource 28/58,
<N16> answered with numbers 7/8, <E11_32> emitted twice as an answer 5/8.
Token-row surgery on the pairs_v2 final (no training, evals under
/runs/qwen-series-eval/pairs-v2/): rowswap6 (six rows from v3) fixed the
E11_32 doubling, helped N16, left T10 at 50% but flipped which head it
answers with. meanfix10 (twelve rows realigned to the global atlas mean)
made T10 answer "empty" on both tile heads: the global mean is node/edge
dominated. Family analysis: every tile input row sits at +0.22..+0.30 to
the tile centroid except T10 at -0.10 (v3: -0.19); N26 and E08_27 were
anti-aligned to their families in v3, N16 became so during pairs; output
rows are normal. famfix (13 rows realigned to their own family centroid at
the family median, residual kept): T10 14/22 and 22/36, number head 100%
and resource head answering numbers; N16 slightly up; others unchanged.
Across all three surgeries T10 answers one head's type to both prompts and
the row picks which head. The suffix-conditioning failure for T10 is
upstream of its row. Row-surgery line closed.
Run complete: step 384 loss 0.047 / row exact 0.960; step 512 loss 0.030 /
row exact 0.972. Final bundle at `.../d430f310de8a/final`; eight-set panel
(label pairs-v2-final, app ap-iKwx2e32A6pT3lzn851xJV) launched on it.

(An earlier launch at 13:47, app ap-XmbzzSYHOxaU5ZLDzFskw8, was interrupted
during upload before any call spawned and is stopped.)
- [x] Step 3 prep: `--slice-stages` in the curriculum builder; rungs a
  (setup + grounding, 4,960 rows), b (sparse, 3,424), c (dense, 8,096)
  written under replay_v1/production_curriculum_v1_rung_{a,b,c}; launcher
  dry-run against validation_v1 passes. Occlusion controls showed the
  residual positive failure is clutter sensitivity, so the 98% pair gate is
  no longer a prerequisite; rung a starts from the pairs_v2 final bundle.
- [ ] Step 3: after the pair gate (positives and negatives >= 98%, no hop-1
  false positives), run production_curriculum_v1 empty/setup + sparse from
  the pair bundle, then dense.

Panel pairs-v2-final (recorded under pairs_v2_run.panel_final in the report):
real boards node.occupancy 0.797 / edge.owner 0.789 (v3 0.625 / 0.602,
pairs_v1 0.75 / 0.65), 53 misses (v3 99): false negatives 24, neighbor
confusions 12, elsewhere 15; roads 45/64, bronze 7/8; tiles unchanged;
robber/ports still 0 (untrained). pairs_v2 validation positives 0.928,
adjacent 0.911; far-pair control scores like touching pairs; single-piece
v3 positives 0.963 / negatives 0.960. Decision: rung a starts from the
pairs_v2 final bundle (`.../d430f310de8a/final`). Launch awaits the user.

# Overnight 2026-09-04: token-init experiment, stages skipped

Two runs from the base model (no initial bundle) straight onto
spatial_localization_v3 train (24,080 rows), 512 steps, batch 16 x 2, eval
and save every 128 on v3 validation (1,980 rows):
- `catan-qwen38-sl-v3-direct-familyinit-s512-20260904`, `--token-init
  family_words`, app ap-K0OaRzUNteoi3apL3WG6EM, call
  fc-01M1NTRHHHQZHYAJEHWNHGPXR5, output
  `/runs/catan-vision-sft/catan-qwen38-sl-v3-direct-familyinit-s512-20260904/020dd5ad268f`.
- `catan-qwen38-sl-v3-direct-meaninit-s512-20260904`, default init, app
  ap-Ocnfz9zmNomkiQFMls8V90, call fc-01M1NTRXY4NEZ00HXV948G5PNW, output
  `/runs/catan-vision-sft/catan-qwen38-sl-v3-direct-meaninit-s512-20260904/020dd5ad268f`.
Reference: staged v3 run (from v2 ck256) had loss 0.226 / row exact 0.770 at
128, 0.101 / 0.876 at 256, 0.037 / 0.949 at 512.
User clarified they meant a distributional init: the family_words run
(ap-K0OaRzUNteoi3apL3WG6EM) was stopped within minutes and replaced by
`catan-qwen38-sl-v3-direct-gaussinit-s512-20260904` with `--token-init
vocab_gaussian` (rows drawn at the vocabulary's per-dimension mean and std):
app ap-cJCHyYUTNRNuK3jD9H3jTf, call fc-01M1NVCS7ZSY07Y4MZHXC8F467, output
`/runs/catan-vision-sft/catan-qwen38-sl-v3-direct-gaussinit-s512-20260904/020dd5ad268f`.
Step 128: meaninit loss 0.445 / row exact 0.476; gaussinit loss 0.543 /
row exact 0.497 (staged v3 reference 0.226 / 0.770).
Step 256: meaninit 0.267 / 0.625; gaussinit 0.265 / 0.637 (staged v3 reference 0.101 / 0.876).
Step 384: meaninit 0.234 / 0.646; gaussinit 0.181 / 0.727 (earlier note of 0.128 / 0.811 at 384 was the 512 line caught early).
Step 512 (final): meaninit 0.204 / 0.686; gaussinit 0.128 / 0.811 (staged
v3 reference 0.037 / 0.949). Row geometry of the finals: meaninit rows are
still nearly all entangled (149/154 with a partner >0.25, pairwise cos
0.21); gaussinit rows have none (pairwise cos 0.011), T10 sits at the tile
median (+0.26), family centroids are distinct (NT 0.27 vs 0.83). Recorded
under token_init_experiment_20260904 in the report. Per-task on the
gaussinit final: localization heads 0.87 to 1.00, tile_to_token 0.97,
tile_number 0.92, but occupancy positives 0.60 (single) / 0.38 (pairs),
negatives 0.53 / 0.56, tile_resource 0.63. T10 shows ordinary errors, no
head flip. Query-side grounding is what the skipped marker stage provides.
Recommendation: gaussian init as default for from-base runs; a full staged
run with it if the token anomalies are to be removed for good. No GPU
running as of this note.
Morning checks: side-by-side in-run curves; row geometry of both final
bundles (tile rows vs tile centroid, T10 in particular); panel on the better
bundle. The pairs_v2 final panel (ap-iKwx2e32A6pT3lzn851xJV) decides the
rung a starting bundle.

# Gaussian-init ladder with failure-mode gates (approved plan, 2026-09-04)

Plan file: ~/.claude/plans/async-wandering-phoenix.md. Gates are failure-mode
scorecards, never loss.
- [x] Step 0: sft/scripts/inspect_token_rows.py and sft/scripts/failure_scorecard.py
  with tests (11 passing); scorecards backfilled under reports/sft/scorecards/
  for the marker control, v3 final, pairs_v1 final, pairs_v2 final, and the
  gaussian direct eval.
- [x] Step 1 launched: `catan-qwen38-gauss-s1-marker-20260904`, from the base
  model, `--token-init vocab_gaussian`, v1 marker stage-1 data (24,080 rows),
  256 steps, eval and save every 64, app ap-TTCqSzF68fAtxwbTIUpQdJ, call
  fc-01M1PXB6MBB8HDRGGEE63RGEYE, output
  `/runs/catan-vision-sft/catan-qwen38-gauss-s1-marker-20260904/334d305b1c43`.
- [x] User asked for road-shaped markers: `--entity-markers` in the marker
  exporter draws edge markers as bars along the edge at its true angle and
  tile markers as tile-scale hexagons; exported as spatial_localization_v1e.
  The diamond-marker stage-1 run (ap-TTCqSzF68fAtxwbTIUpQdJ) was stopped at
  about step 60 and stage 1 relaunched on v1e with the eval only at the end:
  `catan-qwen38-gauss-s1-markers-entity-20260904`, app ap-kO4nZVjPpdt6jiPIqzjp8A,
  call fc-01M1PZJJ9YHDSFB7EZAK7MT0M0, 256 steps, save at 128 and 256, output
  `/runs/catan-vision-sft/catan-qwen38-gauss-s1-markers-entity-20260904/<identity>`.
- [x] Step 1 gate passed: v1e validation exact 0.991 (marker_to_token 0.987,
  token_to_marker 0.995); diamond-glyph transfer 0.877 (edges 0.764);
  gray-dot probes 0.179 / candidate 0.310 (glyph control: 0.928 / 0.639);
  scorecard modes all zero; rows 0 twins, 0 anti-aligned. Grounding is more
  marker-shape-specific than the glyph control; accepted for the ladder.
- [x] Step 2 launched: `catan-qwen38-gauss-s2-v3-20260904` from the stage-1
  final, v3 data, 512 steps, eval at the end only; app
  ap-caHTX43hyS76PjHelxu6VH, call fc-01M1QABS61GZDMXNKNHPJCRETC, output
  `/runs/catan-vision-sft/catan-qwen38-gauss-s2-v3-20260904/020dd5ad268f`.
  Stopped by the user within minutes of launch (2026-09-04); no checkpoint written.
- [x] Budget-capped final marker diagnostic (2026-09-04): 308 paired rows across
  all 154 tokens and five boards; free generation 303/308, candidate 304/308.
  Marker-to-token 149/154; token-to-marker 154/154. Four wrong locations
  (E22_23, N03, N08, P05) and one node-to-edge answer (N43 -> E43_47), no
  malformed/repeated output. Slanted queries 95/96, vertical 48/48; actual-road
  transfer remains untested. App ap-XgCM5XzAhsdEgK8G2F2WPQ completed and stopped,
  192s runtime (~$0.25 compute estimate, not settled billing), no retries.
  Report: reports/sft/2026-09-04-gaussian-entity-marker-mini.md. This small
  diagnostic does not close the full step-1 gate or launch the next stage.
- [x] Step 2 redefined by the user as terrain readout (tiles and ports only,
  full readouts, no inverse, no pieces, split by layout): exporter
  `terrain_readout.py`, evaluator long-answer routing, panel set `terrain`.
  Launched `catan-qwen38-gauss-s2-terrain-20260904` from the stage-1 final,
  49,152 rows, 512 steps, eval at the end only; app ap-zUI8zNEA48pTRdWvaZkyec,
  call fc-01M1QG8NPW77M1YJRZA1JBGYCH, output
  `/runs/catan-vision-sft/catan-qwen38-gauss-s2-terrain-20260904/398f0a023ec9`.
- [x] Loss at 1e-5 on the 77 training layouts raised a memorisation flag;
  checkpoint-128 on the 5 unseen validation layouts: tile.number 0.996,
  tile.resource 0.991, port.port_type 0.953 (16 of 23 misses answer "wood
  port"), so it reads rather than memorises. Readouts by items 1479/1792 =
  0.825, 4 of 64 fully correct; misses are omissions (ports dropped, desert
  dropped, half the readouts stop early at 19-27 items). The first ck128 eval
  OOMed on the readout rows (batch 48 at 2,048 tokens); evaluator now
  sub-batches long rows and scores readouts by items. terrain_readout_v2
  (900 synthetic train layouts, 40 val, 40 test; 92,352 rows) is exported
  and dry-run clean but NOT launched: the user wants checkpoint-256 of the
  current run first.
- [x] Checkpoint-256 on the same 5 unseen layouts (eval
  `/runs/qwen-series-eval/gauss-ladder/s2-terrain-ck256`, 3,072 rows):
  tile.number 1216/1216, tile.resource 1216/1216, port.port_type 576/576, all
  five layouts perfect on every short head. Readouts 63 of 64 exact, items
  1779/1792 = 0.993, zero extra items. The one miss (layout 194209320, state
  246) is a sequence skip: <T03> got <T04>'s answer and the shift ran to <T06>,
  then the model jumped to <P05> and stopped, dropping <T18> and eight ports.
  Rows at ck256: input pair mean 0.012, 0 twins, 0 below the family floor,
  T10 +0.244 on the tile centroid; output 0.062, 0 twins. It generalises
  across layouts.
- [x] Run stopped by the user at step 416 (2026-09-04 21:50 PDT, from the
  progress bar in the final log): the last 96 steps sit under a quarter of peak learning rate and checkpoint-384
  (21:34) was complete on the volume. Checkpoints 256 and 384 remain; 128 was
  rotated out by the two-checkpoint save limit. ck384 rows are identical to
  ck256 to three decimals (input 0.012 / 0 twins / T10 +0.244; output 0.062 /
  0 twins). Stage-2 gate launched on ck384: full 9-set panel, original and
  blank, label `gauss-s2-terrain-ck384`; app ap-K55aKeyzYWoa9A3fYpcmG9, call
  fc-01M1QZYZQZCVGMRKMGP1FQ8Z6R, output
  `/runs/qwen-series-eval/regression-panel/gauss-s2-terrain-ck384`.
- [x] Gate, first four sets (2026-09-04 23:05 PDT, panel still running): the
  terrain rung forgot the marker grounding. Diamond marker set, ck384 versus
  the stage-1 final on the same 1,540 rows: marker->token 17/770 = 0.022
  (was 0.768), token->marker 552/770 = 0.717 (was 0.986; tiles 1.000, ports
  0.844, edges 0.725, nodes 0.585). The marker->token misses are head flips:
  446 answer a tile token (<N00> -> <T05>), 307 answer a resource word
  ("sheep"). Gray-dot probes 0.000 (was 0.179). Tile heads transfer to the
  v3 renders at 1.000; piece heads on single-v2/v3 are 0.000 as expected for
  a bundle that has never seen a piece. Row drift from the stage-1 final:
  node/edge input rows did not move at all (cos 1.000, norm unchanged) and
  their output rows barely (cos 0.993, norm -1%); tile/port rows moved (cos
  0.90, input norm +12%, output norm +7-9%). So the forgetting lives in the
  shared LoRA, vision tower and merger plus the grown tile/port output rows,
  not in row entanglement (still 0 twins). Failing mode: head flip /
  forgetting. The fix it names is rehearsal: mix marker and terrain rows into
  every later rung instead of training each rung on its own data alone.
  Decision pending with the user before stage 3 launches.
- [x] Step 2 gate PASSED on checkpoint-384 (panel `gauss-s2-terrain-ck384`,
  scorecard `reports/sft/scorecards/gauss-s2-terrain-ck384.json`): terrain
  set 3,072/3,072 on the 5 unseen layouts, readouts 64/64 exact, 0 sequence
  skips, 0 head flips, 0 glitches; blank-image control 5.4%, so it reads
  the board. Rows 0 twins, 0 below the family floor. Full-board set: port
  1.000, tile number 1.000, tile resource 1.000; node occupancy and edge
  owner 0.000 (never trained on pieces), robber 0.25, spatial yes/no 0.48.
  Side effects recorded, not gated: markers 0.369 (marker-to-token 0.022),
  probes 0.000, single-v2 0.001, single-v3 0.304 (tile heads only), pairs
  0.211, pairs-control 0.209. These are the stage-3 side-effect baseline.
- [x] Step 3 redefined again by the user (2026-09-05): not pairs. The node
  and edge recognition rung trains on real replay boards, token as query
  only, terrain-style: `<N17> building?`, `<E17_18> road?`, plus a 54-item
  node readout and a 72-item edge readout per image with explicit empties.
  Spec in the plan file (every row, colour validation, gate).
- [x] Exporter `data_pipeline/board_recognition/node_edge_readout.py` and
  tests (6): train capped at 4 occupied + 4 empty per family per image (empty
  quota adjacent 2 / cross-type or hop-2 1 / far 1), eval splits with
  every node and edge; splits train / validation / test / color_diagnostic.
- [x] Evaluator: `node_readout`/`edge_readout` in LONG_ANSWER_TASK_TYPES;
  metadata whitelist gains state_id, layout_id, piece_count, item_count,
  occupied_count.
- [x] Panel sets `node-edge` and `node-edge-colors`; scorecard reads
  every `.readout` category and counts sequence skips (dropped tokens,
  values shifted onto the previous token). Trainer row guard raised from
  512 to 2,048 answer characters (`MAX_ANSWER_CHARACTERS`); readouts run
  to 1,525.
- [x] Exported `node_edge_readout_v1` (2026-09-05): 77/5/5/16 layouts;
  train 17,708 rows (886 of 1,024 images at the full 18; empties 6,123
  adjacent / 830 cross-type / 1,239 far), validation, test and
  color_diagnostic 8,192 rows each; 30 sampled rows agree with their
  contracts; readout answers 700 to 1,525 characters. Launcher dry run
  clean: 17,708 train / 8,192 eval rows from terrain checkpoint-384, 512
  steps, eval and save every 128, output
  `/runs/catan-vision-sft/catan-qwen38-gauss-s3-nodes-edges-20260905/e64dcbc2cc4c`.
- [x] Baseline, pairs_v2 final on `node-edge` (validation, 8,192 rows, every
  node and edge of 64 images; label `pairs-v2-final-node-edge`, app
  ap-ehWrCwFf8DOxe4kIPddtkM; scorecard
  `reports/sft/scorecards/pairs-v2-final-node-edge.json`). These are the
  stage-3 gate bars: node occupancy 0.749 (empty boards 1.000, setup 0.817,
  sparse 0.686, dense 0.623; occupied 0.852, empties 0.730); edge owner
  0.828 (setup 0.889, sparse 0.768, dense 0.736; roads 0.565, empties
  0.872). Modes: blindness 317 (roads 254, settlements 53, cities 10),
  neighbor confusion 650 (hop-1 360, hop-2 268, wrong piece 22), far false
  positives 665, other 28, head flips 0, glitches 0; edge error vertical
  0.124 versus slanted 0.196; colour recall min 0.493 (green), bronze 0.60,
  blue 0.67, mystic blue 0.88; occupied recall city 0.887, settlement 0.842,
  road 0.565. Readouts 0 of 128 (never trained). Colour-diagnostic set
  (8,192 rows, 16 engine layouts, all 11 colours): node occupancy 0.824,
  edge owner 0.791; blindness 596, neighbor 537, far 408; occupied recall
  settlement 0.757, city 0.764, road 0.419; per-colour recall from green
  0.433 to pink 0.582, every colour in the 0.43 to 0.58 band, so the bar
  is min recall 0.433 and the failure there is global, not one colour. Scorecard fix on the way: rows were treated as
  synthetic because they carry target_token/piece/color; `is_synthetic`
  now keys on the grounding stage (single_piece, adjacent_pair).
- [x] User approved piece-recognition launch (2026-09-05). Reweighted input
  `node_edge_readout_reweighted_v1`: 17,708 rows / 1,024 images, completion-token
  exposure approximately 50% occupied / 25% empty / 25% readout, colour x piece
  balancing; the full-readout subset is 81 rows. Held-out files unchanged.
  Run `catan-qwen38-gauss-s3-nodes-edges-rw-20260905`, dataset `bbbc99d8bc8c`,
  from terrain checkpoint-384, 512 steps, batch 16 x 2, eval/save every 128.
  App `ap-ZwkBRDHVyLMbNl3fg4BiuH`; training call
  `fc-01M1SQEX7D43F1JGWZSMA6YKVT`; CPU watchdog
  `fc-01M1SQEXBHVSB8Y4ZNHCX2RF7X`. $79 plan, $25 reserved for later evals;
  training bounded by 7-hour execution timeout, 16-core/128-GiB ceilings,
  no function-error retries and absolute cancellation at 21:36:43 PDT.
  A timeout means incomplete training; no automatic resume or extra run.
  Local receipt `artifacts/runs/sft/gauss-s3-nodes-edges-rw-20260905/launch.json`.
- [x] Startup verified at 14:31 PDT: step 3/512 observed; CPU watchdog persisted
  `watching` at 14:27:12. Live trainable-scope report has no errors; all vision
  (327 tensors), merger (6), language LoRA (992) and atlas-row tensors are FP32.
  This confirms startup and precision, not recognition accuracy.
- [x] User requested removal of budget guard at 14:35 PDT. Paused the CPU
  watchdog process before cancelling `fc-01M1SQEXBHVSB8Y4ZNHCX2RF7X` with
  container termination, avoiding its fail-closed training cancellation.
  Watchdog call confirmed cancelled; original training call remains active
  (step 9/512 observed), with no restart. Absolute watchdog deadline no longer
  enforced; native seven-hour per-attempt timeout and resource limits remain.
  The original $79 plan is not an active aggregate spending guarantee.
- [ ] Read piece-recognition checkpoint metrics;
  final node/edge, colour and terrain-retention evals within the $25 reserve,
  scorecard/rows and gate table here. No standalone eval jobs auto-launched.
- [ ] After piece recognition improves: new combined readout of all 154
  locations followed by `robber <Txx>` (155 entries, existing atlas token for
  the robber tile). User explicitly deferred this until after the current rung;
  do not mix it into the approved piece-recognition run.
- [ ] Step 4: rungs a, b, c from the winner.
- [x] The long gaps in the loss prints are the in-run eval (2026-09-05
  18:00 PDT): the full 8,192-row validation at eval batch 32 is 256 batches
  at about 7.6 s each, roughly 33 minutes of silence at steps 128, 256, 384
  and 512, about 2.2 hours of the run. Checkpoint-128 landed 16:17; the
  step-256 eval was at 191/256 at 17:58. Next launch: pass a 1,024-row
  eval sample for the in-run eval; the full-coverage scoring is the side
  evaluator's job.
- [ ] Checkpoint-128 side eval launched 18:05 (`node-edge` set, original,
  label `gauss-s3-rw-ck128`, app ap-BPlyYX5M4JOnZNX894MlkO, call
  fc-01M1T3RCRSK5MXQ57TWSF1D9MN); ck256, ck384 and the final to follow;
  final panel with `node-edge` and `node-edge-colors`, scorecard against
  `pairs-v2-final-node-edge`, rows; gate table here.
- [x] Stage-3 run stopped by the user at 18:25 PDT after terrain collapsed
  (checkpoint-256 on the terrain set, first 1,336 rows: port 0/252, tile
  number 0.183, tile resource 0.083, the wrong answer is `empty`; the
  stage-2 bundle was 1.000 on all three). Checkpoint-256 is complete on the
  volume; checkpoint-128 node-edge eval and checkpoint-256 terrain +
  node-edge evals kept running. Early piece numbers at ck128 (3,604 rows):
  edge owner 0.865, node occupancy 0.869 versus pairs_v2 0.828 / 0.749, with
  272 of 276 edge misses being `empty` on a road. Mechanism: no tile or
  port rows in the mix, shared LoRA / vision tower / merger overwritten,
  `empty` became the default answer for unrecognised prompts; token rows
  unchanged. Fix for every later rung: rehearsal of every earlier head in
  the mix, or the combined all-location readout.
- [ ] Additional possible contributor, recorded 2026-09-05: stage 3
  continued terrain checkpoint-384's existing rank-8 LoRA; it did not
  merge the language adapter into frozen base weights and start a fresh
  rank-8 adapter. Adapter reuse may contribute to interference, but this
  is untested, not a missing-load bug or a confirmed explanation. Candidate
  comparison: continuation versus merge-plus-fresh-LoRA from the same
  parent, matching supervision and other trainable components; measure
  piece acquisition and terrain/port retention. No run authorized or
  launched. Evidence, caveats and controls are in
  `reports/sft/2026-09-05-node-edge-recognition-rung.md`, under
  "Possible contributor: reusing the same LoRA across stages".
- [x] Stage-3 checkpoint-256 on the node-edge validation (8,192 rows;
  scorecard `reports/sft/scorecards/gauss-s3-rw-ck256.json`, baseline
  pairs_v2): node occupancy 0.924 (pairs_v2 0.749), edge owner 0.881
  (0.828), errors 937 (1,788). Far false positives 5 (665), neighbor
  confusion 314 (650), head flips 0, glitches 0, empties 0.95 to 1.00 by
  kind. But blindness 495 (317): road recall 0.418 (0.565), settlement
  0.763 (0.842), city 0.913 (0.887); colour recall min 0.429 mystic blue,
  black 0.54, blue 0.64, red 0.57 (all down), green 0.82 (up). Road recall
  by density: setup 0.20 / sparse 0.39 / dense 0.52 versus pairs_v2 0.90 /
  0.57 / 0.44; settlements setup 0.67 (0.98), dense 0.85 (0.71). The
  reweighted train set is 71% dense-board rows (12,609 of 17,708; setup
  1,930), so the model learned a density prior: on a sparse board it
  answers `empty`. Readouts: node 5/64 exact, items 0.854; edge 0/64,
  items 0.334 with 683 extra items (the 81-row readout budget). Terrain at
  ck256: port 1/576, number 0.215, resource 0.083, readouts 0/64 with the
  response `empty port; empty port; ...`.
  Gate verdict at 256: fails on blindness and colour, passes neighbor, far,
  head flip, glitch. Mix recipe must stratify by density bin as well as
  colour x piece, and keep readouts at hundreds of rows.
- [x] Rung 3b built (2026-09-05 evening): `mix_rung_data.py` + recipe
  `configs/sft/mix_rung3b_v1.json` -> `replay_v1/mixed_rung3b_v1`, 14,770
  rows over 1,024 images, density near-equal (setup 4,563 / sparse 4,833 /
  dense 5,024 / empty 350). Estimated token shares: piece short 38% (7,000
  occupied, colour x piece balanced with 3x oversampling for rare colours,
  4,500 empties hardest-first), terrain 25% (3,000 short + 120 readouts),
  node/edge readouts 37% (80 + 70). In-run eval sample 1,683 rows (every
  7th node-edge validation row incl. node readouts, every 6th terrain row).
  Source pool `node_edge_readout_pool_v1` (train full coverage, 131,072
  rows). Launcher dry run clean from stage-3 checkpoint-256, 512 steps,
  eval/save every 128, run name `catan-qwen38-gauss-s3b-mixed-20260905`.
  NOT launched: waits for the user.
- [ ] Hybrid bundles on the volume for the forgetting ablation, eval only:
  `qwen-series-eval/experiments/hybrid-lora256-vision384` (stage-3 ck256
  LoRA + rows, terrain ck384 vision) and `hybrid-lora384-vision256`.
  Terrain + node-edge sets on each, ~55 min and ~$5 each. Waits for the go.
- [ ] Merge-and-unload path for rank changes (user: LoRA 16 or more):
  Modal merge job folding LoRA, rows and vision into a full checkpoint on
  the volume, launcher support for a volume-path model, token-init mode
  that keeps merged rows. About a day; for the rung after 3b.
- [x] Stage-3 checkpoint-128 on the node-edge validation (scorecard
  `reports/sft/scorecards/gauss-s3-rw-ck128.json`): node occupancy 0.835,
  edge owner 0.854, but blindness 1,103, road recall 0.04, settlement 0.03
  to 0.09, colour min recall 0.0: at step 128 the model answered `empty`
  almost everywhere and scored 85% because 70% of the rows are empties.
  By 256: blindness 495, roads 0.42, settlements 0.76. The run was stopped
  at about step 300 mid-ramp; ck256's blindness is a point on a steep
  curve, not a plateau. Edge readouts went the other way (items 0.834 ->
  0.334 with 683 extras) as the model began emitting pieces inside them.
  The single-set panel run writes to the label root
  (`gauss-s3-rw-ck128/summary.json`), not a set subdirectory.
- [x] Eval redesign after the user's call on the 70%-empty sets (2026-09-05
  evening): node-edge eval splits are balanced (every occupied location
  plus as many empties by kind shares adjacent 0.5 / cross 0.15 / hop2 0.1
  / hop3 0.05 / far 0.2): validation 2,622 rows, test 2,790, colour 2,801;
  `--eval-coverage full` kept for diagnostics. Train empties use the same
  shares. Evaluator: per-class recall and precision, balanced accuracy,
  occupied versus empty readout items. Scorecard leads with road,
  settlement, city recall and empty precision. Bundles rescored on the
  balanced validation (`reports/sft/scorecards/balanced-node-edge-*.json`):
  pairs_v2 errors 819/2,622, roads 0.565, settlements 0.842, cities 0.887,
  empty precision 0.753; stage-3 ck128 errors 1,297, roads 0.035; ck256
  errors 742, roads 0.418, settlements 0.763, cities 0.913, empty precision
  0.702, far false positives 0. Mixed rung 3b re-exported with the balanced
  in-run eval sample (887 rows); dry run clean; still not launched.
- [x] Scorecard gains tile resource, dice number and port recall beside the
  piece recalls (2026-09-05). One table, balanced node-edge validation plus
  the terrain set (pairs_v2's terrain from the full-board set, 43/42/128
  rows): pairs_v2 roads 0.565 / settlements 0.842 / cities 0.887 / empty
  precision 0.753 / resource 1.000 / number 0.976 / port 0.000 (never
  trained on ports); stage-2 ck384 pieces 0 / terrain 1.000 / 1.000 /
  1.000, readouts 64/64; stage-3 ck256 roads 0.418 / settlements 0.763 /
  cities 0.913 / empty precision 0.702 / resource 0.083 / number 0.215 /
  port 0.002, readouts 5/128.
- [x] O-LoRA trainer profile `olora_frozen_bundle` (2026-09-06): frozen
  stage-2 bundle (exact vision weights, language LoRA and rows merged in
  memory), fresh rank-16 adapters on language layers and the vision tower's
  qkv/proj/fc1/fc2 and merger projections, atlas rows kept, orthogonality
  penalty (lambda 0.5) against the frozen adapter's lora_A rows and the
  visual-delta SVD lora_A rows; bundles carry `frozen_adapter/`; evaluator,
  initial-bundle loading and reload validation merge it first. Launcher
  gains `--frozen-bundle --visual-delta-factors --orthogonal-lambda
  --vision-lora-learning-rate --lora-rank --lora-alpha --lora-dropout` and
  uploads the factor file with the dataset. Tests: 26 trainer tests incl.
  penalty math, bases loading, vision targets and categories. Dry run clean
  on mixed rung 3b: run `catan-qwen38-olora-s3c-mixed-20260906`.
- [ ] Workspace move to the user's `icebear5h` profile (own account):
  volumes created, stage-2 checkpoint-384 bundle uploaded to the same path;
  the base model re-downloads into the new HF cache on the first run.
  Preflight 2026-09-06 morning: bundle at
  `/runs/catan-vision-sft/catan-qwen38-gauss-s2-terrain-20260904/398f0a023ec9/checkpoints/checkpoint-384`
  carries the five files the frozen loader needs (adapter config and
  weights, visual_model, training_config, trainable_parameters) with sizes
  matching tetracorp; the optimizer, tokenizer and processor files were not
  copied and are not needed. Token inventory for `--token-init keep` is
  `ms_swift_bidirectional_v1/trainable_tokens.json` (sha matches the
  parent's). 38 trainer and budget tests pass. Dry run on `icebear5h`
  stops at `Secret 'catan-hf' not found`; the launcher requires that name
  with key `HF_TOKEN`. `huggingface-secret-2` exists there but its key
  name is unverified. User's call: fall back to `tetracorp` for this run.
  Dry run clean there (`MODAL_PROFILE=tetracorp`, run name
  `catan-qwen38-olora-s3c-mixed-20260906`, plan shows profile
  olora_frozen_bundle, rank 16 / alpha 32, lambda 0.5, token_init keep,
  factors uploaded as dataset `c0858f119641`). Awaiting the go for
  `--no-dry-run --spawn-training` with `modal run --detach`.
- [x] O-LoRA loader startup fixes (2026-09-06, found by the other session's
  smoke, finished here): the parent's `visual_model.safetensors` is keyed by
  PEFT-wrapped names, so it is restored while the parent `PeftModel` is
  still attached (`apply_frozen_adapter(..., restore_visual=True)`), fp32
  promoted first; then merge, then fresh adapters. The LoRA target lists are
  computed once before `get_peft_model`, which renames the targets in place
  (`base_layer`, `lora_A`), so the report no longer recounts and the vision
  count no longer hits "no vision linear modules". Covered by a real-PEFT
  test on a Qwen-shaped tiny model (parent bundle written through PEFT,
  exact vision restore, merged language and atlas rows, 2 + 6 targets, 2
  protected modules, smoke forward). Needed peft 0.20 locally (Modal pins
  0.20.0; 0.17 rejects trainable-token rows on an untied `lm_head`), so the
  `sft` extra pins `peft>=0.20.0` and `[tool.uv] conflicts` separates it
  from the llamafactory `pretraining` extra. 39 trainer and budget tests pass.

---

# Pi Minecraft sounds in global OpenCode (2026-09-07)

- [x] Inspect Pi's active sounds and OpenCode's supported TUI plugin API;
  user approved the full port, with model-switch audio on next submission.
- [x] Copy the nine active assets and register one global TUI plugin under
  `~/.config/opencode`, preserving 0.65 volume, 30-second long-task timing,
  serialized subagent cues, and parent completion suppression.
- [x] Verify event handling, config/plugin loading, audio, and independent review.
- Scope: leave Pi, permission policies, desktop notifications, and project
  runtime code untouched. Manual `!` exit failures and pre-start background
  cancellations lack reliable public events in OpenCode 1.18.29.
- Review: 29 tests / 226 assertions pass; strict TypeScript passes. The real
  isolated TUI Plugins dialog reports `minecraft-sounds` active. All nine
  copied files are byte-identical and pass real `afplay` playback; completion
  was also played at 0.65 volume. Independent review found no remaining blockers.
  Setup/limitations are documented in `~/.config/opencode/README.md`.
# Live checkpoint navigation and recorded token usage (2026-09-15)

- [x] Inspect dirty tree, navigation, provider usage, and trace persistence.
- [x] Separate browse-only presentation from runtime; reuse navigator in live dock.
- [x] Add lightweight usage projection, per-step totals and covered-step averages.
- [x] Verify aggregation, routes and isolated mounted navigation; document review.

Design: history GETs never restore a sandbox or replace active inference settings.
Latest returns to the runtime view; progression is guarded while browsing/loading.
Metrics use canonical model-call rows (decision retries plus communication), never
duplicated result payloads. Failure-only calls are reported separately, excluded
from completed-step averages. Unknown provider usage remains unknown.

Review: reused TraceStepNavigator/SavedStepReasoningTrace, with one navigator in
the dock. Runtime board/configuration stays independent of the browse view;
Latest uses no load POST. Initial runtime identity gates default checkpoint
selection, request generations discard stale detail responses, and failed history
GETs keep progression locked until Latest. Metrics use the existing trace GET's
usage-only projection; old baseline rows are excluded and failure batches retain
distinct decision/speech identities. README documents denominator and limitations.

Verification: 44 frontend Node tests; 63 trace-store/live-route tests; 13 isolated
mounted browser tests (1280/1600 widths, real engine board states, exact saved
messages/reasoning, usage, socket updates during history, failed GET recovery,
startup into an existing game, and existing autoplay regressions). Production
TypeScript/Vite build and focused ESLint passed. Existing dirty work preserved.
No live server restart or user's game mutation. An already-running old backend
must pick up the usage projection before game averages become available; per-step
usage works through the existing checkpoint endpoint. No transport usage estimates.
# Compact board control strip (2026-09-15)

- [x] Flatten status/actions and inline checkpoint navigation; suppress empty history.
- [x] Show available token summaries with optional coverage details; retain warnings.
- [x] Verify unit/build/scoped lint and isolated desktop/mobile browser flows; review.

Design: retain existing palette and runtime handlers, use one busy status and stable
button labels, truncate secondary model settings, wrap controls within the dock.

Review: one status, stable Step/Auto-play labels, inline history, available-only
token summaries and native details disclosure. Preserved runtime/navigation guards
and alert visibility. Desktop populated strip fits two rows (<90px); empty/busy
strip stays <85px at 390/1600px. Mobile wraps and enabled dock hit targets pass.
45 Node tests, TypeScript/Vite build, scoped ESLint and diff check passed. All 16
isolated browser cases passed; after final CSS compaction, all five affected
desktop/mobile layout/history cases passed again. Browser build uses a temporary
output directory. Updated the existing fixture's required result field; mobile
exact-request expansion remains covered on desktop because its inspector collapses.
Screenshots: pytest-57/test_live_history_board_exact_{0,1,2}/compact-history-*.png
and test_compact_empty_and_busy_co{0,1}/compact-controls-*.png under the local
pytest temp root. No resident backend restart or live game mutation.
# Message board shows the game step (2026-09-16)

- [x] Stamp live speech log rows with the recorded trace step index after record_step.
- [x] Rewrite the recorded checkpoint so browsed steps carry the same labels.
- [x] Frontend: derive table talk in a pure module; render "step N", falling back to "event #N".
- [x] Unit/store/route/frontend tests for stamping, relabelling and fallback.

Design: the trace step index only exists after `record_step` commits, so speech rows
keep being logged with their engine-event sequence during `analyze_transitions`, and
`stamp_message_step_indexes` labels the rows appended since a per-step mark. The step
row is already durable, so the follow-up `update_step_public_state` write is best
effort - a failed relabel costs the label, not the step. Rebuilding the snapshot after
stamping keeps the POST response, the socket broadcast and the stored public state
identical, which the existing checkpoint equality assertion enforces.

Review: no second source of truth - the label is derived from the recorded index, not
recomputed in the UI. Rows without a recorded step (no trace store, or games saved
before this change) fall back to the engine-event sequence instead of inventing a step.
Derivation moved out of App.tsx into `src/tableTalk.ts` so it is directly testable,
matching traceUsage/liveStepErrors.

Verification: 48 Node tests, `tsc -b`, and 104 passing route/store/logging/fresh-notes
tests plus 115 reactive-speech/action-batch/trade/checkpoint-audit tests. Two failures
in tests/test_live_sandbox_routes.py (invalid-attempt diagnostics) are pre-existing in
this dirty tree: uncommitted `_retry_feedback` in cle/sandbox/catan.py appends "Your
currently legal tools: ..." to the surfaced validation_error, which those expectations
predate. Unrelated to this change; left for the author of that work to settle.

Follow-up (same day): the message board rendered empty because the log is broadcast,
not accumulated - both the socket snapshot and `/api/state` sent only the last 50 rows,
and a 384-step game's tail held no speech (12 messages existed, the newest at step 318).
Moved the Messages panel under Game log, then dropped the tail: both snapshots now carry
the whole log. A dedicated speech list was built first and then reverted - with an
unbounded log it was a second source of truth for the same rows. Storage cost: the
per-step checkpoint's public_state grows with the log (~500 bytes/row) against the
~2.1 MB pickled snapshot each step already writes, so ~13% on a 1.4 GB trace DB.

## 2026-09-16 Persistent auto-play (unlimited retries until game complete)
- [x] Keep-awake: standalone `caffeinate -dims` (pid 76877, "asserting forever" per `pmset -g assertions`) already covers the machine; no second one started
- [x] `autoPlay.ts`: `retryable` on step results, optional `retryPause(failures)`; unbounded retries, count resets after success, game-over checked before failure
- [x] `liveStepErrors.ts`: warnings retryable (action applied, next Step advances); `liveStepFailureRetryable` stops only when `retryable=false` and `checkpoint_saved=false` on a traced game
- [x] `App.tsx`: fetch/guard failures retryable, cancellable exponential backoff (1s doubling to 30s), retry notice state, socket notices no longer cancel auto-play unless persistence failed
- [x] `App.tsx`: `snapshotKey` echo-dedupe now includes `last_live_step_error`, `live_inference`, `player_types` (the pre-existing gap noted above; notice-only broadcasts were being dropped)
- [x] `BoardControlsDock`: retry countdown line (`role=status`), "Auto-play retrying" status, updated tooltips
- [x] Tests: `npm test` 60/60; `tsc -b` clean; eslint clean on touched files; browser suite 22/22 with new 422/500 retry and stop-during-wait cases
- [x] README live-failure section documents the retry policy and the single hard stop

Review: frontend-only, so the live backend (pid 76710) and its in-memory game
were not restarted. Vite hot-reloaded `App.tsx`, which remounts the app and
ends any auto-play that was running at that moment; Auto-play must be clicked
once more in the tab to pick up the persistent loop.
# Miles eval-only integration (2026-09-21)

User requested code for Miles evaluations and supplied the upstream CLI Eval
reference. Use the built-in evaluation flow: one SGLang GPU, zero training
rollouts, greedy/no-thinking generation, existing Catan scorers, saved raw answers.
Pin the inspected upstream interface to Miles commit
`12754e9507e64d5e537288da17793246e913c525`.

- [x] Inspect eval-only driver, dataset fields, reward hook, and eval-log hook.
- [x] Add a small `sft/miles_eval/` package: dataset projection, scoring/result
      hooks, and a prepare/launch CLI using the documented eval flags.
- [x] Preserve Catan IDs/metadata/golds and complete matched-pair admission;
      reject media, training splits, unsupported scoring, and adapter-only models.
- [x] Document a one-GPU invocation using a complete merged HF serving checkpoint.
      The current PEFT bundle must be exported separately; this task is eval code.
- [x] Verify with real local Catan rows and upstream API contracts; run focused
      tests and `uv run --no-sync python -m scripts.quality`, reporting failures.

No inference or training job is requested here. Keep the new implementation in
files of at most 300 lines and do not alter existing experiment data or scorers.

## Review

Implemented the eval-only CLI, exact existing scorer hooks, raw JSON results,
paired summaries, and CPU checkpoint/token-budget checks. Prepared the actual
400-row comparison at `artifacts/generated/sft/miles_coordinate_eval_v1/`; local
dry-run invocation works without Miles/GPU execution. All 21 focused tests pass;
targeted Ruff and strict mypy pass across all nine new Python source/test files.
The required full quality gate ran: 223 structural violations and existing
Ruff/mypy failures remain elsewhere, with no diagnostics for the new Miles files.

Upstream source review confirmed the eval-only flags/hook contracts and corrected
two details: dry-run now prints a self-contained wrapper invocation, and the docs
state that Miles still initializes a lightweight FSDP trainer actor under its
debug path even though training weights/optimizer are never loaded. Real
Miles/SGLang inference and checkpoint export/parity remain unverified; no GPU job
was launched and no inference result is claimed.
# Stock-Qwen exact Cartesian evaluation (2026-09-21)

User explicitly selected stock Qwen3.8-27B, no fine-tuning, and ordinary equal-scale
2D Cartesian coordinates using sqrt(3), rather than a representation sweep.
Reuse the existing 200 canonical cases as a new immutable Cartesian-only panel.

- [x] Confirm stock versus existing Catan checkpoint and inspect current contracts.
- [x] Add exact rational/sqrt(3) tile/node/edge/port positions and an inspectable
      200-case dataset, with board center (0,0), top corner (0,1), y pointing up.
- [x] Reuse the original task oracle; parse exact coordinate answers without
      numeric approximation, and preserve original case/split/provenance identities.
- [x] Wire the panel into Miles scoring and permit a stock tokenizer when model
      inputs/answers contain no atlas tokens; keep prior panel admission intact.
- [x] Generate the new panel/Miles projection, run focused tests and quality gate,
      and document the stock-checkpoint command and remaining GPU verification.

Existing generated comparisons and r04 experiment evidence remain historical;
the new panel is separately versioned. This is evaluation preparation, not SFT.

## Review

Implemented `sft/cartesian_eval/` and integrated its exact scorer/admission with
Miles. Active artifacts are `artifacts/generated/sft/cartesian_eval_v2/` and
`artifacts/generated/sft/miles_cartesian_eval_v2/`. All 200 original cases and
their 198-test/2-validation provenance are retained. The v2 legend explicitly
distinguishes entity sets from color answers; the earlier v1 draft was never
evaluated. Fraction arithmetic verifies the 154 typed positions and 72 unit roads.

Stock-only tokenizer preflight adds no vocabulary and skips the legacy atomic
atlas requirement; legacy panels still enforce their saved 154 token IDs. The
stock-HF command's local dry run passes. Focused verification: 29 tests passed,
Ruff and strict scoped mypy clean on all 17 source/test files, `git diff --check`
clean. Required full quality gate executed and remains red: 198 structural
violations plus Ruff/mypy failures elsewhere; none reference the new eval files.
Real stock-tokenizer token sizing and Miles/SGLang GPU inference remain pending.
No fine-tuning, adapter loading, vocabulary expansion, or GPU job was performed.
# Execute stock Cartesian evaluation (2026-09-21)

User explicitly requested the GPU run. Use existing 200-case Cartesian v2 data,
stock Qwen3.8-27B revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, and Miles
revision `12754e9507e64d5e537288da17793246e913c525`. One H200; no training.

- [x] Verify Modal workspace and cached stock checkpoint (all 18 shards present).
- [x] Pin official Miles amd64 image digest and implement bounded Modal wrapper.
- [x] Run CPU dependency/token/checkpoint preflight in the same image.
- [x] Launch one 30-minute-bounded H200 evaluation, monitor logs, and retain receipts.
- [x] Download/rescore every answer, report per-task accuracy and measured compute
      estimate, verify app termination, and document any launch fixes.

Image: `radixark/miles@sha256:6628bff749ffd32e6a62b479a1128daee25a8c0e3c28eb86301a0d620f5dd598`.
GPU execution limit 1,800 seconds, startup limit 600 seconds, retries=0;
CPU preflight limit 1,200 seconds. No automatic replacement GPU job.

## Review

Completed `miles-cartesian-stock-20260921-r01`: 143/200 exact (71.5%), with
direction 55/64, adjacency 7/32, incidence 3/16, and ownership 78/88. All raw
outputs/metadata/golds were downloaded and rescored; all stored scores agree.
No truncation; 12 predictions referenced unknown Cartesian points and one failed
atom syntax. Actual generation/scoring loop ~41 seconds; H200 function 379.92s,
CPU preflight 104.46s. Recorded function-window cost estimate ~$0.65, excluding
image building/startup/termination/storage and not an actual bill.

Official image import initially exceeded the local CLI timeout before any
function ran; reused the cached image and installed extra packages into the
image's `/opt/sglang` interpreter. Only one GPU evaluation was launched.
App `ap-c299dzp6nqmgY5ce4W4hsN` in `icebear5h` is stopped with zero tasks;
the empty image-import app is also stopped. User requested the default profile
switch to `tetracorp`, which is activated and verified. Historical artifacts
were retrieved using an explicit `icebear5h` environment override.

Report: `reports/sft/2026-09-21-stock-cartesian-eval.md`.
New Modal wrapper passes targeted Ruff/strict mypy and `git diff --check`.
Required repo-wide quality gate ran and remains red on 194 structural violations
plus existing Ruff/mypy failures elsewhere; no diagnostics name the new wrapper.
# Sparse h Cartesian rerun (2026-09-21)

User requested the next eval and confirmed continuing after the tile-coordinate
clarification. Stock Qwen3.8-27B, same 200 cases and inference conditions; new
input uses h=sqrt(3)/4, exact decimal y, and omits only empty dynamic node/edge
records. Static inventories retain 154 entities. All 19 tile records remain.

- [x] Verify active `tetracorp` profile and its complete cached stock checkpoint.
- [x] Generate immutable shorthand panel with exact geometry/source/gold checks.
- [x] Integrate scorer/tokenizer admission, prepare Miles projection, and verify
      real tokenizer savings plus scoped tests before launch.
- [x] Run bounded CPU preflight and one H200 job in `tetracorp`.
- [x] Download/rescore all answers, compare paired improved/regressed cases and
      pure-readout accuracy, record cost and app shutdown.

This measures the combined sparsity + h-spelling change, not h spelling alone.

## Review

Completed `miles-cartesian-h-stock-20260921-r01` in `tetracorp`: 153/200 strict
(76.5%) versus 143/200 (71.5%). Paired outcomes: 19 improved, 9 regressed,
134 both correct, 38 both wrong. Incidence rose 3/16→10/16; neighbors stayed 7/32.
All 200 raw responses were downloaded and rescored, and model/tokenizer hashes
and runtime revisions match the baseline. Mean prompt tokens 2,287.145→1,315.51;
dynamic mean 2,514.59→920.90. No truncations. CPU preflight 121.73s; GPU function
523.71s; recorded compute-window estimate ~$0.89 excluding startup/build/storage.

Pure readouts are 69/72 strictly, but two misses are only `green`/`GREEN` and
`blue`/`BLUE`; a labeled case-insensitive-color diagnostic gives 71/72 readouts
and 155/200 overall. Official scores remain unchanged. The remaining real readout
miss omits the second BLACK building. App `ap-motz8vXjClLdX0T17KmCa7` stopped
at 19:14:46 UTC with zero tasks. Profile remains `tetracorp`.

Code verification: 35 focused tests passed; scoped Ruff/strict mypy clean on all
22 source/test files. Required full quality gate ran and remains red on 194
structural violations and existing Ruff/mypy errors elsewhere. Report:
`reports/sft/2026-09-21-sparse-h-cartesian-eval.md`.
# Scaled coordinate accuracy comparison (2026-09-21)

User requested actual evaluations of both measured integer-scaling variants.
Same stock Qwen3.8-27B revision, 200 cases per variant, sparse state, no thinking
or fine-tuning. Run both panels together on one bounded H200 in `tetracorp`.

- [ ] Finish and verify immutable scaled-h and bare-integer panels against the
      completed sparse-h parent, retaining exact geometry, cases, and oracle.
- [ ] Integrate both schemas/variants and generalize the Modal total-row check
      to the admitted panel manifest; prepare and launch 400 total generations.
- [ ] Download/rescore all outputs, compare paired accuracy and token budgets
      with the 153/200 strict sparse-h baseline, and verify app shutdown.

Scaled-h uses side4, h=sqrt(3), integer y. Bare-integer coordinates use the same
physical scale with physical position (sqrt(3)*x,y). No new questions are added.
