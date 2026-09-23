# Stricter linter: implementation and cleanup checklist

Updated: 2026-09-21. This is the active checklist for the strict quality work.
Historical run details remain in [todo.md](todo.md).

## Required rules

- [x] Enforce at most **300 physical lines per source file**, including comments
  and blank lines.
- [x] Enforce at most **15 direct authored files per source folder**.
- [x] Require explicit Python parameter and return types with Ruff and strict
  mypy; reject explicit `Any` and unparameterized containers.
- [x] Require module-level imports and deterministic lint diagnostics.
- [x] Guard dependency-list and lockfile edits through OpenCode; use `uv add`,
  `uv remove`, and uv-managed lock updates.
- [x] Give OpenCode structural feedback after edits and block its ordinary
  commits unless the fully staged worktree passes the full quality gate.
- [x] Provide the `/quality` command and document the required OpenCode restart.
- [x] Apply the rules repo-wide, with **no legacy baseline** or changed-file
  exemption. Generated/data/vendor exclusions are explicit in
  `scripts/quality/inventory.py`.

## Status

The gate is still **failing**. Installation of the checks is complete; repository
cleanup is ongoing.

| Checkpoint | Oversized files | Overfull folders | Structural failures |
| --- | ---: | ---: | ---: |
| Before cleanup | 213 | 10 | 223 |
| First completed pass | 190 | 4 | 194 |
| Latest reported second-pass run | — | — | 188 |
| Engine package typed and split (2026-09-21 evening) | — | 0 | 179 |
| SFT folders reorganized; partial SFT file splits | — | 0 | 152 |
| Fresh full run 2026-09-21 | 170 | 0 | 170 |
| Interim 2026-09-22 (four owners still finishing) | 6 | 0 | 6 |

The second pass still needs a final merged verification. Do not treat the latest
intermediate count or scoped passing checks as a repo-wide pass. The first-pass
full run also reported 5,750 Ruff errors and strict mypy failures; fresh full
run 2026-09-21 reports 5,367 Ruff diagnostics and 9,301 strict mypy diagnostics
(structure is now 170 max-lines, 0 max-files). Interim 2026-09-22 measurement
with data pipeline, scripts, SFT, and playground owners still finishing:
6 structural (two data-pipeline package inits at 313/303, `App.tsx`, one
browser test, plus two in flight), 85 Ruff (all the evals `ascii_variations`
ANN401 boundary), 7,877 strict mypy of which about 6,150 are under `tests/`
(`index`, `union-attr`, `arg-type` on JSON-shaped payloads; the pending tests
policy decision), about 1,340 under `sft/` (owner mid-pass), about 285 under
`evals/` (the deferred JSON boundary).

## Immediate next steps

- [x] Finish verification of the action-tool helper-override repair.
  `cle/harness/action_tools/parser.py` performs late-bound package lookups
  (`action_tools._TOOL_TYPES`, `_require_arguments`, `_spatial_values`,
  `_color`, `_resource`, `_bundle`, `_card_counts`, etc.); preserves overrides
  such as `action_tools._spatial_values`. Verified 2026-09-21: 324 passed in
  `tests/harness_interfaces/test_helper_patch_sites.py` + `tests/sandbox` +
  `tests/sandbox_contracts`; scoped Ruff and strict mypy pass on `parser.py`.
- [x] Resolve the discard annotation mismatch in
  `cle/game_engine/state/robber.py` around `apply_discard`/`validate_discard`:
  `validate_discard` declares `tuple[FastResource, ...]` (lines 30-32), matching
  both `apply_discard` branches (`hand[:num]` from `list[FastResource]` and
  `rng.sample` tuples). Runtime behavior preserved, no suppressions; scoped
  Ruff and strict mypy pass on `robber.py` + `parser.py`.
- [x] Recheck the integrated runtime changes: sandbox, action tools, board
  presentation, formatter, and typed state helpers. Verified 2026-09-21:
  302 passed in `tests/harness_interfaces tests/formatter_contracts
  tests/engine tests/engine_contracts`; `git diff --check` passes.
- [x] Recheck the known uncapped-reasoning regression:
  `tests/evals/reasoning/test_initial_settlement_reasoning.py::test_capture_trace_omits_max_tokens_and_separates_native_reasoning`.
  Failed on the 2026-09-21 recheck because the per-decision reasoning budget
  set `max_tokens=8192` on setup requests. Resolved by the feature owner on
  2026-09-22: `reasoning_budget_for_context` in `cle/harness/reasoning.py`
  now leaves completion uncapped everywhere and paces the model through
  effort tiers. The probe passes again; verified directly.
- [x] Run the complete quality gate and record fresh structural, lint, and type
  results here. Report unresolved failures explicitly. Fresh run 2026-09-21:
  170 structural (all max-lines, 0 max-files), 5,367 Ruff, 9,301 mypy;
  `git diff --check` passes. Gate still failing.

## Third pass (2026-09-21 evening): parallel area agents

Orchestration: one agent per area with disjoint file ownership, file-by-file
completion, and a progress log per agent in the session scratchpad. A first
wave of ten agents was killed by the account session limit mid-edit; the
interrupted SFT split left two monkeypatch-seam regressions (repaired) and
several consistent same-name packages (kept). Wave one relaunched with five
Opus agents: cle/replay, cle/harness+cle/prompts, cle/traces+agents+sandbox,
oversized tests, sft files. Wave two pending: evals, scripts, data_pipeline,
playground, frontends.

- [x] `cle/game_engine`: Ruff 0, strict mypy 0 across every file. Oversized
  `state`, `models/board`, `models/actions`, `features`, `game`, and
  `public_board` became same-name packages with explicit `__all__`, pinned
  `__module__` for pickled classes, and a new pickle-identity test. Engine
  289 passed; consumer suites unchanged. Behavior parity for actions/features
  proven over 2,400 seeded steps against HEAD.
- [x] Stub packages added through `uv add --optional dev`: types-networkx,
  types-jsonschema, types-PyYAML, types-requests, pandas-stubs,
  types-Flask-Cors, types-Flask-SocketIO, google-api-python-client-stubs.
  `NodeGraph` alias in `cle/game_engine/models/board/graph.py` spells all
  three `nx.Graph` parameters because the stub defaults are Any-based.
- [x] `pyproject.toml` `[tool.mypy]` gained `plugins = ["pydantic.mypy"]` with
  `[tool.pydantic-mypy] init_typed = true`: every pydantic model otherwise
  trips explicit-any through its synthesized `__init__`. Verified on a bare
  model; tightens constructor checking rather than relaxing anything.
- [x] `sft/`, `sft/launchers/`, `sft/scripts/` under the folder cap via
  ownership subfolders (`board/`, `analysis/`, `launchers/{board_fluency,
  full_board,qwen_series,spatial}/`, `scripts/{builders,eval,train,report,
  render}/`); `sft/diagnostics` avoided because the layout verifier forbids it.
- [x] `sft/scripts/builders/build_symbolic_board_dataset.py` `engine_board`
  provenance path repointed from the removed `models/board.py` to the package.

- [x] `evals/` Python: 14 oversized modules became same-name packages with exact
  namespace parity; tests at baseline. Correction to earlier notes: the frozen
  scorer is `evals/catan_board_bench/ascii_variations/scoring.py`, whose
  `strict_scorer_digest` hashes five function sources; `catan_board_bench/scoring.py`
  carries no fingerprint. The frozen bytes were not touched; digest still
  b80971f0... Remaining evals debt: 84 ANN401 and 212 explicit-any on the
  `JsonDict` alias in `ascii_variations`, the heterogeneous board-fact
  boundary. Retyping it properly means TypedDicts for the whole fact schema
  and touches renderers with byte witnesses: a separate project, not cleanup.
- [x] `cle/replay`, `cle/harness`, `cle/prompts`, `cle/traces`, `cle/agents`,
  `cle/sandbox`: Ruff 0, strict mypy 0, structure 0. Replay runtime proven
  byte-identical over 53 real Colonist archives (26.8 MB digest, 36,842 stdout
  lines) with `PYTHONHASHSEED=0`; note playable-action ordering is otherwise
  nondeterministic per process (pre-existing). Harness context formatting and
  prompt suite rendering proven byte-identical over 400 steps / 150 contexts.

- [x] `playground/` Python (game viewer): structure 13 to 0, Ruff 303 to 0,
  mypy 533 to 14 (all in the dead `playground/game_orchestrator.py`). Replay
  loaders verified byte-identical on curated game 242781000 (111 traces across
  121 window cursors; 659 transcript segments across 201 windows). Suites run
  identically with all proxies pointed at a dead port. Recovered three
  uncommitted `follow_latest_enabled` lines a predecessor had overwritten.
- [x] `tests/`: 52 oversized modules to 1 (the SFT-owned symbolic dataset test),
  Ruff 3,949 to 47 (same file), all 3,464 baseline node ids collect. Mypy
  under tests rose 5,084 to 5,839 once fixtures were typed: JSON-shaped
  payload access under `disallow_any_explicit`. Decision pending (below).

- [x] `scripts/` (except the gate): Ruff 62 to 0, structure 17 to 0, mypy 237
  to 7 (5 need `fastapi`, 2 pre-existing Modal decorator errors on HEAD). All
  16 entry points answer `--help` byte-identically via `prog=`. Load-bearing
  package structure documented in `scripts/README.md` (retargetable dataset
  names, the self-hashing tile-prompt scorer kept in its `__init__`, `httpx`
  re-exports for patched clients). Two real bugs fixed: `excluded_models` is a
  dict keyed by model, not a list; the text-format optimizer retargeted the
  evaluator by assigning package attributes, which would have shadowed the
  forward after the split. One pre-existing quirk preserved and flagged: the
  board-parts probe's `SETTLEMENTS\\s+` regex never matches, so its
  settlement/city counts are always None.

- [x] Decision 2026-09-22 (user approved): `pyproject.toml` gained a
  tests-only `[[tool.mypy.overrides]]` for `tests.*` with
  `disallow_any_explicit = false`. Source stays fully strict; tests may type
  JSON-shaped fixtures and HTTP payloads as `dict[str, Any]` at the fixture
  boundary. Rationale: a narrowing helper across 220 sites cut only 24 of the
  ~6,150 test-side errors; typing every payload would be a schema project.

Known non-lint failures, all pre-existing or owned elsewhere (do not "fix" in
cleanup work):
- `tests/replay/test_replay_trading.py` (2): trade payload shape from the
  uncommitted targeted-offer work.
- `tests/viewer/routes/test_live_sandbox_routes.py` (2): validation hint text
  from the uncommitted action-tools parser work; fails identically at HEAD.
- `tests/test_verify_spatial_extension.py` (4) and
  `tests/test_modal_spatial_extension.py` (6): local receipts under
  `artifacts/runs/sft/spatial-continuation-20260909-r01/` predate the
  `representation` (2a27dbc) and `operation` (6703e43) summary dimensions, so
  the verifier's key-set comparison fails at HEAD too; skipped in clean
  checkouts. Receipts are historical and must not be regenerated in cleanup.
- `tests/data_pipeline/recognition/export/test_reweight_node_edge.py` (11
  setup errors): obsolete `full_coverage` keyword.

- [x] Frontends: `playground/frontend` types, App.css, PromptSuiteStudio (tsx
  and css), ReplayTranscriptPanel.css split; 61 node tests unchanged; built
  CSS byte-identical (sha dec89013...); JS bundle delta of 1,515 bytes fully
  attributed to four extracted PromptSuiteStudio subcomponents. The board
  bench UI was red on arrival from an earlier split (three type errors) and
  builds green again.

Open policy questions for the user:
- `playground/frontend/src/App.tsx` is 1,550 lines after extracting
  constants, snapshot types and API helpers; the remaining ~1,050 lines are
  one stateful component (about 20 hooks, a 424-line render tree, and
  `applyStateSnapshot` writing 15 setters). Reaching 300 lines needs a
  deliberate decomposition into roughly ten domain hooks plus panel
  components with hook order preserved, and the only end-to-end net is an
  8-test Playwright suite. Deferred rather than rewritten unverified; decide
  whether to schedule it with added tests or accept it as standing debt.
- `evals/catan_board_bench/ascii_variations` `JsonDict` boundary: 84 ANN401
  and 212 explicit-any findings; retyping it needs TypedDicts for the whole
  board-fact schema and touches renderers with byte witnesses. A project, not
  cleanup.
- Optional extras not installed here (ms-swift `swift`, `qwen_vl_utils`,
  `fastapi`) are import-not-found under strict mypy and the gate has no
  mechanism for them: install the extra in the dev environment, or add
  per-module `ignore_missing_imports` overrides for exactly those packages.
- `playground/game_orchestrator.py` cannot import (references a nonexistent
  `playground.agents` package and a catanatron path); dead code awaiting a
  delete decision.

## Gate green (2026-09-23)

`uv run --no-sync python -m scripts.quality` passes: structure 0, Ruff 0, mypy 0.
Final steps: `App.tsx` split 1,550 -> 66 lines into `src/app/use*.ts` hooks and
`src/components/workspace/` (built CSS byte-identical; typecheck, build, 61 node
tests pass; 3 pre-existing eslint `no-explicit-any` in `src/types/game.ts` remain,
not part of the gate). Local `sft` extra now pins transformers==5.16.1 / trl==1.12.0 to match the
Modal image (`_common.py`), so the `_snapshots` exemption was removed; the six
new transformers-5 typing errors were fixed in code (`LanguageHiddenCapture`
moved to `_visual.py` to keep `_model_tokens.py` under 300). Tests remain
Ruff-only.

## Prompt feature cleanup (2026-09-23)

- One compile step: `cle/harness/prompt_store/compile.py`
  (`compile_runtime_suites`, `compile_active_suites`) replaced six copies; one
  digest (`source_sha256`); env pin names are constants in `resolution.py`.
- Legacy pair-mode *editing* retired: the Studio API only accepts the shared
  document; local `decision.yaml`/`communication.yaml` overrides are ignored with
  a warning. Legacy suites still run/replay via explicit pins or paths and show
  read-only (`mode: "legacy", read_only: true`) in the Studio.
- One strict YAML loader for all suites: `cle/harness/yaml_source.py`.
- `cle/players/agent.py` no longer pairs a shared decision suite with legacy
  communication_v5 silently.
- Frontend `components/prompts/`: legacy UI removed, `usePromptSuite` split
  (`promptSuiteApi.ts`, `useUnsavedGuard.ts`, `useCandidatePreview.ts`,
  `LegacySuiteView.tsx`); folder is at the 15-file cap.
- Pending user action: delete dead `cle/prompts/` (unreferenced).

## Wave 3 notes (2026-09-23)

Wave 3 (three Opus owners plus forks, no pytest by user's call) brought all
non-test source to strict mypy/Ruff zero except three known items. Fresh full
gate after wave 3: structure 1 (`App.tsx`), Ruff 38 (all in
`playground/frontend/tests/*shared_prompt_browser*`), mypy ~3,470 of which
~3,450 are under `tests/` (tests/sft 2,544).

Source leftovers (not code-fixable or awaiting the user):
- `playground/game_orchestrator.py` (14): dead, unimportable; user approved
  deletion but auto mode blocks `rm`, so the user runs it.
- `sft/launchers/board_fluency/modal_board_fluency_eval/_snapshots.py:40`:
  `AutoModelForMultimodalLM` needs transformers 5.x (Modal pin); local is 4.57.
  Legacy Modal SFT path; training moved to miles, so accepted as-is.
- Data drift: `full_graph_format_probe/metadata.json` source_lock hashes do not
  match the committed ascii_variation_probe manifest/qa (pre-existing).
- Spatial continuation compares `modal_full_board_pilot.py` against a stale
  `legacy_pilot_sha256` (pre-existing; legacy path).

Narrow config exemptions added in wave 3 (each commented in pyproject): frozen
ASCII scorer (explicit-any/no-any-return + Ruff ANN401), `sft.ms_swift_plugin`
and the TRL `_trainer` (subclassing Any bases), `accelerate` missing stubs,
Modal probe `call-arg`. `sft/miles_eval/` belongs to Codex; lint workers stay out.

Tests exempted from mypy (user decision 2026-09-23): `tests.*` and
`playground.frontend.tests.*` use `ignore_errors`; Ruff still applies. CLAUDE.md
updated. `playground/game_orchestrator.py` deleted by the user.

Progress logs: `tasks/stricter-linter-progress/progress-w3-*.md`.

## Previous resume point (session restart 2026-09-22 evening)

State at handoff: structure 1 violation (`playground/frontend/src/App.tsx`,
deferred by decision), Ruff 85 (all the evals `ascii_variations`
boundary), strict mypy about 7,900 before the tests-only override landed.
Per-agent progress logs are copied to `tasks/stricter-linter-progress/`
(one line per completed file; read the matching log before touching an area).

Unfinished at handoff, by area:
- [ ] `tests/` (excluding `tests/sft`): bring mypy to zero under the new
  tests-only override by typing JSON-shaped fixtures and payload helpers as
  `dict[str, Any]` at the fixture boundary. Three residual calls into untyped
  source functions to name. Log: `progress-w1-tests.md`.
- [ ] `sft/` and `tests/sft`: mypy pass (was 1,379 and dropping; count
  optional-extra import-not-found separately), split
  `tests/sft/data/test_symbolic_board_dataset.py` (404 lines, 47 Ruff), Ruff
  remainder in three `tests/sft` files, package-layout note in `sft/README.md`.
  Preserve the late-binding seams in the spatial launchers and
  `modal_catan_vision_sft`. Log: `progress-w1-sft.md`.
- [ ] `data_pipeline/`: final verification only (Ruff/mypy/structure,
  `tests/data_pipeline tests/recognition_contracts
  tests/replay/test_replay_playwright_scraper.py` including the four
  byte-parity tests, HASHED_SOURCES file test). All seven hashed modules are
  thin files at their paths with `*_impl/`, `source_lock/` subpackages beside
  them; they must stay files. Log: `progress-w2-data-pipeline.md`.
- [x] `playground/frontend/tests/test_live_autoplay_browser.py` split into
  `playground/frontend/tests/live_autoplay/` (9 files, 23 node ids identical,
  23 passed before and after; build/browser fixtures widened to session scope
  to keep a single npm build). Structure is now 1 violation: `App.tsx`.
- [ ] `playground/frontend/tests/test_shared_prompt_browser.py` and
  `shared_prompt_browser_fixture.py`: 38 Ruff and 30 mypy findings, never
  assigned to an owner (Python browser tests under the frontend folder).
- [ ] `evals/`: the remaining 85 Ruff / ~310 mypy findings are not a
  mechanical pass. Replacing the `Any` boundary in `evals/replay_action_diff`
  raised mypy 59 to 90 by exposing about 31 unchecked dereferences of an
  optional engine and optional replay archive (real latent crashes). Deciding
  what happens when no replay is loaded is an owner-level design call; the
  probe edit was reverted by hand. One kept edit: `evals/catan_board_bench`
  metadata module typed against the installed eval library under a
  type-checking guard (Ruff 15 to 0 there); re-verify tests/evals.
- [ ] Eight errors no code edit can clear: 5 need `fastapi` (`uv add
  --optional dev fastapi` if the user agrees), 1 needs a `tokenizers` mypy
  override (no stub package exists), 2 are real Modal API drift in the VLM
  benchmark probe (`Secret.from_name(required_hint=...)` and a constructor
  argument the installed Modal rejects) that only someone who can run Modal
  should fix.
- [x] Frozen ASCII scorer (`evals/catan_board_bench/ascii_variations/scoring.py`):
  hashed helpers keep `Any`; exempted by a one-module mypy override
  (`explicit-any`, `no-any-return`) and a Ruff per-file ANN401 ignore, so the
  fingerprint is unchanged (user-approved 2026-09-22). A typed v2 scorer is the
  path if the exemption ever needs to go.
- [ ] Final: `uv run --no-sync python -m scripts.quality` and the full test
  suite in one sitting; record numbers here.

Orchestration rules that held up: one owner per directory (never per file);
file-by-file completion with a progress line per file; check `ListAgents`
before spawning any "resume" (limit-hit "failed" notices are transient and
agents resume on their own); a subagent's `model: "opus"` alias resolves to
Opus 5 (1M), not 5.5; the Agent tool cannot pin 5.5, so set the session or
subagent default model first.

Verification for every split: namespace parity against the pre-split module
(`dir()` plus resolving every `from pkg import X` across the repo), monkeypatch
seams late-bound through the module tests patch, node-id sets identical for
test splits, and a run with all proxies pointed at 127.0.0.1:9 to prove no
network access.

## Completed cleanup

### Organization and first pass

- [x] Reorganize 44 scripts by ownership; `scripts/` has 12 direct files.
- [x] Reorganize 124 tests; `tests/` has 13 direct authored files.
- [x] Group frontend components into feature folders and bring both frontend
  folder layouts under the 15-file limit.
- [x] Split map/trade/player implementations and introduce concrete local types.
- [x] Convert two benchmark format modules into same-name packages; retain exact
  scorer fingerprints and rendered/generated output parity.
- [x] Convert eight recognition generators into same-name packages; retain
  dataset bytes, source contracts, pickle identities, and sampler override seams.
- [x] Split nine oversized frontend components/stylesheets across two passes;
  verify unchanged complete production CSS bytes.
- [x] Update live script commands and test paths in the relevant READMEs.

### Runtime second pass

- [x] Type `SandboxSnapshot.player_states` using the existing `PlayerSnapshot`
  contract; replace bare automatic-action payload dictionaries with TypedDicts.
- [x] Split `cle/sandbox/catan.py` into a nine-file same-name package; preserve
  atomic commits, cancellation cleanup, callback dispatch, and snapshot identity.
- [x] Split action tools and board presentation into same-name packages, bringing
  the harness root from 17 to 15 direct files.
- [x] Split the observation formatter into ten files; preserve both legacy and
  engine observations, exact strings, and historical class identities.
- [x] Split and type the engine state helpers; remove the formatter's 18 upstream
  untyped-call errors without adding ignores.
- [x] Restore late-bound state-helper overrides; verify a real one-card draw
  calls the original-module wrapper exactly once.
- [x] Restore sandbox context-builder and other original helper overrides;
  verify their invocation and modified-context propagation.

## Remaining repository backlog

- [x] Organize `sft/`, `sft/launchers/`, and `sft/scripts/` under the folder cap.
  Coordinate with active SFT work and preserve historical source-hash contracts.
- [ ] Split the remaining oversized runtime modules, including engine state,
  replay step execution, trace storage, and the legacy LLM player.
- [ ] Split remaining oversized scripts, tests, and frontend modules by coherent
  responsibility; preserve test collection and CSS/rendering order.
- [ ] Replace remaining broad JSON/`Any` contracts with concrete, truthful types.
- [ ] Finish annotations and type correctness in engine state and deck helpers;
  their source files still have existing diagnostics.
- [ ] Resolve frozen scorer annotation debt through deliberate versioning and
  compatibility work. Never rewrite historical manifests to hide source drift.
- [ ] Repair remaining Ruff failures, including missing annotations and imports,
  without changing behavior or adding blanket ignores.
- [ ] Make the full repository quality command exit zero.

## Verification evidence and procedure

Use the selected interpreter explicitly; the local `pytest` console script has
a stale shebang pointing at another checkout.

```bash
# Full gate: required before declaring code work complete or committing.
uv run --no-sync python -m scripts.quality

# Faster whole-tree structural report; output limits do not limit evaluation.
uv run --no-sync python -m scripts.quality --structure-only --limit 30

# Runtime regressions; use relevant selections for the area being changed.
uv run --no-sync python -m pytest -q tests/sandbox tests/sandbox_contracts
uv run --no-sync python -m pytest -q tests/harness_interfaces tests/formatter_contracts
uv run --no-sync python -m pytest -q tests/engine tests/engine_contracts

git diff --check
```

Recorded evidence:
- Quality integration: 38 Python tests and 26 Node plugin tests passed.
- First-pass combined regression: 862 passed, 1 skipped, 1 failed; four expensive
  full-dataset export parity tests were verified separately, not rerun together.
- Correct-interpreter first-pass collection: 3,429 tests, no collection errors.
- Catan extraction: 235 existing tests before/after; new compatibility tests pass.
- Harness interfaces: 535 existing tests before/after, 62 provider/trade tests,
  and three new compatibility tests passed before the helper-override follow-up.
- Formatter: 24,700 strings / 13,656,754 bytes matched the original behavior.
- State helpers: 290 matched baseline/post-change tests; the later override
  repair passed its 21-test focused selection.
- Sandbox typed-contract selection: 100 tests passed; scoped Ruff/mypy passed.
- Sandbox override repair: 37 focused tests and scoped Ruff/mypy passed.

Before checking off a remaining item, preserve current user changes, verify the
affected behavior, and record the actual result. Existing failures remain visible;
repair edits stay possible while the commit gate is red.
