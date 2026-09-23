# w2-playground progress

- baseline: ruff 303, mypy 533, structure 13, tests 317 passed / 2 pre-existing failures
- current_game fix applied to playground/game_viewer/state.py (init + reset), verified reads None
- done: async_runtime.py, routes/inject.py, app.py, state.py (clean ruff+mypy)
- done: routes/websocket.py (clean except 4 downstream untyped-dep errors)

## resumed agent (2026-09-22)
- STEP 1 DONE: game_logging package repaired. Chunks _p1/_p2/_p3 proved to be exact
  sequential slices of HEAD (1-140, 142-361, 363-608); deleted. Built sink.py (31),
  players.py (103), analysis.py (219), __init__.py (65) from HEAD slices via script.
  All 21 HEAD top-level names re-exported; AST diff shows 0 missing.
  Body deltas are annotation-only (4 mine) + predecessor's 5 (see report).
  Fixed 4 mypy errors in predecessor's normalization.py (matched_index rename, payload cast).
  Package clean: ruff + mypy strict. tests/viewer collects 219, app imports.
- current_game fix confirmed already applied by predecessor (state.py:24 init, :81 reset),
  annotation GameEngine | None matches ReplayRuntimeState protocol. No hasattr reliance found.
- structure violations now 12 (game_logging resolved).
- openrouter_client.py (467) -> package: config.py 105, completions.py 164, tool_loop.py 163,
  benchmark.py 126, __init__.py 41. httpx re-exported in __init__ (tests patch
  openrouter_client.httpx). ruff+mypy clean. tests/harness/providers/test_openrouter_tool_client.py passes.
- board_renderer.py (443) -> package: palette.py 55, geometry.py 78, shapes.py 103,
  renderer.py 278, __init__.py 51. _draw_settlement/_draw_city/_draw_legend became module
  functions in shapes.py (no self use except canvas_size/font, now explicit kwargs).
  ruff+mypy clean; smoke test renders a real 24KB PNG. No importers anywhere in repo.
- structure violations now 10.
- commentary/semantic_display.py (366) -> package: patterns.py 160, utterances.py 29,
  segmentation.py 186, traces.py 77, __init__.py 25. Private module constants became public
  names inside the package (re-exported). ruff+mypy clean; smoke test produces traces.
- commentary/references.py (356) -> package: mentions.py 132, corners.py 134, grounding.py 123,
  __init__.py 28. GOTCHA: _CORNER_MAP_PATH used Path(__file__).parents[1]; the extra package
  level required parents[2]. Verified it resolves to game_viewer/corner_to_node_map.json (54 entries).
  ruff+mypy clean; tests/viewer/commentary 35 passed.
- NEXT: contextualizer.py needs replay/transcript.py annotated first (strict
  disallow_untyped_calls on select_guarded_transcript_evidence / guarded_transcript_bounds).
- replay/transcript.py (449) -> package: pairs.py 48, segments.py 112, bounds.py 72,
  pairing.py 115, window.py 171, __init__.py 30. GOTCHA: PROJECT_ROOT parents[3] -> parents[4].
  Verified it resolves to the repo root and the curated replay path exists. ruff+mypy clean.
- commentary/contextualizer.py (507) -> package: models.py 180, evidence.py 169,
  session.py 258, __init__.py 26. CausalCommentarySession split by a base class
  (CommentarySessionBase = the read-only half) so no call sites changed.
  archive()/engine()/engine_state() are cast-only accessors; runtime attribute chain is unchanged.
  ruff+mypy clean; tests/viewer/commentary 35 passed.
- replay/narrator_reasoning.py (656) -> package: schema.py 62, readers.py 74, validation.py 205,
  artifact.py 211, runs.py 74, window.py 130, __init__.py 31. PROJECT_ROOT parents[3]->[4].
- replay/model_traces.py (1001) -> package: schema.py 75, readers.py 73, indexing.py 153,
  ledgers.py 139, overrides.py 200, repairs.py 100, traces.py 210, artifact.py 174,
  runs.py 97, window.py 91, __init__.py 31. The 584-line load_model_trace_artifact was split
  into phases: read_side_ledgers -> validate_plan_identity -> build_setup_stages ->
  validate_setup_overrides -> validate_rationale_repairs -> build_traces, with SideLedgers /
  OverrideIndex / TraceIndex dataclasses carrying the locals between phases.
- DIFFERENTIAL VERIFICATION (old module vs new package on real curated data, game 242781000):
  model traces payload byte-identical (111 traces, all ready), 121 window cursors identical;
  transcript payload identical (659 segments, 737 timings), 201 windows + 200 guarded-evidence
  selections identical; narrator reasoning identical. End-to-end /api/load-replay: 200,
  narrator reasoning complete with 189 paragraphs, model traces complete with 111 ready.
- structure violations now 4 (all in routes/).

## resumed agent, part 2
- FIXED the broken import: the PromptSuiteEditError slice in prompt_studio/errors.py had lost its
  `def __init__` line, leaving a bare `super().__init__(message)` in the class body.
- routes/prompt_suite.py (666 at HEAD) -> prompt_suite.py 225 + prompt_studio/ package
  (errors 59, access 68, editor 80, previews 208, edits 139, __init__ 47).
  PATCH SEAMS: tests patch `_saving_locked`, `_decision_preview`, `_communication_preview`,
  `save_shared_prompt_override` and `resolve_prompt_suites` on the prompt_suite module, so
  `_studio_payload`, `_active_suites`, `_save_prompt_suite_transaction` and
  `_reset_prompt_suite_transaction` all stay in prompt_suite.py and resolve those globals there.
- RESTORED a lost uncommitted change: overwriting prompt_suite.py dropped the
  `follow_latest_enabled` import and the `if follow_latest_enabled(): resolve_prompt_suites(
  follow_latest=True)` branch in `_active_suites`. Both are back in prompt_suite.py.
- EXPORT PARITY: all 11 packages now expose every name their HEAD module defined (0 gaps).
  Re-added 59 private names, incl. `_CURATED_TRACE_RUNS` (the model_traces test uses
  monkeypatch.setitem, so the plain re-export shares the same dict object).
  semantic_display also keeps its historical `_UPPER` aliases.
- BLOCKED, needs the cle owner: ServerState.current_sandbox cannot be narrowed from `object`.
  `ReplayRuntimeState.current_sandbox` in cle/replay/contracts.py:85 is a mutable protocol
  attribute, so it is invariant; typing it `CatanSandbox | ReplaySandbox | None` makes
  ServerState stop satisfying ReplayRuntimeState and breaks every replay_step_logic /
  bump_replay_revision / ReplaySandbox call site. Fix is a one-line change in cle: make it a
  read-only property on the protocol (covariant), then I can narrow. Until then the tests owner
  should narrow at the use site. current_game is already `GameEngine | None` and conforms.
- tests/viewer + tests/replay/model_traces + tests/replay/transcript: 240 passed, 2 failed
  (the known live_sandbox pair). Identical with HTTPS_PROXY/HTTP_PROXY/ALL_PROXY=127.0.0.1:9.
- routes/live_game.py (943) -> package: blueprint 96, failures 163, start 171, advance 292,
  step 66, traces 282, __init__ 59. GOTCHA: my ast slicer matched only ast.FunctionDef, so the
  `async def _step_with_committed_result` was silently dropped; restored and re-exported.
  All 26 HEAD names re-exported. 7 routes register; tests/viewer 217 passed / 2 known failures.
- routes/live_game.py (943) -> package (blueprint 96, failures 163, start 171, advance 298,
  step 66, traces 282, __init__ 59); routes/bench.py (1737) -> bench.py 267 + bench_api/
  (blueprint 7, paths 153, common 133, qa 110, board_bench 177, reasoning 245, text_format 250,
  sft_rows 197, readiness 133, sft 249, spatial 181, __init__ 21).
  bench PATCH SEAMS: tests patch `_load_latest_eval_summaries`,
  `INITIAL_SETTLEMENT_REASONING_RUNS` and `DEFAULT_INITIAL_SETTLEMENT_REASONING_RUN` on the
  bench module, so those and the routes that read them stay in bench.py; the two reasoning
  handlers resolve the run there and delegate the payload to bench_api/reasoning.py.
- FINAL: structure 13 -> 0, ruff 303 -> 0, mypy 533 -> 14 (all 14 in the dead
  playground/game_orchestrator.py, which cannot import: `from ..agents.llm_agent_impl import
  StrategicLLMAgent` reaches beyond the top-level package and playground/agents/ does not exist).
- Export parity 0 gaps across all 13 split packages/modules.
- UNBLOCKED AND DONE: `ReplayRuntimeState.current_sandbox` is now a read-only property returning
  `GameEngineHolder | None`, so the covariant narrowing is accepted.
  `ServerState.current_sandbox` is `CatanSandbox | ReplaySandbox | None` in both `__init__`
  and `reset()` (playground/game_viewer/state.py). Verified ServerState still satisfies
  ReplayRuntimeState under mypy strict.
  Removed the workarounds it made unnecessary: routes/health.py and routes/mapping.py now read
  `sandbox.game_engine` directly (the GameEngineHolder casts and imports are gone), and
  live_game/blueprint.py types `_applied_prompt_config`/`_sync_live_inference` as CatanSandbox
  instead of `object`. The remaining casts in prompt_studio/previews.py, live_game/step.py,
  routes/replay/llm_response.py and contextualizer/models.py are real union-member narrowings,
  not workarounds, so they stay.
  Counts after: ruff 0, mypy 14 (all in the dead game_orchestrator.py), structure 0.
  tests/viewer + tests/replay/model_traces: 230 passed, 2 failed (the known live-sandbox pair).
  The tests owner can now narrow against the concrete union.
