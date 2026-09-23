# w1-tests split progress

Global: ran mechanical annotation passes over tests/ (ANN + fixture/parametrize type inference).
Ruff on tests: 3949 -> see per-file notes. Backup of pre-change tests/ at scratchpad/w1t/backup_tests.

- tests/engine/test_engine_boundaries.py (955) -> tests/engine/engine_boundaries/{__init__,conftest,support,test_engine_boundaries_forcing,test_engine_boundaries_discard,test_engine_boundaries_events,test_engine_boundaries_messaging,test_engine_boundaries_snapshots,test_engine_boundaries_road_restore}.py ; 98 node ids identical; 98 passed; ruff clean
- tests/engine/test_trade_window.py (839) -> tests/engine/trade_window/{__init__,support,test_trade_window_offers,test_trade_window_selection,test_trade_window_wildcards,test_trade_window_preflight,test_trade_window_identity}.py ; 68 node ids identical; 68 passed; ruff clean
- tests/engine/test_game_supply_limits.py (560) -> tests/engine/supply_limits/{__init__,support,test_game_supply_limits_pools,test_game_supply_limits_pieces,test_game_supply_limits_production,test_game_supply_limits_discard}.py ; 48 node ids identical; 48 passed; ruff clean
- tests/audits/test_engine_rule_audit.py (841) -> tests/audits/engine_rule_audit/{__init__,conftest,support,players,test_engine_rule_audit_roads,test_engine_rule_audit_terminal,test_engine_rule_audit_economy,test_engine_rule_audit_full_game}.py ; 52 node ids identical; 20 passed 32 skipped; ruff clean
- tests/audits/test_harness_boundary_audit.py (545) -> tests/audits/harness_boundary_audit/{__init__,conftest,support,test_harness_boundary_audit_parsing,test_harness_boundary_audit_trading,test_harness_boundary_audit_protocol}.py ; 17 node ids identical; 17 passed; ruff clean
- tests/audits/test_replay_checkpoint_audit.py (453) -> tests/audits/replay_checkpoint_audit/{__init__,conftest,support,test_replay_checkpoint_audit_rollback,test_replay_checkpoint_audit_reuse,test_replay_checkpoint_audit_continuation}.py ; ruff clean
- tests/evals/formats/test_catan_full_graph_formats.py (356) -> tests/evals/formats/full_graph_formats/{__init__,support,test_catan_full_graph_formats_parsing,test_catan_full_graph_formats_jobs}.py ; 7 node ids identical; 7 passed; ruff clean
- tests/evals/formats/test_catan_tile_prompt_ablation.py (324) -> tests/evals/formats/tile_prompt_ablation/{__init__,support,test_catan_tile_prompt_ablation_factors,test_catan_tile_prompt_ablation_jobs}.py ; 7 node ids identical; 7 passed; ruff clean
- tests/evals/reasoning/test_inspect_archives.py (301) -> tests/evals/reasoning/inspect_archives/{__init__,support,test_inspect_archives_round_trip,test_inspect_archives_importer}.py ; ruff clean
- tests/data_pipeline/recognition/export/test_build_catan_board_recognition_eval_suite.py (515) -> tests/data_pipeline/recognition/export/build_eval_suite/{__init__,conftest,support,..._build,..._cli,..._validation}.py ; 20 node ids identical; 20 passed; ruff clean; ROOT parents[4]->parents[5] adjusted for the extra directory level
- tests/data_pipeline/recognition/tasks/test_board_recognition_adjacent_pair.py (324) -> tests/data_pipeline/recognition/tasks/adjacent_pair/{__init__,support,..._sampling,..._rows,..._rendering}.py ; ruff clean; verify BLOCKED by another agent's cle.replay.runtime.step_executor refactor (ModuleNotFoundError), re-check pending
- tests/sft/data/test_spatial_continuation_dataset.py (389) -> tests/sft/data/spatial_continuation_dataset/{__init__,conftest,support,..._fixtures,..._corpus}.py ; ruff clean; verify BLOCKED by cle.replay.runtime.step_executor refactor
- tests/sft/data/test_symbolic_board_tasks.py (319) -> tests/sft/data/symbolic_board_tasks/{__init__,support,..._geometry,..._routes,..._projection}.py ; ruff clean; verify BLOCKED
- tests/sft/eval/test_eval_candidate_scoring.py (320) -> tests/sft/eval/candidate_scoring/{__init__,support,..._candidates,..._confusion,..._readouts}.py ; ruff clean; verify BLOCKED
- tests/sft/eval/test_spatial_tasks.py (540) -> tests/sft/eval/spatial_tasks/{__init__,conftest,support,..._graph,..._scoring,..._production,..._dispatch}.py ; ruff clean; verify BLOCKED
BLOCKER: cle.replay.runtime.step_executor.action_matcher missing (another agent's in-flight refactor) blocks pytest collection for most of tests/. Re-verify all "BLOCKED" entries once it lands.

## Root-level relocations (new paths for other agents)
- tests/test_board_atlas.py -> tests/sft/data/board_atlas/ (5 test modules + conftest + support)
- tests/test_eval_visual_precision.py -> tests/sft/eval/visual_precision/{test_eval_visual_precision_preservation,..._generation,..._cli}.py + support.py
- tests/test_trl_catan_text.py -> tests/sft/training/trl_catan_text/{test_trl_catan_text_contract,..._peft,..._trainer}.py + support.py
- tests/test_qwen_vision_sft.py -> tests/sft/training/qwen_vision_sft/{..._commands,..._fingerprints,..._scope,..._launch}.py + support.py
- tests/test_modal_spatial_continuation.py -> tests/sft/launchers/modal_spatial_continuation/{..._config,..._identity,..._coordination,..._audits}.py + conftest/fixtures/support
- tests/test_modal_spatial_extension.py -> tests/sft/launchers/modal_spatial_extension/{..._plan,..._preflight,..._training,..._coordination}.py + conftest/support
  NOTE: the old cross-module `import test_modal_spatial_continuation as previous` is now
  `from modal_spatial_continuation.fixtures import no_remote as offline_boundaries, plan as base_plan`
  and `from modal_spatial_continuation.support import FakeCall, fake_volumes, write_json, write_rows`
  (tests/sft/launchers is the pytest basedir for both packages).
- tests/test_verify_spatial_extension.py -> tests/sft/launchers/verify_spatial_extension/{..._status,..._safety,..._rescoring,..._receipts}.py + conftest/support
Root tests/ now holds 6 direct files: conftest.py, test_board_packs.py, test_eval_qwen_text.py, test_full_board_new_layouts.py, test_modal_catan_budget.py, README.md
- tests/sft/training/test_trl_catan_vision.py (886) -> tests/sft/training/trl_catan_vision/{..._contracts,..._bundles,..._embeddings,..._objectives,..._lora}.py + support.py ; ruff clean (function-level imports hoisted to module scope for PLC0415)
- tests/harness/contracts/test_action_tools.py (1200) -> tests/harness/contracts/action_tools/{..._resolution,..._spatial,..._resources,..._trades,..._audiences,..._execution}.py + support.py ; 517 node ids identical; 517 passed; ruff clean
- tests/harness/contracts/test_harness_context.py (1254) -> tests/harness/contracts/harness_context/{..._suites,..._setup,..._prompts,..._conversation,..._action_parsing,..._parsing,..._discard}.py + conftest/support ; 158 node ids identical; 158 passed; ruff clean
- tests/harness/contracts/test_knight_tool.py (567) -> tests/harness/contracts/knight_tool/{..._bundle,..._retries,..._traces}.py + support.py ; 47 node ids identical; 47 passed; ruff clean
- tests/harness/contracts/test_response_xml.py (316) -> tests/harness/contracts/response_xml/{..._fields,..._validation,..._echo}.py ; 233 node ids identical; 233 passed; ruff clean
- tests/harness/contracts/test_shared_fresh_contract.py (396) -> tests/harness/contracts/shared_fresh_contract/{..._freshness,..._trades,..._prompts}.py + support.py ; 13 node ids identical; 13 passed; ruff clean
- tests/harness/contracts/test_shared_prompt_components.py (450) -> tests/harness/contracts/shared_prompt_components/{..._notes,..._bundle,..._compilation,..._rendering}.py + conftest/support ; 43 node ids identical; 43 passed; ruff clean
- tests/harness/providers/test_harness_openrouter.py (349) -> tests/harness/providers/harness_openrouter/{..._requests,..._surfaces}.py + support.py ; 7 node ids identical; 7 passed; ruff clean
- tests/harness/providers/test_openrouter_http_failure.py (542) -> tests/harness/providers/openrouter_http_failure/{..._reasons,..._redaction,..._retries}.py + conftest/support ; 109 node ids identical; 109 passed; ruff clean
- tests/harness/providers/test_openrouter_tls.py (478) -> tests/harness/providers/openrouter_tls/{..._recovery,..._budget,..._cleanup}.py + conftest/support ; 46 node ids identical; 46 passed; ruff clean
ALL tests/harness oversized files done.
- tests/sandbox/test_catan_sandbox.py (1701) -> tests/sandbox/catan_sandbox/{..._stepping,..._trading,..._concurrency,..._speech,..._barriers,..._capacity,..._offers,..._attempts,..._validation,..._discards}.py + support.py ; 78 node ids identical; 78 passed; ruff clean

## resumed session (w1-tests, Opus)
- tests/replay/test_replay_model_traces.py (618) -> tests/replay/model_traces/{__init__ 0,support 205,test_replay_model_traces_windows 84,..._validation 136,..._overrides 89,..._curated 150}.py ; 13 node ids identical; 13 passed; ruff clean; mypy 31 -> 2 (index on dict[str, object] / Optional, assertions unchanged)
- tests/replay/test_replay_core_boundaries.py (352) -> tests/replay/core_boundaries/{__init__ 0,conftest 63,support 21,test_replay_core_boundaries_navigation 176,..._isolation 120}.py ; 10 node ids identical; 10 passed; ruff clean; mypy 29 -> 22 lines (residual: ReplaySandbox.step declared tuple in source but indexed by str at runtime -> call-overload; NOT a test bug)
- tests/replay/test_replay_llm_response.py (543) -> tests/replay/llm_response/{__init__ 0,conftest 17,support 103,test_replay_llm_response_rolls 164,..._activity 135,..._routes 155}.py ; 11 node ids identical; 11 passed; ruff clean; mypy 26 -> 17
NOTE baseline for tests/replay is 13F/164P/1S (11 of the failures are test_replay_action_diff.py, new since the predecessor's log; 2 are the known trade-payload ones).
- tests/replay/test_replay_transcript.py (449) -> tests/replay/transcript/{__init__ 0,support 23,test_replay_transcript_parsing 95,..._windows 106,..._cursors 81,..._routes 171}.py ; 10 node ids identical; 10 passed; ruff clean; mypy 8 -> 10 (index/len on object from source Dict[str, Any]|None)
- tests/replay/test_replay_action_diff.py (799) -> tests/replay/action_diff/{__init__ 0,support 76,test_replay_action_diff_matching 136,..._consumers 171,..._semantic 98,..._receipts 157,..._historical 56,..._reporting 201}.py ; 100 node ids identical; 11F/89P == the same 11 pre-existing failures by name; ruff clean; mypy 18 -> 18
- tests/replay/test_replay_trading.py (1395) -> tests/replay/trading/{__init__ 0,support 241,test_replay_trading_closures 143,..._exchange 157,..._offers 216,..._ledger 183,..._windows 242,..._navigation 181,..._corpus 170}.py ; 24 node ids identical; 2F/21P/1S == the same 2 known trade-payload failures; ruff clean; mypy 178 -> 181 (all Optional-narrowing on source types: TradeWindow|None, GameEngine|None, ServerState.current_players)
tests/replay structure violations: 6 -> 0. tests/replay now 12 direct files/folders.
- tests/viewer/routes/test_prompt_suite_routes.py (314) -> tests/viewer/routes/prompt_suite/{__init__ 0,conftest 28,support 43,test_prompt_suite_routes_editing 172,..._previews 101}.py ; 7 node ids identical; 7 passed; ruff clean; mypy 33 -> 22
- tests/viewer/traces/test_game_logging.py (350) -> tests/viewer/traces/game_logging/{__init__ 0,support 15,test_game_logging_trades 203,..._speech 155}.py ; 11 node ids identical; 11 passed; ruff clean (converted the file's one E731 `said = lambda ...` into an annotated nested def, same body); mypy 39 -> 29
- tests/viewer/commentary/test_commentary_contextualizer.py (434) -> tests/viewer/commentary/contextualizer/{__init__ 0,conftest 24,support 50,test_commentary_contextualizer_adapter 187,..._invalidation 127,..._replay 119}.py ; 18 node ids identical; 18 passed; ruff clean; mypy 34 -> 16
- tests/viewer/routes/test_shared_prompt_routes.py (446) -> tests/viewer/routes/shared_prompt/{__init__ 0,conftest 41,test_shared_prompt_routes_schema 182,..._sources 133,..._locking 132}.py ; 31 node ids identical; 31 passed; ruff clean; mypy 44 -> 82. The rise is expected: the `studio` fixture parameter was untyped, so mypy skipped every test body; annotating it (Studio = tuple[FlaskClient, SimpleNamespace]) exposed 59 latent `TestResponse.json` is `Any | None` index errors plus PromptSuiteDocument|None union-attr. Fixing them needs new asserts, which the brief forbids.
- tests/viewer/routes/test_fresh_notes_routes.py (918) -> tests/viewer/routes/fresh_notes/{__init__ 0,conftest 58,support 152,test_fresh_notes_routes_lifecycle 214,..._migration 134,..._failures 263,..._attempts 79,..._replay 138}.py ; 32 node ids identical; 32 passed; ruff clean; mypy 115 -> 376. Same cause as shared_prompt: the `live_app` fixture parameter was untyped so mypy skipped every test body. Annotating it (LiveApp = tuple[FlaskClient, ServerState, RecordingSocket, LocalTransport]) exposed latent errors that all trace to source types: TestResponse.json is `Any | None` (79), ServerState.current_sandbox / live_trace_store are `object` / Optional (77). Fixing them needs new asserts or source changes, both out of scope here.
- tests/viewer/traces/test_live_trace_store.py (1122) -> tests/viewer/traces/live_trace_store/{__init__ 0,support 21,test_live_trace_store_persistence 249,..._communication 199,..._context 224,..._failures 241,..._images 132,..._migrations 192}.py ; 16 node ids identical; 16 passed; ruff clean; mypy 1142 -> 1142 (all from indexing the source JsonValue unions returned by cle.traces; unresolvable without new asserts)
- tests/sandbox/test_action_batches.py (555) -> tests/sandbox/action_batches/{..._setup,..._queue,..._envelope,..._continuation}.py + support.py ; 49 node ids identical; 49 passed; ruff clean
- tests/sandbox/test_communication.py (500) -> tests/sandbox/communication/{..._suites,..._parsers,..._sandbox}.py + support.py ; 57 node ids identical; 57 passed; ruff clean
- tests/sandbox/test_players.py (534) -> tests/sandbox/players/{..._context,..._communication,..._materializer,..._receipts}.py + conftest/support ; 41 node ids identical; 41 passed; ruff clean
- tests/sandbox/test_reactive_speech.py (361) -> tests/sandbox/reactive_speech/{..._branches,..._triggers}.py + support.py ; 13 node ids identical; 13 passed; ruff clean
- tests/sandbox/test_trade_preauthorization.py (419) -> tests/sandbox/trade_preauthorization/{..._barrier,..._consumption,..._parsing}.py + support.py ; 42 node ids identical; 42 passed; ruff clean
SCOPE: tests/viewer reassigned to w1-runtime (I never touched it). tests/sandbox fully done by me before the handover.
- tests/viewer/routes/test_live_sandbox_routes.py (2234) -> tests/viewer/routes/live_sandbox/{__init__ 0,conftest 129,test_live_sandbox_routes_routes 222,..._config 249,..._agent 200,..._attempts 180,..._failures 287,..._sources 157,..._unapplied 239,..._barrier 165,..._postaction 222,..._transport 205,..._reasoning 194,..._recovery 145}.py (14 files, limit 15) ; 53 node ids identical; 2F/51P == the same 2 known failures; ruff clean; mypy 406 -> 407
  FIXED (thanks w1-runtime for flagging it): my first pass imported CompletionTransport from cle.sandbox.contracts; the real home is cle.harness. 3 modules were uncollectable for a few minutes.
tests/viewer structure violations: 7 -> 0.

## FINAL (w1-tests resumed session)
- structure under tests/: 14 -> 1 (the 1 is tests/sft/data/test_symbolic_board_dataset.py, SFT agent's, not touched)
- ruff tests/replay + tests/viewer: clean. ruff tests/ overall: 73 left, 47 of them in the SFT-owned file; the other 26 are in tests/sft/training/{test_ms_swift_core,test_olora_bundle_integration}.py and tests/sft/eval/test_failure_scorecard.py.
- pytest tests/replay 13F/164P/1S and tests/viewer 2F/217P, both identical to the pre-split baselines by test name.
- git diff --check clean.
- CONCURRENCY: from 12:04 to 12:07 another agent ran a global annotation sweep over tests/ (data_pipeline/recognition/tasks, evals/reasoning, viewer/traces, sft/{data,eval}), landing the annotations seconds before mine each time. I stopped my sweep rather than clobber them and told team-lead + w2-data-pipeline.

## Final state (w1-tests)
- Structure gate for tests/: only tests/sft/data/test_symbolic_board_dataset.py (404 lines) remains, owned by the SFT agent.
- Ruff on tests/: 3949 -> 47, and all 47 remaining are in that same SFT-owned file.
- Collection: all 3464 original node ids still collected; total is 3473 because other agents added 9 tests. Nothing lost or renamed.
- tests/sft/training/test_eval_qwen_text.py moved from the root so it sits beside its helper package (it does `from trl_catan_text.support import ...`). Root tests/ now holds conftest.py, test_board_packs.py, test_full_board_new_layouts.py, test_modal_catan_budget.py and README.md.
- mypy --strict on tests/: 5084 -> 5332. Annotation errors (no-untyped-def + no-untyped-call, 3290 at baseline) are gone; the strict annotations expose index/union-attr/arg-type errors on JSON-shaped test data that were previously masked by implicit Any. See the report for detail.

## Final verification (w1-tests, end of session)
- Ruff on tests/: 3949 -> 47. All 47 sit in tests/sft/data/test_symbolic_board_dataset.py (w3-sft). Every other test file passes E4,E7,E9,F,I,ANN,PLC0415.
- Structure gate on tests/: 52 oversized files -> 1, the same SFT-owned file. No tests/ folder exceeds 15 direct authored files.
- Collection: 3464 baseline node ids, 0 lost, 9 added by other agents; 3473 total.
- Suite (tests/ minus tests/sft and recognition_contracts): 2733 passed, 34 skipped, 4 failed, 11 errors.
  The 4 failures and 11 errors are exactly the pre-existing set named in the brief.
- git diff --check: clean.
- mypy --strict on tests/: 5084 -> 5839 total, but missing-annotation errors went 3290 -> 38.
  35 of the 38 are in tests/sft; the other 3 are calls into untyped SOURCE functions
  (get_benchmark_metadata, _get_state_snapshot, _failed_attempt_payload) owned by other agents.
- Tried and REVERTED: a typed `payload(response)` narrowing helper over 220 `TestResponse.json`
  sites in tests/viewer/routes. It cut mypy by only 24 because the errors move one level down to
  indexing JsonValue, and it collided with local variables named `payload` in 8 files.
  Viewer is back to Ruff clean, 217 passed, the 2 known failures.
