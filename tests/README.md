# Test ownership and commands

Run commands from the repository root. Pytest discovers these directories through
the existing `testpaths = ["tests"]` configuration and default `prepend` import
mode. The ownership directories deliberately have no `__init__.py`, so test
module names stay unchanged and cannot shadow application packages such as `sft`.
`conftest.py` keeps the repository root importable. Use `python -m pytest`: the
local `pytest` console script has a stale interpreter path into another checkout.

```sh
# Collection, including optional-dependency diagnostics
uv run --no-sync python -m pytest --collect-only -q

# Select an ownership area or an individual test
uv run --no-sync python -m pytest -q tests/engine tests/harness
uv run --no-sync python -m pytest -q tests/evals/board_bench/test_catan_board_bench_naming.py

# Relocation-sensitive paths and imports (offline)
uv run --no-sync python -m pytest -q \
  tests/evals/board_bench/test_catan_board_bench_naming.py \
  tests/evals/board_bench/test_catan_board_bench_render_variant.py \
  tests/evals/board_bench/test_catan_strict_vision_probe.py \
  tests/evals/board_bench/test_catan_strict_text_probe_novita.py \
  tests/evals/board_bench/test_catan_unified_benchmark_summary.py \
  tests/evals/reasoning/test_initial_settlement_reasoning.py \
  tests/data_pipeline/recognition/export/test_build_catan_board_recognition_eval_suite.py \
  tests/sft/data/test_sft_tooling.py \
  tests/sft/eval/test_board_readouts.py \
  tests/replay/test_replay_core_boundaries.py::test_legacy_viewer_core_shims_are_removed \
  tests/replay/test_replay_core_boundaries.py::test_core_runtime_has_no_viewer_or_web_framework_imports

# Requires the project's SFT dependencies, including peft
uv run --no-sync python -m pytest -q tests/sft/eval/test_spatial_tasks.py

# Repository-wide deterministic quality gate
uv run --no-sync python -m scripts.quality
```

## Root and existing suites

The root holds 12 Python files plus this README (13 direct authored files):
`conftest.py`, `test_board_atlas.py`, `test_board_packs.py`,
`test_eval_visual_precision.py`, `test_full_board_new_layouts.py`,
`test_modal_catan_budget.py`, `test_modal_spatial_continuation.py`,
`test_modal_spatial_extension.py`, `test_qwen_vision_sft.py`,
`test_verify_spatial_extension.py`, `test_eval_qwen_text.py`, and
`test_trl_catan_text.py`. The two text tests stay together for their direct helper
import; the spatial launcher tests also retain their direct helper import.

Existing `audits/`, `quality/`, `engine_contracts/`, `map_contracts/`,
`player_contracts/`, `format_contracts/`, and `miles_eval/` keep their ownership.
Every newly organized directory has at most 15 direct authored files. Individual
test files retain their contents and test names; oversized files await a separate
split. Repository-relative `__file__` paths still resolve to the same project root.

## Relocation verification (2026-09-21)

- All 124 moved files match their captured working-tree contents after the six
  necessary `__file__.parents` depth adjustments. Existing caller/import edits
  were preserved. The 12 retained root Python files also match the snapshot.
- Baseline and post-move `pytest --collect-only -q`: **2,867 tests, 18 collection
  errors** each. Comparing multisets of node IDs after mapping destination paths
  back to their original paths found zero lost, added, or duplicated identities;
  the collection-error modules also match exactly. These used the stale pytest
  console script, whose interpreter lacks `peft`; they do not establish the
  current environment's dependency availability. Final collection through
  `uv run --no-sync python -m pytest --collect-only -q` succeeds: **3,429 tests**.
- The offline relocation command above: **64 passed**, three existing prompt-suite
  deprecation warnings. All six moved repository-root expressions were also
  checked to resolve to the same absolute project root, including
  `test_spatial_tasks.py`. This 64-test run also used the stale console script;
  see the final explicit-interpreter verification in `../tasks/todo.md`.
- `uv run --no-sync python -m scripts.quality` ran and failed: 210 repository-wide
  structure violations, 5,782 Ruff errors, and strict mypy failures. The test root
  and all new ownership directories meet the folder cap; their largest direct
  file count is 13. Existing oversized tests and lint/type failures remain.

## Complete root-file move map

Every filename below moved from `tests/<filename>` to the heading directory,
without a filename change.

### `tests/engine/`

- `test_board_rules.py`, `test_engine_boundaries.py`, `test_engine_rng.py`
- `test_game_engine_events.py`, `test_game_force.py`, `test_game_supply_limits.py`
- `test_map_terrain_inventory.py`, `test_state_copy.py`, `test_trade_window.py`

### `tests/harness/contracts/`

- `test_action_tools.py`, `test_board_surface.py`, `test_contract_compatibility.py`
- `test_harness_context.py`, `test_harness_reasoning.py`, `test_knight_tool.py`
- `test_numeric_response_limits.py`, `test_prompt_suite_store.py`, `test_response_xml.py`
- `test_shared_fresh_contract.py`, `test_shared_prompt_components.py`, `test_suite_status.py`
- `test_v11_response.py`

### `tests/harness/providers/`

- `test_harness_cerebras.py`, `test_harness_openrouter.py`
- `test_openrouter_http_failure.py`, `test_openrouter_tls.py`, `test_openrouter_tool_client.py`

### `tests/sandbox/`

- `test_action_batches.py`, `test_catan_sandbox.py`, `test_communication.py`
- `test_players.py`, `test_reactive_speech.py`, `test_sandbox_decision.py`
- `test_sandbox_pool.py`, `test_trade_preauthorization.py`

### `tests/replay/`

- `test_replay_action_diff.py`, `test_replay_audit.py`, `test_replay_core_boundaries.py`
- `test_replay_llm_response.py`, `test_replay_model_traces.py`, `test_replay_playwright_scraper.py`
- `test_replay_trading.py`, `test_replay_transcript.py`

### `tests/viewer/routes/`

- `test_catan_sft_data_routes.py`, `test_catan_text_format_routes.py`, `test_decision_eval_routes.py`
- `test_fresh_notes_routes.py`, `test_game_viewer_server.py`, `test_initial_settlement_reasoning_routes.py`
- `test_live_sandbox_routes.py`, `test_prompt_suite_routes.py`, `test_shared_prompt_routes.py`

### `tests/viewer/commentary/`

- `test_commentary_contextualizer.py`, `test_commentary_episodes.py`
- `test_commentary_references.py`, `test_narrator_reasoning.py`

### `tests/viewer/traces/`

- `test_compact_live_traces.py`, `test_game_logging.py`
- `test_live_color_palette.py`, `test_live_trace_store.py`

### `tests/evals/board_bench/`

- `test_catan_board_bench.py`, `test_catan_board_bench_naming.py`, `test_catan_board_bench_openrouter_eval.py`
- `test_catan_board_bench_render_variant.py`, `test_catan_board_bench_run_summary.py`
- `test_catan_strict_text_probe_novita.py`, `test_catan_strict_vision_probe.py`
- `test_catan_unified_benchmark_summary.py`, `test_openrouter_model_sweep.py`

### `tests/evals/formats/`

- `test_catan_ascii_variations.py`, `test_catan_full_graph_formats.py`
- `test_catan_text_format_optimization.py`, `test_catan_tile_prompt_ablation.py`
- `test_catan_tokenizer_integration.py`, `test_catan_tokens.py`

### `tests/evals/reasoning/`

- `test_decision_buckets.py`, `test_initial_settlement_reasoning.py`, `test_inspect_archives.py`
- `test_transcript_observation_assembly.py`, `test_transcript_reasoning.py`

### `tests/data_pipeline/`

- `test_data_pipeline_layout.py`

### `tests/data_pipeline/recognition/datasets/`

- `test_board_recognition_density_curriculum.py`, `test_board_recognition_production_curriculum.py`
- `test_board_recognition_replay_dataset.py`, `test_board_recognition_sources.py`
- `test_catan_board_recognition_curriculum.py`, `test_catan_board_recognition_dataset.py`
- `test_full_board_diversify.py`

### `tests/data_pipeline/recognition/tasks/`

- `test_board_recognition_adjacent_pair.py`, `test_board_recognition_inverse_grounding.py`
- `test_board_recognition_node_edge_readout.py`, `test_board_recognition_query_schedule.py`
- `test_board_recognition_semantics.py`, `test_board_recognition_single_piece.py`
- `test_board_recognition_spatial_localization.py`, `test_board_recognition_spatial_robber.py`
- `test_board_recognition_terrain_readout.py`

### `tests/data_pipeline/recognition/export/`

- `test_board_recognition_ms_swift.py`, `test_build_catan_board_recognition_eval_suite.py`
- `test_mix_rung_data.py`, `test_reweight_node_edge.py`

### `tests/sft/data/`

- `test_coordinate_comparison.py`, `test_sft_tooling.py`, `test_spatial_continuation_dataset.py`
- `test_symbolic_board_dataset.py`, `test_symbolic_board_tasks.py`

### `tests/sft/eval/`

- `test_analyze_occupancy_misses.py`, `test_behavior_diagnostics.py`, `test_board_readouts.py`
- `test_eval_candidate_scoring.py`, `test_eval_regression_panel.py`, `test_failure_scorecard.py`
- `test_full_board_readout.py`, `test_marker_diagnostics.py`, `test_spatial_grounding_gates.py`
- `test_spatial_tasks.py`, `test_symbolic_eval_integration.py`, `test_visual_rank_eval.py`

### `tests/sft/training/`

- `test_extract_visual_delta.py`, `test_gradient_diagnostics.py`, `test_inspect_token_rows.py`
- `test_ms_swift_core.py`, `test_olora_bundle_integration.py`, `test_trl_catan_vision.py`
