# Repository script entrypoints

Run Python entrypoints from the repository root using `python -m` so local
packages resolve in both the parent process and spawned evaluators:

```sh
uv run python -m scripts.board_bench.builders.build_catan_board_bench --help
uv run python -m scripts.board_bench.run.run_catan_openrouter_model_sweep --help
uv run python -m scripts.board_recognition.build_catan_board_recognition_dataset --help
uv run python -m scripts.reasoning.eval_catan_initial_settlement_reasoning --help
uv run python -m scripts.probes.inspect_board_pixels --help
```

The Modal probe remains a Modal entrypoint:
`modal run scripts/probes/vlm_benchmark_modal.py --help`.

The root owns development, deployment, live-trace maintenance, replay diffing,
archive import/visualization, artifact-layout verification, and
`gen_prompt_studio_types` (regenerates the Prompt Studio's shared-suite TS types
from the pydantic model; `--check` fails when stale, and
`tests/viewer/prompt_contract` runs the same check). `quality/` owns
the existing quality tooling.

Direct authored file counts, including initializers and documentation: root 13;
`board_bench` 2; `board_bench/builders` 8; `board_bench/run` 5;
`board_recognition` 10; `reasoning` 7; `probes` 5; `quality` 7. A script that
grew past the 300-line file cap became a same-name package, so its `python -m`
path and every name it exported are unchanged; same-name packages do not count
toward a folder's direct-file cap.

## Split scripts and their gotchas (2026-09-22)

Seventeen oversized scripts are now packages. Each has a `__main__.py`, an
`__init__.py` that re-exports every pre-split name with an explicit `__all__`,
and an `ArgumentParser(prog=...)` pinned to the old filename so `--help` text
is byte-identical. `scripts/board_bench/shapes.py` holds the JSON narrowing
helpers those packages share.

Three packages carry deliberate structure because something patches them at
runtime. Read these before moving code between their modules:

- `eval_catan_board_bench_full_graph_formats` keeps the seven retargetable
  dataset names in `dataset_config.py`. Readers resolve them as
  `dataset_config.NAME` at call time and the package `__init__` forwards reads
  through `__getattr__`. Retarget the suite by patching `dataset_config`;
  assigning the name on the package shadows the forward and never reaches the
  submodule readers. `eval_catan_strict_text_probe_novita` and
  `eval_catan_text_format_optimization` both retarget it through
  `configured_text_evaluator()`.
- `eval_catan_tile_prompt_ablation` keeps its whole scorer, the condition sets,
  and the four regexes in `__init__.py`, because `scorer_sha256` hashes them out
  of its own module globals and the factor tests patch them on the package. The
  sibling modules are imported at the bottom of `__init__` so they can import
  those names back.
- Packages whose tests patch `<module>.httpx.Client` re-export `httpx` from
  `__init__` and list it in `__all__`.

## Exact relocation map (2026-09-21)

Every filename below moved from `scripts/<filename>` into the indicated
directory with its basename preserved. The old module paths have no wrappers.

### `scripts/board_bench/builders/` — 10 scripts

- `build_catan_board_bench.py`
- `build_catan_board_bench_ascii_variations.py`
- `build_catan_board_bench_full_graph_formats.py`
- `build_catan_board_bench_leakage_and_presft.py`
- `build_catan_board_bench_piece_visuals.py`
- `build_catan_text_format_optimization.py`
- `export_catan_tokens.py`
- `rebuild_catan_board_bench_questions.py`
- `render_catan_board_bench_variant.py`
- `render_catan_strict_vision_probe.py`

### `scripts/board_bench/run/` — 12 scripts

- `eval_catan_board_bench_ascii_variations.py`
- `eval_catan_board_bench_full_graph_formats.py`
- `eval_catan_board_bench_openrouter.py`
- `eval_catan_board_bench_text_formats.py`
- `eval_catan_strict_text_probe_novita.py`
- `eval_catan_strict_vision_probe.py`
- `eval_catan_text_format_optimization.py`
- `eval_catan_tile_prompt_ablation.py`
- `rescore_piece_semantic_aliases.py`
- `run_catan_openrouter_model_sweep.py`
- `summarize_catan_board_bench_runs.py`
- `summarize_catan_unified_benchmark.py`

### `scripts/board_recognition/` — 10 scripts

- `audit_catan_board_recognition_sources.py`
- `build_catan_board_recognition_curriculum.py`
- `build_catan_board_recognition_dataset.py`
- `build_catan_board_recognition_density_curriculum.py`
- `build_catan_board_recognition_eval_suite.py`
- `build_catan_board_recognition_production_curriculum.py`
- `export_catan_board_recognition_ms_swift.py`
- `export_catan_board_recognition_sft.py`
- `export_catan_inverse_grounding_ms_swift.py`
- `export_catan_spatial_robber_sft.py`

### `scripts/reasoning/` — 6 scripts

- `build_decision_spot_checks.py`
- `build_transcript_observation_assembly.py`
- `build_transcript_reasoning.py`
- `complete_transcript_observation_assembly_with_pi.py`
- `complete_transcript_reasoning_with_pi.py`
- `eval_catan_initial_settlement_reasoning.py`

### `scripts/probes/` — 6 scripts

- `inspect_board_pixels.py`
- `probe_catan_board_coverage.py`
- `probe_catan_board_mech_geometry.py`
- `probe_catan_board_parts.py`
- `vlm_benchmark.py`
- `vlm_benchmark_modal.py`

## Runtime and provenance constraints

- The five moved scripts with `__file__`-relative repository roots account for
  their new directory depths. Artifact and dataset paths are unchanged.
- The model sweep spawns the canonical evaluator modules with `sys.executable
  -m`. The text-format optimization entrypoint still configures the same shared
  full-graph evaluator module object.
- The tile-ablation scorer's source-hashed functions retain their exact source
  text. Frozen benchmark scorer implementations and historical manifests are
  not part of this relocation.
- The recognition curriculum's generator value remains
  `scripts/build_catan_board_recognition_curriculum.py`: it is a schema-bound
  provenance identifier, not a current execution command.
- Authored source uses the specific `builders/` name because generic `build/`
  directories are excluded by both Git and the quality inventory. The package
  requires no ignore exceptions or quality exclusions changes.

## Documentation follow-up

Current recognition command examples in `sft/README.md` and
`data/curriculum/board_recognition/README.md` use the new module paths.
Historical artifact receipts, run commands, and generated
dataset documentation retain their recorded paths; do not bulk-rewrite them.

## Verification scope

The focused selection is the 15 test modules directly calling moved scripts,
plus `tests/evals/reasoning/test_inspect_archives.py` for the archive adapter's transitive import.
Before and after relocation: **96 passed**, with the same three prompt-suite
deprecation warnings. Tests now live in the ownership folders below.

```sh
.venv/bin/python -B -m pytest -q -p no:cacheprovider \
  tests/evals/board_bench/test_catan_board_bench_naming.py \
  tests/evals/board_bench/test_catan_board_bench_render_variant.py \
  tests/sft/data/test_sft_tooling.py \
  tests/data_pipeline/recognition/datasets/test_catan_board_recognition_curriculum.py \
  tests/data_pipeline/recognition/export/test_build_catan_board_recognition_eval_suite.py \
  tests/evals/formats/test_catan_ascii_variations.py \
  tests/evals/formats/test_catan_full_graph_formats.py \
  tests/evals/board_bench/test_catan_board_bench_openrouter_eval.py \
  tests/evals/board_bench/test_catan_strict_text_probe_novita.py \
  tests/evals/board_bench/test_catan_strict_vision_probe.py \
  tests/evals/formats/test_catan_tile_prompt_ablation.py \
  tests/evals/board_bench/test_catan_board_bench_run_summary.py \
  tests/evals/board_bench/test_catan_unified_benchmark_summary.py \
  tests/evals/board_bench/test_openrouter_model_sweep.py \
  tests/evals/reasoning/test_initial_settlement_reasoning.py \
  tests/evals/reasoning/test_inspect_archives.py
```

Additional verification:

- All 44 moved files compile; 30 retain byte-identical content. All 480 unchanged
  top-level function/class definitions retain exact source text; only the three
  model-sweep spawn builders changed inside function definitions.
- All five repository-root constants resolve to the actual repository. All
  three generated spawn prefixes identify the correct canonical modules; their
  remote evaluation commands were not executed.
- 43 modules cold-import with networking blocked. The Modal probe fails on
  `modal.Secret.from_name(..., required_hint=...)` with the installed SDK;
  executing the original HEAD source reproduces the identical `TypeError`.
  Its API compatibility repair is separate from relocation.
- Six representative CLI help paths pass with networking blocked: benchmark
  builder, strict-vision evaluator, model sweep, recognition dataset builder,
  initial-settlement evaluator, and pixel inspector. No remote jobs launched.
- Repository source scans find no remaining imports of the moved flat script
  modules; the historical curriculum generator string is intentional.

The full quality gate is deferred to the main cleanup owner. This wave addresses
folder ownership; existing oversized script bodies and broader lint/typing debt
still require follow-up.

The follow-up `build` → `builders` package correction passed 37 focused tests
across naming, variant rendering, SFT tooling, strict vision/text probes, unified
summaries, and archive integration. The unchanged quality inventory includes all
10 builder modules plus their initializer in both code and Python source scope.
The temporary package-local Git ignore exception was removed.
