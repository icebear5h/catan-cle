# Catan SFT Tooling

`Sft` contains Catan-specific dataset builders, conversion utilities, and the
current pinned Qwen-Series Modal launchers. Generated datasets, diagnostics,
checkpoints, and reports live outside this installable package.

The first training target is board grounding: stable atlas tokens identify board
entities, while text or pixels supply their transient state. Strategy imitation
comes only after grounding and leakage checks pass. See
[`EXPERIMENT_DESIGN.md`](EXPERIMENT_DESIGN.md) for the curriculum and trainable
scope ladder.

## Safety contract

CatanBoardBench-100 is held out. Never train on its game IDs, images, contracts, QA,
or derived text. The required fail-closed ledger is:

```text
data_pipeline/catan_board_bench/datasets/catan_board_bench_100/leakage/benchmark_game_ids.json
```

Fresh replay candidates come from:

```text
artifacts/raw/colonist/indexes/4p_games_training_candidates.json
```

`build_vlm_sft_dataset` refuses excluded IDs and now errors if the held-out
ledger is missing. Do not create a bypass for production training.

## Layout

- `sft/scripts/`: dataset, conversion, renderer, training-wrapper, and eval code.
- `sft/modal_qwen_series_train.py`: preferred pinned upstream trainer launcher.
- `sft/modal_qwen_series_eval.py`: adapter evaluation launcher.
- `configs/sft/renderer_style.json`: accepted renderer calibration.
- `artifacts/generated/sft/`: ignored, regenerable SFT datasets.
- `artifacts/fixtures/sft/`: tracked smoke and renderer fixtures.
- `artifacts/diagnostics/sft/`: ignored local diagnostic images.
- `artifacts/runs/sft/`: ignored checkpoints/logs plus compact accepted evidence.
- `reports/sft/`: tracked run reports.

Image paths in JSONL should be relative to the dataset file when possible. SFT
upload/conversion tools resolve dataset-relative paths first and repository-
relative paths second; do not embed a developer-specific checkout path.

## Build deterministic atlas data

```bash
uv run python -m sft.scripts.build_atlas_topology_dataset \
  --output artifacts/generated/sft/atlas_topology/catan_atlas_topology.jsonl
```

The canonical build contains 390 text-only rows spanning tile, node, edge, and
port topology.

## Build and render node-factor data

```bash
uv run python -m sft.scripts.build_node_factor_dataset
uv run python -m sft.scripts.render_contract_images
```

Defaults write to `artifacts/generated/sft/node_factors/` and use
`configs/sft/renderer_style.json`. The standard build creates 594 controlled
board contracts and 3,564 visual QA rows. Rendered chat rows use relative image
references; the renderer does not duplicate the answer key.

Tune renderer dimensions locally with:

```bash
uv run python -m sft.scripts.renderer_tuning_app --port 8765
```

The tuner reads tracked calibration contracts from
`artifacts/fixtures/sft/render_contracts/`. Regenerate the dense fixture with:

```bash
uv run python -m sft.scripts.build_colonist_dummy_fixture
```

## Build leakage-checked replay QA

Only use fresh non-benchmark QA and manifest rows:

```bash
uv run python -m sft.scripts.build_vlm_sft_dataset \
  --qa-jsonl data_pipeline/catan_board_bench/datasets/my_fresh_train/questions/qa.jsonl \
  --manifest-jsonl data_pipeline/catan_board_bench/datasets/my_fresh_train/manifest.jsonl \
  --image-root data_pipeline/catan_board_bench/datasets/my_fresh_train \
  --output artifacts/generated/sft/replay_qa/train.jsonl
```

Keep curriculum stages separate until each path trains and evaluates cleanly:

```text
Phase 0: text-only atlas topology
Phase 1: post-atlas node visual grounding
Phase 2: replay-derived image and targeted visual QA
```

## Preferred Qwen-Series training path

Install optional local dependencies with `uv sync --extra sft --extra modal`.
The launcher pins `2U1/Qwen-VL-Series-Finetune` to a recorded commit, converts
local message JSONL to its conversation format, adds Catan tokens before PEFT,
and writes remote checkpoints to the `catan-sft-runs` Modal volume.

Run the self-contained infrastructure smoke first:

```bash
uv run python -m modal run sft/modal_qwen_series_train.py \
  --train-jsonl artifacts/fixtures/sft/modal_vlm_smoke/train.jsonl \
  --max-steps 1
```

Keep a hard step cap until upload, tokenizer resize, adapter initialization,
checkpoint saving, and adapter reload all pass. The superseded custom TRL
launcher was removed; historical smoke results remain in
`reports/sft/2026-05-14-modal-sft-smoke-runs.md`.

For standalone conversion:

```bash
python -m sft.scripts.convert_to_qwen_series_sft \
  --input artifacts/fixtures/sft/modal_vlm_smoke/train.jsonl \
  --output /tmp/catan_qwen_series_train.json \
  --copy-images-to /tmp/catan_qwen_series_images
```

Evaluate a base model or saved adapter through the current Modal eval launcher:

```bash
uv run python -m modal run sft/modal_qwen_series_eval.py \
  --eval-jsonl artifacts/fixtures/sft/modal_vlm_smoke/train.jsonl \
  --limit 4
```

The smoke fixture validates infrastructure only; it is not a model-quality
benchmark. The quarantined historical short train/held-out files share source
images and must not be reported as an independent visual split.
