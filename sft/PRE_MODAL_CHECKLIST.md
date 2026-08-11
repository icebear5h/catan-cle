# Pre-Modal SFT Checklist

Use this as the gate before creating/spending on a Modal training run.

## 0. Decide The Smoke Target

- [ ] Smoke run target is Phase 0 atlas topology, not visual QA.
- [ ] `--max-steps` is set to `5` for the first Modal run.
- [ ] GPU is `L40S` unless there is a specific reason to use A100/H100.
- [ ] Expected output is only "training loop works and saves checkpoint", not model quality.

## 1. Local Repo Hygiene

- [ ] Confirm current branch/worktree state.
- [ ] Confirm SFT files exist:
  - [ ] `sft/modal_train.py`
  - [ ] `sft/scripts/train_qwen_vl_sft.py`
  - [ ] `sft/scripts/build_atlas_topology_dataset.py`
  - [ ] `sft/scripts/build_vlm_sft_dataset.py`
  - [ ] `sft/configs/qwen3_vl_8b_qlora.yaml`
- [ ] Confirm generated training data is git-ignored.
- [ ] Confirm no CatanBench generated QA is committed into `sft/data`.

## 2. Phase 0 Atlas Data

- [ ] Build atlas topology dataset:

```bash
uv run python sft/scripts/build_atlas_topology_dataset.py \
  --output sft/data/catan_atlas_topology.jsonl
```

- [ ] Confirm row count is nonzero, currently expected `390`.
- [ ] Inspect 5 examples manually.
- [ ] Confirm examples are text-only and use Catan tokens.
- [ ] Confirm categories include tile, node, edge, and port topology.

## 3. Phase 1 Visual QA Leakage Guard

- [ ] Run the builder against CatanBench-100 and confirm it refuses:

```bash
uv run python sft/scripts/build_vlm_sft_dataset.py \
  --qa-jsonl data_pipeline/catanbench/datasets/catanbench_100/questions/qa.jsonl \
  --manifest-jsonl data_pipeline/catanbench/datasets/catanbench_100/manifest.jsonl \
  --image-root data_pipeline/catanbench/datasets/catanbench_100 \
  --output /tmp/catan_leak_check.jsonl \
  --limit 3
```

- [ ] Expected result: exits nonzero and reports held-out game IDs.
- [ ] Do not bypass this for real training.

## 4. Local Script Checks

- [ ] Compile SFT scripts:

```bash
uv run python -m py_compile \
  sft/modal_train.py \
  sft/scripts/train_qwen_vl_sft.py \
  sft/scripts/build_atlas_topology_dataset.py \
  sft/scripts/build_vlm_sft_dataset.py
```

- [ ] Trainer CLI help works:

```bash
uv run python sft/scripts/train_qwen_vl_sft.py --help
```

- [ ] Dataset builder CLI help works:

```bash
uv run python sft/scripts/build_vlm_sft_dataset.py --help
```

## 5. Modal Account And Cost Controls

- [ ] Modal CLI is installed locally.
- [ ] Modal login/setup is complete.
- [ ] Modal project/app name is `catan-qwen-vl-sft`.
- [ ] Confirm current Modal credits/budget.
- [ ] Set a personal stop rule before launch:
  - [ ] kill if image build loops
  - [ ] kill if model download fails repeatedly
  - [ ] kill if GPU starts but no train step logs within 15 minutes
- [ ] First run uses `--max-steps 5`.

## 6. Modal Volumes

The first run creates these volumes automatically:

- [ ] `catan-hf-cache`
- [ ] `catan-sft-data`
- [ ] `catan-sft-runs`

Confirm after the first run:

- [ ] training JSONL uploaded
- [ ] referenced images uploaded when using visual QA
- [ ] model cache persists
- [ ] checkpoint appears in `catan-sft-runs`

## 7. First Modal Smoke Command

```bash
modal run sft/modal_train.py \
  --train-jsonl sft/data/catan_atlas_topology.jsonl \
  --max-steps 5
```

Expected:

- [ ] Modal image builds.
- [ ] Dataset uploads.
- [ ] Qwen3-VL model loads with 4-bit quantization.
- [ ] Catan tokens are added and embeddings resized.
- [ ] QLoRA adapter initializes.
- [ ] At least one train step completes.
- [ ] Checkpoint/adapter saves to `catan-sft-runs`.

## 8. After Smoke

- [ ] Record Modal runtime and approximate cost.
- [ ] Record errors/warnings.
- [ ] Confirm output directory contents.
- [ ] Do not run a longer job until the saved adapter can be loaded for inference.
- [ ] Next run should be either `max_steps=50` atlas topology or a fresh-game visual QA smoke, depending on blocker status.

## 9. Before Any Real Visual QA SFT

- [ ] Pull fresh non-held-out games.
- [ ] Build a train manifest with explicit game IDs.
- [ ] Prove no overlap with `data_pipeline/catanbench/datasets/catanbench_100/leakage/benchmark_game_ids.json`.
- [ ] Build visual QA JSONL from fresh games.
- [ ] Inspect positive examples for roads, settlements, cities, ports, robber, and tile numbers.
- [ ] Keep CatanBench-100 untouched as eval-only.
