# Catan VLM SFT Workspace

This folder is the clean workspace for Catan-specific VLM supervised fine-tuning.
It is intentionally separate from benchmark generation and OpenRouter eval code.

## Goal

Train an open-weight VLM to bind board pixels to Catan atlas tokens:

```text
<N11> -> stable board position
pixels over <N11> -> current transient state
answer -> <BLACK> <SETTLEMENT> / EMPTY / etc.
```

The first target is board-state extraction, not strategy imitation.

## Leakage Rule

CatanBench-100 is held out. Do not train on its game IDs, board images, contracts,
QA rows, or derived captions.

The canonical held-out list is:

```text
data_pipeline/catanbench/datasets/catanbench_100/leakage/benchmark_game_ids.json
```

Training data should come from fresh replay pulls using:

```text
data_pipeline/bootstrapping/scrapers/4p_games_training_candidates.json
```

## Layout

| Path | Purpose |
| --- | --- |
| `configs/qwen3_vl_8b_qlora.yaml` | First Qwen3-VL-8B QLoRA SFT config. |
| `scripts/build_vlm_sft_dataset.py` | Converts Catan QA rows into TRL-style VLM JSONL while checking game-ID leakage. |
| `scripts/train_qwen_vl_sft.py` | TRL/PEFT QLoRA training entrypoint. |
| `data/` | Local generated training JSONL files. |
| `outputs/` | Local checkpoints/logs. |

## Dataset Build

Build Phase 0 atlas/token topology data:

```bash
uv run python sft/scripts/build_atlas_topology_dataset.py \
  --output sft/data/catan_atlas_topology.jsonl
```

Build the second dataset: Phase 1 post-atlas node visual grounding.

```bash
uv run python sft/scripts/build_node_factor_dataset.py \
  --output-dir sft/data/synthetic_node_factors
```

This assumes the atlas already exists. It does not teach topology from scratch;
it trains the model to use stable node tokens as visual handles, then read
transient local facts around those handles. By default it creates one controlled
contract per node/occupancy case:

```text
54 nodes * (EMPTY + 5 colors * SETTLEMENT/CITY) = 594 board contracts
594 contracts * 6 QA rows = 3,564 QA rows
```

Use `--variants-per-case N` to multiply board shuffles for the same stable node
token and transient occupancy target. Use `--assume-rendered-images` only when a
frontend render step will fill `images/<sample>.png`; otherwise the generated QA
rows stay contract-only with `image_path: null`.

Render contract datasets into image-backed QA:

```bash
uv run python sft/scripts/render_contract_images.py \
  --dataset-dir sft/data/synthetic_node_factors \
  --style-config sft/configs/renderer_style_current.json
```

This uses the Python HexBoard-compatible renderer: same frontend assets, same
geometry constants, and no Playwright loop. The original contract-only rows are
preserved; image-backed copies are written as `qa_with_images.jsonl` and
`messages_with_images.jsonl`.

Tune the Python renderer dimensions with a local slider UI:

```bash
uv run python sft/scripts/renderer_tuning_app.py \
  --dataset-dir sft/data/synthetic_node_factors \
  --port 8765
```

Use this before bulk rerenders when roads, docks, or output image size need
manual calibration against the frontend look. The tuner also loads persistent
calibration fixtures from `sft/fixtures/render_contracts/`; regenerate the dense
Colonist-like dummy board with:

```bash
uv run python sft/scripts/build_colonist_dummy_fixture.py
```

Build Phase 1 visual QA data only once fresh non-benchmark QA rows exist:

```bash
uv run python sft/scripts/build_vlm_sft_dataset.py \
  --qa-jsonl data_pipeline/catanbench/datasets/my_fresh_train/questions/qa.jsonl \
  --manifest-jsonl data_pipeline/catanbench/datasets/my_fresh_train/manifest.jsonl \
  --image-root data_pipeline/catanbench/datasets/my_fresh_train \
  --output sft/data/catan_vlm_sft_train.jsonl
```

The script exits if any source game ID appears in the held-out CatanBench ledger.

Keep Phase 0 and Phase 1 as separate files/runs at first:

```text
Phase 0: text-only atlas topology
Phase 1: post-atlas node visual grounding
Phase 2: replay-derived image + targeted visual QA
```

Do not mix them until each path trains and evaluates cleanly.

## First Training Run

Install training dependencies separately when running on a GPU machine:

```bash
uv sync --extra sft
```

Then:

```bash
uv run python sft/scripts/train_qwen_vl_sft.py \
  --config sft/configs/qwen3_vl_8b_qlora.yaml \
  --train-jsonl sft/data/catan_vlm_sft_train.jsonl
```

## Modal Smoke Run

Use Modal first for capped smoke tests:

```bash
modal run sft/modal_train.py \
  --train-jsonl sft/data/catan_vlm_sft_train.jsonl \
  --max-steps 5
```

The Modal entrypoint uploads the JSONL and referenced images into the
`catan-sft-data` volume, caches Hugging Face weights in `catan-hf-cache`, and
writes checkpoints to `catan-sft-runs`.

Keep `--max-steps` set until the data path, tokenizer resize, QLoRA setup, and
checkpoint save path are all proven.

## Preferred Qwen-VL-Series-Finetune Path

The TRL runner above is a local smoke scaffold. The preferred training backend is
the pinned upstream Qwen-VL-Series-Finetune stack because it owns the
Qwen-specific dataset, collator, model-loader, LoRA/QLoRA, vision-tower, and
merger training behavior.

Convert local JSONL into the upstream conversation JSON format:

```bash
python sft/scripts/convert_to_qwen_series_sft.py \
  --input sft/data/catan_vlm_sft_train.jsonl \
  --output sft/data/catan_vlm_qwen_series_train.json \
  --copy-images-to sft/data/catan_vlm_qwen_series_images
```

Run the upstream-based Modal smoke:

```bash
.venv/bin/modal run sft/modal_qwen_series_train.py \
  --train-jsonl sft/data/modal_vlm_smoke.jsonl \
  --max-steps 1
```

This Modal launcher converts the JSONL during upload, clones
`2U1/Qwen-VL-Series-Finetune` at the pinned commit, adds Catan atlas tokens before
PEFT wraps the model, and runs the upstream `train_sft.py` entrypoint through a
small wrapper.

## Qwen-Specific Tooling

| Tool | Why we use it |
| --- | --- |
| `transformers>=4.57.1` | Native Qwen3-VL processor/model support. |
| `qwen-vl-utils` | Official Qwen image/video utility package; useful for Qwen message/image preprocessing and future multi-image work. |
| `trl>=0.21.0` | VLM-compatible `SFTTrainer` scaffold. |
| `peft>=0.17.0` | LoRA/QLoRA adapters and saved embedding/head modules for added Catan tokens. |
| `bitsandbytes` | 4-bit QLoRA loading. |
| `accelerate` | Device mapping and trainer launch glue. |
| `tensorboard`/`wandb` | Training telemetry. |

Qwen's official finetuning framework also exposes `tune_mm_llm`,
`tune_mm_vision`, and `tune_mm_mlp`. Our first TRL/PEFT ablation approximates
that by using LoRA while freezing vision-like modules. If roads, dice numbers,
and ports still fail, the next run should deliberately unfreeze or LoRA the
vision/projector path.

## Intended Ablations

1. Atlas topology SFT with Catan tokens.
2. Visual QA QLoRA SFT with vision frozen and Catan tokens added.
3. Same visual QA data without Catan tokens, to measure token-anchor value.
4. Vision/projector LoRA if roads, ports, and dice numbers still fail.
5. Spatial-ID auxiliary loss only after basic SFT has a clean baseline.
