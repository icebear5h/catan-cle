# Miles topology SFT

The new SFT path uses **Miles + Ray + Megatron LoRA**. Its first run is a
two-update, single-H200 smoke. Generation is absent during SFT; SGLang evaluation
is a subsequent stage. Real GPU acceptance is separate from local contract tests.

## Training boundary

`symbolic_board_v2/train.jsonl` contains 3,200 rows. This projection admits 1,600:

| Primitive | Rows |
| --- | ---: |
| Direction | 400 |
| Direction choice | 400 |
| Incidence | 400 |
| Neighbors | 240 |
| Oriented step | 160 |

The remaining state-conditioned/readout/composition rows are excluded and counted
in the preparation receipt. Source train splits, IDs, ordering and hashes are
retained. Native no-thinking tokenization masks the prompt and supervises the
complete answer including end-of-turn/newline. Overlong examples fail rather
than truncate. This is fixed-atlas topology SFT, not coordinate SFT or GRPO.

## Selected initialization

The user selected **merge r04, then train a fresh adapter**. The exporter consumes:

```text
Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
board-fluency-extension-20260915-r04/training/checkpoints/checkpoint-512
```

It merges every independent language LoRA update and replaces the 154 selected
rows in both untied token matrices. The full vocabulary remains 248320. The
merged language weights/rows are BF16; inherited vision tensors remain FP32.
This is a functional warm-start, **not bitwise equivalence to live FP32 adapter
arithmetic**: the manifest reports merge rounding. No optimizer state is imported.

Fresh standard Megatron LoRA uses rank 16, alpha 32, dropout 0, and six explicit
language-decoder module patterns including GDN. The merged base, token matrices,
vision and MTP are frozen. Fused LoRA changes the parameterization from the old
independent HF adapters; it is not exact adapter-factor continuation.

Bridge can omit MTP and cast vision during export. Therefore the final model is
assembled from **only the admitted trained language projections** plus every other
tensor and inference asset copied exactly from the immutable merged base. A
composition manifest and completion seal bind the result. See
[`runtime/README.md`](runtime/README.md) for actual gradient/update/save witnesses.

## Local commands

Inspect the split without loading weights:

```bash
uv run --no-sync python -m sft.miles_sft inspect \
  artifacts/generated/sft/symbolic_board_v2/train.jsonl
```

On a machine with the actual model and historical bundle:

```bash
python -m sft.miles_sft merge --base /models/qwen-stock \
  --adapter /models/r04 --output /models/r04-merged
python -m sft.miles_sft prepare \
  --source artifacts/generated/sft/symbolic_board_v2/train.jsonl \
  --checkpoint /models/r04-merged --output /data/miles-topology
python -m sft.miles_sft run --checkpoint /models/r04-merged \
  --data /data/miles-topology --output /runs/miles-sft-smoke
```

`run` defaults to command-only planning. Add `--execute` inside the pinned GPU
runtime to perform the update. `preflight` accepts the same model/data/run paths
and validates all hashes and native tokenization on CPU. Outputs must be fresh.

## Modal smoke

The launcher uses the active workspace's `catan-hf-cache` and `catan-sft-runs`.
The historical r04 bundle originally exists in **icebear5h**; its required model
assets must be copied into **tetracorp** before using the default workspace.
The pinned stock model is already cached in tetracorp.

```bash
MODAL_PROFILE=tetracorp uv run --no-sync python -m modal run --detach \
  -m sft.miles_sft.modal_run --run-name miles-topology-smoke-r01 --prepare-only

# Reuse a successfully completed CPU preparation:
MODAL_PROFILE=tetracorp uv run --no-sync python -m modal run --detach \
  -m sft.miles_sft.modal_run --run-name miles-topology-smoke-r01 --use-prepared --execute
```

Without `--prepare-only` or `--execute`, the entrypoint prints a plan. The CPU
stage merges/validates weights and encodes the admitted corpus. GPU execution is
two updates, batch 2, microbatch 1, fixed LR 5e-5, with no warmup, no generation,
no retries and a 1,740-second worker deadline inside a 1,800-second H200 function.
This deliberately small constant-LR smoke is not a matched performance benchmark
or a full continuation of the historical cosine schedule.

Runtime pins live in `config.py`: immutable base image plus exact Miles,
Megatron-Bridge and Megatron-LM revisions. The image creates separate pinned
checkouts; it does not silently use whichever sources happened to ship in the base.
Bridge `40b93089` and Core `73b54618` are the compatible pair documented in Miles
PR #3336. The newer Bridge `2e09c234` does not work with that Core. Native backend
imports require `libcuda`, so the GPU worker checks the actual provider before
scanning/loading weights. A failed attempt is preserved; use a new
`--training-name training-r03` with `--use-prepared` to reuse the completed merge.
Pass the same `--training-name` to `modal_eval`.

## Outputs and acceptance

```text
/runs/miles-sft/<name>/
  prepare.json
  merged-base/                     # prior learned state, now frozen
  data/input.jsonl, metadata.json
  training/launch.json, worker.log
  training/receipts/               # actual scope, gradients, updates, final cursor
  training/checkpoints/iter_0000001/adapter/
  training/exports/rollout-1/bridge/ # raw upstream export
  training/exports/rollout-1/model/  # complete, provenance-bound serving checkpoint
```

Completion requires finite real loss/gradients, updates in every target family,
native adapter/optimizer/scheduler artifacts, complete merged export, and a saved
data-source cursor. These checks do not establish native-resume numerical parity
or HF-versus-Megatron logit parity; those need real model execution.

The reload smoke reuses [`miles_eval`](../miles_eval/README.md), with a separate
32-question held-out topology panel and its own proven evaluation runtime:

```bash
uv run --no-sync python -m sft.miles_sft.evaluation \
  --source artifacts/generated/sft/symbolic_board_v2/validation.jsonl \
  --output artifacts/generated/sft/miles_topology_reload_eval_v1
MODAL_PROFILE=tetracorp uv run --no-sync python -m modal run --detach \
  -m sft.miles_sft.modal_eval --run-name miles-topology-smoke-r01
```

The evaluator resolves the final `model/` directory from a completed training
receipt and checks its composition identity on CPU before allocating a GPU.
Never point post-SFT evaluation at `merged-base/` or the raw `bridge/` directory.
SGLang reload/generation and training throughput remain GPU checks.

## Local verification

```bash
uv run --no-sync python -m pytest tests/miles_sft -q
uv run --no-sync python -m scripts.quality
```

Current local result: **90 passed, 1 skipped**; the skipped test requires real
Miles. Full repository quality gate passes. Tests exercise real small Torch and
safetensors updates, native token/mask contracts, exact frozen export preservation,
artifact corruption rejection and the single-GPU command contract.

The real saved r04 tokenizer also encoded all 1,600 admitted training examples:
82,386 total tokens, 7,084 supervised tokens, maximum sequence length 65, without
truncation. Artifacts: `artifacts/generated/sft/miles_topology_sft_v1/`.

Real CPU merge/preflight passed on `tetracorp` for
`miles-topology-smoke-20260923-r03`, after copying and hash-verifying the historical
bundle. Its merged manifest is
`1b76b77ef0bea13523bd0133ce5f7e2d9c3e74aa8cf6e2d71d91b135e8255267`.
The receipt is `artifacts/runs/sft/miles-topology-smoke-20260923-r03/prepare.json`.
The H200 training smoke and post-training reload are separate acceptance steps.

Current remote status: the corrected GPU attempt
`ap-vla1cKW0SyrnwkN6x2aJnu` passed provider construction and reached fresh LoRA
creation, but was terminated with SIGTERM/"function is stopped" before any
optimizer-step receipts. `training-r02/receipts/initialize-rank-0.json` exists;
successful training/export/reload is **not yet verified**. No Catan Miles app
remains active; termination intent must be clarified before relaunching.
