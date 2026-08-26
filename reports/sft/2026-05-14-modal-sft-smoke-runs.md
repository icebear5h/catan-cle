# Catan SFT Training Run

Date: 2026-05-14

## Saved Renderer Baseline

The accepted renderer style is saved at:

```text
configs/sft/renderer_style.json
```

The Phase 1 post-atlas node visual grounding data has been regenerated with
that style:

```text
artifacts/generated/sft/node_factors/messages_with_images.jsonl
artifacts/generated/sft/node_factors/render_metadata.json
```

Current rendered data:

```text
594 board images
3,564 visual QA rows
512x512 images
```

## Smoke Inputs

Text-only atlas smoke:

```text
artifacts/generated/sft/atlas_topology/catan_atlas_topology.jsonl
390 rows
```

Visual smoke:

```text
artifacts/fixtures/sft/modal_vlm_smoke/train.jsonl
24 rows
4 examples per visual category
```

Categories:

```text
board_atlas_bboxes -> full_board_atlas_bbox_map
node_bbox -> atlas_coordinate_grounding
node_occupancy -> transient_node_state_readout
node_adjacent_tile_resource_numbers -> local_tile_readout
node_incident_road_owners -> local_edge_readout
node_local_state_json -> local_state_composition
```

## Recommended First Modal Run

The following historical 8B QLoRA smoke used the now-removed custom
`sft/modal_train.py` launcher. The pinned Qwen-Series launcher superseded it:

```bash
.venv/bin/modal run sft/modal_train.py \
  --train-jsonl artifacts/fixtures/sft/modal_vlm_smoke/train.jsonl \
  --max-steps 5
```

This uses:

```text
model: Qwen/Qwen3-VL-8B-Instruct
gpu: L40S
quantization: 4-bit NF4
LoRA: enabled
Catan tokens: added, with embed_tokens and lm_head saved
```

Do not run the full 3,564-row visual QA job until this smoke saves an adapter and
we can reload it for inference.

## Cost Guard

Stop the job if:

```text
image build loops
model download fails repeatedly
GPU starts but no train step logs appear within 15 minutes
CUDA OOM repeats after restart
```

For the next step after smoke, prefer `--max-steps 50` before a full epoch.

## 2026-05-14 Modal Smoke Result

Historical command (retained as run provenance, no longer runnable):

```bash
.venv/bin/modal run sft/modal_train.py \
  --train-jsonl artifacts/fixtures/sft/modal_vlm_smoke/train.jsonl \
  --max-steps 5
```

Run URL:

```text
https://modal.com/apps/icebear5h/main/ap-3dGxbAF0nPnxKANLTUQJ9W
```

Result:

```text
completed: yes
uploaded_rows: 20
uploaded_images: 4
model: Qwen/Qwen3-VL-8B-Instruct
added_catan_tokens: 197
frozen_vision_like_parameters: 156,608,240
train_runtime_seconds: 38.77
train_loss: 6.045
mean_token_accuracy: 0.4682
output_dir: /runs/qwen3-vl-8b-catan-qlora
```

Warnings to track:

```text
HF Hub ran unauthenticated, so downloads may be slower or rate-limited.
PEFT warned that tied embeddings are part of the adapter without ensure_weight_tying.
Modal worker reported kernel 4.4.0 below the recommended 5.5.0.
```

## 2026-05-14 Qwen3.5 Modal Smoke Result

Command:

```bash
.venv/bin/modal run sft/modal_qwen_series_train.py \
  --train-jsonl artifacts/fixtures/sft/modal_vlm_smoke/train.jsonl \
  --model-id Qwen/Qwen3.5-9B \
  --max-steps 1 \
  --remote-dir catan-qwen-series-sft/qwen3.5-9b-smoke \
  --output-dir /runs/qwen3.5-9b-catan-smoke-1step
```

Run URL:

```text
https://modal.com/apps/icebear5h/main/ap-R48z9ojREzfuVNAvMMfofM
```

Result:

```text
completed: yes
uploaded_rows: 20
uploaded_images: 4
model: Qwen/Qwen3.5-9B
added_catan_tokens: 197
max_steps: 1
train_runtime_seconds: 3.483
train_loss: 8.632
output_dir: /runs/qwen3.5-9b-catan-smoke-1step
qwen_series_commit: 130ad7ccafe06a2ae5377bad9d512027d8225dc5
```

Notes:

```text
First attempt failed before model download because the Modal image could not import
the top-level catan_board_bench compatibility package. Fixed the wrapper to import
data_pipeline.catan_board_bench.tokens directly.
HF Hub ran unauthenticated, so downloads may be slower or rate-limited.
Qwen3.5 fell back to the torch implementation because FLA/causal-conv1d fast
kernels were not installed.
Modal worker reported kernel 4.4.0 below the recommended 5.5.0.
```

## 2026-05-14 Qwen3.5 Modal Throughput Smoke

Command:

```bash
.venv/bin/modal run sft/modal_qwen_series_train.py \
  --train-jsonl artifacts/fixtures/sft/modal_vlm_smoke/train.jsonl \
  --model-id Qwen/Qwen3.5-9B \
  --max-steps 10 \
  --remote-dir catan-qwen-series-sft/qwen3.5-9b-throughput-10step \
  --output-dir /runs/qwen3.5-9b-catan-throughput-10step
```

Run URL:

```text
https://modal.com/apps/icebear5h/main/ap-Yd50o4HOztjNo46IXMkZ5b
```

Result:

```text
completed: yes
uploaded_rows: 20
uploaded_images: 4
model: Qwen/Qwen3.5-9B
added_catan_tokens: 197
max_steps: 10
weight_load_seconds: about 12
train_runtime_seconds: 13.75
train_steps_per_second: 0.727
seconds_per_step: 1.38
train_samples_per_second: 0.727
train_loss: 4.957
output_dir: /runs/qwen3.5-9b-catan-throughput-10step
```

Cost estimate:

```text
Modal L40S listed price on 2026-05-15: $0.000542/sec.
Trainer-only GPU cost for 13.75s: about $0.0075.
Approx steady-state GPU cost: about $0.00075/step, or about $0.75/1k steps.
Actual billed run cost is higher because it includes startup, CPU, memory, and
cached model load time. Terminal wall time was roughly 40s, implying about
$0.02 GPU-only for this short run.
```

Warnings to track:

```text
Fast FLA/causal-conv1d path still unavailable; Qwen3.5 used torch fallback.
Modal worker still reported kernel 4.4.0 below the recommended 5.5.0.
HF Hub ran unauthenticated.
```
