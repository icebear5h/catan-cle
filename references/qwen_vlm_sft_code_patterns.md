# Qwen VLM SFT code patterns used by the Catan launchers

Status: implementation audit, 2026-08-26.

This note records the upstream code patterns used by
`sft/modal_qwen_series_vision_train.py`. It is not a benchmark claim and does not
replace the pinned source code. This native short-answer path is related to the
optional decoder bridge in `catan_direct_vision_tower_readout.md`; the direct
slot classifier remains a separate primary perception experiment.

## Audited revisions

| Repository | Revision | Relevant code |
| --- | --- | --- |
| [2U1/Qwen-VL-Series-Finetune](https://github.com/2U1/Qwen-VL-Series-Finetune) | `130ad7ccafe06a2ae5377bad9d512027d8225dc5` | `scripts/finetune_lora.sh`, `src/train/train_sft.py`, `src/trainer/sft_trainer.py`, `src/dataset/sft_dataset.py`, `src/utils.py` |
| [QwenLM/Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) | `96588727e44c78b25ba03ea03b8e12f7e64fd0da` | `qwen-vl-finetune/qwenvl/train/train_qwen.py`, `trainer.py` |
| [huggingface/peft](https://github.com/huggingface/peft) | `v0.15.2` / `3c7b6e7f0252cb18386d86b056a9b100a8160792` | `TrainableTokensConfig`, `LoraConfig.trainable_token_indices`, trainable-token tests |
| [hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) | `a18110d2f064b1518ac313eeb2ba980946467b4d` | `model/adapter.py`, `model/model_utils/visual.py` |
| [modal-labs/modal-examples](https://github.com/modal-labs/modal-examples) | fetched 2026-08-26 | `06_gpu_and_ml/unsloth_finetune.py` |
| [modelscope/ms-swift](https://github.com/modelscope/ms-swift) | `4.5.2` / `ff5128777dd3dc5538f1a95fbcc29442c327d7d2` | exact Qwen3.8 registration/template, `lora_llm`, multimodal optimizer, visual checkpoint save/reload |

## Provenance decision for Qwen3.8-27B

The pinned 2U1 trainer remains useful historical pattern evidence but is not
admitted for the 27B pilot: it claims generic Qwen3.5 support without naming or
testing `Qwen/Qwen3.8-27B`. The official Qwen3-VL fine-tuning script is also not
a drop-in replacement because it does not register Qwen3.8's `qwen3_5`
architecture.

Released ms-swift 4.5.2 explicitly registers the exact checkpoint, dedicated
`qwen3_8` template, full vision/aligner plus language-LoRA tuner, separate
learning rates, and `vit.safetensors` save/reload. It is the selected migration
target, not yet an active launcher. A local external tuner must still restrict
training to the versioned Catan position-token rows, audit the effective scope,
and pass save/reload and dry-run gates before any paid GPU smoke.

## Pattern 1: freeze scope explicitly

The pinned 2U1 trainer has separate controls for the language base, vision
tower, and merger. Its canonical `finetune_lora.sh` uses the configuration needed
for joint visual adaptation:

```text
freeze_llm          = true
freeze_vision_tower = false
freeze_merger       = false
lora_enable         = true
vision_lora         = false
```

This means full vision and merger weights train while LoRA is attached to the
frozen language base. `train_sft.py` re-enables visual parameters after PEFT
wrapping because PEFT freezes base parameters during adapter insertion.

The official Qwen3-VL trainer expresses the same decomposition as
`tune_mm_llm`, `tune_mm_vision`, and `tune_mm_mlp`. LLaMA-Factory keeps explicit
architecture-specific lists of vision, projector, and language module keys.
These are safer patterns than deciding scope from one broad `model.parameters()`
operation.

Catan implementation:

- `vision_only`: full vision + merger + selective Catan token rows;
- `vision_language_lora`: the same, plus language LoRA;
- the base language embedding matrix, decoder weights, and LM head remain frozen
  in both profiles.

Before the trainer creates its optimizer, the wrapper writes
`trainable_parameters.json` and fails if the effective scope differs from the
selected profile.

## Pattern 2: use separate optimizer groups

Both the pinned 2U1 trainer and official Qwen3-VL trainer separate parameters by
name and apply dedicated learning rates to the vision tower and merger. The
Catan defaults are deliberately conservative:

```text
language LoRA or token rows: 1e-4
multimodal merger:           1e-5
full vision tower:           1e-6
```

These are starting values, not an optimum. The tower rate should be swept before
unfreezing additional language capacity.

## Pattern 3: token loss does not require base-language updates

The pinned SFT dataset masks the user prompt and visual placeholder positions
with `IGNORE_INDEX`; cross-entropy is computed on assistant response tokens.
Gradients still pass through the frozen decoder to the merger and vision tower.
Freezing language weights therefore does not disconnect token supervision.

Qwen3.8 supports an explicit no-reasoning prefill. Catan perception rows set
`enable_reasoning=false`; they should contain short exact answers rather than
teacher chain-of-thought.

## Pattern 4: train both sides of an untied token

PEFT 0.15.2 supports two relevant mechanisms:

- standalone `TrainableTokensConfig` for selected embedding/head rows;
- `LoraConfig.trainable_token_indices` for selected rows alongside LoRA.

Qwen may use untied input embeddings and output LM-head weights. Training only
`embed_tokens[token_id]` is insufficient for learning a new output token. The
Catan wrapper therefore targets both input and output rows when they are untied.
It relies on PEFT's automatic tied-weight handling when they share storage.

The `vision_only` profile uses standalone TrainableTokens through the upstream
PEFT save lifecycle. The internal `lora_enable=true` flag selects that upstream
lifecycle; the wrapper replaces the would-be LoRA configuration, so no dummy or
hidden language LoRA parameters are created.

## Pattern 5: preserve non-LoRA trainable weights

When PEFT and full visual tuning are combined, adapter files do not contain the
full vision and merger weights. The pinned trainer stores those parameters in:

```text
non_lora_state_dict.bin
```

The upstream loader restores that state into the resized base model first, then
loads the PEFT adapter. Reversing or omitting this order silently evaluates the
base visual path rather than the trained one. The Catan evaluator now follows
that order and records load evidence in its summary.

## Pattern 6: do not quantize a trainable vision tower

The pinned 2U1 README explicitly rejects combining 4-bit or 8-bit quantization
with full vision training, vision LoRA, or top-k vision unfreezing. Catan vision
SFT is therefore hard-coded to BF16/`bits=16`.

The old `sft/modal_qwen_series_train.py` remains the separate 4-bit,
frozen-vision infrastructure smoke. It is not reused as the visual-training
profile.

## Pattern 7: make cloud runs immutable and resumable

The Modal launcher follows the official example's storage structure:

- one volume for Hugging Face cache;
- one volume for converted data and images;
- one volume for checkpoints and final artifacts;
- a local CLI that constructs a typed run configuration;
- checkpoint discovery and resume delegated to the pinned trainer.

Before launch, the local path validates one image and one user/assistant pair
per row, rejects targets over 512 characters, and hashes the source JSONL plus
every referenced image. This guard matters because the audited upstream SFT
dataset does not currently apply its declared sequence truncation helper. The
remote path stores an identity over the dataset digest, complete training
configuration, hardware profile, and pinned upstream revision. A non-empty
output directory without that manifest, or a mismatched identity, fails closed.

A completed run is accepted only if it contains the PEFT adapter, tokenizer,
`non_lora_state_dict.bin`, and the trainable-parameter audit. Dry-run is the CLI
default, so constructing and reviewing a command does not allocate a training
GPU.

## Hardware profiles

The launcher provides two explicit single-GPU functions:

```text
l40s  -> Qwen3-VL-4B architecture smoke
h200  -> Qwen/Qwen3.8-27B BF16 pilot
```

The 27B checkpoint is rejected on the L40S profile before upload. An H200 is an
initial fit hypothesis, not a measured throughput result; the one-step smoke
must record peak memory and step time before a full run is budgeted.
