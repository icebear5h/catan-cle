# Miles Catan evaluations

Latest completed evaluation: **sparse h Cartesian, 153/200 (76.5%)**, on stock
Qwen3.8-27B in `tetracorp`. See the
[paired result](../../reports/sft/2026-09-21-sparse-h-cartesian-eval.md).

Uses Miles' built-in evaluation flow and SGLang generation with the existing Catan
scorers. One local GPU, zero training rollouts, one greedy completion per prompt,
thinking disabled, 4,096-token context, and 512 output tokens. No training-model
weights or optimizer are loaded.

Upstream interface inspected at Miles commit
`12754e9507e64d5e537288da17793246e913c525`:
[CLI Eval](https://miles.radixark.com/docs/user-guide/cli-reference#eval),
[eval-only driver](https://github.com/radixark/miles/blob/12754e9507e64d5e537288da17793246e913c525/train.py),
[custom hooks](https://miles.radixark.com/docs/user-guide/customization).

## Files

- `run.py`: prepare panels, print the invocation, or execute it.
- `data.py`: project Catan text rows into Miles prompt/label/metadata JSONL.
- `contracts.py`: typed interfaces and dispatch to the original exact scorers.
- `hooks.py`: Miles reward callback and complete-panel JSON result writer.
- `preflight.py`: validate the serving bundle, tokenizer, and context budget on CPU.
- `worker.py`: start a private one-GPU Ray runtime and run upstream Miles.
- `modal_run.py`: pinned official image, CPU preflight, and bounded H200 execution.

## 1. Prepare a panel locally

Run from this repository with its existing generated datasets available:

```bash
uv run --no-sync python -m sft.miles_eval.run prepare \
  --panel cartesian_h=artifacts/generated/sft/cartesian_h_eval_v1/eval.jsonl \
  --output artifacts/generated/sft/miles_cartesian_h_eval_v1
```

The selected experiment is [exact Cartesian coordinates](../cartesian_eval/README.md)
on **stock Qwen3.8-27B**, without fine-tuning or added tokenizer tokens. It uses
200 cases, ordinary equal-scale x/y axes, and exact fractions/`sqrt(3)` positions.

Repeat `--panel NAME=PATH` to include multiple datasets, such as the board-fluency
review or validation panel. Supported contracts are the stock Cartesian panel, the matched coordinate panel,
board-fluency review/SFT evaluation splits, and symbolic evaluation splits. Images,
training rows, unknown scoring contracts, and incomplete coordinate pairs fail.

Only the user message becomes the model prompt. Gold answers, source rows, and
scoring metadata remain in the separate label/metadata fields. The prepared
manifest pins source and projected file hashes. Preparation runs no model.

## 2. Run in the GPU environment

Use a Miles installation at the pinned commit and install this repository into
the same Python environment. Both code checkouts, prepared panels, and the model
directory must be accessible to local Ray workers. The intended GPU is one H200.

**`--hf-checkpoint` must be a complete local HF model directory.** For the selected
Cartesian-only experiment, use the stock model and original tokenizer:

```bash
hf download Qwen/Qwen3.8-27B \
  --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --local-dir /models/Qwen3.8-27B
```

For historical atlas-trained evaluations, an adapter must instead be exported
with its language update, both learned atlas row tables, inherited visual weights,
and saved tokenizer/config. Adapter-only directories are rejected. This package
does not perform that merge or establish native-PEFT/SGLang numerical parity.

```bash
python -m sft.miles_eval.run run \
  --miles-root /root/miles \
  --hf-checkpoint /models/Qwen3.8-27B \
  --data-dir artifacts/generated/sft/miles_cartesian_h_eval_v1 \
  --output artifacts/runs/sft/miles-cartesian-h-stock-r02 \
  --execute
```

Omit `--execute` to validate the prepared data and print a reproducible invocation
without loading a tokenizer, starting Ray, or allocating a GPU. Execution checks
the Miles revision, hashes the local serving files, and verifies every prompt has
room for the full output allowance. No rows are silently filtered or truncated
to fit. Cartesian-only panels need no atomic atlas vocabulary; legacy panels
still require their 154 saved atlas token IDs. Output directories must be new.

The generated Miles flags include:

```text
--debug-rollout-only --num-rollout 0 --eval-interval 1
--rollout-num-gpus 1 --rollout-num-gpus-per-engine 1 --eval-num-gpus 0
--eval-prompt-data cartesian_h /absolute/path/to/prepared/cartesian_h.jsonl
--eval-input-key prompt --eval-label-key label --metadata-key metadata
--apply-chat-template --apply-chat-template-kwargs '{"enable_thinking":false}'
--n-samples-per-eval-prompt 1 --eval-temperature 0 --eval-top-k 1
--custom-rm-path sft.miles_eval.hooks.reward
--custom-eval-rollout-log-function-path sft.miles_eval.hooks.log_results
```

`--eval-num-gpus 0` shares the single inference engine; it does not disable eval.
`--train-backend fsdp` selects Miles' HF-aware configuration without Megatron
conversion. Under this upstream debug path, a lightweight trainer actor still
initializes CUDA/distributed state, but it returns before loading training weights
or an optimizer. This is not a multi-GPU FSDP training recipe.

## Results

The Modal wrapper selects a prepared dataset using `CATAN_MILES_DATASET` and
uses the active/explicit Modal profile. Its default dataset remains the historical
dense panel. For a fresh sparse-h run with the cached stock weights in `tetracorp`:

```bash
MODAL_PROFILE=tetracorp CATAN_MILES_DATASET=miles_cartesian_h_eval_v1 \
  uv run --no-sync python -m modal run --detach -m sft.miles_eval.modal_run \
  --run-name miles-cartesian-h-stock-20260921-r02
```

The wrapper performs CPU preflight before one H200 call (30-minute execution
limit, ten-minute startup limit, zero retries). Run names/output directories must
be fresh. Raw scores and the historical source panels are never overwritten.

The first completed stock-Cartesian run scored **143/200 (71.5%)**, with all
answers independently rescored. See the [run report](../../reports/sft/2026-09-21-stock-cartesian-eval.md).
The default Modal profile is now `tetracorp`; this completed historical run is
stored in `icebear5h` and its app is stopped.

`launch.json` records source/checkpoint identity, token lengths, and completion or
failure. Miles retains its raw sample dump under `debug/`. The log hook writes:

```text
results/eval-0/
├── summary.json
└── cartesian_h/
    ├── records.jsonl
    └── summary.json
```

Records include raw responses, original IDs/metadata/golds, exact scores,
completion status, truncation, and prompt/completion token counts. Summaries
include operation/family/representation accuracy and paired coordinate outcomes.
Every expected ID must appear exactly once. Malformed answers score zero;
truncated answers remain in the denominator and retain their strict answer score.
Aborted/unfinished or incomplete panels prevent a completion summary.

CPU contracts can be checked with `uv run --no-sync pytest tests/miles_eval -q`.
Real Miles/SGLang execution is verified on stock Qwen3.8-27B and one H200.
