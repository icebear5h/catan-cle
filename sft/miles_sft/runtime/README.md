# Miles SFT runtime

## Implementation plan

- [x] Inspect Miles commit `24ec7d8666c2c4ca9986addd197523b89d67621d`,
  its real Sample/rollout interfaces, hooks, LoRA saver, and HF export.
- [x] Implement preencoded real-Sample ingestion and strict training configuration admission.
- [x] Audit actual trainable parameters and collect finite-gradient/update witnesses.
- [x] Validate native adapter shards and merged HF exports; finalize only after cursor save.
- [x] Exercise CPU contracts and run the repository quality gate.

The full-checkpoint merge (including historical token rows), launcher, CLI,
image/Bridge pin, and preflight are owned by main. This package owns runtime
verification of fresh standard fused rank-16/alpha-32 language-only adapters.

## Upstream contracts inspected

All Miles paths below refer to commit `24ec7d8666c2c4ca9986addd197523b89d67621d`:

- `miles/rollout/sft_rollout.py`, `base_types.py`, `data_source.py`, and
  `inference_rollout/compatibility.py`: synchronous legacy-signature functions
  are automatically adapted by the default v2 loader; no legacy environment
  switch is necessary. Return the real `RolloutFnTrainOutput`.
- `miles/utils/types.py`: real Sample status is `Sample.Status.COMPLETED`;
  loss masks cover the response suffix, not the full token sequence.
- `miles/backends/megatron_utils/initialize.py`: `custom_init(args)` runs after
  distributed setup and before model construction.
- `miles/backends/megatron_utils/model.py`: before-step hook runs after zeroing
  gradients; optimizer.step returns `(update_successful, grad_norm, num_zeros)`;
  train_one_step returns `(loss_dict, grad_norm, outcome)` after zeroing gradients.
- `miles/backends/megatron_utils/actor.py`: post-save hook is
  `(args, rollout_id, checkpoint_dir, hf_checkpoint_dir)`, on rank zero only.
- `miles/backends/megatron_utils/checkpoint.py` and `lora/utils.py`: native
  adapter shards and training state live in `iter_NNNNNNN/adapter/`.
- `miles/backends/megatron_utils/hf_export.py`: merged HF export uses Bridge,
  and writes `.complete` only after export calls finish. Native saver can
  swallow PEFT export errors, so runtime must verify actual artifacts.
- `train.py`: trainer save precedes rollout data-source save. Post-save evidence
  alone cannot certify a complete checkpoint/run.

## Launcher integration

Use these exact CLI/callable pairs:

| Miles flag | Callable |
| --- | --- |
| `--rollout-function-path` | `sft.miles_sft.runtime.rollout.generate_rollout` |
| `--custom-megatron-init-path` | `sft.miles_sft.runtime.hooks.initialize` |
| `--custom-megatron-before-train-step-hook-path` | `sft.miles_sft.runtime.hooks.before_train_step` |
| `--custom-megatron-post-save-hook-path` | `sft.miles_sft.runtime.hooks.post_save` |

Signatures:

```python
generate_rollout(args, rollout_id, data_source, evaluation=False)  # real RolloutFnTrainOutput
initialize(args) -> None
before_train_step(args, rollout_id, step_id, model, optimizer, opt_param_scheduler) -> None
post_save(args, rollout_id, checkpoint_dir, hf_checkpoint_dir) -> None
```

After **successful Miles driver exit**, main calls
`sft.miles_sft.runtime.receipts.finalize_run(Path(receipt_dir))` to validate the
complete run and write/return `run.json`. It raises if cursor state, any rank's
step evidence, or final native/HF artifacts are missing or inconsistent.

Set **`MILES_SFT_RECEIPT_DIR`** to a fresh, shared absolute directory in both
trainer workers and the launcher. This is the only custom required environment
variable. No `MILES_USE_LEGACY_ROLLOUT_V1` setting is required: the default loader
adapts the callable. The imported Miles checkout must retain Git metadata and
have HEAD exactly at the pinned commit; runtime resolves HEAD from the module
actually imported. Bridge installation/pin adaptation belongs to preflight.

Required launch constraints are checked by `admission.validate_args`:

- `--train-backend megatron --megatron-to-hf-mode bridge --debug-train-only
  --lora-train-only --lora-type lora --lora-rank 16 --lora-alpha 32`.
- **`--start-rollout-id 0` explicitly**: Miles' HF loader reports iteration 0;
  without this override the controller would choose start rollout 1.
- `--load` and `--hf-checkpoint` must name the same local merged full base.
  No `lora_adapter_path`/old adapter import; LoRA B must be zero-initialized.
- `--target-modules` must equal `runtime.REQUIRED_TARGETS`: exact
  `language_model.decoder.layers.*.` paths for `self_attention.linear_qkv`,
  `self_attention.linear_proj`, `self_attention.in_proj`, `self_attention.out_proj`,
  `mlp.linear_fc1`, and `mlp.linear_fc2`. No exclusions or canonical/split LoRA.
- Global dataset, `messages` input, `metadata` metadata, one sample per prompt;
  no upstream chat-template application or prompt-length filtering. `seq_length`
  is the complete encoded-sequence cap. Batch size is divisible by global batch size.
- `sft_loss`, per-token loss, disabled advantages/returns; no MTP, critic, KL,
  teacher, old-actor, or eval loop; no rollout/eval GPUs.
- One actor node and one tensor-parallel replica (TP divides 16); PP/CP/EP=1,
  no virtual pipeline/independent-DP/fully-async/overlapped parameter gathering.
- Native `--save`, per-rollout `--save-hf`, and optimizer state saving are required.
  Final save must cover `num_rollout - 1`.

## Evidence and limits

### Complete HF composition

Configure `--save-hf` as `OUTPUT/exports/rollout-{rollout_id}/bridge`. The
post-save hook checks that exact configured raw path and assembles the sibling
`OUTPUT/exports/rollout-N/model` as the final, complete HF checkpoint.

```python
from sft.miles_sft.export import assemble_complete_export, validate_complete_export

manifest_path = assemble_complete_export(base, bridge_export, bridge_export.parent / "model")
manifest = validate_complete_export(base, bridge_export.parent / "model")
```

All arguments are `Path`s. Assembly returns the new `composition_manifest.json`
path. It requires a fully validated merged base with its sealed `merge_manifest.json`;
the source merge manifest is the authority for the trained-key and asset inventories.
Production has 496 trained HF keys, selected by `operation == "lora_fp32_then_bf16"`.
Every selected key must exist in Bridge with the base shape/dtype. Unexpected
Bridge tensor keys fail. Bridge may omit frozen tensors or round frozen FP32 values.

Composition processes one base-shaped shard at a time, replacing **only** admitted
trained keys. All other tensors—including MTP, FP32 visual weights, complete
embedding/head matrices with historical token rows, and norms—come directly from
the base. Source and serialized output tensor bytes are SHA256-compared without
dtype conversion. The HF index is rebuilt in full; config, tokenizer, template
(including nested `chat_templates/`), and processor assets are byte-copied from the
base manifest's asset/output inventory. Bridge's reserialized assets are not used.

Outputs must be fresh. Raw Bridge files are read-only. A failed assembly can leave
an incomplete output directory, and retries reject that directory. Base merge
manifests/markers are not copied into the new model. The new composition manifest
records the base manifest hash, source/output file SHA256s, and per-tensor
source selection, shape, dtype, shard, and source/output byte hashes. Its own
`.complete` seal is written last, after payload validation.

Checkpoint receipts record `bridge_checkpoint_dir` separately from final
`hf_checkpoint_dir`, and `hf.composition_manifest_sha256` pins composition identity.
`finalize_run` derives both paths from the configured raw path and revalidates the
final export against this pinned digest; both paths are also top-level fields in
the returned `run.json`. `validate_complete_export` additionally accepts keyword
arguments `bridge_export` and `expected_sha256` for these bindings. Validation
checks coverage/dtypes, manifest-authorized frozen/trained identities, byte-identical
asset identities, and all composed/raw tensor-file hashes. This relies on identity
established during composition and its externally pinned manifest hash; it is not
an authenticity claim about an untrusted, independently resealed manifest.

Offline admission/audits/artifact validation do not import Miles or Megatron.
Only `rollout.py` and `hooks.py` bind real remote modules at module import time;
there are no fake Sample classes, fake adapter classes, or fallback loaders.

Every before-step audit visits actual parameters, checks A/B pairs on every
admitted target, and rejects any trainable vision/MTP/full-token/base parameter.
The first step also checks finite nonzero A / zero B; real Bridge adapter objects
must report dimension 16 and alpha 32. Scope is checked on subsequent steps.

The wrapper clones **only adapter tensors**, reads `main_grad` before `.grad`,
requires finite gradients and nonzero gradients in every target family (including
GDN), calls the real optimizer once, and checks finite updated adapter tensors.
It returns optimizer/train-step results unchanged. Loss receipts contain the
real forward loss reported by Miles (before that update), not an invented
post-update loss. A zero-update warmup step is recorded honestly; each rank must
show an actual update somewhere before run finalization succeeds.

Files are exclusive, flushed writes; a reused directory or interrupted partial
JSON fails closed. Before/after step receipts distinguish failed/skipped steps.
Checkpoint receipts stay `complete: false`, `model_saved_cursor_pending` even
when model artifacts pass. Finalization rechecks native adapter tensor
names/shapes/finiteness, optimizer/scheduler state and hashes, the complete HF
composition described above, and cursor counters against all completed rollouts.
The 27B base shards are hash-checked as they are consumed during composition;
finalization binds to that base manifest without rehashing the full base. Large
raw/final artifact hashing occurs only at export/finalization, never each step.

## Verification review

- `uv run --no-sync python -m pytest tests/miles_sft/test_runtime.py -q`:
  **6 passed, 1 skipped**. Tests cover a real CPU Torch backward/SGD update,
  Megatron `main_grad` precedence, missing/nonfinite gradients, skipped optimizer
  rejection, trainable scope, frozen matrices, GDN coverage, native serialization,
  and cursor-gated finalization. The real-Miles Sample integration test is skipped
  because Miles is not installed locally; it never substitutes a fake module.
- `uv run --no-sync python -m scripts.quality`: **passed**, including strict mypy,
  Ruff, the 300-line source cap and 15-direct-file cap.
- Remote Miles/Megatron/Bridge execution remains the main-owned runtime preflight.

Focused composition repair review:

- `uv run --no-sync python -m pytest tests/miles_sft/test_export.py tests/miles_sft/test_runtime.py -q`:
  **22 passed, 1 skipped** (real Miles is not installed locally).
- Real safetensors tests exercise different base/Bridge shard layouts, dropped MTP,
  changed/rounded frozen tensors, exact asset copies, shape/dtype/key admission,
  fresh-only output, incomplete-source rejection, payload/manifest tampering, and
  raw/final path binding through cursor-gated run finalization.
- Scoped Ruff and strict mypy pass for the repair. The repository-required quality
  check passes structure and mypy; Ruff reports unrelated import-order failures in
  `tests/traces_journal/test_calls.py`, `test_commands.py`, and `test_transactions.py`.
