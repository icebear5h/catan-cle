# Miles SFT runtime

## Implementation plan

- [x] Inspect Miles commit `24ec7d8666c2c4ca9986addd197523b89d67621d`,
  its real Sample/rollout interfaces, hooks, LoRA saver, and HF export.
- [ ] Implement preencoded real-Sample ingestion and strict training configuration admission.
- [ ] Audit actual trainable parameters and collect finite-gradient/update witnesses.
- [ ] Validate native adapter shards and merged HF exports; finalize only after cursor save.
- [ ] Exercise CPU contracts and run the repository quality gate.

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
