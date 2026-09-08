# O-LoRA loader repair and smoke validation

Status: **passed**. The repaired loader completed the actual Qwen two-step
H200 smoke, checkpoint saves and production reload validation. Modal confirmed
the app stopped with zero tasks. Full training was not launched.

## Repairs

The stage-2 parent stores visual keys under `base_model.model.model.visual`.
The original O-LoRA loader attempted to restore them into a bare model with
`model.visual` keys, so strict loading failed before training.

`apply_frozen_adapter(..., restore_visual=True)` now restores the visual state
while the parent PEFT wrapper is attached. It promotes the visual module to
FP32 before copying, then merges the parent adapter and adds fresh adapters.
Strict state-key validation remains in place.

The loader also captures language/vision target lists before `get_peft_model`
mutates the module tree. Previously its report recomputed vision targets after
wrapping, when their names no longer matched, causing another startup failure.

The production launcher accepts `CATAN_HF_SECRET_NAME` (default `catan-hf`).
On `icebear5h`, set it to the verified existing `huggingface-secret-2`; this
avoids requiring a second copy of the secret under a different name.
The [full-launch dry-run receipt](../../artifacts/runs/sft/olora-smoke-20260907/full-launch-dry-run.json)
confirms successful planning with that secret: 14,770 training rows, 887
evaluation rows, rank 16 / alpha 32, batch 16 × accumulation 2, and 512 steps.
It records `dry_run=true` and `paid_gpu_requested=false`. This checks launch
configuration and data contracts; the two-example smoke batch does not prove
that batch 16 will fit or establish the full run's runtime/budget.

## Local validation

`tests/test_olora_bundle_integration.py` exercises actual PEFT wrapping with a
small random language model and visual projections, using both FP32 and BF16
base weights. It verifies:

- Exact FP32 restoration of visual weights that are not BF16-representable.
- Correct target counts, complete protected-basis coverage and trainable scope.
- Fresh-adapter logits equal the merged-parent reference.
- One optimizer update changes vision LoRA, language LoRA and token rows,
  while every frozen parameter remains unchanged.
- Checkpoint reload produces identical logits; corrupt visual keys still fail.

62 focused tests passed across trainer, integration, extractor, budget and
candidate-scoring tests. Ruff, Python compilation and `git diff --check` passed
for the repair and smoke runner. These CPU tests do not establish actual Qwen
throughput, memory requirements, or retention under training.

## Actual-model smoke

Runner: `sft/modal_olora_smoke.py`. Workspace: `icebear5h`.

Parent: terrain checkpoint-384 under
`catan-vision-sft/catan-qwen38-gauss-s2-terrain-20260904/398f0a023ec9/checkpoints/checkpoint-384`.

The new workspace initially lacked the Qwen3.8-27B cache. A CPU-only job downloaded revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` using its existing
`huggingface-secret-2` secret. No secret values are exposed or copied.

The GPU phase ran two optimizer steps on four short mixed-rung training rows
(node occupancy, edge owner, tile resource and port type), saved each step and
the final bundle, then invoked the production reload validation. It checked all
333 frozen visual tensors against the parent and verified effective updates
between checkpoints in all three trainable groups.

Results:

- Function elapsed time: **294.83 seconds**; trainer time: 36.20 seconds.
- Fresh rank-16 adapters: 496 language targets and 110 vision targets.
- Protected modules: **606**; unprotected modules: **0**.
- All **333** parent visual tensors remained FP32 and bit-for-bit unchanged
  in checkpoint-2. Parent and protection-factor provenance hashes matched.
- Changed tensors between steps 1 and 2: **220 vision LoRA**, **992 language
  LoRA**, and **2 token-row** tensors. Saved adapter tensors were finite.
- Production final-bundle reload: **valid**, with all **553** expected visual
  state tensors (333 frozen tensors plus 220 adapter tensors).
- Runtime versions matched the training pins: torch 2.13.0, transformers
  5.16.1, TRL 1.12.0, PEFT 0.20.0, datasets 5.0.1, accelerate 1.14.0.

Smoke app: [ap-SA7PXTGrW6AGfAfXs08dYc](https://modal.com/apps/icebear5h/main/ap-SA7PXTGrW6AGfAfXs08dYc).
Function call: `fc-01M1X63N2147XB0BAGAV2CSHGP`.
CPU cache-preparation app: `ap-uJEBxFV981bklkTS2fRpIi`.

Remote output: `catan-sft-runs:catan-vision-sft/olora-smoke-20260907`.
Local [result](../../artifacts/runs/sft/olora-smoke-20260907/result.json),
[launch receipt and source hashes](../../artifacts/runs/sft/olora-smoke-20260907/launch.json),
[initialization audit](../../artifacts/runs/sft/olora-smoke-20260907/frozen_bundle.json),
[trainable scope](../../artifacts/runs/sft/olora-smoke-20260907/trainable_parameters.json),
and [optimizer coverage](../../artifacts/runs/sft/olora-smoke-20260907/optimizer_coverage.json).

Limits: one H200, 16 CPU cores, 128 GiB RAM, 900-second function timeout,
300-second startup timeout, no configured retries. CPU preparation has a
1,200-second timeout, four CPU cores and 8 GiB RAM. Full training and held-out
retention evaluation are outside this smoke.

The production trainer retains its existing export convention: intermediate
checkpoints preserve FP32 visual state; the final export stores BF16 visual
state. Exact frozen-weight checks use checkpoint-2. The small-model logit
round-trip test likewise uses the FP32 checkpoint format.
