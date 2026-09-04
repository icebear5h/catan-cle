# Modal board-recognition training preflight

Date: 2026-08-27

No Modal job was launched for this estimate.

## Recommendation

Do not launch the 27B profile or a full epoch yet. The first paid path should be
Qwen3-VL-4B on one L40S, first for one optimizer update and then for 50 updates.
Use the measured 50-update throughput and peak memory to replace the ranges in
this report before authorizing a full epoch.

The direct slot-classifier remains the preferred architecture, but only its
dataset/DataLoader exists today. The current Modal implementation is an
autoregressive Qwen SFT trainer and therefore repeats each image 16 times.

## Current workload

The training split contains:

```text
states / unique images                  512
SFT rows                              8,192
queries per image                        16
image size                        1024x1024
approximate visual tokens / row        1,024
prompt characters                 204--313
answer characters                    1--17
effective batch                            8
microbatches / epoch                   8,192
optimizer updates / epoch              1,024
visual tokens / epoch              8,388,608
```

The visual-token estimate follows Qwen3-VL's effective 32x32-pixel token grid
and agrees with the observed hosted 1024px count of 1,026 image tokens.

The state-batch direct head would process only 512 image states per epoch, or
524,288 visual tokens, before supplying all 16 targets. The SFT projection is
therefore exactly 16x more expensive in repeated visual encoding before also
paying for the language decoder.

## Modal rates

Rates fetched from Modal's public pricing and resource documentation on
2026-08-27:

```text
L40S GPU                 $0.000542 / sec = $1.9512 / hour
H200 GPU                 $0.001261 / sec = $4.5396 / hour
physical CPU core         $0.0000131/sec = $0.047160/hour
RAM                      $0.00000222/GiB/sec = $0.007992/GiB/hour
```

Modal bills CPU and memory using the greater of requested and actual use. The
current vision launcher explicitly reserves:

```text
L40S: GPU + 8 CPU +  32 GiB = $2.5842/hour
H200: GPU +16 CPU + 128 GiB = $6.3171/hour
```

Thus CPU and RAM are not free. The current 12-hour function timeout is a
worst-case cap of approximately $31.01 for one L40S invocation or $75.81 for
one H200 invocation. Those timeouts are too high for a first smoke.

The current Starter plan advertises $30/month of compute credit. The estimates
below are gross usage before credits. The 349 MB corpus and model cache remain
well below Modal's currently included first TiB of Volume storage, so storage is
not the controlling cost.

Primary Modal sources:

- https://modal.com/pricing
- https://modal.com/docs/guide/resources
- https://modal.com/docs/guide/gpu

## Cost ranges

These are planning ranges, not benchmark results. They use the repository's
historical Qwen3.5-9B L40S smoke rate of 1.38 seconds per sample as an anchor,
then widen the range for 1024 visual tokens, full vision/merger backpropagation,
checkpoint I/O, and startup.

### Qwen3-VL-4B, full BF16 vision + merger, language LoRA, one L40S

```text
one-update smoke (8 microbatches)    $0.10--$0.75
50-update profile (400 rows)         $0.75--$2.00
one epoch (8,192 rows)               $7.00--$20.00
three epochs                        $21.00--$60.00
```

The estimated steady-state range is roughly 1--3 seconds per microbatch, or
2.3--6.8 compute hours per epoch before extra checkpoint/startup allowance. A
$25 hard budget is appropriate for the first full 4B epoch only after the
50-update profile supports it.

A 48 GiB L40S is a plausible fit, not yet proven. The 4B BF16 weights are about
8 GB; gradients and Adam states apply only to the roughly 0.3--0.4B visual and
merger parameters, while activation memory remains the major uncertainty. The
official ms-swift Qwen3-VL-4B language-LoRA example reports 21 GiB/GPU with a
frozen visual path. Our full visual path should be profiled with batch one and
gradient checkpointing before increasing the batch.

### Qwen/Qwen3.8-27B full visual profile, one H200

```text
one-step/cold-load smoke               $0.75--$3
50-update throughput profile              $3--$10
one epoch                                $30--$95
three epochs                             $90--$285
```

This range is intentionally wide because no native 27B throughput result exists
for this corpus. The present 12-hour H200 timeout can terminate before one epoch
and still bill about $76. Qwen3.8 Max, used in the hosted evaluation, is not the
same artifact as the open 27B checkpoint. The Max result is not a reason to pay
for the 27B run without a separate baseline.

### Direct slot classifier

A frozen-tower feature-cache pass sees 512 train images once, and subsequent
head training is small. A first 4B/L40S cache-and-head pilot should be on the
order of $1--$5 including model startup. End-to-end visual tuning with the
direct head should still be substantially cheaper than row-wise SFT because it
omits the language decoder and removes 16x repeated visual encoding. This is a
lower-confidence estimate because the direct-head Modal trainer has not yet
been implemented.

## Kernel recommendation

### Qwen3-VL-4B

Required baseline:

- BF16;
- gradient checkpointing;
- TF32 matmul;
- PyTorch SDPA as a known-safe fallback;
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`;
- fixed 1024px image bounds and batch size one.

Profile after the safe smoke:

- FlashAttention 2 for the language attention path;
- Liger Kernel 0.7.0 for Qwen3-VL fused operations/loss.

Qwen's model card recommends FlashAttention 2 for speed and memory savings. The
official fine-tuning framework lists `flash_attn==2.7.4.post1`, Triton 3.2, BF16,
and gradient accumulation. The pinned 2U1 trainer supports dense Qwen3-VL and
defaults `use_liger_kernel=true`.

The current Modal image, however, installs neither `flash-attn` nor
`liger-kernel`; both launchers explicitly disable them. Do not simply flip the
flags. The current image uses Torch 2.8.0, Transformers 5.3.0, and Triton 3.4.0,
so kernel wheels/imports and one real forward/backward must pass in a pinned
CUDA image first. Keep SDPA as the control and accept a kernel only if it
preserves loss while improving measured step time or peak memory.

### Qwen3.8 / Qwen3.5 hybrid architecture

The hybrid decoder uses Gated DeltaNet layers. Its useful fast path needs both:

- `flash-linear-attention` (`fla`);
- `causal-conv1d`.

The repository's previous Qwen3.5 run lacked both and used the slower Torch
fallback. Liger is explicitly disabled by the pinned trainer for `qwen3_5`, and
that trainer recommends disabling FlashAttention 2 after observed CUDA errors.
A 27B launch is a no-go until FLA and causal-conv1d import and execute on the
actual H200 image, with no fallback warning.

### Not recommended for this run

- Do not quantize a trainable vision tower. The native visual profiles correctly
  use BF16 rather than QLoRA.
- Do not add xFormers, Unsloth, or `torch.compile` to the first controlled run.
  They add a second trainer/runtime variable without addressing the current
  bottlenecks.
- Do not enable image/text packing in this pinned trainer without proving that
  multimodal boundaries and masks remain correct.

Primary Qwen sources:

- https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-finetune/README.md
- https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
- https://github.com/2U1/Qwen-VL-Series-Finetune/tree/130ad7ccafe06a2ae5377bad9d512027d8225dc5

## Pre-launch blockers

1. **Image paths do not currently resolve through the Modal uploader.**
   `qwen_sft/train.jsonl` contains `images/...`, but from that nested file the
   existing resolver looks for `qwen_sft/images/...`; the real images are one
   directory higher. The local preflight currently reports `exists=False`.
2. **Production checkpoint cadence is unsafe.** The one-step smoke default
   `save_steps=1` would serialize and rotate visual weights on every one of the
   1,024 optimizer updates in a full epoch. Use approximately 128--256 steps,
   while retaining step one for the smoke.
3. **The kernel diagnostic is CPU-only.** `check_fast_kernel_deps_remote` has no
   GPU allocation, so it cannot validate CUDA execution, compatibility, or
   speed.
4. **No peak-memory/phase timing receipt exists.** Record model-load,
   preprocessing, forward/backward, optimizer, checkpoint, maximum allocated
   VRAM, maximum reserved VRAM, and total container wall time.
5. **The direct-head training path is not implemented.** The DataLoader exists,
   but the current Modal launchers consume conversation rows only.
6. **A full run has no in-training validation path configured.** Reserve the
   1,024 validation rows for exact classification evaluation and do not train on
   validation/test.

## Safe launch ladder

1. Fix and revalidate portable image paths locally.
2. Add a real GPU kernel/import/forward/backward preflight.
3. Keep SDPA and Liger off for the first one-update 4B smoke; cap it at 30
   minutes (about $1.29 at the current reserved L40S rate).
4. Verify adapter, non-LoRA visual state, tokenizer, trainable-scope audit, and
   reload evaluation.
5. Run 50 optimizer updates (400 examples), once with SDPA and once with the
   candidate FlashAttention/Liger image if the kernel preflight passes. Cap each
   at $5.
6. Compute the full forecast from measured values:

```text
epoch compute seconds = 8,192 / measured samples_per_second
optimizer updates      = ceil(8,192 / 8) = 1,024
cost                    = total wall seconds * $2.5842 / 3,600
```

7. Authorize at most one 4B epoch with a $25 cap. Evaluate before adding epochs.
8. Do not authorize the 27B/H200 run until the 4B run shows a real held-out
   perception gain and the 27B fast-kernel profile provides an exact forecast.
