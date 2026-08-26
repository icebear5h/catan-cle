# Modal cost model for the humans& NVFP4 RL recipe

Checked: 2026-08-16 22:36 UTC

## Scope

This memo prices the published Qwen3-30B-A3B ablation from:

- humans&: <https://humansand.ai/blog/nvfp4-rl>
- LMSYS/Miles companion: <https://www.lmsys.org/blog/2026-07-29-mxfp8-nvfp4-rl/>
- Modal pricing: <https://modal.com/pricing>
- Modal GPU allocation: <https://modal.com/docs/guide/gpu>

It does not claim that the published timings transfer to dense multimodal Qwen3.8-27B.

## Published workload

The companion post specifies:

- synchronous GRPO-style RL;
- Qwen3-30B-A3B (30.5B total, about 3.3B active parameters);
- DAPO-math-17k;
- eight rollout samples per prompt;
- maximum response length 8,192 tokens;
- eight B200 GPUs on one node;
- four GPUs for rollout and four for training;
- approximately 400 plotted rollout steps.

The articles do not publish a tabular total wall time or dollar cost. The phase-time means below were digitized from the thick smoothed lines in the published charts, so they are approximate.

## Modal list rates

Modal lists B200 at $0.001736 per GPU-second:

- one B200: $6.2496/GPU-hour;
- eight B200s: $49.9968/hour, effectively $50/hour.

Modal documents requests of up to eight B200s on the same physical machine. `B200+` may place the job on B200 or B300 while billing the B200 rate, but the image must also support B300/CUDA 13.1.

Base Function CPU and memory are additional:

- physical CPU core: $0.0000131/second = $0.04716/hour;
- memory: $0.00000222/GiB-second = $0.007992/GiB-hour.

For illustration, 64 physical cores plus 512 GiB add $7.11/hour. Sandbox/Notebook CPU and memory rates are higher than base Function rates. Explicit region pinning multiplies all resources by 1.5 for broad regions and 1.75 for narrow regions.

## Chart-derived synchronous cost

Assumption: rollout and train durations are exclusive sequential phases, and both four-GPU groups remain allocated throughout both phases. Under that assumption:

`cost = steps * (rollout_seconds + train_seconds) * 8 * $0.001736`

| Recipe | Rollout s/step | Train s/step | 400-step hours | GPU cost |
|---|---:|---:|---:|---:|
| BF16 train + BF16 rollout | 109.52 | 64.31 | 19.31 | $966 |
| MXFP8 high-precision backward | 93.44 | 49.52 | 15.88 | $794 |
| MXFP8 dequantized backward | 93.48 | 54.59 | 16.45 | $823 |
| MXFP8 end-to-end | 94.21 | 53.88 | 16.45 | $823 |
| NVFP4 high-precision backward | 85.09 | 65.95 | 16.78 | $839 |
| NVFP4 dequantized backward | 85.00 | 70.62 | 17.29 | $865 |
| NVFP4 4/6 high-precision backward | 84.18 | 65.95* | 16.68 | $834 |
| NVFP4 4/6 dequantized backward | 83.80 | 70.62* | 17.16 | $858 |

`*` The 4/6 chart adds rollout curves but not separate training curves. These rows provisionally reuse the corresponding base-NVFP4 training mean and are not measured end-to-end totals.

The measured implementation therefore suggests roughly 11-14% lower synchronous GPU cost for the final NVFP4 4/6 variants versus BF16, not a 4x reduction. MXFP8 high-precision backward is the cheapest plotted configuration because the current NVFP4 training path is slower even though its rollout is faster.

## Scale examples for the final 4/6 recipe

GPU-only, article topology and timings:

| Steps | High-precision backward | Dequantized backward |
|---:|---:|---:|
| 10 | $21 | $21 |
| 25 | $52 | $54 |
| 50 | $104 | $107 |
| 100 | $209 | $214 |
| 400 | $834 | $858 |
| 1,000 | $2,085 | $2,145 |

Five repeated 400-step 4/6-dequantized runs would be about $4,289 GPU-only. Reproducing the eight plotted configurations once each is about $6,800 GPU-only. The full research program contains more ablations than these eight curves, so its historical total cannot be reconstructed from the posts.

## Parallelism sensitivity

The published benchmark is synchronous. If a genuinely asynchronous pipeline overlaps trainer and sampler phases while retaining all eight GPUs, the ideal steady-state duration approaches `max(rollout, train)` rather than their sum. Ignoring fill/drain, weight synchronization, and staleness costs:

- BF16: about 12.17 hours and $608 for 400 steps;
- NVFP4 4/6 high-precision: about 9.35 hours and $468;
- NVFP4 4/6 dequantized: about 9.31 hours and $466.

These are optimistic pipeline bounds, not results reported by the article. They change the policy-staleness regime and therefore are not an apples-to-apples substitute for the synchronous learning curves.

If Modal could allocate only the active four-GPU phase and release the idle half instantly, the theoretical resource cost would be half the synchronous eight-GPU estimate. Miles normally keeps distributed trainer and rollout services resident, and repeated model startup/weight loading makes that elastic bound unrealistic without a purpose-built architecture.

## Excluded costs

The chart-derived totals exclude or may undercount:

- container startup, image pulls, CUDA/JIT compilation, and model loading;
- weight synchronization outside the two chart timers;
- reward computation, evaluation, checkpointing, failures, and retries;
- CPU, RAM, volume storage, and data transfer;
- Modal plan fees, credits, taxes, and explicit-region multipliers.

A practical budget for one 400-step article-style NVFP4 run is therefore about $1,000-$1,200 without explicit region pinning, and $1,250-$1,500 with modest retry/engineering allowance. Pinning a broad or narrow region alone moves the $834-$858 GPU subtotal to about $1,251-$1,287 or $1,460-$1,501 respectively, before CPU and RAM.

## Applicability to Qwen3.8-27B Catan RL

Only the Modal rate arithmetic transfers directly.

The published model is an MoE with about 3.3B active parameters per token, and the NVFP4 recipe targets routed MoE experts. Qwen3.8-27B is a dense native VLM with 27.78B active parameters, a vision encoder, 48 Gated DeltaNet layers, and 16 full-attention layers. The published kernels, phase times, and speedup percentages therefore cannot be assigned to Qwen3.8 without a new benchmark.

A naive `27.78 / 3.3` runtime multiplier is not defensible because attention, Gated DeltaNet state, vision encoding, communication, batching, and memory bottlenecks do not scale linearly with active parameter count. Conversely, Catan actions and reactions may use much shorter outputs than the article's 8,192-token maximum.

Use reservation costs until measured throughput exists:

| B200 count | Cluster rate | 4 hours | 12 hours | 24 hours |
|---:|---:|---:|---:|---:|
| 2 | $12.50/hour | $50 | $150 | $300 |
| 4 | $25.00/hour | $100 | $300 | $600 |
| 8 | $50.00/hour | $200 | $600 | $1,200 |

The first Qwen3.8 Catan RL benchmark should measure rollout tokens/second, optimized tokens/second, game/trajectory length, weight-sync time, and trainer/sampler utilization for 10-25 steps. Extrapolate only after that benchmark. GGUF is not a training format, and using a quantized rollout policy with a different-precision trainer requires explicit log-probability/mismatch handling.
