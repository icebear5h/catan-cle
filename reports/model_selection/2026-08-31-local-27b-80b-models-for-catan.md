# Local 27B–80B models for a context-heavy Catan policy

Checked: **2026-08-31 UTC**

This audit covers publicly downloadable, official model weights with roughly
27B–80B **total** parameters. For MoE models, both total and active parameters
are reported because total parameters drive resident weight memory while active
parameters are a better, but incomplete, proxy for per-token compute.

The primary policy path is assumed to be symbolic text generated from the Catan
engine. Native vision is treated as an optional capability, not the source of
truth for board state. Quantizations and community merges are deployment
formats, not separate model-training candidates.

## Executive answer

There is no defensible single winner before a matched Catan evaluation. The
first bakeoff should cover six distinct roles rather than pretend an ordinal
internet ranking exists:

- **Integrated incumbent:** `Qwen/Qwen3.8-27B`.
- **Local context-rich agent:** `meta-models/Muse-Glimmer-30B`.
- **Documented text action/environment training:**
  `ibm-granite/granite-4.2-30b`.
- **Published long-context plus native-vision evidence:**
  `google/gemma-4-31B-it`.
- **Fully open post-training control:** `allenai/Olmo-3.1-32B-Instruct` and
  `...-Think`.
- **Higher-compute text prior/teacher control:**
  `nvidia/Llama-3_3-Nemotron-Super-49B-v1_5`.

Add these specialized ablations:

- `Qwen/Qwen-AgentWorld-35B-A3B` for language-world-model and state-transition
  training, not as the authoritative Catan environment.
- `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` for cheap multi-seat rollouts.
- `moonshotai/Kimi-Linear-48B-A3B-Instruct` for extreme long-context efficiency.
- `ByteDance-Seed/Seed-OSS-36B-Instruct` as a native-512K full-attention control.

The provisional default remains **Qwen3.8-27B**, because it is dense, native
multimodal, Apache-2.0, already integrated into this project, and unusually
memory-efficient for four long-lived seats. That is an engineering default, not
a claim that it has the strongest Catan prior. Its exact post-training recipe is
not public, and existing project evidence shows significant Catan weaknesses.

## Two naming corrections

### Qwen3.7 is not a local checkpoint

`qwen3.7-plus`, `qwen3.7-flash`, and `qwen3.7-max` are hosted API/service model
IDs. No official downloadable `Qwen/Qwen3.7-*` weight repository, config, or
license was found.

The local open-weight models closest to that generation are:

- `Qwen/Qwen3.5-27B`
- `Qwen/Qwen3.5-35B-A3B[-Base]`
- `Qwen/Qwen3.6-27B`
- `Qwen/Qwen3.6-35B-A3B`
- `Qwen/Qwen3.8-27B`
- `Qwen/Qwen3-Next-80B-A3B-{Instruct,Thinking}`

Sources:
[Alibaba model catalog](https://www.alibabacloud.com/help/en/model-studio/models),
[Qwen3.8 repository](https://github.com/QwenLM/Qwen3.8), and
[Qwen Hugging Face organization](https://huggingface.co/Qwen).

### “80% context / 20% generation” is not a model class

The phrase can mean three different things:

1. **Window allocation.** Use 80% of `prompt + output` capacity for input and
   reserve 20% for reasoning, the final answer, retries, and tool results.
2. **Serving workload.** Eighty percent of processed tokens are prefill and 20%
   are decode. This affects TTFT, decode latency, KV/state memory, and the value
   of prefix caching.
3. **Information source.** The decision should be driven mostly by authoritative
   in-context evidence, while weights contribute strategic priors.

None of these ratios describes how a model was necessarily trained. Official
cards almost never disclose a stable prompt-to-target token ratio across all
post-training stages.

For Catan, reserving 20% of a 256K window would allow roughly 51K output tokens,
which is far too much for one action. The useful policy is:

- treat 70–80% occupancy as a **compaction trigger**;
- reserve enough room for a bounded reasoning pass and retry;
- generate a few hundred to a few thousand tokens, not 20% of the window;
- retain the complete authoritative event ledger outside model-authored memory.

The existing stateless Qwen3.8 replay run used 164,201 input and 113,004 output
tokens across 111 scored decisions—about 59% input and 41% output—because it
requested verbose visible rationales. That is evidence that output policy, not
window size, controls the realized ratio. Source:
`data_pipeline/training/reasoning/pilots/_2n5F2DxtPI/action_selection_diff/qwen3_8_27b_blue_20260817/summary.json`.

## Evidence conventions

The report uses four evidence levels:

- **Exact:** an official source assigns the stage to the exact checkpoint.
- **Family-level:** an official source describes a family recipe, but does not
  prove every stage was applied identically to this revision.
- **Config-derived:** architecture/context follows from the released config.
- **Undisclosed:** no primary source establishes the claimed stage or optimizer.

An instruct model is not “SFT-only” merely because a downstream derivative added
only SFT. Its parent may already contain rejection sampling, DPO, RLHF, or RLVR.
Likewise, a student distilled from an RL teacher is not student-side RL unless it
was itself rolled out and optimized against rewards.

## What post-training stages actually buy

- **Continued pretraining or mid-training:** internalizes domain language,
  transitions, and long-context distributions. This is the closest stage to
  building a Catan-specific prior.
- **Instruction SFT:** teaches response behavior, demonstrations, action schemas,
  and tool protocols. It does not by itself optimize game return.
- **Distillation/rejection sampling:** transfers selected teacher behavior into
  weights. It can create strong “out-of-the-box” intuition, but also transfers
  the teacher's errors and style.
- **DPO and related preference methods:** improve relative preference,
  instruction following, tone, and response selection. They are not environment
  interaction training.
- **RLVR:** optimizes against verifiable rewards. Math/code RLVR transfers only
  partially to Catan; engine-grounded Catan RLVR would be directly relevant.
- **Agentic/environment RL:** trains multi-turn action → observation → recovery
  loops. This is more structurally relevant than static benchmark RL.
- **Model merging:** combines checkpoints but makes causal attribution harder.
- **Pruning/NAS plus distillation:** transfers a larger model's behavior into a
  smaller deployment shape; useful for strong priors per resident byte.

## Primary candidates, named by what was actually done

### Qwen3.8-27B — opaque-post-trained dense hybrid VLM

**Identity**

- ID: [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B)
- Size: about 27.8B dense parameters; all language parameters are active.
- Modality: text, image, and video input; text output.
- License: Apache 2.0.
- Context: 262,144 native; supported to one million with static YaRN.
- Architecture: 64 language layers: 48 Gated DeltaNet and 16 gated full-attention
  layers, plus a native vision encoder.

**Documented lineage**

`multimodal pretraining → post-training → released checkpoint`

The exact CPT, SFT, preference, distillation, RL, and merge sequence is
**undisclosed**. Qwen's family material discusses scaled RL and million-agent
environments, but that does not establish the exact Qwen3.8-27B recipe. The
older Qwen3 four-stage pipeline must not be copied onto this checkpoint.

**Designed strength**

A single local model for reasoning, tools, coding/agents, images, video, and long
context. The hybrid stack reduces attention KV growth while preserving periodic
full-attention retrieval. It supports thinking on/off, reasoning effort, and
historical thinking preservation.

**Catan judgment**

Best integrated prototype, but not a proven Catan prior. Existing project runs
show why:

- only 28/89 nontrivial action choices matched one human replay; human agreement
  is not optimality, but the gap is large;
- one opening response called `SHEEP + WOOD + WOOD` three distinct resources;
- another assumed a setup road selected the later second-settlement location,
  which is false under Catan setup rules;
- the same run returned valid recoverable actions but had 35 format warnings;
- visual board runs remained weak on exact node/road binding.

Use `preserve_thinking=False` or explicit structured memory for persistent seats;
old private reasoning is a stale hypothesis, not authoritative state.

### Muse Glimmer 30B — teacher-distilled local-agent VLM

**Identity**

- ID: [`meta-models/Muse-Glimmer-30B`](https://huggingface.co/meta-models/Muse-Glimmer-30B)
- Size: about 29.6B total, including a roughly 1.8B perception encoder; dense.
- Modality: text and image input; text output.
- License: Apache 2.0.
- Context: 131,072+.
- Architecture: 52 language layers with repeating local/local/local/global
  attention, a 2,048-token local window, and only two KV heads.

**Documented lineage**

`Muse Spark logit-distilled pretraining → long-context/agent-heavy mid-training
→ post-training combining SFT, on-policy distillation, and RL`

Meta separately documents safety SFT/RL as train-time mitigations, but does not
establish them as a final chronological stage. It does not disclose the complete
datasets, reward construction, or exact RL optimizer.

**Designed strength**

Local autonomous agents that consume private context, call tools, recover from
failure, and use images. Official 4-bit packages target 24–32GB systems. Its
local/global attention and low KV-head count are attractive for four concurrent
seat histories.

**Catan judgment**

This is the closest product-design match to “feed it a lot of context and let it
act.” It deserves a top-three test. It is newer and less independently tested
than Qwen, and its agent training targets personal/computer tasks rather than
stochastic, adversarial games.

Source: [Meta training overview](https://research.meta.ai/blog/introducing-muse-glimmer-open-agentic-model).

### Granite 4.2 30B — multi-environment GRPO/RLHF text agent

**Identity**

- ID: [`ibm-granite/granite-4.2-30b`](https://huggingface.co/ibm-granite/granite-4.2-30b)
- Size: 30B dense.
- Modality: text only.
- License: Apache 2.0.
- Context: 128K native; IBM describes a 512K extension.

**Documented lineage**

The public provenance wording is inconsistent: the checkpoint card names
`Granite-4.1-30B-Base` as its base, while IBM's same-generation training account
says the Granite 4.2 models were pretrained from scratch on roughly 15T tokens.
The downstream chronology is clear:

`base checkpoint → broad SFT → second agent-upsampled 30B SFT → RLVR ×3 →
instruction/code boosters → SWE RL → terminal RL → search RL → final RLHF`

Every RL stage uses asynchronous GRPO and warm-starts the next stage. Rewards
include exact verifiers, unit tests, structured-output checks, outcome rewards,
LLM judges, and a final generative preference/safety reward model.

**Designed strength**

Exact tools, structured outputs, multi-turn environment interaction, recovery,
and agentic reasoning. It has the most directly inspectable post-training design
for the intended Catan control loop.

**Catan judgment**

Strongest text-only action-policy comparator in the proposed bakeoff and the
best-documented general agentic-RL template to adapt and test for Catan. It still
has no learned Catan belief or negotiation objective, and its full-attention
cache is more expensive than Qwen's or Muse's for four long histories.

Source: [IBM's complete training account](https://huggingface.co/blog/ibm-granite/granite-4-2).

### Gemma 4 31B IT — sparse-global long-context multimodal model

**Identity**

- IDs: [`google/gemma-4-31B`](https://huggingface.co/google/gemma-4-31B) and
  [`google/gemma-4-31B-it`](https://huggingface.co/google/gemma-4-31B-it)
- Size: 31B-class dense; official totals vary slightly depending whether
  embeddings, vision, and drafter components are counted.
- Modality: text/images; video represented as frames.
- License: Apache 2.0.
- Context: 256K native.
- Architecture: local sliding-window attention with periodic global attention,
  variable image-token budgets, and a native thinking mode in `-it`.

**Documented lineage**

- Base: multimodal pretraining only.
- IT: base → Gemma-3-like instruction post-training with thinking behavior.

The exact SFT corpus, preference optimizer, RL method, teacher distillation, and
merge stages for Gemma 4 are **undisclosed**. Gemma 3's named BOND/WARM/WARP
recipe must not be automatically transferred to Gemma 4.

**Designed strength**

Long documents, OCR, pointing, charts, images/video frames, function calls, and
reasoning. Google reports RULER, LOFT, and multi-needle MRCR results, giving it
broader long-context evidence than Qwen3.8's current public card. These remain
vendor-reported evaluations.

**Catan judgment**

Primary multimodal/context challenger. In one strict 60-question hosted
raw-image cohort, Gemma led the four tested open VLM families at 28/60; that was
a complete-stack test with formatting confounds, not a universal comparison or
a reliable board parser result. Tied embeddings and chat/thinking-template
behavior make selective token adaptation less straightforward than Qwen.

Sources: [Gemma 4 report](https://arxiv.org/html/2607.02770v2) and
[official model card](https://ai.google.dev/gemma/docs/core/model_card_4).

### OLMo 3.1 32B — fully open SFT → DPO → RLVR model flow

**Identity**

- IDs: `allenai/Olmo-3-1125-32B`,
  `allenai/Olmo-3.1-32B-Instruct`, and
  `allenai/Olmo-3.1-32B-Think`.
- Size: 32B dense; text only; Apache 2.0.
- Context: 65,536.

**Documented lineage**

`5.5T pretraining → two 100B mid-training branches and merging → 100B
long-context training and checkpoint merging → branch-specific SFT → DPO →
RLVR`

Instruct and Think have separate SFT/DPO/RL paths. Ai2 releases datasets, code,
intermediate checkpoints, and model-flow details.

**Designed strength**

Scientific control over the model lifecycle. It is the clearest top candidate
for comparing base, SFT, DPO, and RLVR effects without reconstructing an opaque
vendor pipeline.

**Catan judgment**

Best research substrate. Use Instruct for concise deployed actions and Think for
offline planning/teacher ablations. Its shorter context is still adequate for a
compact Catan trajectory, but full-attention KV and verbose reasoning make raw
unbounded history expensive.

Sources: [OLMo 3 paper](https://arxiv.org/abs/2512.13961),
[Think card](https://huggingface.co/allenai/Olmo-3.1-32B-Think), and
[Instruct card](https://huggingface.co/allenai/Olmo-3.1-32B-Instruct).

### Seed-OSS 36B — native-512K dense context model, opaque alignment

**Identity**

- IDs: `ByteDance-Seed/Seed-OSS-36B-Base`, `...-Base-woSyn`, and
  `...-Instruct`.
- Size: 36B dense; text only; Apache 2.0.
- Context: trained up to 512K natively.

**Documented lineage**

The standard Base includes synthetic instruction data during pretraining;
Base-woSyn removes it. Instruct is post-trained from Base. The safety disclosure
names SFT and RLHF/PPO, but the complete Instruct chronology, data, rewards, and
hyperparameters are **undisclosed**.

**Designed strength**

Full global attention over an unusually long trained window, adjustable thinking
budget, reasoning, coding, and agent tools. It is a clean control against hybrid
recurrent models.

**Catan judgment**

Strong context-stress candidate. Its exact global access may help event and trade
retrieval, but four-seat KV memory grows linearly and rapidly. Base-woSyn is an
interesting scientific control, not a ready policy.

Sources: [official repository](https://github.com/ByteDance-Seed/seed-oss) and
[Instruct card](https://huggingface.co/ByteDance-Seed/Seed-OSS-36B-Instruct).

### Falcon-H1 34B — trained-long-context SFT → DPO hybrid

**Identity**

- IDs: `tiiuae/Falcon-H1-34B-Base` and `...-Instruct`.
- Size: 33.6B dense; text only; Falcon-LLM License.
- Context: trained to 256K.
- Architecture: all 72 blocks combine parallel Mamba-2 and attention.

**Documented lineage**

`approximately 18T base pretraining + explicit 32K/128K/256K stages → 3GT SFT
at 16K → 3GT SFT at 128K → normalized DPO`

**Designed strength**

Long-context language with a comparatively conservative alignment recipe and
broad deployment/fine-tuning support.

**Catan judgment**

Best SFT+DPO control for testing whether heavy reasoning RL has harmed general
language or negotiation. Because every block still has attention, its KV cache
is not as small as “Mamba hybrid” might imply.

Sources: [Falcon-H1 report](https://arxiv.org/abs/2507.22448) and
[post-training details](https://github.com/tiiuae/Falcon-H1/blob/main/docs/post_training_details.md).

## Efficient and high-capacity specialists

### Nemotron 3 Nano 30B-A3B — SFT → RLVR → RLHF → RLVR hybrid MoE

- ID: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`.
- Size accounting: about 31.6B total and 3.2–3.6B active depending whether
  embeddings are counted.
- Architecture: Mamba-2/MoE plus six GQA attention layers.
- Context: 256K default; vendor evaluations/support up to one million.
- License: NVIDIA Open Model License; marked for commercial use.
- Lineage: `pretraining → SFT → initial RLVR → RLHF with a generative reward
  model → second RLVR`; synchronous GRPO is documented.

**Use case:** inexpensive multi-seat rollout and explicit post-training research.
**Risk:** 3B active compute may cap nuanced negotiation and opponent modeling.

Sources: [model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)
and [technical report](https://arxiv.org/abs/2512.20848).

### Kimi Linear 48B-A3B — multi-stage SFT → RLVR with PTX protection

- IDs: `moonshotai/Kimi-Linear-48B-A3B-{Base,Instruct}`.
- Size: 48B total, 3B active; MIT; text only.
- Architecture: 20 Kimi Delta Attention plus seven global MLA layers.
- Context: one million.
- Lineage: `5.7T pretraining → multi-stage broad/reasoning SFT → math/code/STEM
  RLVR with concurrent PTX/SFT loss`.

**Use case:** longest-context and recurrent-state efficiency experiment.
**Risk:** exact old-event copying and belief retention are empirical; public RULER
figures vary across report settings, and the serving stack uses custom KDA/FLA
kernels.

Sources: [model card](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct)
and [report](https://arxiv.org/abs/2510.26692).

### Jamba2 Mini — long-document mid-training → SFT → DPO → on-policy RL

- ID: `ai21labs/AI21-Jamba2-Mini`.
- Size: 52B total, 12B active; text only; Apache 2.0.
- Architecture: 28 Mamba and four attention layers, with dense and MoE blocks.
- Context: 256K.
- Lineage: `Jamba 1.5 initialization → 500B-token long-document mid-training
  with state passing → cold-start SFT → DPO → multiple on-policy RL phases`.

**Use case:** higher active capacity than A3B models with small KV growth.
**Risk:** less common stack and weaker direct Catan evidence than the primary
shortlist.

Source: [Jamba2 card](https://huggingface.co/ai21labs/AI21-Jamba2-Mini).

### Qwen3-Next 80B-A3B — sparse long-context ceiling, post-training opaque

- IDs: `Qwen/Qwen3-Next-80B-A3B-{Instruct,Thinking}`.
- Size: 80B total, 3B active; Apache 2.0; text only.
- Architecture: 36 Gated DeltaNet and 12 attention layers; 512 experts.
- Context: 262K native, one million with YaRN.
- Lineage: `15T pretraining → post-training`; detailed order is undisclosed.
  The Thinking card explicitly names GSPO, but the Instruct card does not expose
  a complete recipe.

**Use case:** long-context capacity/throughput ceiling.
**Risk:** roughly 80B of weights remain resident despite 3B active compute. It is
not automatically a stronger policy prior than a dense 27–32B model.

### Hunyuan-A13B — reasoning SFT/RL → general SFT/RL 80B MoE

- IDs: `tencent/Hunyuan-A13B-{Pretrain,Instruct}`.
- Size: 80B total, 13B active; text only; custom Tencent license.
- Context: up to 256K; released configs may default lower for memory.
- Lineage: `20T+ pretraining/annealing/context extension → reasoning SFT →
  reasoning GRPO RL → all-scenario SFT → all-scenario RL`.

**Use case:** more active capacity than A3B models at efficient MoE compute.
**Risk:** 80B resident memory, custom licensing, and a less mature local stack.

Source: [official repository](https://github.com/Tencent-Hunyuan/Hunyuan-A13B).

### GLM-4.7-Flash — 30B-A3B text rollout specialist, recipe undisclosed

- ID: `zai-org/GLM-4.7-Flash`.
- Size: 30B total, 3B active; text only; MIT.
- Context: 202,752 config limit; official hardware guidance validates 128K.
- Architecture: sparse MoE with MLA.
- Lineage: exact parent, SFT, preference, RL, and distillation sequence is
  undisclosed. Do not assign the GLM-4.5 recipe to it.

**Use case:** high-throughput text-only self-play after parser/tool validation.
**Risk:** only 3B active, newer serving requirements, and weak recipe auditability.

Source: [model card](https://huggingface.co/zai-org/GLM-4.7-Flash).

## Models that explicitly target priors or transition prediction

### Qwen-AgentWorld 35B-A3B — CPT → next-state SFT → GSPO world model

This is the closest official model to the user's “feed history and generate what
happens next” description.

- ID: [`Qwen/Qwen-AgentWorld-35B-A3B`](https://huggingface.co/Qwen/Qwen-AgentWorld-35B-A3B)
- Parent: `Qwen3.5-35B-A3B-Base`.
- Size: 35B total, 3B active; text-only checkpoint; Apache 2.0.
- Context: 262,144.
- Exact lineage: `environment CPT → next-state-prediction SFT → GSPO RL for
  simulation fidelity`.
- Objective: predict the next environment observation from action plus history
  across tool, search, terminal, SWE, Android, web, and OS domains.

This is a **world model**, not primarily an action policy. For Catan, the engine
already gives exact public transitions, chance outcomes, and legality. Replacing
it with a language simulator would reduce correctness. The proposed transfer hypothesis to test is:

- ablate its CPT → transition SFT → RL curriculum on Catan representations;
- use a learned model for opponent intent, hidden-state beliefs, and sampled
  continuation hypotheses;
- keep dice, cards, resources, legal actions, and actual state transitions in
  the engine.

A Catan world model must represent a distribution over stochastic multi-agent
futures, not narrate one deterministic continuation as truth.

### Nemotron Super 49B — compressed Llama-70B prior plus mixed alignment

- ID: `nvidia/Llama-3_3-Nemotron-Super-49B-v1_5`.
- Parent: Llama-3.3-70B-Instruct lineage.
- Size: 49B dense; text only; NVIDIA and Llama terms; 128K.
- Exact lineage: `NAS/block search → block-wise distillation → 40B-token KD →
  SFT → RPO for chat → RLVR for reasoning → iterative DPO for tools → checkpoint
  merge`.

**Use case:** strongest practical high-prior teacher/control before a 70B model.
It transfers a 70B aligned model into a 49B deployment shape and then adds
reasoning/tool stages. It is too expensive to assume as the four-seat default,
but it is a valuable upper bound.

Sources: [model card](https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1_5)
and [NVIDIA overview](https://developer.nvidia.com/blog/build-more-accurate-and-efficient-ai-agents-with-the-new-nvidia-llama-nemotron-super-v1-5/).

### Cogito v2 70B — IDA-internalized reasoning prior

- ID: `deepcogito/cogito-v2-preview-llama-70B`.
- Size: about 71B dense; text only; Llama 3.3 terms; 128K.
- Lineage: an aligned Llama-family parent followed by Iterated Distillation and
  Amplification (IDA). Exact datasets, iteration count, and hyperparameters are
  not public.

**Use case:** explicit “internalized intuition” comparator rather than only
longer test-time search. **Risk:** preview status, opaque details, and high cost.

### Tülu 3 70B — transparent Llama base → SFT → DPO → RLVR

- ID: `allenai/Llama-3.1-Tulu-3-70B`.
- Parent: Llama-3.1-70B Base.
- Lineage: `SFT → length-normalized DPO → RLVR/GRPO`.
- License: Llama terms; text only; 128K architecture/config, although the public
  SFT and DPO recipes use much shorter training sequences.

**Use case:** transparent 70B post-training upper bound.

Source: [Tülu 3 report](https://arxiv.org/abs/2411.15124).

### DeepSeek R1 distills — teacher-trace SFT, not student RL

- IDs: `DeepSeek-R1-Distill-Qwen-32B` and
  `DeepSeek-R1-Distill-Llama-70B`.
- Exact local delta: about 800K DeepSeek-R1-generated/curated examples followed
  by SFT only; no student-side RL.
- Inherited lineage: Qwen2.5-Math base or Llama-3.3-Instruct respectively.

**Use case:** reasoning-trace distillation control. **Risk:** math/reasoning bias,
long outputs, and weaker negotiation/action-policy fit.

Source: [DeepSeek-R1 paper](https://arxiv.org/abs/2501.12948).

## Native-vision track

If raw board images remain mandatory, the serious additional candidates are:

### Qwen3-VL 30B-A3B and 32B

The best-documented Qwen visual pipeline:

`merger-only alignment → approximately 1T multimodal pretraining → another
approximately 1T long-context stage → 100B ultra-long 256K stage → SFT at 32K
and 256K → off-/on-policy distillation → SAPO reasoning RL → general and
visual-agent RL`.

Strengths include OCR, boxes/points, grounding, documents, video, interleaving,
and explicit visual-agent training. The project's 32B raw-board result remained
poor, so these are perception ablations, not trusted parsers.

Sources: [Qwen3-VL report](https://arxiv.org/abs/2511.21631) and
[repository](https://github.com/QwenLM/Qwen3-VL).

### InternVL3/3.5 38B and InternVL3 78B

- InternVL3 Pretrained: native multimodal pretraining only.
- InternVL3 Instruct: pretraining → SFT.
- InternVL3 suffix-free: pretraining → SFT → Mixed Preference Optimization.
- InternVL3.5 MPO: multimodal pretraining → SFT → offline MPO.
- InternVL3.5 suffix-free: MPO → online GSPO Cascade RL.

Strengths: dynamic high resolution, OCR, grounding, video, and explicit variant
separation. Context is around 32–40K, so use a compact policy history.

Sources: [InternVL3 report](https://arxiv.org/abs/2504.10479) and
[InternVL3.5 report](https://arxiv.org/abs/2508.18265).

### Supervised-only visual controls

- `allenai/Molmo-72B-0924`: multimodal pretraining → all-parameter SFT; no RLHF.
  Strong pointing/counting, but only about 2.3K training context.
- `nvidia/NVLM-D-72B`: aligned Qwen2-72B-Instruct parent → connector alignment
  → multimodal/text SFT; no separate preference/RL stage reported.
- `CohereLabs/aya-vision-32b`: Aya Expanse parent → connector alignment →
  multilingual visual SFT → cross-modal model merge; 16K and noncommercial.
- `llava-onevision-qwen2-72b`: projector alignment → high-quality visual
  mid-training → single/multi-image/video SFT; no preference/RL stage reported.
- `nyu-visionx/cambrian-34b` and `VILA1.5-40b`: older supervised visual
  fallbacks with short context and custom stacks.

These models are useful controls for whether heavy RL helps or harms language and
perception. None should replace the symbolic board contract.

## Broader eligible census

The exhaustive audit also found these official families in range. They are lower
priority because they are older, specialized, license-constrained, or dominated
by a newer sibling:

- `GLM-4-32B-0414`, `GLM-Z1-32B-0414`, and rumination variants.
- `EXAONE-Deep-32B`, `EXAONE-4.0-32B`, and multimodal `EXAONE-4.5-33B`.
- `CohereLabs/c4ai-command-r-*` 32–35B and `aya-expanse-32b`.
- `swiss-ai/Apertus-70B-2509` and multimodal `Apertus-v1.5-70B`.
- `Qwen2/2.5` 32B/57B/72B text and vision families.
- `Llama-2/3/3.1/3.3-70B` base/chat/instruct families.
- `Mixtral-8x7B` at roughly 47B total/13B active.
- `Yi/Yi-1.5/Yi-VL-34B`, Falcon-40B, MPT-30B, Yuan 40B/51B,
  DeepSeek-VL2 27.5B, ERNIE-VL 28B-A3B, Qianfan-VL-70B, and older regional
  language models.

The machine-readable-style census and exact IDs were reviewed separately; no
community quantization or trivial merge was counted as a new model.

## Important exclusions

Below the lower bound:

- Gemma 4 26B-A4B is actually about 25.2B total.
- Mistral Small/Magistral 24B, Aria 25.3B, and InternVL 26B are below 27B.

Above the upper bound, even when active compute is small:

- Qwen3.8-Flash-Next is roughly 125B main weights plus 51B n-gram embeddings.
- Llama 4 Scout is about 109B total/17B active.
- Mistral Small 4 is about 119B total.
- GPT-OSS-120B, GLM-4.5-Air, Command A, and larger frontier MoEs exceed 80B.

Unavailable, superseded, or unsuitable:

- Qwen3.5/3.6 27B and 35B-A3B checkpoints were audited but omitted from the main
  bakeoff because Qwen3.8 supersedes the ready general checkpoints; retain the
  Qwen3.5 35B-A3B Base only for an explicit base/CPT ablation;
- exact `Nemotron-H-56B-VLM` weights were not verified; Cosmos Reason is a
  separate post-trained derivative;
- Solar Pro 2 is an API model, not a downloadable 31B checkpoint;
- code-only, math-only, reward, UI-only, theorem-proving, and robotics models are
  not general Catan policy candidates.

## Which strength matters for Catan

### Opening-board prior

This is a closed-book strategy test. The model should evaluate both settlements
and roads as one portfolio, including production, resource diversity, number
correlation, starting resources, ports, expansion, blocking, and plausible
routes to 10 VP. Long context is almost irrelevant here; dense capacity,
distillation, reasoning post-training, and Catan adaptation matter.

Best candidates to test: Qwen3.8, Muse, Granite, Gemma, and Nemotron Super 49B.

### Exact observation and belief tracking

Exact own cards, public events, legal actions, and board occupancy must remain
engine/harness facts. The model should maintain probabilistic opponent resource,
dev-card, goal, and trade-willingness beliefs with provenance and confidence.

Best architecture tests: Muse local/global attention, Qwen hybrid GDN/attention,
Seed full attention, Kimi linear/global hybrid, and Nemotron Nano.

### Trade denial and negotiation

“Freeze out” or “ice” a trade should normally mean refusing, warning, pricing,
or coordinating within ordinary table talk. Standard Catan has no enforceable
embargo action. The model must estimate the counterfactual value of the trade to
both parties and the probability it enables an immediate city, road, award, or
win.

Preference-trained language may improve negotiation style; only Catan
outcome-based training can establish whether the policy makes strategically good
trades.

### Long-horizon planning

Catan is stochastic and partially observed. One narrated future is not a value
estimate. The policy needs a distribution over dice, cards, opponent actions,
and hidden hands. Multiple engine continuations or a calibrated learned value
head are more appropriate than trusting one generated trajectory.

### Structured action execution

Granite's structured-output/environment RL and Qwen/Muse tool behavior are
relevant. Still require the exact engine-generated menu and validate one indexed
action. A model should never manufacture legality from prose.

## Recommended Catan runtime design

Use three different kinds of state:

1. **Authoritative exact state:** engine state, perspective-safe event log, own
   private cards, public board, exact legal menu, accepted trades, and RNG/event
   provenance.
2. **Typed uncertain beliefs:** per-opponent resource intervals/distributions,
   dev-card probabilities, goals, route threats, trade likelihood, and confidence.
3. **Strategic memory:** current plan, alternatives, commitments, triggers, and
   reasons to replan.

The model may update items 2 and 3. It must never overwrite item 1.

Maintain one persistent private session per seat. Share weights and the exact
public world state, but not private decoder history. Provider KV or recurrent
state is a losable acceleration artifact; the harness must be able to rebuild it
from the event ledger.

At every decision:

- append unread perspective-safe events;
- attach an authoritative current-state checkpoint;
- attach the exact legal action menu;
- include typed beliefs with provenance;
- strip or compact old native thinking;
- use high reasoning effort for setup, robber, pivotal trade, and win/loss
  decisions; use low/off for routine actions and silence gates.

## How to create the strongest Catan prior

No audited model has a strong Catan prior today. Build it explicitly:

1. Start from the strongest **post-trained** dense candidate unless a base-model
   ablation is the purpose of the experiment.
2. Use Catan continued/mid-training for rules, terminology, event sequences,
   strategic commentary, opponent models, and trajectory distributions.
3. Use engine-verified SFT or self-distillation for state → belief → action
   demonstrations and exact tool schemas.
4. Include counterfactual states differing in one board fact, resource event,
   trade, or hidden-state hypothesis.
5. Run seat-relative, engine-grounded online RL/self-play only after legality,
   state tracking, and belief calibration pass.
6. Keep board perception as a separately scored auxiliary objective or parser.

A useful Qwen-inspired experiment is:

`Qwen3.5-35B-A3B-Base → Catan CPT → next-event/belief SFT → simulator RL`

That is the AgentWorld recipe transferred to Catan. It should be compared with a
dense post-trained policy such as Qwen3.8 or Granite. The simulator must not
replace the deterministic engine.

## Matched evaluation required before selection

### Phase A — prior-only opening read

Use at least 100 held-out boards with no demonstrations or game history.

Score:

- ranked settlement-pair/road portfolio regret versus a search/rollout oracle;
- resource/number/port/expansion calculations;
- calibration over alternative plans;
- sensitivity to seat order and likely opponent picks.

### Phase B — causal context tracking

Create exact histories differing in one roll, trade, steal, discard, dev-card
play, robber move, or build. Test at 4K, 16K, 32K, 64K, and 128K, with the key
event at the beginning, middle, and end.

Score:

- exact own/public state;
- conservation and event-order violations;
- opponent-belief Brier score/log loss;
- stale-state override rate;
- cross-seat and future-information leakage;
- action changes and non-target invariance.

### Phase C — trade and opponent modeling

Use scenarios where a legal trade helps the acting seat but may enable an
opponent's city, road, Largest Army, Longest Road, or immediate win.

Score acting-seat continuation value, value conceded, refusal quality,
negotiation outcome, bluff sensitivity, and calibration—not prose persuasiveness.

### Phase D — full policy games

After the offline gate, run seat-balanced games against a fixed opponent league
with common boards/seeds and exact behavior-policy metadata.

Report:

- win share and final VP by seat;
- illegal/parse/retry rate;
- belief calibration;
- opponent-enabled wins after trades;
- tokens and latency per decision;
- four-seat throughput and cache/state memory;
- variance over boards, dice, and opponent lineups.

### Initial go/no-go gates

These are project proposals, not universal standards:

- 100% legal action after bounded retry and at least 99.5% first-attempt binding.
- Zero hidden-state, future-event, or cross-seat leakage.
- At least 99.5% exact own state and 99% exact public event state.
- No more than five percentage points of middle-position degradation.
- Positive opening regret improvement over pip-only and heuristic baselines with
  a confidence interval above zero.
- Full-game improvement against a frozen league with no material legality or
  privacy regression.

## Deployment implications

Raw weight payloads, before quantization metadata and runtime overhead:

| Total parameters | BF16 | 8-bit/FP8 | raw 4-bit |
| ---: | ---: | ---: | ---: |
| 27B | 54 GB | 27 GB | 13.5 GB |
| 30B | 60 GB | 30 GB | 15 GB |
| 36B | 72 GB | 36 GB | 18 GB |
| 49B | 98 GB | 49 GB | 24.5 GB |
| 70B | 140 GB | 70 GB | 35 GB |
| 80B | 160 GB | 80 GB | 40 GB |

Actual residency is larger. Add quantization scales, embeddings, load buffers,
activations, CUDA graphs, workspaces, vision components, KV/recurrent state, and
allocator fragmentation.

Weights are shared by four seats; KV/recurrent state is not. At four seats × 16K
history:

- Qwen3.8's 16 full-attention layers use about 4 GiB BF16 attention KV before
  recurrent state and runtime overhead;
- a conventional 64-layer, eight-KV-head dense model with 128-dimensional heads
  uses about 16 GiB BF16 KV;
- Muse's two KV heads plus bounded local windows should be substantially smaller,
  but must be measured in the exact runtime;
- MoE active count reduces compute, not resident expert-weight memory.

Practical starting hardware:

- 32GB Blackwell: the
  [third-party Qwen3.8 NVFP4 quantization](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4)
  is weight-fit in principle, but its published validation is backend- and
  hardware-specific and does not prove four-seat service; Muse's official 4-bit
  package is the more directly packaged consumer path.
- 48GB: Qwen3.8 FP8/4-bit and 27–36B 4-bit models.
- 80–96GB: comfortable 27–36B BF16/FP8 policy testing and 49B FP8.
- 141GB H200: 48–56B BF16; 70B BF16 leaves too little runtime headroom.
- Multi-GPU: robust 70–80B serving, especially at long context.

Measure TTFT, decode latency, cache rebuild, four-seat concurrency, quantized
policy quality, and prefix/state-cache behavior. Do not infer them from parameter
count alone.

## Final recommendation

The project should not switch models from internet evidence alone. Run a six-way
matched bakeoff:

- Qwen3.8-27B
- Muse Glimmer 30B
- Granite 4.2 30B
- Gemma 4 31B IT
- OLMo 3.1 32B Instruct/Think
- Nemotron Super 49B

Add Nemotron Nano or GLM-4.7-Flash only when rollout throughput is the main axis,
and Qwen-AgentWorld only as a transition/world-model adaptation experiment.

If one model must be selected before that test, retain **Qwen3.8-27B**. It has
the strongest integrated engineering fit among the audited options and the
lowest migration cost; this is not a validated Catan-quality rank. If text-only symbolic
state is permanently confirmed, **Granite 4.2-30B** is the most important new
challenger. If the work is primarily scientific post-training research rather
than immediate deployment, **OLMo 3.1 32B** is the cleanest foundation.

The strongest Catan prior will come from Catan CPT/SFT/self-play, not from a
marketing context window. Context should carry current evidence; weights should
carry strategy; the engine should carry truth.
