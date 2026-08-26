# Catan VLM Model Selection

Date reviewed: 2026-05-14

## Decision

Use **Qwen3-VL-8B-Instruct** as the first serious SFT target, with
**InternVL3.5-8B-HF** as the strongest local-training backup, and keep
**Gemini, Grok 4.3, Claude Sonnet 4.6, and Gemma 4 31B** as eval/comparison
models rather than main students.

This is not because Qwen3-VL-8B won zero-shot CatanBoardBench. It did not. The reason
is operational: it is closest to the requested 9B size, Apache-2.0, has active
adapter/fine-tune ecosystem support, and has explicit VLM SFT recipes in NVIDIA
NeMo. For this project, base zero-shot performance matters less than whether the
model can be cheaply adapted on engine-labeled board images.

## Local Evidence

Current clean 40-question visual smoke:

| Model | Exact | Component | Notes |
| --- | ---: | ---: | --- |
| `openrouter/google/gemini-3.1-pro-preview` | 67.5% | 72.5% | Strong ceiling; still fails road-location/count questions. |
| `openrouter/x-ai/grok-4.3` | 30.0% | 39.4% | Best non-Gemini run; slow and token-heavy. |
| `openrouter/anthropic/claude-sonnet-4.6` | 25.0% | 29.7% | Strongest practical frontier comparison after Grok. |
| `openrouter/anthropic/claude-opus-4.7` | 17.5% | 25.8% | Tied Gemma/Qwen on exact, higher component. |
| `openrouter/google/gemma-4-31b-it` | 17.5% | 24.8% | Best Gemma result; still weak on coordinate-grounded tasks. |
| `openrouter/qwen/qwen3-vl-235b-a22b-instruct` | 17.5% | 22.5% | Larger Qwen helps exact versus 32B, but still fails key slots. |
| `openrouter/mistralai/mistral-large-2512` | 15.0% | 23.7% | Mid-tier frontier result. |
| `openrouter/meta-llama/llama-4-maverick` | 12.5% | 18.8% | Better than Scout, still weak. |
| `openrouter/qwen/qwen3-vl-32b-instruct` | 12.5% | 16.3% | Larger dense Qwen did not fix slot binding. |
| `openrouter/google/gemma-3-4b-it` | 12.5% | 21.3% | Small baseline. |
| `openrouter/google/gemma-4-26b-a4b-it` | 10.0% | 18.3% | Cheaper MoE Gemma 4 route, weaker than 31B. |
| `openrouter/nvidia/nemotron-nano-12b-v2-vl:free` | 10.0% | 21.7% | Similar to other small VLMs. |
| `openrouter/qwen/qwen3-vl-8b-instruct` | 10.0% | 18.0% | Poor zero-shot, but still the best SFT-shaped base. |
| `openrouter/openai/gpt-5.5` | 10.0% | 13.0% | Poor fit for this strict token-answer prompt. |
| `openrouter/meta-llama/llama-4-scout` | 10.0% | 14.4% | Fast and cheap, but poor visual parser here. |
| `openrouter/google/gemma-3-12b-it` | 10.0% | 17.2% | No clear benefit over smaller Gemma 3. |
| `openrouter/qwen/qwen3-vl-30b-a3b-instruct` | 7.5% | 17.5% | MoE Qwen route was weakest on exact score. |
| `openrouter/z-ai/glm-5v-turbo` | 2.5% | 6.8% | Unusable with the current prompt. |

Source artifact:

- `reports/catan_board_bench/2026-05-13-small-vlm-visual-smoke.md`

Interpretation:

- General VLM benchmarks are not enough. CatanBoardBench is a fixed-slot board parsing
  task with tile/node/edge/port binding.
- Gemini proves that frontier VLMs can extract much of the board, but even it is
  not reliable enough for a parser.
- The student should be chosen by adaptation path plus eventual full CatanBoardBench
  score, not by out-of-the-box benchmark claims.
- The 2026-05-14 follow-up runs argue that simple model scale within Qwen is not
  solving this task zero-shot. Test InternVL3.5 locally before committing all
  training effort to a Qwen-family base.
- Big proprietary models improve the ceiling but do not solve the board parser:
  Grok 4.3 reached only 30.0% exact and still scored 0/4 on robber-tile exact.

## Catan-Specific Selection Criteria

Hard requirements:

- Open weights or at least fine-tunable weights.
- Native image plus text input.
- Good structured-output behavior after SFT.
- Can support exact board tokens: `<T00>`, `<N17>`, `<E12_17>`, `<WOOD>`,
  `<MYSTIC_BLUE>`.
- Practical QLoRA/SFT path on affordable hardware.
- Vision pathway preserves enough spatial detail for 512px to 768px board crops.

Important but secondary:

- Good public VLM benchmarks.
- Long context.
- Thinking/reasoning mode.
- Tool use/function calling.

For this project, the model must first become a reliable **public board parser**.
Strategy imitation and expert-game training come later and should use a validated
symbolic observation stream.

## Candidate Ranking

### 1. Qwen3-VL-8B-Instruct

Recommendation: **primary SFT target**.

Why:

- Fits the intended 9B class. Hugging Face lists the model size as 9B params.
- Apache-2.0 license.
- Qwen3-VL technical report describes dense 2B/4B/8B/32B variants and MoE
  variants, with interleaved text/image/video context and explicit spatial-temporal
  architecture work.
- NVIDIA NeMo AutoModel has Qwen3-VL SFT recipes for 4B and 8B.
- Hugging Face shows many adapters/fine-tunes/quantizations already attached to
  the model card, which is a practical signal that the ecosystem is alive.

Risks:

- Our zero-shot CatanBoardBench result was bad.
- The 8B model may not have enough visual precision without a curriculum.
- Thinking variants can be slower and more expensive; our earlier Qwen thinking
  smoke was not worth trusting because it came from a now-excluded leaky run.

What to test next:

- Run clean full selected visual suite, not only 40Q smoke.
- Re-test after an overlay/slot-label curriculum, where Qwen may benefit from
  explicit token grounding even though zero-shot failed.
- Compare against InternVL3.5-8B locally before making Qwen the only SFT base.

Sources:

- Qwen3-VL-8B-Instruct model card:
  https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
- Qwen3-VL technical report:
  https://arxiv.org/abs/2511.21631
- NVIDIA NeMo Qwen3-VL fine-tuning recipes:
  https://docs.nvidia.com/nemo/automodel/0.4.0/model-coverage/vlm/qwen/qwen3-vl.html

### 2. InternVL3.5-8B-HF

Recommendation: **best backup for local adaptation**.

Why:

- 8B-ish scale and explicitly image-text-to-text.
- InternVL3.5 model card describes a dynamic high-resolution strategy and a
  ViT-MLP-LLM architecture initialized from Qwen3-series language models.
- Axolotl has an InternVL3.5 QLoRA example and reports about 8.21 GiB VRAM for
  the 8B QLoRA config.
- Strong fit for a board parser because the family is built around high-resolution
  visual understanding.

Risks:

- Not currently in our OpenRouter availability query, so evaluation may require
  local serving or another provider.
- More custom model path than Qwen3-VL.

What to test next:

- Run a local CatanBoardBench smoke as soon as local inference is available.
- If it beats Qwen3-VL-8B zero-shot and QLoRA is smooth, promote it to primary.

Sources:

- InternVL3.5-8B-HF model card:
  https://huggingface.co/OpenGVLab/InternVL3_5-8B-HF
- Axolotl InternVL3.5 fine-tuning guide:
  https://docs.axolotl.ai/docs/models/internvl3_5.html
- InternVL3.5 report:
  https://arxiv.org/abs/2508.18265

### 3. Gemma 4 31B IT

Recommendation: **strong comparison model, not first student**.

Why:

- Best non-Gemini result in our clean 40Q smoke so far.
- Google/Hugging Face docs describe Gemma 4 as multimodal with E2B, E4B, 26B A4B,
  and 31B variants.
- The vision design is interesting for Catan: variable aspect ratio, fixed-budget
  soft tokens, and configurable image token budgets. The default is 280 soft tokens
  per image, with larger settings available.
- OpenRouter has `google/gemma-4-31b-it` and `google/gemma-4-26b-a4b-it`.

Risks:

- Too large for the 9B target.
- NVIDIA's Gemma 4 31B fine-tuning guide lists 8x A100 80GB or 8x H100 for its
  full FSDP recipe. QLoRA may be possible, but the 31B path is not the cheap first
  loop.
- Our CatanBoardBench category breakdown is still poor: 0/4 robber tile, 0/4 node
  occupancy, 0/4 tile_has_robber, 0/4 color road locations.

What to test next:

- Keep dense `google/gemma-4-31b-it` as the stronger Gemma comparison point. The
  `google/gemma-4-26b-a4b-it` follow-up scored 10.0% exact and 18.3% component,
  below dense 31B.
- If local Gemma 4 E4B vision is easy to run, test it as a small Google-family
  student.

Sources:

- Gemma 4 31B model card:
  https://huggingface.co/google/gemma-4-31B-it
- Hugging Face Gemma4 docs:
  https://huggingface.co/docs/transformers/model_doc/gemma4
- NVIDIA Gemma 4 31B fine-tuning guide:
  https://docs.nvidia.com/nemo/automodel/latest/guides/vlm/gemma4.html
- Google Gemma vision QLoRA guide:
  https://ai.google.dev/gemma/docs/core/huggingface_vision_finetune_qlora

### 4. Llama 4 Scout

Recommendation: **evaluate, but do not start SFT here**.

Why:

- Natively multimodal, MoE, strong public model card claims for text/image.
- OpenRouter lists `meta-llama/llama-4-scout` with cheap token pricing and large
  context.

Risks:

- 109B total parameters, 17B active. Not a 9B-class training target.
- Custom Llama 4 license and gated model access.
- The model card says image understanding has been tested up to 5 input images,
  but that is not the same as exact board-slot parsing.

What to test next:

- Llama 4 Scout already scored 10.0% exact and 14.4% component on the 40Q smoke.
  Do not prioritize it unless a later prompt/curriculum change affects all
  models and needs a broad rerun.

Sources:

- Llama 4 Scout model card:
  https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E

### 5. MiniCPM-V 4.5 / MiniCPM-o 4.5

Recommendation: **watchlist / local candidate**.

Why:

- MiniCPM-V 4.5 is an 8B model focused on efficiency.
- MiniCPM-o 4.5 is a 9B omni-modal model; the paper claims strong open-source
  performance at its scale and edge-device feasibility under 12GB RAM.

Risks:

- Not in our current OpenRouter model query.
- Omni-modal/full-duplex strengths are not directly useful for still-image board
  parsing.
- Need to verify training tooling before committing.

Sources:

- MiniCPM-V 4.5 paper:
  https://arxiv.org/abs/2509.18154
- MiniCPM-o 4.5 paper:
  https://arxiv.org/abs/2604.27393

### 6. Molmo 7B-D

Recommendation: **interesting grounding baseline, not first choice**.

Why:

- AllenAI describes Molmo as a fully open vision-language model family trained on
  a curated PixMo dataset.
- 7B-D is close to the desired scale and uses Qwen2-7B plus a CLIP vision backbone.

Risks:

- Older than Qwen3-VL / InternVL3.5 / Gemma 4.
- More custom inference path.
- Less obvious current SFT path for this repo than Qwen or InternVL.

Sources:

- Molmo 7B-D model card:
  https://huggingface.co/allenai/Molmo-7B-D-0924

### 7. Pixtral 12B

Recommendation: **skip unless it is already convenient**.

Why:

- 12B multimodal model with image understanding and 128K context.

Risks:

- Mistral docs now mark Pixtral 12B as deprecated with Ministral 3 14B as the
  replacement.
- It is older and not clearly better suited to exact Catan board slots than newer
  Qwen/InternVL/Gemma options.

Sources:

- Pixtral 12B model card:
  https://docs.mistral.ai/models/model-cards/pixtral-12b-24-09

## Why Spatial Benchmarks Change The Decision

The literature supports caution: current VLMs can look good on broad benchmarks
while failing precise low-level spatial tasks.

Relevant evidence:

- "Vision Language Models Are Blind" reports that strong VLMs struggle on simple
  geometry tasks such as overlap/intersection/counting under resolution and line
  width variation.
- "The Spatial Blindspot of Vision-Language Models" argues that common CLIP-style
  VLM recipes flatten images into 1D patch sequences and can discard 2D structure
  needed for spatial reasoning.
- "The Dual Mechanisms of Spatial Reasoning in Vision-Language Models" finds that
  spatial signal is dominated by the vision encoder, while language-backbone
  spatial mechanisms are secondary.
- SPHERE shows that VLMs still struggle as spatial tasks require distance,
  perspective, and logic composition.

Implication for Catan:

- Do not rely on broad MMMU/MathVista/DocVQA scores.
- Prefer models whose vision path and training tooling let us directly supervise
  board slots.
- Mechanistic checks should focus on whether board facts are decodable from
  visual-token and question-token activations, not only on attention maps.

Sources:

- Vision Language Models Are Blind:
  https://github.com/anguyen8/vision-llms-are-blind
- The Spatial Blindspot of Vision-Language Models:
  https://arxiv.org/abs/2601.09954
- The Dual Mechanisms of Spatial Reasoning in Vision-Language Models:
  https://arxiv.org/abs/2603.22278
- SPHERE:
  https://arxiv.org/abs/2412.12693

## OpenRouter Availability Snapshot

Queried from `https://openrouter.ai/api/v1/models` during this research pass.

| Model | Prompt $/token | Completion $/token | Context |
| --- | ---: | ---: | ---: |
| `google/gemma-4-31b-it` | 0.00000012 | 0.00000037 | 262144 |
| `google/gemma-4-26b-a4b-it` | 0.00000006 | 0.00000033 | 262144 |
| `qwen/qwen3-vl-8b-instruct` | 0.00000008 | 0.0000005 | 131072 |
| `qwen/qwen3-vl-8b-thinking` | 0.000000117 | 0.000001365 | 131072 |
| `qwen/qwen3-vl-32b-instruct` | 0.000000104 | 0.000000416 | 131072 |
| `qwen/qwen3-vl-30b-a3b-instruct` | 0.00000013 | 0.00000052 | 131072 |
| `qwen/qwen3-vl-235b-a22b-instruct` | 0.0000002 | 0.00000088 | 262144 |
| `x-ai/grok-4.3` | 0.00000125 | 0.0000025 | 1000000 |
| `anthropic/claude-sonnet-4.6` | 0.000003 | 0.000015 | 1000000 |
| `anthropic/claude-opus-4.7` | 0.000005 | 0.000025 | 1000000 |
| `openai/gpt-5.5` | 0.000005 | 0.00003 | 1050000 |
| `meta-llama/llama-4-scout` | 0.00000008 | 0.0000003 | 327680 |
| `meta-llama/llama-4-maverick` | 0.00000015 | 0.0000006 | 1048576 |
| `mistralai/mistral-large-2512` | 0.0000005 | 0.0000015 | 262144 |
| `mistralai/pixtral-large-2411` | 0.000002 | 0.000006 | 131072 |
| `z-ai/glm-5v-turbo` | 0.0000012 | 0.000004 | 202752 |

## Completed Follow-Up Runs

These were run on the same clean 40-question visual suite after the initial
selection memo:

| Model | Exact | Component | Log |
| --- | ---: | ---: | --- |
| `openrouter/qwen/qwen3-vl-32b-instruct` | 12.5% | 16.3% | `logs/2026-05-13T17-22-52-07-00_catan_board_bench-visual_3sEP25BLYGN6ZeySusTk6C.json` |
| `openrouter/qwen/qwen3-vl-30b-a3b-instruct` | 7.5% | 17.5% | `logs/2026-05-13T17-23-32-07-00_catan_board_bench-visual_Ss7WUPEEbw2wZV7TgQ4TUF.json` |
| `openrouter/google/gemma-4-26b-a4b-it` | 10.0% | 18.3% | `logs/2026-05-13T17-24-05-07-00_catan_board_bench-visual_68coXZJ79X4D7FHGsmbXx4.json` |
| `openrouter/meta-llama/llama-4-scout` | 10.0% | 14.4% | `logs/2026-05-13T17-24-54-07-00_catan_board_bench-visual_GuAB6L9YPhRHji2nPrVhtG.json` |

Additional big-model runs:

| Model | Exact | Component | Log |
| --- | ---: | ---: | --- |
| `openrouter/qwen/qwen3-vl-235b-a22b-instruct` | 17.5% | 22.5% | `logs/2026-05-13T17-41-41-07-00_catan_board_bench-visual_cM6wEnPB7rfntyHZBpoTxq.json` |
| `openrouter/meta-llama/llama-4-maverick` | 12.5% | 18.8% | `logs/2026-05-13T17-42-21-07-00_catan_board_bench-visual_a26Ds3bshcScSb6rDNNSQE.json` |
| `openrouter/anthropic/claude-opus-4.7` | 17.5% | 25.8% | `logs/2026-05-13T17-42-49-07-00_catan_board_bench-visual_6qw6xQVLL9weZk72PDxB74.json` |
| `openrouter/anthropic/claude-sonnet-4.6` | 25.0% | 29.7% | `logs/2026-05-13T17-43-47-07-00_catan_board_bench-visual_m2yWLSUSFx4Y4dmdxnmP59.json` |
| `openrouter/openai/gpt-5.5` | 10.0% | 13.0% | `logs/2026-05-13T17-45-02-07-00_catan_board_bench-visual_DzTSeLgfiA46857Tt3d6TV.json` |
| `openrouter/x-ai/grok-4.3` | 30.0% | 39.4% | `logs/2026-05-13T17-48-00-07-00_catan_board_bench-visual_GTkFqMxuEKLz2cjSuay8Uo.json` |
| `openrouter/mistralai/mistral-large-2512` | 15.0% | 23.7% | `logs/2026-05-13T17-55-42-07-00_catan_board_bench-visual_Kb9Prgw3FwVdtZJ3EjnDPR.json` |
| `openrouter/z-ai/glm-5v-turbo` | 2.5% | 6.8% | `logs/2026-05-13T17-56-21-07-00_catan_board_bench-visual_Nyi8jmGuWJNcd3pD9ncNvF.json` |

The next useful eval is no longer another larger general VLM on the same prompt.
It should be one of:

- InternVL3.5-8B local smoke, because it is the strongest untested trainable
  visual-parser candidate.
- The same candidate set on 768px and overlay-labeled boards, to isolate whether
  the bottleneck is visual resolution, atlas binding, or prompt format.
- A text-only logic suite run for the intended student, to verify the language
  side can reason over perfect symbolic observations.

The previous command template was:

```bash
uv run --extra eval dotenv run -- bench eval catan_board_bench \
  --model openrouter/qwen/qwen3-vl-32b-instruct \
  --temperature 0 \
  --max-tokens 256 \
  --max-connections 2 \
  --log-format json \
  --no-log-images \
  -T suite=visual \
  -T max_requests=40 \
  -T include_system=false
```

## Training Path

1. Keep visual parsing separate from game strategy.
2. Generate engine-labeled visual data from rendered board contracts.
3. Train micro-QA first:
   - tile resource/number
   - robber tile
   - node occupancy
   - edge owner
   - port type and port occupancy
   - color road locations
4. Train full public-board extraction second:
   - canonical `PublicBoardContract` JSON
   - exact token IDs
   - invalid topology checks
5. Train game logic/action imitation only after visual extraction passes.

Data curriculum:

- Start with 768px boards and optional tile/node/edge overlays.
- Move to 512px tight crops without overlays.
- Add color/asset/style randomization.
- Add counterfactual pairs where one board fact changes.
- Keep a held-out real/replay set separate from generated training boards.

## Success Criteria Before Scaling

Before spending serious compute on expert-game SFT:

| Metric | Minimum |
| --- | ---: |
| Full selected visual suite exact | > 90% |
| Tile resource/number component | > 98% |
| Robber tile exact | > 98% |
| Node occupancy component | > 97% |
| Edge road owner component | > 97% |
| Port type/occupancy component | > 95% |
| Color road location component | > 90% |
| Logic/text-only suite exact | > 99% |

If the model cannot reach these parser metrics, strategy training on expert games
will mostly teach it to compensate for bad perception.

## Bottom Line

The best current plan is:

1. **Benchmark InternVL3.5-8B locally** before locking the base model. The
   larger OpenRouter candidates improved the ceiling but still did not produce a
   reliable board parser.
2. **Start SFT engineering on Qwen3-VL-8B** because it is the right size and has
   the cleanest adaptation path, but keep this reversible until InternVL is
   tested.
3. **Add an overlay/768px curriculum eval** because the plain 512-ish visual
   prompt is exposing atlas-binding failure, not just generic reasoning weakness.
4. **Use Gemini, Grok, and Claude Sonnet as ceiling/teacher candidates**, never
   as the ground-truth judge. The engine remains the oracle.
