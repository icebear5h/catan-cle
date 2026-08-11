# Gemini SOTA Evidence For Catan VLM Ceiling Eval

Date reviewed: 2026-05-13

## Why this matters

Gemini is not a public open-weight student model for our Catan work, but it is useful as a
ceiling model. If Gemini 3.1 Pro cannot reliably extract a clean 768px Catan board contract,
then the problem is probably not "pick a better 9B VLM"; it is more likely a need for
Catan-specific atlas grounding, overlays, or a task-specific training curriculum.

The ground truth still must come from the engine, not Gemini.

## Official DeepMind Evidence

### Gemini 3.1 Pro model card

Source: https://deepmind.google/models/model-cards/gemini-3-1-pro/

DeepMind describes Gemini 3.1 Pro as the next iteration in the Gemini 3 series and, at the
model card publication date, Google's most advanced model for complex tasks. The card says it
handles multimodal sources including text, audio, images, video, and code repositories, with
up to a 1M-token input context and 64K-token output.

Relevant official benchmark claims from the February 2026 model card:

| Benchmark | Notes | Gemini 3.1 Pro Thinking High | Gemini 3 Pro Thinking High |
| --- | --- | ---: | ---: |
| Humanity's Last Exam | full set, text + multimodal, no tools | 44.4% | 37.5% |
| ARC-AGI-2 | abstract reasoning puzzles | 77.1% | 31.1% |
| GPQA Diamond | scientific knowledge, no tools | 94.3% | 91.9% |
| MMMU-Pro | multimodal understanding and reasoning, no tools | 80.5% | 81.0% |
| MRCR v2 8-needle | 128K long-context average | 84.9% | 77.0% |

Interpretation for Catan:

- The model card supports using Gemini 3.1 Pro as a strong multimodal/reasoning ceiling.
- The MMMU-Pro score is relevant but not decisive: Catan board extraction is a fixed-slot,
  symbolic binding task, not general multimodal QA.
- The result that 3.1 Pro is slightly below 3 Pro on MMMU-Pro means we should benchmark both
  if both are available, instead of assuming the newer model dominates every visual task.

### Gemini 3 / Gemini 3.1 product page

Source: https://deepmind.google/models/gemini/

DeepMind's Gemini page positions Gemini 3 / 3.1 as state-of-the-art across broad benchmarks
and emphasizes advanced multimodal understanding, images/video/audio/code inputs, tool use,
structured output, and agentic workflows.

Relevant details for our eval design:

- Model information lists text, image, video, audio, and PDF inputs.
- The page lists structured output and code execution/tool use support.
- The performance table repeats Gemini 3.1 Pro and Gemini 3 Pro comparisons across reasoning,
  agentic, coding, multimodal, and long-context benchmarks.

Interpretation for Catan:

- Structured output support is useful for forcing `PublicBoardContract` JSON.
- Broad agentic/coding results are less relevant to board pixel extraction.
- The model is still a good "can a frontier VLM see this board at all?" ceiling.

### Gemini 3.1 Pro eval methodology

Source: https://deepmind.google/models/evals-methodology/gemini-3-1-pro

DeepMind links a methodology PDF for the benchmark table. We should cite the methodology when
making broad benchmark claims, but the methodology does not replace our domain-specific eval.

## Catan-Specific Eval Implication

Use Gemini as a contestant, not a judge.

Recommended comparison:

1. Generate engine oracle:
   - `PublicBoardContract`
   - 768px board image
   - render manifest if available
   - QA pairs
2. Ask every model for the same strict JSON schema:
   - no explanation
   - no hidden state
   - use `null` if uncertain
3. Score against engine oracle:
   - tile resource accuracy
   - tile number accuracy
   - robber location exact match
   - port resource accuracy
   - port attached node-id accuracy
   - node occupancy accuracy
   - edge road-color accuracy
   - invalid JSON rate
   - illegal topology rate
   - null rate

If Gemini is strong but 9B models are weak, that supports distillation/SFT from Gemini-like
outputs or a stronger student base. If Gemini is weak on node/edge binding, the next move is
not just a bigger model; it is atlas curriculum and explicit coordinate grounding.

