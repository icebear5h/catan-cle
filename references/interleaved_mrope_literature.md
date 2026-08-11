# Interleaved-MRoPE Literature Map

Date: 2026-05-14

Prompt: Qwen3-VL model card says:

> Interleaved-MRoPE: Full-frequency allocation over time, width, and height via
> robust positional embeddings, enhancing long-horizon video reasoning.

## Short Explanation

RoPE rotates attention queries/keys by position, so attention can depend on
relative position. Standard RoPE is 1D: one sequence position.

MRoPE extends this to multimodal positions:

```text
token position = (time, height, width)
```

The older/chunked version assigns contiguous embedding dimensions to each axis:

```text
[time frequencies][height frequencies][width frequencies]
```

The issue is frequency imbalance: one axis may get mostly low or high rotary
frequencies, so the model does not use the full frequency spectrum for every
axis.

Interleaved-MRoPE spreads the axes across the frequency spectrum:

```text
[time low][height low][width low][time mid][height mid][width mid]...
```

So time, height, and width all get access to low/mid/high positional frequencies.
That is what the model card means by "full-frequency allocation."

## Most Direct Sources

### Qwen3-VL Technical Report

- URL: https://arxiv.org/abs/2511.21631
- Why it matters: official Qwen3-VL report. It lists enhanced interleaved-MRoPE
  as one of the three architecture upgrades, alongside DeepStack and text-based
  timestamp alignment.
- Local implication: this is the model-family claim, but not the deepest
  standalone explanation.

### Qwen3-VL-8B-Instruct Model Card

- URL: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
- Why it matters: the exact line that triggered this note. It states that
  Interleaved-MRoPE gives full-frequency allocation over time/width/height and
  improves long-horizon video reasoning.
- Useful adjacent fact: the model card also links Qwen2-VL and Qwen2.5-VL as
  prior architecture papers.

### Revisiting Multimodal Positional Encoding in Vision-Language Models

- URL: https://arxiv.org/abs/2510.23095
- OpenReview: https://openreview.net/forum?id=sCCF4ygDAw
- ICLR poster: https://iclr.cc/virtual/2026/poster/10007080
- Code: https://github.com/JJJYmmm/Multimodal-RoPEs
- Why it matters: closest focused literature. It analyzes multimodal RoPE along
  two axes: position design and frequency allocation. It proposes:
  - `MHRoPE`: multi-head axis allocation
  - `MRoPE-I`: interleaved axis allocation
- Key guidelines:
  - positional coherence
  - full frequency utilization
  - preservation of textual priors
- This is the source to read if you want to understand why the interleaving
  matters, not just that Qwen3-VL uses it.

## Predecessors In Qwen

### Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution

- URL: https://arxiv.org/abs/2409.12191
- Why it matters: introduces Qwen2-VL's dynamic-resolution processing and
  Multimodal Rotary Position Embedding.
- Relevance: baseline MRoPE idea for arbitrary-resolution images/videos.

### Qwen2.5-VL Technical Report

- URL: https://arxiv.org/abs/2502.13923
- Why it matters: extends the Qwen2-VL line toward long-video comprehension,
  dynamic resolution, absolute time encoding, and localization by boxes/points.
- Relevance: bridge between Qwen2-VL MRoPE and Qwen3-VL's interleaved-MRoPE +
  text-timestamp alignment.

## Foundational RoPE / Vision RoPE

### RoFormer: Enhanced Transformer with Rotary Position Embedding

- URL: https://arxiv.org/abs/2104.09864
- Why it matters: original RoPE paper. RoPE encodes absolute position through
  rotations while making attention depend naturally on relative offsets.
- Read first if the rotation math is fuzzy.

### Rotary Position Embedding for Vision Transformer

- URL: https://arxiv.org/abs/2403.13298
- Why it matters: studies RoPE in ViTs and 2D vision. Shows RoPE can help image
  resolution extrapolation and dense vision tasks.
- Relevance: explains why rotary embeddings are not just for text.

## Video-Specific RoPE Literature

### VideoRoPE: What Makes for Good Video Rotary Position Embedding?

- URL: https://arxiv.org/abs/2502.05173
- Why it matters: asks almost the exact general question: what makes a good video
  rotary position embedding?
- Relevance: useful comparison to Qwen's interleaved-MRoPE. Focuses on preserving
  spatiotemporal structure for video tasks.

### VRoPE: Rotary Position Embedding for Video Large Language Models

- URL: https://arxiv.org/abs/2502.11664
- Why it matters: argues that adapting RoPE to video is hard because video has
  structured space-time axes; proposes a video-tailored RoPE to reduce positional
  bias and video-text transition problems.
- Relevance: independent evidence that naive 3D RoPE extensions can have
  attention-bias and modality-transition issues.

## Related RoPE Extensions

### ComRoPE

- URL: https://arxiv.org/abs/2506.03737
- Why it matters: generalizes RoPE with trainable commuting angle matrices and
  focuses on positional robustness/scalability.
- Relevance: not Qwen-specific, but useful for understanding "robust positional
  embeddings" language.

### Rethinking RoPE: A Mathematical Blueprint for N-dimensional Positional Embedding

- URL: https://arxiv.org/abs/2504.06308
- Why it matters: broader mathematical framing for N-dimensional RoPE.
- Relevance: background if we later care about graph/hex-board positional encodings.

## What To Look For In The Papers

When reading, search within PDFs for:

```text
frequency allocation
full frequency utilization
positional coherence
textual priors
MRoPE-I
MRoPE-Interleave
Multi-Head RoPE
spatial reset
temporal height width
video-text transition
long video
```

## Catan Relevance

For Catan screenshots, the time axis is mostly irrelevant unless we feed video or
multi-turn board sequences. The useful part is the principle:

```text
separate position axes should not be starved of useful frequency bands
```

For our board parser:

- width/height positional fidelity matters for tile/node/edge/port grounding
- long video reasoning is less relevant than stable 2D spatial indexing
- MRoPE-like ideas suggest why Qwen3-VL might be better at layout than older VLMs
- but this does not prove it can resolve tiny roads/ports at 512px

Important practical point:

Qwen3-VL's processor still has patch size `16` and spatial merge size `2` in the
Hugging Face config, so a merged visual token is effectively around `32 x 32`
pixels. Interleaved-MRoPE helps encode where visual tokens are; it does not
recover information lost because the board was too low-resolution or too densely
packed.

## Suggested Catan Experiment

Use CatanBench to test whether the positional encoding advantage matters:

1. Run the same visual QA on:
   - Qwen2.5-VL
   - Qwen3-VL
   - non-Qwen VLM of similar size
2. Keep image size, prompt, and decoding fixed.
3. Compare categories that are mainly positional:
   - `node_occupancy`
   - `edge_road_owner`
   - `port_type_nodes`
   - `robber_tile`
4. Add a target-highlight ablation:
   - if Qwen3-VL improves mainly without highlight, positional encoding may help
   - if all models improve similarly with highlight, the bottleneck is likely
     search/local crop rather than global positional encoding
5. Add a rotated/flipped board stress test if renderer supports it. A robust
   spatial encoding should not collapse when absolute pixel coordinates shift.

## Reading Order

1. RoFormer: https://arxiv.org/abs/2104.09864
2. Rotary Position Embedding for Vision Transformer: https://arxiv.org/abs/2403.13298
3. Qwen2-VL: https://arxiv.org/abs/2409.12191
4. Qwen2.5-VL: https://arxiv.org/abs/2502.13923
5. Revisiting Multimodal Positional Encoding: https://arxiv.org/abs/2510.23095
6. Qwen3-VL Technical Report: https://arxiv.org/abs/2511.21631
7. VideoRoPE: https://arxiv.org/abs/2502.05173
8. VRoPE: https://arxiv.org/abs/2502.11664
