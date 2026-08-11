# VLM Spatial Representation Literature Keywords

Use this as a reading map for the Catan residual-stream idea: stable atlas
geometry, pixel-conditioned transient bindings, and downstream reasoning.

## Core Keywords

- `spatial IDs in vision language models`
- `VLM spatial ID textual activations`
- `vision-language mechanistic interpretability`
- `causal tracing vision-language models`
- `activation patching multimodal LLM`
- `visual token representations across layers`
- `linear probes multimodal representation dynamics`
- `visual grounding task reasoning answer decoding`
- `object localization improves spatial reasoning VLM`
- `coordinate instruction tuning visual LLM`
- `board state probes world model sequence model`
- `linear representations of space and time LLM`
- `spatial VQA supervision vision language model`
- `guided visual search multimodal LLM`

## Best Starting Point

1. **Linear Mechanisms for Spatiotemporal Reasoning in Vision Language Models**
   - URL: https://arxiv.org/abs/2601.12626
   - Code: `references/external/linear-mech-vlms`
   - Why it matters: closest match to our framing. The paper argues that VLMs can
     linearly bind spatial IDs to textual activations and then reason through
     language tokens.
   - Catan translation: `<N11>`, `<T09>`, and `<E23>` should behave like stable
     atlas/spatial IDs; pixels should write transient state onto those IDs.

2. **How Multimodal LLMs Solve Image Tasks: A Lens on Visual Grounding, Task Reasoning, and Answer Decoding**
   - URL: https://arxiv.org/abs/2508.20279
   - Why it matters: uses layerwise probes to separate visual grounding, reasoning,
     and answer decoding stages.
   - Catan translation: probe whether tile/node/edge facts appear in early,
     middle, or answer-position states after SFT.

3. **Towards Interpreting Visual Information Processing in Vision-Language Models**
   - URL: https://arxiv.org/abs/2410.07149
   - Why it matters: studies visual tokens inside the language-model component and
     shows object information becoming more text-aligned across layers.
   - Catan translation: test whether visual board regions become aligned with
     resource, number, road, settlement, and robber tokens.

4. **Towards Vision-Language Mechanistic Interpretability: A Causal Tracing Tool for BLIP**
   - URL: https://arxiv.org/abs/2308.14179
   - Why it matters: direct precedent for causal tracing on image-conditioned text
     generation.
   - Catan translation: patch board-state activations and check whether answers
     flip for robber/tile/node/edge questions.

5. **What Do VLMs NOTICE? A Mechanistic Interpretability Pipeline for Gaussian-Noise-free Text-Image Corruption and Evaluation**
   - URL: https://arxiv.org/abs/2406.16320
   - Why it matters: uses semantically meaningful text-image corruption and causal
     mediation rather than noisy pixel corruption.
   - Catan translation: create clean minimal pairs where only one road, node,
     robber, or port changes.

## Spatial Training And Evaluation

6. **Learning to Localize Objects Improves Spatial Reasoning in Visual-LLMs**
   - URL: https://arxiv.org/abs/2404.07449
   - Why it matters: argues that coordinate/localization instruction tuning can
     improve spatial reasoning and reduce hallucination.
   - Catan translation: local atlas grounding tasks may be a prerequisite for
     policy reasoning.

7. **SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning Capabilities**
   - URL: https://arxiv.org/abs/2401.12168
   - Why it matters: trains VLMs on spatial VQA supervision and evaluates
     quantitative/spatial reasoning.
   - Catan translation: SFT should include explicit spatial/graph questions, not
     only strategy descriptions.

8. **V*: Guided Visual Search as a Core Mechanism in Multimodal LLMs**
   - URL: https://arxiv.org/abs/2312.14135
   - Why it matters: focuses on high-resolution visual details and selective
     visual search.
   - Catan translation: roads, ports, and number pips may require crop/search
     curricula, not only full-board images.

## Board/World-Model Analogies

9. **Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task**
   - URL: https://arxiv.org/abs/2210.13382
   - Why it matters: Othello-GPT develops an internal board-state representation
     recoverable with probes and controllable by interventions.
   - Catan translation: board tensor reconstruction from hidden states is a valid
     experimental target.

10. **Language Models Represent Space and Time**
    - URL: https://arxiv.org/abs/2310.02207
    - Why it matters: finds linear spatial/temporal representations in LLM hidden
      states.
    - Catan translation: use linear probes first for stable atlas geometry and
      transient image-conditioned board facts.

## Search Strings To Reuse

```text
"spatial IDs" "vision language models" "textual activations"
"visual grounding" "task reasoning" "answer decoding" "linear classifiers"
"vision-language mechanistic interpretability" "causal tracing"
"activation patching" "vision-language models" "visual question answering"
"visual token representations" "last token" "prediction" "VLM"
"coordinate instruction tuning" "spatial reasoning" "Visual-LLMs"
"spatial VQA supervision" "vision-language model"
"guided visual search" "multimodal LLM" "high-resolution"
"emergent world representations" "board state" "probes"
"linear representations" "space and time" "large language models"
```

## Catan-Specific Hypothesis Language

The model should learn a stable atlas manifold for Catan board positions, then
condition that manifold on pixels to produce transient board-state bindings.
SFT should improve image-to-atlas binding, not teach fixed answers like
`<N11> = BLACK`.

The probe target is not just "can a hidden state say the answer?" It is whether
the set of image-conditioned atlas-token states can reconstruct the current board
tensor: tile resources/numbers, node occupancy, edge owners, robber, and ports.

## Special Tokens As The Catan Spatial IDs

The literature's object-word framing can be translated into explicit Catan atlas
tokens.

Generic VLM version:

```text
activation("cup") ~= object_semantics("cup") + spatial_id(current cup location)
```

Catan version:

```text
activation("<N11>") ~= stable_spatial_id(<N11>)
                    + transient_visual_state(pixels over <N11>)
                    + question/task context
```

This is why custom tokens matter for the tuned model. They are not just nicer
answer formatting. They give us stable text-token handles for board positions:

- `<N00>` through `<N53>` for vertices
- `<T00>` through `<T18>` for hexes
- edge tokens for roads
- `<P00>` through `<P08>` for ports

The expected result is that token representations for nearby board positions have
nearby or structured geometry in activation space, and that image conditioning
adds the transient state currently visible at those positions.
