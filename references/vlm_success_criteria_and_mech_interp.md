# Catan VLM Success Criteria And Mech Interp Plan

## Goal

Train and evaluate a multimodal Catan model that can bind a Colonist-style board image to
the engine's symbolic board atlas:

- tile ids: `<T00>` through `<T18>`
- node ids: `<N00>` through `<N53>`
- edge ids: canonical edge tokens like `<E12_17>`
- port ids: `<P00>` through `<P08>`

Loss is useful training telemetry, but it is not the success criterion. The real test is
engine-scored information accuracy plus causal evidence that board facts are represented in
model activations.

## Behavioral Success Criteria

Default image setting:

- 512x512 tight Colonist-style board crop
- board and ports visible
- no hidden player hand/dev-card information

Recommended extraction targets:

| Metric | Target |
| --- | ---: |
| Invalid JSON rate | < 1% |
| Tile resource accuracy | > 99% |
| Tile number accuracy | > 99% |
| Robber exact location | > 99% |
| Node occupancy accuracy | > 98% |
| Edge road-color accuracy | > 98% |
| Port resource accuracy | > 97% |
| Port attached-node accuracy | > 95% |

Track full-board exact match too, but do not rely only on it. One wrong edge makes a full
board fail, so per-field accuracy is the main diagnostic.

## Agent Success Criteria

For policy/agent evaluation, use the validated engine contract plus legal actions.

| Metric | Target / Use |
| --- | --- |
| Legal action rate | > 99.5% |
| Expert action top-1 | useful but noisy |
| Expert action top-3/top-5 | better imitation metric |
| Winrate / ELO | final policy metric |

Keep perception metrics separate from policy metrics. A model can see the board correctly and
still play badly, or play decently by relying on text while failing vision.

## Evaluation Ladder

Run the same board states through each condition:

1. **Clean symbolic render**
   - Low visual noise.
   - Tests whether the task/schema is learnable.
2. **Colonist-style board crop, 512**
   - Main target.
   - Tests realistic visual parsing.
3. **Colonist-style board crop, 768**
   - Ceiling/hard verification.
   - If 512 fails but 768 works, pixel budget is the bottleneck.
4. **Colonist-style board crop, 256**
   - Cheap ablation only.
   - Expected to lose ports and building/road boundaries.
5. **Overlay curriculum**
   - tile ids on/off
   - node ids on/off
   - edge ids on/off
   - faded overlays

Interpretation:

- Clean works, Colonist fails: asset/style problem.
- 768 works, 512 fails: resolution/effective-pixel problem.
- Overlays work, clean board fails: atlas-binding problem.
- Contract extraction works, action choice fails: policy/strategy problem.

## Mech Interp Evidence Ladder

Attention maps are weak evidence. They are useful diagnostics, but not proof of causality.

### Weak Evidence

- Attention or attribution highlights the correct visual region.
- Example: answer mentions `<N22>`, attention visually overlaps node 22.

This is not enough, because attention weights do not necessarily explain model decisions.

Reference:

- Jain & Wallace, "Attention is not Explanation" - https://arxiv.org/abs/1902.10186

### Medium Evidence

Train linear probes on cached activations:

- Does this hidden state encode whether `<N22>` is occupied?
- Does it encode whether `<N22>` touches a port?
- Does it encode the owner of `<E12_17>`?
- Does it encode the robber tile id?

Probe targets should come from the engine oracle, not model outputs.

### Strong Evidence

Use activation patching on counterfactual board pairs.

Example pair:

- Board A: `<N22>` touches a wood port.
- Board B: same layout except the relevant port/node fact differs.

Patch activations from A into B and test whether only the target answer flips:

```text
B clean answer: "<N22> is not a port node"
B with A target activation patched: "<N22> is a port node"
```

Controls:

- Patch unrelated layers/regions.
- Patch unrelated board slots.
- Patch same region for a different question.
- Verify output changes are specific, not global degradation.

### Strongest Evidence

Sparse autoencoder or circuit features:

- Feature fires for "port node".
- Feature fires for "blue road on edge".
- Feature fires for "robber on tile".
- Intervening on the feature predictably changes the model's answer.

This is closest to evidence that the model learned meaningful board circuits rather than just
surface correlations.

References:

- OpenAI, "Multimodal Neurons in Artificial Neural Networks" - https://openai.com/index/multimodal-neurons/
- Prisma vision mech-interp toolkit - https://arxiv.org/abs/2504.19475
- Sparse autoencoders in VLMs - https://huggingface.co/papers/2504.02821

## Counterfactual Dataset Design

Create paired board images with exactly one semantic fact changed:

- same board, robber moved from `<T03>` to `<T04>`
- same board, `<E12_17>` road owner changes from `<RED>` to `<BLUE>`
- same board, `<N22>` changes from empty to `<BLUE>` settlement
- same board, port resource at `<P03>` changes from `<WOOD>` to generic 3:1
- same board, a port's attached nodes change in the visual overlay/crop condition

For each pair, cache activations and score:

- output delta
- probe delta
- patched-output delta
- specificity against unrelated questions

## Symbol Recognition SOTA Context

Catan board extraction combines three hard subproblems:

1. **OCR/text-rich image recognition**
   - number tokens and port text/icons
   - benchmark analogue: OCRBench v2
   - source: https://arxiv.org/abs/2501.00321

2. **GUI/icon grounding**
   - mapping visual symbols to fixed locations
   - benchmark analogue: ScreenSpot-Pro
   - source: https://arxiv.org/abs/2504.07981

3. **Fixed-layout symbolic board parsing**
   - exact tile/node/edge/port binding
   - practical ceiling is known geometry plus slot classifiers
   - VLM goal is not just local recognition, but atlas binding in model activations

For this project, public benchmarks are useful context, but the real benchmark is
engine-scored Catan board facts.

## Working Definition Of Success

The Catan VLM is successful when:

1. It extracts the public board contract from 512x512 Colonist-style crops with high exact
   engine-scored accuracy.
2. It generalizes to held-out real replay states and generated legal edge cases.
3. Its internal activations support probes for tile/node/edge/port facts.
4. Activation patching can causally flip targeted board answers without unrelated damage.
5. The policy model can use the validated board contract plus private/legal-action context to
   choose strong legal actions.

