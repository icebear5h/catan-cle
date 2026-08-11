# Residual Stream Board Representation Experiment

## Research Question

Can a 9B VLM hold a meaningful enough Catan board representation in its residual stream for text-token reasoning to answer engine-verifiable board questions?

The desired evidence chain is:

1. The model answers board questions correctly.
2. Board facts are decodable from hidden states.
3. Patching or ablating those hidden states changes the answer in the predicted direction.

## Hypothesis

After atlas and board-QA SFT, mid-to-late residual stream states at answer-relevant text positions will linearly encode tile/node/edge facts, and causal patching those states will control board-QA answers.

## Representation Framing

Do not frame the target representation as a permanent fact like:

```text
<N11> = BLACK SETTLEMENT
```

That is a transient board-state binding, not the stable meaning of `<N11>`.

The stable representation should be the Catan atlas manifold:

```text
<T09> -> fixed hex location, neighboring tiles, neighboring nodes, board coordinate
<N11> -> fixed vertex location, adjacent tiles, adjacent edges, coast/interior/port status
<E23> -> fixed edge segment, endpoint nodes, adjacent tiles
<P06> -> fixed port slot and touching nodes
```

The image-conditioned representation should then bind current pixels into that
stable atlas:

```text
pixels over atlas region <T09> -> current resource/number/robber state
pixels over atlas region <N11> -> current owner/building state
pixels over atlas region <E23> -> current road owner/empty state
pixels over atlas region <P06> -> current trade resource/ratio state
```

So the claim is not that the model should memorize that `<N11>` is black. The
claim is that it should learn a stable geometry for `<N11>`, look at the pixels
covering that region in this image, and bind the transient visual state back to
the atlas token.

Some priors can still be stable, such as whether a node is coastal, interior,
port-adjacent, high-degree, or historically often occupied. Those are heuristics,
not answers. The model must override them with pixels.

The clean mechanistic target is:

1. Atlas tokens form a structured board-position manifold.
2. Image evidence writes transient object/color/resource/number facts onto that manifold.
3. Probes can reconstruct the current board tensor from image-conditioned token states.
4. Causal patches to those states change board answers in the expected direction.

## Special Tokens As Spatial IDs

The VLM spatial-ID literature studies whether object-word activations contain a
linearly extractable location component. A rough form is:

```text
activation("cup") ~= object_semantics("cup") + spatial_id(current cup location)
```

Catan gives a cleaner handle because atlas positions are already named. Our
special tokens should become explicit spatial IDs:

```text
activation("<N11>") ~= stable_spatial_id(<N11>)
                    + transient_visual_state(pixels over <N11>)
                    + question/task context
```

This is stronger than generic object words because `<N11>` always refers to the
same board coordinate, while an object word like "cup" can appear anywhere.

The goal of adding tokens like `<N11>`, `<T09>`, `<E23>`, and `<P06>` is not only
format cleanliness. They provide intervention/probing anchor points where we can
measure whether the model has learned:

- stable atlas geometry for each board position
- visual binding from pixels to the current state at that position
- downstream use of that binding when answering board questions

Extraction plan:

1. Collect hidden states at each atlas token across many boards.
2. Average across varied occupants/resources/colors to estimate stable spatial
   identity.
3. Compare same-token states across different board images to estimate transient
   visual bindings.
4. Check whether token-distance geometry matches board topology.
5. Patch token states across contrastive boards and verify answer flips.

## Data

Generate board images from engine states and label them from engine state.

Minimum labels:

- tile resource by tile ID
- tile number by tile ID
- robber tile coordinate and tile ID
- node owner and building type by node ID
- edge owner by edge ID or node-pair
- port type by port ID and port nodes
- visible VP by player
- longest visible road length by player
- official Longest Road holder
- legal settlement nodes for current player
- legal road edges for current player

Use held-out splits by board seed and game trajectory, not by question row, so the eval tests visual generalization rather than duplicate memorization.

## QA Tiers

### Tier 0: Atlas

Tests memorized fixed topology.

- Which nodes touch tile 8?
- Which edge connects node 31 and node 32?
- Which tile is at cube coordinate (-1, 0, 1)?
- Which port slot touches nodes 12 and 13?

### Tier 1: Atomic Perception

Tests direct visual facts.

- Where is the robber?
- What resource is on tile 8?
- What number is on tile 8?
- Who owns node 42?
- Who owns edge (31, 32)?

### Tier 2: Grounded Lists

Tests complete extraction.

- List all red settlement node IDs.
- List all blue road edges.
- List every ore tile and number.
- List all occupied nodes.

### Tier 3: Derived Board Facts

Tests graph reasoning over perceived state.

- How long is red's longest visible road chain?
- Which player has the longest visible road chain?
- Which settlements touch the robber-blocked tile?
- Which player has the most visible public VP?

### Tier 4: Affordance

Tests board reasoning against engine legality.

- Where can red legally build a settlement?
- Which red road endpoints allow expansion?
- Which open settlement nodes touch ore?
- Which ports does blue have access to?

## Model Conditions

Run each benchmark under these conditions:

1. Base 9B VLM, image only.
2. Base 9B VLM, image plus atlas prompt.
3. SFT 9B VLM after atlas curriculum.
4. SFT 9B VLM after atlas plus board QA.
5. Gold symbolic text only.
6. Image plus gold symbolic text.

This separates visual failure from reasoning failure.

## Hidden-State Probing

For every example, save hidden states at:

- visual token positions
- question token positions
- last prompt token before answer
- first generated answer token
- final answer token

Train probes by layer and token group:

```text
residual[layer, token] -> robber_tile_id
residual[layer, token] -> tile_8_resource
residual[layer, token] -> node_42_owner
residual[layer, token] -> edge_17_owner
residual[layer, token] -> red_longest_road_length
```

Start with linear probes. Use nonlinear probes only as a diagnostic. If a fact needs a large nonlinear probe, it may exist in the representation without being conveniently usable by the language head.

## Causal Tests

Create contrastive pairs:

```text
Board A: robber on tile 3
Board B: robber on tile 11
Question: "Where is the robber?"
```

Patch activations from Board A into Board B at selected layers/tokens.

Evidence levels:

- Weak: answer accuracy improves.
- Medium: linear probes recover the fact.
- Strong: activation patching flips the answer in the expected direction.

Repeat for:

- robber location
- node owner
- edge owner
- port type
- longest road length

## Expected Positive Pattern

After SFT, we want:

- higher QA accuracy
- higher layerwise probe accuracy
- board facts decodable earlier or more cleanly
- stronger causal patching effects
- less reliance on text priors when image labels conflict with prompt priors

## Failure Modes

- The model memorizes answer priors, such as fixed port slots, but fails current port type.
- The model answers easy captions but misses exact node/edge IDs.
- Board facts are decodable from visual tokens but not answer-position tokens.
- Probes work, but patching does not change answers; the model has information it does not use.
- Derived questions fail even when atomic perception works, meaning graph reasoning needs symbolic scaffolding or explicit chain data.

## First Implementation Target

Build a dataset generator that emits:

```json
{
  "state_id": "seed_123_step_45",
  "image_path": "data/vlm_board_qa/images/seed_123_step_45.png",
  "question": "Where is the robber?",
  "answer": "tile_id:12",
  "answer_type": "tile_id",
  "tier": "atomic_perception",
  "engine_label": {"robber_coordinate": [0, -2, 2], "robber_tile_id": 12}
}
```

Keep answer formats narrow and exact so eval does not depend on LLM judgment.
