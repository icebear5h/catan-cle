# Operationalizing a Learned Catan Board Representation

Date: 2026-08-12

## Proposed claim

For this project, use the following preregistered definition:

> A model has learned and uses the Catan board representation on task family
> `F` when its internal entity states encode the atlas's typed incidence,
> orientation, and graph-distance variables in a low-complexity form on held-out
> states, and interventions on the aligned internal variables cause the model's
> answers or legal-action scores to change according to the corresponding
> symbolic board intervention while preserving unrelated facts.

This claim is always qualified by:

- the task family tested;
- the atlas and orientation version;
- the observation modality;
- the layers/entity positions tested;
- the intervention distribution.

It is stronger than behavioral accuracy or probing alone and weaker than
claiming a complete, general-purpose world model.

## What the representation is

There is no meaningful statement that `25 is less than/below 27` merely because
`25 < 27`. Canonical IDs are names, not coordinates. The base atlas has:

- a stable **topological structure**: typed incidence, adjacency, paths, and
  graph distance;
- a conventional **oriented embedding**: six screen directions and absolute
  vertex positions under `engine-v1` orientation;
- changing **state bindings**: resource/number, port type, robber, roads, and
  buildings.

For every directed playable road `(u,v)`, the current atlas's derived vertex
coordinate displacement belongs to exactly one of six values:

```text
UP          ( 1, 1,-2)
UP_LEFT     (-1, 2,-1)
UP_RIGHT    ( 2,-1,-1)
DOWN_LEFT   (-2, 1, 1)
DOWN_RIGHT  ( 1,-2, 1)
DOWN        (-1,-1, 2)
```

Names are conventional; the six displacement classes are exact. Thus:

```text
N25 -> N26 = DOWN
N26 -> N27 = DOWN_LEFT
N25 -> N27 = not adjacent; shortest path length 2
```

Separate claims must be made for topology and orientation. A graph policy can
know routes perfectly while being invariant to whether a renderer calls one
side “top.” Absolute orientation is meaningful only after anchoring the atlas or
camera frame and otherwise is identifiable up to an atlas automorphism.

## Evidence ladder

### Level 0 — architecturally supplied geometry

If the model receives adjacency masks, coordinates, graph-relative attention
biases, or one learned embedding per canonical ID, that geometry is supplied.
Successful use is valuable but not emergent.

Audit and report geometry available at:

- raw ID embeddings;
- coordinate/positional embeddings;
- graph encoder input and attention masks;
- base/pre-fine-tuning checkpoint;
- untrained control encoder;
- every trained layer.

Claim language: “The policy successfully uses supplied board geometry.”

### Level 1 — behavioral knowledge

Evaluate exhaustive engine-labeled relations:

- 72 node-node adjacencies among all 1,431 unordered node pairs;
- edge endpoints;
- tile-node and tile-edge incidence;
- port-edge and port-node attachment;
- six directed road orientations;
- all-pairs shortest-path distances;
- route and legal-target questions under dynamic occupancy.

Use held-out games and minimally changed state pairs. Include arbitrary label
permutations only when the topology is supplied separately; otherwise a
fixed-ID model is intentionally atlas-specific.

Suggested gate:

- static relation accuracy at least 99%;
- 95% Wilson lower bound at least 98%;
- counterfactual answer consistency at least 95%;
- no invalid target output after the engine mask.

Claim language: “The model answers fixed-atlas geometric queries accurately.”

### Level 2 — low-complexity decodability

At each layer/entity position, freeze the model and fit preregistered linear
probes for:

- canonical identity;
- `(x,y)` or named vertex coordinate;
- six direction classes for ordered adjacent pairs;
- adjacency and typed incidence;
- graph distance;
- transient resource, number, port type, owner, building, road, and robber.

Use linear probes first. Powerful nonlinear probes can compute the task rather
than read it. Follow Hewitt and Liang with random-label control tasks and report
selectivity. Compare with raw embeddings, explicit coordinate features,
untrained model, and degree-only baselines.

A fixed atlas makes “leave-node-out” geometry decoding subtle: if every entity
has an arbitrary learned ID embedding, unseen IDs cannot be expected to
extrapolate. Use both:

1. held-out board-state examples per entity, measuring stable decodability;
2. held-out entities or relabeled/topology-supplied variants, measuring
   algorithmic generalization beyond ID lookup.

Report AUROC/AUPRC/MCC for sparse adjacency, not accuracy alone. For graph
metric, report shortest-path Spearman correlation, exact distance, and MAE.
For oriented geometry, Procrustes-align coordinate predictions before scoring
because rotations/reflections and some latent basis changes are not substantive.

Claim language: “Atlas topology/orientation is linearly decodable from layer L.”

### Level 3 — representation geometry

Ask whether relations are reflected in latent geometry, without requiring a
one-dimensional ordering.

- Compute cross-board entity centroids or crossvalidated representational
  distances.
- Compare their representational dissimilarity matrix with graph shortest-path
  distance using RSA and a graph-aware/QAP permutation test.
- Fit a structural probe whose transformed squared distances predict graph
  distance, analogous to structural syntax probes.
- Compare the learned subspace with atlas coordinate or graph-Laplacian
  eigenspaces using orthogonal Procrustes/CKA.
- Test six displacement directions as approximately consistent transformations:
  `h(v) ~= A_direction h(u)` for every edge `u -> v`.
- Test all six valid base-atlas `D3` automorphisms for equivariance.

Do not require nearest latent neighbors to exactly equal graph neighbors.
Task-relevant similarity can cluster all ports, occupied nodes, or resource
classes. Geometry may live in a subspace rather than dominate raw cosine
similarity.

Claim language: “A low-dimensional subspace has geometry aligned with the atlas.”

### Level 4 — causal use

This is the operational threshold for “learned and uses.”

Construct clean/source board pairs that differ in exactly one interpretable
variable. Examples:

```text
source: N27 is the destination reached from N26 by DOWN_LEFT
base:   N42 is the destination reached from N26 by DOWN_LEFT

source: P00 is 2:1 BRICK
base:   P00 is 2:1 ORE
```

Use activation patching or Distributed Alignment Search (DAS):

1. Define a symbolic high-level causal model.
2. Run base and source inputs.
3. Copy only the aligned geometry/state subspace from source to base.
4. Compute the symbolic counterfactual answer/action under the same variable
   replacement.
5. Test whether neural and symbolic counterfactual outputs agree on held-out
   base/source pairs.

Report interchange intervention accuracy (IIA), answer flips, normalized logit
recovery, and side effects. Include:

- random subspaces of equal rank;
- wrong layers and wrong entity positions;
- matched-norm random perturbations;
- on-manifold source activation swaps;
- causal erasure/ablation of the proposed subspace;
- specificity on facts that should not change.

Suggested exploratory gate:

- expected answer/action flip in at least 80% of initially correct pairs;
- median normalized logit recovery at least 0.5 with bootstrap lower bound over
  0.3;
- non-target facts preserved in at least 95%;
- geometry-subspace ablation causes at least a 20-point topology-task drop;
- equal-rank random ablations cause no more than a 5-point drop.

These numerical thresholds are project acceptance criteria, not field-wide
standards. Preregister and revise only between experiments.

Claim language: “A localized/distributed atlas representation is causally used
for task family F.”

### Level 5 — a mechanism or circuit

To claim a mechanism, identify how input facts become atlas variables and how
those variables cause outputs:

```text
entity state
  -> identity/topology/state features
  -> route or production feature
  -> candidate action value
  -> selected legal action
```

Use sparse features/SAEs or transcoders to generate hypotheses, feature
attribution/path patching to localize paths, and interventions in the underlying
model to validate predicted downstream changes.

Anthropic's current circuit-tracing workflow is:

1. decompose activations into candidate interpretable features;
2. link active features in prompt-local attribution graphs;
3. label and group candidate mechanisms;
4. perturb the underlying model and compare actual downstream effects to graph
   predictions;
5. explicitly report reconstruction error, missing attention-QK computation,
   and incomplete global coverage.

Claim language: “Circuit C implements part of relation/action computation R on
this evaluated distribution.”

## Attention-map policy

Attention is a routing operation, not a representation definition or a complete
explanation. A node query attending to nearby nodes is an encouraging diagnostic,
but not sufficient evidence because:

- attention weights omit the values being transported;
- residual and MLP paths can dominate the answer;
- different attention maps can yield equivalent outputs;
- a head may attend to an entity to suppress, copy, compare, or ignore it;
- architectural graph masks can make local attention tautological.

For each head/layer and query type:

- compute attention mass by graph distance 0, 1, 2, and 3+;
- normalize by the number of eligible entities at each distance;
- compare with degree-, token-position-, and mask-matched nulls;
- evaluate consistency across held-out boards and question phrasings;
- inspect value/output vectors, not only attention probabilities;
- ablate or patch the head's value/output path and test predicted answer/logit
  changes.

Safe language before causal validation: “Head H often routes information from
adjacent entities.” Never: “The attention map proves the model reasons locally.”

## Required experiment matrix

```text
Checkpoint/input             Behavior  Probe  Geometry  Causal
base LLM + text IDs             x        x       x        x
SFT LLM + canonical DSL         x        x       x        x
board encoder before training   x        x       x        x
trained board encoder + LLM     x        x       x        x
trained model, shuffled state   x        x       x        x
trained model, random topology  x        x       x        x
```

For every condition, test:

- atlas-static facts separately from dynamic state binding;
- topology separately from absolute orientation;
- state decoding separately from derived reasoning;
- answer correctness separately from action value;
- text, exact structured state, and vision modalities where applicable.

## Strongest Catan-specific intervention

The most compelling Othello-style demonstration would be:

1. Give a legal Catan state and a route query/action choice.
2. Patch the internal representation so one edge endpoint or occupancy binding
   matches a counterfactual but coherent donor board.
3. Observe the model choose a candidate legal in the counterfactual topology or
   state and not in the original.
4. Show the exact corresponding route/production features changed, while
   unrelated tile, port, and player facts stayed fixed.
5. Repeat over hundreds of held-out pairs, layers, and random controls.

Because actual Catan topology is fixed, topology interventions may be
out-of-distribution. Prefer dynamic binding interventions first (robber, road,
building, port type), then use atlas automorphisms and topology-supplied relabeling
for geometry. State clearly when a topology edit creates a hypothetical atlas.

## Literature precedents

- **Othello-GPT:** Li et al. trained a sequence model only on moves, decoded the
  64-square board from hidden states, and then altered probe-aligned activations
  so outputs followed counterfactual boards. This is the closest operational
  precedent for a board representation. Nanda and later work found a simpler
  player-relative linear ontology (`mine/theirs`, not merely `black/white`), a
  warning that the right probe labels may differ from the researcher's first
  ontology.
- **Language Models Represent Space and Time:** Gurnee and Tegmark linearly
  decoded geographic coordinates across scales and prompts and found
  spatially-selective dimensions. This supports low-complexity decodability,
  but the paper does not establish causal use for spatial answering.
- **Structural probes:** Hewitt and Manning fit a linear transformation whose
  squared distances encode parse-tree distances. This is a direct template for
  testing whether graph shortest-path distance is embedded in Catan entity
  states.
- **Probe controls:** Hewitt and Liang show high probe accuracy can come from
  probe memorization; random-label controls and selectivity contextualize it.
  Amnesic probing demonstrates that conventional probe accuracy need not track
  behavioral importance and tests removal instead.
- **Causal abstraction/DAS:** Geiger et al. define representation claims through
  alignment between high-level causal variables and distributed neural
  subspaces, evaluated by interchange interventions and IIA.
- **Attention is not Explanation:** Jain and Wallace show standard attention
  weights may be uncorrelated with gradient importance and very different
  attention distributions can yield equivalent predictions.
- **Anthropic feature work:** dictionary learning/SAEs expose recurring
  distributed features; activation examples and automated labels generate
  hypotheses; feature amplification/suppression provides causal evidence.
  Anthropic explicitly notes that finding features does not by itself explain
  how they are used.
- **Anthropic circuit tracing:** prompt-local attribution graphs connect features
  into candidate computations and perturbations validate them. The Texas-to-
  California intervention changing Austin to Sacramento is the right form of
  evidence for a Catan route variable. Anthropic also emphasizes that current
  graphs capture only part of computation and can miss how attention patterns
  are formed.
- **Privileged basis/superposition:** residual coordinates can be rotated under
  behavior-preserving reparameterizations, and concepts may be distributed or
  superposed. Do not demand a literal “N25 neuron” or treat raw latent axes as
  objectively meaningful.

## Bottom line

The target is not an objective scalar ordering of `N00...N53`. It is an
interpretable **causal coordinate system** for typed graph variables:

```text
identity(N25)
adjacent(N25,N26)
direction(N25,N26)=DOWN
distance(N25,N27)=2
occupant(N25)=BLUE_SETTLEMENT
port_attached(N25)=P00
```

A nearby-node attention map is a useful visualization. The research-grade claim
requires all of:

```text
correct behavior
+ low-complexity decoding
+ graph-aligned latent geometry
+ counterfactual causal control
+ matched controls and specificity
```

If only the first two pass, say the facts are behaviorally known and decodable.
If interventions also pass, say the representation is causally used. Reserve
“understood the mechanism” for a validated circuit explaining how board facts
flow into the output.

## Sources

- Anthropic, “Mapping the mind of a large language model”:
  <https://www.anthropic.com/research/mapping-mind-language-model>
- Anthropic, “Tracing the thoughts of a large language model”:
  <https://www.anthropic.com/research/tracing-thoughts-language-model>
- Lindsey et al., “Circuit Tracing: Revealing Computational Graphs in Language
  Models”:
  <https://transformer-circuits.pub/2025/attribution-graphs/methods.html>
- Transformer Circuits, “Privileged Bases in the Transformer Residual Stream”:
  <https://transformer-circuits.pub/2023/privileged-basis/>
- Li et al., “Emergent World Representations: Exploring a Sequence Model
  Trained on a Synthetic Task”:
  <https://arxiv.org/abs/2210.13382>
- Nanda, “Actually, Othello-GPT Has A Linear Emergent World Representation”:
  <https://www.neelnanda.io/mechanistic-interpretability/othello>
- Zhang et al., “Linear Latent World Models in Simple Transformers”:
  <https://arxiv.org/abs/2310.07582>
- Gurnee and Tegmark, “Language Models Represent Space and Time”:
  <https://arxiv.org/abs/2310.02207>
- Hewitt and Manning, “A Structural Probe for Finding Syntax in Word
  Representations”:
  <https://aclanthology.org/N19-1419/>
- Hewitt and Liang, “Designing and Interpreting Probes with Control Tasks”:
  <https://aclanthology.org/D19-1275/>
- Elazar et al., “Amnesic Probing”:
  <https://aclanthology.org/2021.tacl-1.10/>
- Geiger et al., “Causal Abstraction: A Theoretical Foundation for Mechanistic
  Interpretability”:
  <https://arxiv.org/abs/2301.04709>
- Geiger et al., “Finding Alignments Between Interpretable Causal Variables and
  Distributed Neural Representations”:
  <https://arxiv.org/abs/2303.02536>
- Park et al., “The Linear Representation Hypothesis and the Geometry of Large
  Language Models”:
  <https://proceedings.mlr.press/v235/park24c.html>
- Jain and Wallace, “Attention is not Explanation”:
  <https://aclanthology.org/N19-1357/>
