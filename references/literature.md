# Catan VLM Literature Map

This folder tracks the papers and claims that matter for a Catan board-understanding VLM. The core claim is narrow: a 9B VLM can learn a fixed Catan board atlas and answer engine-verifiable public-board questions from screenshots.

Project-specific benchmark evidence:

- [Catan Bench 100 Initial Findings](catan_bench_100_findings.md)
- [Catan VLM Model Selection](vlm_model_selection.md)
- [VLM Benchmarking Literature Review For CatanBench](vlm_benchmarking_review.md)

## Board Games and Symbolic State

### Playing Catan with Cross-dimensional Neural Network

- Link: https://arxiv.org/abs/2008.07079
- Authors: Quentin Gendre, Tomoyuki Kaneko
- Why it matters: Catan is explicitly framed as a mixed-structure game with hex faces, vertices, edges, player cards, imperfect information, stochasticity, and a large action space.
- Takeaway for this repo: tile IDs, node IDs, edge IDs, ports, cards, and legal actions should be represented as separate structured objects. Do not collapse the board into a generic caption.

### Determining Chess Game State from an Image

- Link: https://www.mdpi.com/2313-433X/7/6/94
- Authors: Georg Wolflein, Ognjen Arandjelovic
- Why it matters: A board-game vision system maps photos into structured engine-compatible state. It uses synthetic renderings, board localization, occupancy classification, and piece classification.
- Takeaway for this repo: synthetic board images with engine labels are defensible. The output should be structured state, not prose.

### AlphaZero

- Link: https://arxiv.org/abs/1712.01815
- Authors: David Silver et al.
- Why it matters: Strong board-game systems usually consume symbolic board state, not raw screenshots.
- Takeaway for this repo: image understanding should be evaluated by whether it recovers or supports usable board-state representations.

## Diagnostic Visual Reasoning

### CLEVR

- Link: https://web.eecs.umich.edu/~justincj/clevr/
- Authors: Justin Johnson et al.
- Why it matters: Synthetic scenes, generated questions, scene-graph labels, and functional programs create controlled visual reasoning benchmarks.
- Takeaway for this repo: generate Catan QA from engine state: robber location, tile resources, node occupancy, port type, road ownership, longest visible road, and legal settlement affordances.

### CLEVR-Dialog

- Link: https://arxiv.org/abs/1903.03166
- Why it matters: Multi-round visual dialog can be studied with synthetic state and generated dialog traces.
- Takeaway for this repo: multi-chat Catan benchmarks should come after single-turn visual QA works.

### ChartQA

- Link: https://arxiv.org/abs/2203.10244
- Why it matters: Evaluates visual plus logical/arithmetic reasoning on structured graphics.
- Takeaway for this repo: derived board questions should be separated from atomic perception. "Where is robber?" and "who has longest road?" are different skill tiers.

## Neuro-Symbolic Visual Reasoning

### Neural-Symbolic VQA

- Link: https://arxiv.org/abs/1810.02338
- Authors: Kexin Yi et al.
- Why it matters: The system recovers a structural scene representation from an image, parses the question into a program, and executes the program on symbolic state.
- Takeaway for this repo: ideal Catan VQA decomposition is screenshot -> public board state -> symbolic engine query.

### Neuro-Symbolic Concept Learner

- Link: https://arxiv.org/abs/1904.12584
- Authors: Jiayuan Mao et al.
- Why it matters: Object-centric representations plus executable programs support compositional generalization.
- Takeaway for this repo: train from atlas questions into object/graph facts before asking strategic questions.

## VLM Spatial Grounding and Failure Modes

### Vision language models are blind

- Link: https://arxiv.org/abs/2407.06581
- Authors: Pooyan Rahmanzadehgervi et al.
- Why it matters: Modern VLMs can fail simple precise spatial/geometric questions even when visual encoders contain enough information.
- Takeaway for this repo: vague board captions are not enough. Use exact tile/node/edge grounding metrics.

### SpatialVLM

- Link: https://arxiv.org/abs/2401.12168
- Authors: Boyuan Chen et al.
- Why it matters: Spatial ability improves with explicit spatial VQA supervision.
- Takeaway for this repo: Catan spatial data should directly ask about coordinates, adjacency, relative positions, and graph paths.

### Set-of-Mark Prompting

- Link: https://arxiv.org/abs/2310.11441
- Authors: Jianwei Yang et al.
- Why it matters: Marking image regions with speakable alphanumeric labels improves fine-grained visual grounding.
- Takeaway for this repo: use a curriculum from fully labeled boards to faded labels to unlabeled boards. This is especially relevant for node IDs and edge IDs.

### MME and SEED-Bench

- Links:
  - https://arxiv.org/abs/2306.13394
  - https://arxiv.org/abs/2307.16125
- Why they matter: Modern VLM evaluation separates perception and cognition and favors objective answer formats.
- Takeaway for this repo: keep benchmark answers exact and engine-checkable.

## Mechanistic Interpretability

### A Mathematical Framework for Transformer Circuits

- Link: https://transformer-circuits.pub/2021/framework/
- Authors: Nelson Elhage et al.
- Why it matters: The residual stream is the central communication channel in transformer computation.
- Takeaway for this repo: if Catan board facts become usable by text tokens, they should be decodable from residual stream states at relevant layers/tokens.

### Are Vision-Language Transformers Learning Multimodal Representations?

- Link: https://ojs.aaai.org/index.php/AAAI/article/view/21375
- Authors: Emmanuelle Salin et al.
- Why it matters: Probing can reveal whether multimodal representations encode visual facts or rely on language bias.
- Takeaway for this repo: evaluate whether Catan facts are represented in hidden states, not only whether answers are correct.

### Multimodal Neurons in Pretrained Text-Only Transformers

- Link: https://arxiv.org/abs/2308.01544
- Authors: Sarah Schwettmann et al.
- Why it matters: Visual concepts may be translated into text-transformer representations deeper inside the model, with identifiable causal neurons.
- Takeaway for this repo: compare visual-token residual states, question-token states, and answer-token states across layers.

### Towards Automated Circuit Discovery for Mechanistic Interpretability

- Link: https://arxiv.org/abs/2304.14997
- Authors: Arthur Conmy et al.
- Why it matters: Activation patching and automated circuit discovery can identify components causally responsible for behavior.
- Takeaway for this repo: after probes show board facts are decodable, use patching/ablation to test whether those facts are causally used.
