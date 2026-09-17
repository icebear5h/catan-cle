# Symbolic spatial SFT with the existing atlas checkpoint

Research checked September 12, 2026. Scope: reuse the existing checkpoint and
its learned `<N00>`, `<E00_01>`, `<T00>`, and port vocabulary, while changing the
input to explicit symbolic board state. No new training was launched.

No inspected paper establishes this exact Catan checkpoint/input transition.
The relevant precedents cover spatial supervision, pretrained-model adaptation,
discrete graph vocabularies, and causal representation analysis separately.

## Start here

### 1. StepGame — directional question generation

- [AAAI 2022 paper](https://ojs.aaai.org/index.php/AAAI/article/view/21383)
- [Official code](https://github.com/ZhengxiangShi/StepGame)
- [Author-linked updated dataset](https://huggingface.co/datasets/michaelszx/StepGame)

Textual pairwise spatial facts support left/right, above/below, diagonal, and
overlap questions, with variable reasoning depth and distractors. The original
training set spans one to five hops, with tests extending to ten. Use the
author-designated updated dataset rather than known-problematic old wording.

The original work trains supervised reasoning models, not a pretrained decoder
with new spatial vocabulary. Entities are ordinary textual labels. Borrow the
question design and regenerate facts from Catan's actual atlas; arbitrarily
mapping generated layouts onto fixed Catan addresses would contradict their
established geometry. Hop stratification is not proof that staged curriculum
training beats a mixture.

### 2. SpaRTUN — spatial graphs to exact supervised questions

- [EMNLP 2022 paper](https://aclanthology.org/2022.emnlp-main.413/)
- [Official generator](https://github.com/HLR/SpaRTUN)
- [Training implementation](https://github.com/HLR/Spatial-QA-tasks)
- [Earlier SpaRTQA paper](https://aclanthology.org/2021.naacl-main.364/)

Constructs spatial graphs, derives relations with a symbolic reasoner, and
renders text/questions. Covers yes/no and relation-set answers, supporting
facts, distractors, and varied wording. This is a strong data-construction
precedent for board graph -> exact query/answer pairs.

Its supervised transfer experiments use BERT and classification losses, not
chat-style Qwen SFT or newly added address tokens. Adapt relation semantics
carefully: Catan adjacency and incidence are not containment relations.

### 3. GraphWiz / GraphInstruct — actual decoder LLM SFT

- [KDD 2024 paper](https://arxiv.org/abs/2402.16029)
- [Official code](https://github.com/nuochenpku/Graph-Reasoning-LLM)
- [Supervised dataset](https://huggingface.co/datasets/GraphWiz/GraphInstruct-RFT-72K)

Fine-tunes pretrained Llama 2 and Mistral on textual graphs, questions, reasoning,
and answers. Tasks include connectivity, shortest paths, and cycle detection.
Graphs and answers are generated using NetworkX. Reasoning traces are generated
and filtered by final-answer correctness; the paper also evaluates a separate
DPO stage.

Nodes are ordinary integer text, not a custom graph vocabulary. The supervised
format transfers to our checkpoint; DPO is not required to borrow it. For Catan,
derive supporting facts and traces deterministically where possible, since a
correct final answer alone does not validate every reasoning step.

## Strong complementary resources

### 4. SpaRTUN Q-Chain — intermediate spatial questions

- [Findings of NAACL 2025 paper](https://aclanthology.org/2025.findings-naacl.128/)
- [Official code](https://github.com/HLR/SpaRTUNQChain)

Uses chains of intermediate questions plus differentiable logical constraints,
including inverse and compositional relations. Fine-tunes BERT and Flan-T5;
GPT/Llama results are prompting comparisons. No dedicated spatial vocabulary.
Borrowing its question chains for ordinary Qwen SFT is an adaptation, not a
reproduction of the constraint-based objective.

### 5. Alibaba GraphGPT — dedicated discrete graph vocabulary

- [Graph Eulerian Transformer paper, ICML 2025](https://arxiv.org/abs/2401.00529)
- [Official code](https://github.com/alibaba/graph-gpt)

Serializes graph structure and attributes into reversible token sequences,
with dedicated structural/attribute vocabulary. Distinguishes local structural
indices from persistent global identities. A representation reference for
discrete graph tokens, not evidence that numeric-looking IDs carry geometry.

Trains graph models from random initialization, followed by supervised task
heads. This is not a recipe for discarding the current language checkpoint.
Do not copy its local node-reindexing augmentation onto fixed atlas identities.

This differs from [HKUDS GraphGPT](https://arxiv.org/abs/2310.13023), which uses
a graph encoder/projector and continuous node features inserted at special
placeholders. Similar names and token spellings conceal different architectures.

### 6. ChessGPT — reuse a language checkpoint for symbolic game tasks

- [NeurIPS 2023 paper](https://arxiv.org/abs/2306.09200)
- [Official code](https://github.com/waterhorse1/ChessGPT)

Continues RedPajama-3B on chess/game-language data, then instruction-tunes a chat
variant. Uses PGN/UCI move notation and FEN board state with the existing text
tokenizer, not one new token per square. Useful adaptation precedent; continued
language modeling and response SFT are distinct stages. Its FEN reconstruction
metric includes normalized edit similarity, not whole-board exact matching.

### 7. Othello-GPT and linear board representations — causal auditing

- [Original paper, ICLR 2023](https://arxiv.org/abs/2210.13382)
- [Original code](https://github.com/likenneth/othello_world)
- [Linear-representation paper](https://arxiv.org/abs/2309.00941)
- [Author walkthrough](https://www.neelnanda.io/mechanistic-interpretability/othello)
- [Linear-representation code](https://github.com/ajyl/mech_int_othelloGPT)

Othello-GPT learns from move sequences using categorical square tokens. Board
labels train probes afterward; they are not the model's original supervision.
The original GPT is trained from scratch; the linear-representation study
reuses its weights. Useful findings include player-relative representation
targets and activation interventions that change legal-move predictions.

Use these as auditing precedents for the existing Catan model. Decoding an
explicitly supplied board can be copying; derived answers and selective causal
interventions provide stronger evidence that a representation is used.

## Project synthesis

1. Preserve the checkpoint's exact tokenizer/token IDs, trained input/output
   atlas rows, and language adapter. Text-only inputs do not require restarting
   from base. Measure how much the learned representation transfers.
2. Generate board state + question -> answer examples with the engine and atlas
   as the label oracle. Existing atlas tokens remain persistent addresses.
3. Teach direct geometry and inverse questions, then state-conditioned geometry,
   incidence/connectivity, and compositions. This ordering is a proposal, not an
   experimentally established optimal curriculum.
4. Define canonical orientation and ties. Graph connectivity alone cannot
   determine geometric left/right. An atomic `<E00_01>` also does not expose its
   two endpoints as separately readable input tokens.
5. Distinguish fixed-atlas recall from supplied-state use. Both are useful;
   questions about current pieces should change answers when occupancy changes.
6. Hold out board configurations and query compositions; test record-order
   robustness so left/right does not become an input-order shortcut.
7. Keep longest-road reasoning as a separately evaluated graph/rule task.
   Left/right fluency does not establish traversal, branching, or blocking-rule
   competence. The prior failure did not isolate visual from reasoning errors.
