# Catan Research Packet: Evals, VLM SFT, LoRA, Multi-Agent Learning

Date: 2026-05-14

Purpose: collect the most useful papers, docs, blogs, and experiment ideas for
the Catan visual-grounding and multi-agent learning project.

This note is meant to be read tomorrow. It is not a final methodology claim. It
is a map of what to read, why it matters, and how it translates into concrete
CatanBoardBench/SFT work.

## Start Here

Read in this order:

1. **CatanBoardBench eval design**
   - MME: https://arxiv.org/abs/2306.13394
   - SEED-Bench: https://arxiv.org/abs/2307.16125
   - MMBench: https://arxiv.org/abs/2307.06281
   - MM-Vet: https://arxiv.org/abs/2308.02490
   - HallusionBench: https://arxiv.org/abs/2310.14566
   - HELM: https://arxiv.org/abs/2211.09110
   - Inspect: https://inspect.aisi.org.uk/
   - OpenAI Evals: https://platform.openai.com/docs/guides/evals
   - Local note: `references/vlm_benchmarking_review.md`

2. **Qwen/VLM SFT implementation path**
   - TRL VLM SFT guide: https://huggingface.co/docs/trl/v0.21.0/training_vlm_sft
   - Qwen3-VL Transformers docs: https://huggingface.co/docs/transformers/main/en/model_doc/qwen3_vl
   - Qwen3-VL model card: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
   - Qwen3-VL repo: https://github.com/QwenLM/Qwen3-VL
   - Qwen VL finetune scripts: https://github.com/QwenLM/Qwen3-VL/tree/main/qwen-vl-finetune
   - NeMo Qwen3-VL recipes: https://docs.nvidia.com/nemo/automodel/latest/model-coverage/vlm/qwen/qwen3-vl.html

3. **LoRA/QLoRA and trainable-scope decisions**
   - LoRA paper: https://arxiv.org/abs/2106.09685
   - QLoRA paper: https://arxiv.org/abs/2305.14314
   - PEFT LoRA docs: https://huggingface.co/docs/peft/en/developer_guides/lora
   - LoRA Without Regret: https://huggingface.co/docs/trl/lora_without_regret
   - DoRA: https://arxiv.org/abs/2402.09353
   - rsLoRA: https://arxiv.org/abs/2312.03732
   - LoftQ: https://arxiv.org/abs/2310.08659
   - Local note: `sft/EXPERIMENT_DESIGN.md`

4. **Catan strategy, self-play, and multi-agent learning**
   - Agents of Change / HexMachina: https://arxiv.org/abs/2506.04651
   - OpenReview HexMachina page: https://openreview.net/forum?id=V0Fb4pwhS4
   - Catanatron docs: https://docs.catanatron.com/
   - Settlers-RL writeup: https://settlers-rl.github.io/
   - Strategic Dialogue Management via DRL: https://arxiv.org/abs/1511.08099
   - OpenSpiel: https://arxiv.org/abs/1908.09453
   - PSRO: https://arxiv.org/abs/1711.00832

5. **Spatial representations and mech interp**
   - Local note: `references/vlm_spatial_representation_literature.md`
   - Local reference-code note: `references/linear_mech_vlms_reference.md`
   - Linear mechanisms paper: https://arxiv.org/abs/2601.12626

## The Catan Thesis

The project is not "teach a model Catan facts." The first hard problem is:

```text
image pixels -> fixed atlas token -> transient public board state
```

Examples:

```text
pixels over tile slot T09 -> <T09> currently has <WOOD> 11
pixels over node slot N11 -> <N11> currently has <BLACK> <SETTLEMENT>
pixels over edge slot E18_40 -> <E18_40> currently has <RED> road
pixels over port slot P06 -> <P06> currently has <SHEEP> 2:1 touching <N47>/<N45>
```

The model should not memorize that `<N11>` is black. `<N11>` is a stable atlas
coordinate. The owner/building is transient and must come from the pixels.

The right sequence is:

1. Learn atlas topology tokens.
2. Bind board pixels to atlas tokens.
3. Extract the full public board contract.
4. Only then train policy/reasoning from board contract plus rules/history.

This matches the local SFT design in `sft/EXPERIMENT_DESIGN.md`.

## Evaluation Literature: What To Borrow

### Evaluation Engineering

Sources:

- OpenAI Evals guide: https://platform.openai.com/docs/guides/evals
- OpenAI Evals GitHub: https://github.com/openai/evals
- Inspect AI: https://inspect.aisi.org.uk/
- HELM: https://arxiv.org/abs/2211.09110
- VHELM: https://arxiv.org/abs/2410.07112
- METR autonomy eval resources: https://metr.org/blog/2024-03-13-autonomy-evaluation-resources/
- Eval-Driven Development: https://evaldriven.org/

Relevant pattern:

- An eval is a dataset, a grader, and a harness.
- Define correctness before prompt/model iteration.
- Keep raw prompts, completions, scores, and metadata inspectable.
- Treat a benchmark as a living artifact, but freeze eval splits when making
  before/after tuning claims.
- For contamination-sensitive tasks, publish enough examples to explain the task
  but keep some held-out data private.

Catan translation:

- CatanBoardBench should have a formal spec:
  - sample schema
  - answer schema
  - grader semantics
  - leakage boundary
  - human verification protocol
  - report template
- Every SFT claim should link:
  - training game IDs
  - excluded eval game IDs
  - checkpoint
  - prompt template
  - exact and component metrics
  - raw wrong examples
  - subset diagnostics
- Use Inspect/OpenBench-style logs for model comparisons, but keep the engine
  oracle and local scorer as the authority.

### MME

Source: https://arxiv.org/abs/2306.13394

Relevant pattern:

- Split perception from cognition.
- Use concise, quantitative prompts.
- Manually design instruction-answer pairs to reduce leakage.
- Report ability categories, not just one score.

Catan translation:

- Keep separate suites:
  - `visual_perception`: can it see resource icons, dice numbers, robber, road colors, ports?
  - `atlas_binding`: can it map visible objects to `<Txx>`, `<Nxx>`, `<Exx_yy>`, `<Pxx>`?
  - `state_extraction`: can it recover all public board facts?
  - `derived_state`: longest road, counts, legal build regions.
  - `policy`: what should a player do?

### SEED-Bench

Source: https://arxiv.org/abs/2307.16125

Relevant pattern:

- Objective multiple-choice evaluation.
- Human/manual verification of questions.
- Avoids GPT/human judging during scoring.
- Covers spatial/temporal dimensions.

Catan translation:

- CatanBoardBench exact token answers are good for final parser behavior.
- Add a multiple-choice diagnostic mode for "can it see the object?" separate
  from "can it emit our exact token format?"
- Keep human-verifier frontend for samples where we do not trust the engine/render
  alignment yet.

### MMBench

Source: https://arxiv.org/abs/2307.06281

Relevant pattern:

- Fine-grained ability taxonomy.
- Quality-control pipeline.
- CircularEval: rotate answer choices to reduce artifacts.
- Converts free-form predictions into pre-defined choices when needed.

Catan translation:

- If we add multiple-choice CatanBoardBench rows, rotate choices.
- Track invalid format separately from visually wrong answers.
- Use the exact/component split, but also add:
  - format-valid rate
  - abstention/unclear rate
  - positive-node subset
  - occupied-edge subset
  - non-generic-port subset

### MM-Vet

Source: https://arxiv.org/abs/2308.02490

Relevant pattern:

- Tests integrated capabilities, not isolated tags only.
- Gives insight beyond a leaderboard.

Catan translation:

- After targeted QA, add local bundle extraction:

```text
Given <T09>, return:
resource, number, robber?, adjacent occupied nodes, adjacent roads.
```

This is between single-slot QA and full JSON extraction. It tests integrated local
board reasoning without overwhelming the model.

### HallusionBench

Source: https://arxiv.org/abs/2310.14566

Relevant pattern:

- Human-crafted visual questions.
- Control groups.
- Failure-mode analysis, not only accuracy.
- Logical consistency across paired questions.

Catan translation:

- Create paired Catan questions:
  - "Which tile has the robber?"
  - "Does `<T17>` have the robber?"
  - "What resource/number is on `<T17>`?"
- Score consistency across the pair, not only each row independently.
- Add paired counterfactual boards where exactly one road/settlement/robber/port
  changes.

## CatanBoardBench Controls To Add

Highest priority controls:

1. **No-image / text-only control**
   - Run visual questions with no screenshot.
   - If accuracy stays high, the question leaks through priors or prompt text.

2. **Atlas-only control**
   - Provide topology but no board state.
   - Should fail transient facts like road owner and robber tile.

3. **Source PNG vs frontend-rendered board**
   - Already useful in the verifier.
   - Keeps us honest about rendering/annotation mismatch.

4. **Overlay ablation**
   - Image with no labels.
   - Image with nodes only.
   - Image with nodes/edges/tiles.
   - Image with target slot highlighted.

5. **Resolution and crop ablation**
   - 256/512/768 whole-board.
   - Targeted crop around queried slot.
   - Whole-board plus crop.

6. **Counterfactual pairs**
   - Same board, one changed fact.
   - The model must flip only the corresponding answer.

7. **Prompt robustness**
   - Same answer target with 3-5 prompt phrasings.
   - Report mean and worst-case accuracy.

8. **Class-balance reports**
   - Positive node occupancy.
   - Occupied roads.
   - Non-generic ports.
   - Non-`NONE` longest-road/largest-army.

## Qwen3-VL Practical Notes

Sources:

- Transformers Qwen3-VL docs: https://huggingface.co/docs/transformers/main/en/model_doc/qwen3_vl
- Qwen3-VL model card: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
- Qwen3-VL repo: https://github.com/QwenLM/Qwen3-VL
- NeMo Qwen3-VL recipes: https://docs.nvidia.com/nemo/automodel/latest/model-coverage/vlm/qwen/qwen3-vl.html

Important architecture detail:

- Hugging Face documents Qwen3-VL image patch size as `16` and spatial merge size
  as `2`.
- Effective visual token granularity is therefore roughly `32 x 32` pixels after
  patching/merging.
- At `512 x 512`, a whole Catan board is about `16 x 16 = 256` visual grid cells
  before other dynamic processing effects.

Catan implication:

- Full-board 512 may be enough for tile resource/number.
- Roads, ports, and small dice pips are closer to the danger zone.
- The training curriculum should include:
  - full board
  - target highlight
  - local crop
  - full board plus local crop

Qwen3-VL also advertises grounding formats in its cookbooks, including relative
position coordinates, boxes, and points. For our benchmark data generation, boxes
and points are enough. Lines are not required if edge boxes are narrow, centered,
and non-overlapping enough.

## VLM SFT Data Format

The TRL VLM SFT guide uses message-style samples:

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "..."},
        {"type": "text", "text": "What is on <T09>?"}
      ]
    },
    {
      "role": "assistant",
      "content": [{"type": "text", "text": "<WOOD> 11"}]
    }
  ]
}
```

For Catan:

- Use exact canonical answer strings.
- Keep answer formatting stable.
- Use fresh non-benchmark game IDs.
- Store game IDs and split assignment next to the dataset.
- Preserve leakage ledger checks against CatanBoardBench-100.

Recommended Catan SFT phases:

| Phase | Input | Target | Purpose |
| --- | --- | --- | --- |
| 0 | Text-only atlas/topology question | Token answer | Teach stable token geometry. |
| 1 | Image + one slot question | Exact slot fact | Bind pixels to one atlas token. |
| 2 | Image + local neighborhood question | Small JSON or token bundle | Bind local relational state. |
| 3 | Image only or image + extraction instruction | Full public board JSON | Prove complete board parsing. |
| 4 | Board JSON + hand/history/rules | Legal action or strategic answer | Policy/reasoning. |

Do not jump from Phase 1 to strategy SFT and call it intelligence. If full public
state extraction is not reliable, policy training will absorb visual errors.

## LoRA, QLoRA, And Full Fine-Tuning

Sources:

- LoRA: https://arxiv.org/abs/2106.09685
- QLoRA: https://arxiv.org/abs/2305.14314
- PEFT LoRA docs: https://huggingface.co/docs/peft/en/developer_guides/lora
- TRL LoRA Without Regret: https://huggingface.co/docs/trl/lora_without_regret
- DoRA: https://arxiv.org/abs/2402.09353
- rsLoRA: https://arxiv.org/abs/2312.03732
- LoftQ: https://arxiv.org/abs/2310.08659

### What LoRA Is Actually Testing Here

For Catan, LoRA is not just a cheap training trick. It is a diagnostic:

```text
How much trainable capacity is needed before the model develops a usable Catan
atlas and image-to-atlas binding?
```

If a low-rank adapter solves the task and probes show structured atlas geometry,
we do not need full fine-tuning immediately. If behavior improves but probes do
not, that suggests shallow memorization or task-head adaptation. If LoRA cannot
fit held-out atlas topology even with enough data, embeddings/head or full FT may
matter more.

### Trainable Scope Ladder

| Scope | Trainable parts | What it tests | Use when |
| --- | --- | --- | --- |
| A | Added token embeddings only | Can new atlas symbols become anchors? | Phase 0 smoke. |
| B | Added token embeddings + lm_head rows | Can the model emit exact tokens? | Phase 0 smoke. |
| C | Language LoRA | Can language layers learn atlas/topology mappings? | Cheap baseline. |
| D | Language LoRA + projector | Is image-to-language bridge the bottleneck? | Phase 1 visual QA. |
| E | Vision/projector LoRA | Are roads/ports/dice visual features the bottleneck? | If crops/overlays still fail. |
| F | Full bf16 fine-tune | Does the whole model need rewiring? | Only after scoped ablations. |

### Practical Starting Configs

For cheap Qwen3-VL-8B smoke:

```yaml
quantization: 4bit_nf4
compute_dtype: bf16
lora_r: 16
lora_alpha: 16 or 32
lora_dropout: 0.0 to 0.05
target_modules: all-linear
modules_to_save:
  - embed_tokens
  - lm_head
gradient_checkpointing: true
learning_rate: 1e-5 to 2e-5
effective_batch_size: keep modest, then sweep
```

For serious SFT once the data is large enough:

```yaml
lora_r: 64 or 128
target_modules: all-linear
use_rslora: consider for higher ranks
use_dora: consider if normal LoRA plateaus
trainable_token_indices: consider for added Catan tokens
```

For full fine-tuning:

- Treat it as an expensive endpoint, not the first move.
- Use only after a clean train/dev/test split proves the data signal is real.
- Compare against high-rank LoRA. The LoRA Without Regret guidance argues that
  high-rank, all-linear LoRA can be much closer to full FT than the old
  low-rank/attention-only habit.

### 4-Bit QLoRA

Why 4-bit:

- QLoRA freezes a 4-bit quantized base and backpropagates through adapters.
- It saves enough memory to make 8B VLM experimentation affordable.
- It is a budget move, not a claim that 4-bit is more semantically correct.

Trap:

- Quantization can hide whether full model weights would learn better.
- If QLoRA fails, test whether the failure is data, visual resolution, trainable
  scope, or quantization before concluding the model cannot learn.

### How To Know If LoRA Capacity Is Enough

Track:

- Train loss vs held-out loss.
- Exact and component accuracy.
- Positive/occupied/non-generic subsets.
- Format-valid rate.
- Linear probe recovery of atlas IDs.
- Linear probe recovery of transient board state.
- Counterfactual flip accuracy.

Signals:

- **Underfit:** train and validation both bad. Increase data quality, trainable
  scope, rank, or resolution/crop.
- **Memorization:** train good, held-out bad. More fresh boards, paraphrases,
  counterfactuals, leakage checks.
- **Format-only gain:** exact improves mostly from token formatting, component
  visual subsets do not. Add perception-focused data.
- **Visual bottleneck:** text topology works, image QA fails. Try projector/vision
  LoRA, crops, higher resolution, target highlighting.
- **Representation bottleneck:** behavior improves but probes cannot recover atlas
  or board state. Increase scope or add auxiliary representation losses later.

## Spatial IDs And Mech Interp

Local notes:

- `references/vlm_spatial_representation_literature.md`
- `references/linear_mech_vlms_reference.md`
- `references/residual_stream_board_experiment.md`

Core idea:

```text
activation("<N11>") ~= stable_spatial_id(<N11>)
                    + transient_visual_state(pixels over <N11>)
                    + task context
```

Probe targets:

1. Stable atlas geometry
   - Can hidden states identify node/tile/edge/port ID?
   - Do nearby graph positions have structured representations?

2. Transient visual state
   - Can hidden states reconstruct resource, number, robber, road owner, port type,
     settlement/city state?

3. Causal mediation
   - If we patch `<T17>` activation from board A into board B, does the answer
     about robber/resource/number flip?

4. Layer timing
   - Does the information appear near visual-input layers, middle language layers,
     or only answer-token positions?

Do this after Phase 0/1 SFT. Before tuning, the signals may simply be absent.

## Catan-Specific Multi-Agent Learning

### Agents of Change / HexMachina

Sources:

- arXiv: https://arxiv.org/abs/2506.04651
- OpenReview: https://openreview.net/forum?id=V0Fb4pwhS4
- Project page: https://nbelle1.github.io/agents-of-change/

Why it matters:

- Directly uses Settlers of Catan through Catanatron.
- Separates environment discovery from strategy improvement.
- Preserves executable artifacts rather than relying on a prompt-only per-turn
  decider.
- Reported best runs reached 54% against an AlphaBeta baseline in controlled
  Catanatron experiments.

Catan project translation:

- Do not make the VLM be a per-turn planner from pixels forever.
- Use VLM as board-state extractor.
- Use symbolic board contract for strategy, search, and self-play.
- Store strategy artifacts, evaluation logs, and failed-game analyses.

### Catanatron

Source: https://docs.catanatron.com/

Why it matters:

- Open-source Catan simulator.
- Runs many games quickly.
- Has custom bot interfaces and Gymnasium support.
- Useful comparison baseline even if this repo keeps its own engine.

Catan project translation:

- Use our engine for current frontend/benchmark continuity.
- Borrow evaluation ideas:
  - bot-vs-bot tournaments
  - fixed seeds
  - population tables
  - baseline bots
  - replay inspection

### Settlers-RL

Source: https://settlers-rl.github.io/

Why it matters:

- Long practical writeup on training a Catan RL agent.
- Calls out Catan-specific difficulties:
  - huge action space
  - invalid action masking
  - composite actions
  - 4-player hidden/stochastic setting
  - trading and dev cards
  - sparse rewards

Catan project translation:

- Do not use one flat action vector without masks.
- Treat actions as typed heads:
  - build settlement node
  - build road edge
  - city node
  - dev card choice
  - robber tile/player
  - trade proposal/response
  - bank/maritime trade
- Record action masks in training examples and eval failures.

### Strategic Dialogue Management Via DRL

Source: https://arxiv.org/abs/1511.08099

Why it matters:

- Catan-specific trading/dialogue setting.
- Good reminder that trading is a separate strategic-dialogue problem, not just
  board placement.

Catan project translation:

- Policy eval should split:
  - non-trade tactical action quality
  - trade proposal quality
  - trade acceptance/rejection quality
  - negotiation language quality

### OpenSpiel And PSRO

Sources:

- OpenSpiel: https://arxiv.org/abs/1908.09453
- PSRO: https://arxiv.org/abs/1711.00832

Why it matters:

- OpenSpiel gives a clean game/RL vocabulary and algorithm suite.
- PSRO/population methods address overfitting to the current opponent pool.

Catan project translation:

- Track agent performance against a population, not one fixed bot.
- Keep historical checkpoints.
- Evaluate exploitability/proxy exploitability through best-response-ish bots.
- Separate training opponents from evaluation opponents.

## Directed Behavior And Preference Tuning

Sources:

- DPO: https://arxiv.org/abs/2305.18290
- KTO: https://arxiv.org/abs/2402.01306
- ORPO: https://arxiv.org/abs/2403.07691
- SimPO: https://arxiv.org/abs/2405.14734
- GRPO via DeepSeekMath: https://arxiv.org/abs/2402.03300
- TRL trainers: https://huggingface.co/docs/trl/v0.21.0/index

For Catan, "directed behavior" means steering the policy toward better strategic
habits:

- expand toward high-pip ore/wheat
- do not block yourself with the robber
- prefer city plans when ore/wheat income is high
- trade away irrelevant resources for current plan resources
- avoid roads to nowhere unless pursuing Longest Road
- block the leader when robber value is close

Data forms:

1. **Behavior cloning / SFT**
   - `state/history -> expert action`
   - good first phase
   - can imitate bad or inconsistent humans

2. **Reward-weighted SFT**
   - same format, weighted by outcome or heuristic advantage
   - fits the existing credit-assignment note

3. **Pairwise preference**
   - `state -> chosen action vs rejected action`
   - use DPO/ORPO/SimPO style objectives
   - useful when exact expert action is noisy but one move is clearly better

4. **Binary desirable/undesirable labels**
   - use KTO-style data if pairwise comparisons are expensive
   - examples: "robber blocks own best tile" undesirable

5. **Online RL/self-play**
   - only after legal-action and board-state representation are reliable
   - expensive and noisy because Catan is stochastic and multi-agent

## Credit Assignment For Catan

Local note: `tasks/credit_assignment_proposal.md`

Main problem:

```text
win/loss reward = decision quality + dice luck + opponent mistakes + hidden info
```

The existing proposal is the right direction:

- compute position score
- subtract luck delta
- use strategy notes to weight resource relevance
- score trades by plan relevance
- split high-signal actions from low-signal/mandatory events

Add these eval artifacts:

- per-turn decision tags
- legal action mask
- declared plan before action
- action chosen
- top alternative candidates
- heuristic delta
- eventual outcome
- dice/resource events separated from choice

This lets you build preference data without pretending final win/loss alone is a
clean label.

## Recommended Experiment Roadmap

### Phase A: Harden CatanBoardBench

Deliverables:

- no-image control
- overlay/crop ablations
- counterfactual pairs
- positive/occupied/non-generic subset reports
- prompt robustness
- human-verifier queue for suspicious rows

Success:

- We can say exactly what failed:
  - visual primitive
  - atlas binding
  - format
  - reasoning
  - leakage/shortcut

### Phase B: Atlas Token Topology SFT

Data:

- 390 canonical rows is only a smoke test.
- Build 2k, 10k, then 50k paraphrased/inverse topology rows.

Trainable scopes:

- embeddings only
- embeddings + lm_head
- language LoRA

Success:

- held-out topology exact accuracy
- paraphrase robustness
- linear probe topology recovery

### Phase C: Visual QA SFT

Data:

- fresh non-benchmark game IDs only
- full-board QA
- highlighted target QA
- local crop QA
- bbox/point supervision rows
- counterfactual rows

Trainable scopes:

- language LoRA baseline
- language LoRA + projector
- vision/projector LoRA if roads/ports fail

Success:

- CatanBoardBench-100 improves on hard local categories:
  - `tile_resource_number`
  - `robber_tile`
  - `edge_road_owner`
  - `port_type_nodes`
  - positive `node_occupancy`

Not success:

- only `NONE`/`EMPTY` improves
- only format-valid rate improves
- train examples improve but held-out CatanBoardBench does not

### Phase D: Full Board Contract Extraction

Input:

```text
image -> public board JSON
```

Score:

- slot-level exact
- tile resource/number
- node occupancy
- edge road owner
- port type/nodes
- robber
- public point/count consistency
- topology validity

Success:

- The model can recover public state without being prompted slot by slot.

### Phase E: Strategy/Policy

Input:

```text
public board JSON + hand + trade history + current legal actions
```

Targets:

- expert action
- preference pair
- reward-weighted action
- strategic note update

Evaluation:

- legal action rate
- action-type accuracy
- heuristic advantage
- tournament win rate vs fixed baselines
- tournament win rate vs held-out population
- trade-specific quality

## Traps To Avoid

- Benchmark leakage from CatanBoardBench game IDs.
- Tuning on the exact screenshots/contracts used for eval.
- Measuring mostly `EMPTY`/`NONE`.
- Letting answer format dominate the score.
- Assuming a VLM policy is good when board parsing is still bad.
- Treating LoRA failure as model failure before testing data/resolution/scope.
- Treating full fine-tuning as automatically more "deeply ingrained" without a
  probe/eval proving it.
- Making one prompt/template the benchmark.
- Training a per-turn prompt agent when an artifact/compiled policy would be more
  stable.
- Evaluating only against the training opponent.

## Source Catalog

### VLM Evaluation

- OpenAI Evals guide: https://platform.openai.com/docs/guides/evals
- OpenAI Evals GitHub: https://github.com/openai/evals
- Inspect AI: https://inspect.aisi.org.uk/
- HELM: https://arxiv.org/abs/2211.09110
- VHELM: https://arxiv.org/abs/2410.07112
- METR autonomy eval resources: https://metr.org/blog/2024-03-13-autonomy-evaluation-resources/
- Eval-Driven Development: https://evaldriven.org/
- MME: https://arxiv.org/abs/2306.13394
- SEED-Bench: https://arxiv.org/abs/2307.16125
- MMBench: https://arxiv.org/abs/2307.06281
- MM-Vet: https://arxiv.org/abs/2308.02490
- HallusionBench: https://arxiv.org/abs/2310.14566
- MMMU: https://arxiv.org/abs/2311.16502
- MMMU-Pro: https://arxiv.org/abs/2409.02813
- MMStar: https://arxiv.org/abs/2403.20330
- DocVQA: https://www.docvqa.org/
- TextVQA: https://textvqa.org/
- ChartQA: https://arxiv.org/abs/2203.10244

### VLM Training And Grounding

- TRL VLM SFT: https://huggingface.co/docs/trl/v0.21.0/training_vlm_sft
- LLaVA visual instruction tuning: https://arxiv.org/abs/2304.08485
- InstructBLIP: https://arxiv.org/abs/2305.06500
- Qwen3-VL docs: https://huggingface.co/docs/transformers/main/en/model_doc/qwen3_vl
- Qwen3-VL model card: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
- Qwen3-VL repo: https://github.com/QwenLM/Qwen3-VL
- Qwen VL finetune scripts: https://github.com/QwenLM/Qwen3-VL/tree/main/qwen-vl-finetune
- NeMo Qwen3-VL recipes: https://docs.nvidia.com/nemo/automodel/latest/model-coverage/vlm/qwen/qwen3-vl.html
- Learning to Localize Objects Improves Spatial Reasoning: https://arxiv.org/abs/2404.07449
- SpatialVLM: https://arxiv.org/abs/2401.12168
- V*: https://arxiv.org/abs/2312.14135

### LoRA And PEFT

- LoRA: https://arxiv.org/abs/2106.09685
- QLoRA: https://arxiv.org/abs/2305.14314
- PEFT LoRA docs: https://huggingface.co/docs/peft/en/developer_guides/lora
- LoRA Without Regret: https://huggingface.co/docs/trl/lora_without_regret
- DoRA: https://arxiv.org/abs/2402.09353
- rsLoRA: https://arxiv.org/abs/2312.03732
- LoftQ: https://arxiv.org/abs/2310.08659

### Preference Optimization And Directed Behavior

- DPO: https://arxiv.org/abs/2305.18290
- KTO: https://arxiv.org/abs/2402.01306
- ORPO: https://arxiv.org/abs/2403.07691
- SimPO: https://arxiv.org/abs/2405.14734
- DeepSeekMath / GRPO: https://arxiv.org/abs/2402.03300
- TRL trainers: https://huggingface.co/docs/trl/v0.21.0/index

### Catan And Multi-Agent Learning

- Agents of Change / HexMachina: https://arxiv.org/abs/2506.04651
- HexMachina OpenReview: https://openreview.net/forum?id=V0Fb4pwhS4
- Agents of Change project page: https://nbelle1.github.io/agents-of-change/
- Catanatron docs: https://docs.catanatron.com/
- Settlers-RL writeup: https://settlers-rl.github.io/
- Strategic Dialogue Management via DRL: https://arxiv.org/abs/1511.08099
- Monte Carlo Tree Search in Settlers of Catan: https://www.researchgate.net/publication/220716999_Monte-Carlo_Tree_Search_in_Settlers_of_Catan
- A Multi-agent player for Settlers of Catan: https://www.researchgate.net/publication/200493096_A_Multi-agent_player_for_Settlers_of_Catan
- OpenSpiel: https://arxiv.org/abs/1908.09453
- PSRO: https://arxiv.org/abs/1711.00832
- AlphaStar: https://www.nature.com/articles/s41586-019-1724-z
- OpenAI Five: https://openai.com/research/openai-five
- Population Based Training: https://arxiv.org/abs/1711.09846
- QMIX: https://arxiv.org/abs/1803.11485
- VDN: https://arxiv.org/abs/1706.05296
- MADDPG: https://arxiv.org/abs/1706.02275
- COMA: https://arxiv.org/abs/1705.08926

### Mech Interp And Spatial Representations

- Linear mechanisms for spatiotemporal reasoning in VLMs: https://arxiv.org/abs/2601.12626
- Local reference code: `references/external/linear-mech-vlms`
- How multimodal LLMs solve image tasks: https://arxiv.org/abs/2508.20279
- Towards interpreting visual information processing in VLMs: https://arxiv.org/abs/2410.07149
- Causal tracing for BLIP: https://arxiv.org/abs/2308.14179
- What Do VLMs NOTICE?: https://arxiv.org/abs/2406.16320
- Othello-GPT world representations: https://arxiv.org/abs/2210.13382
- Language models represent space and time: https://arxiv.org/abs/2310.02207

## Tomorrow's Concrete Checklist

1. Read the five VLM eval papers and update CatanBoardBench categories into a formal
   capability taxonomy.
2. Add no-image and atlas-only controls to `evals/catan_board_bench/eval`.
3. Generate a fresh non-benchmark game split for SFT and save game IDs.
4. Build Phase 0 topology data at 2k and 10k rows.
5. Run embeddings+lm_head vs LoRA smoke on Phase 0.
6. Add visual QA rows with target highlight and local crop.
7. Decide whether the first Modal SFT run is:
   - token embeddings + lm_head, or
   - QLoRA all-linear with added-token modules saved.
8. After first checkpoint, run:
   - CatanBoardBench-100 hard categories
   - no-image control
   - positive/occupied/non-generic subset reports
   - atlas linear probe
9. Do not claim policy improvement until board contract extraction is reliable.
