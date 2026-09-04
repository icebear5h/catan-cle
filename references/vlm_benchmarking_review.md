# VLM Benchmarking Literature Review For CatanBoardBench

Date reviewed: 2026-05-14

## Bottom Line

CatanBoardBench is directionally strong because it is **domain-specific,
engine-scored, objective, and decomposed by board fact type**. That matches the
best lesson from modern VLM evaluation: broad leaderboards do not reliably tell
us whether a model can solve a specific visual-symbolic task.

The main gap is that CatanBoardBench is currently a good **model smoke test**, but not
yet a fully hardened benchmark. The next work is not more one-off model runs; it
is adding the controls used by the stronger VLM benchmarks:

- no-image / text-only controls for visual indispensability
- resolution and crop ablations
- overlay / slot-label ablations
- counterfactual paired boards
- prompt-template robustness
- format-validity and abstention metrics
- frozen train/dev/test splits with a held-out real replay set

## Current CatanBoardBench Snapshot

Current implementation:

- OpenBench/Inspect task: `evals/catan_board_bench/eval/benchmark.py`
- Questions: `evals/catan_board_bench/datasets/catan_board_bench_100/questions/`
- Rendered contracts/images: `evals/catan_board_bench/datasets/catan_board_bench_100/`
- Main result table: `reports/catan_board_bench/2026-05-13-small-vlm-visual-smoke.md`

Current suites:

- `suite=visual`: image-grounded board parsing questions.
- `suite=logic`: text-only atlas/topology/game-state questions.

Current visual categories:

- `robber_tile`
- `robber_resource_number`
- `tile_resource_number`
- `tile_has_robber`
- `node_occupancy`
- `edge_road_owner`
- `color_road_locations`
- `port_trade_type`
- `port_occupancy`
- `color_building_counts`
- `color_road_count`

Current scoring:

- strict exact accuracy
- component accuracy for structured answers
- category-level breakdown in result docs

Current 40-question visual smoke:

| Model | Exact | Component |
| --- | ---: | ---: |
| `gemini-3.1-pro-preview` | 67.5% | 72.5% |
| `grok-4.3` | 30.0% | 39.4% |
| `claude-sonnet-4.6` | 25.0% | 29.7% |
| `gemma-4-31b-it` | 17.5% | 24.8% |
| `qwen3-vl-8b-instruct` | 10.0% | 18.0% |

Interpretation:

- The benchmark is hard enough to separate frontier VLMs from smaller VLMs.
- It reveals a real visual-symbolic bottleneck: even strong models fail exact
  node/edge/robber binding.
- The 40Q smoke is not enough for final model selection, but it is enough to show
  that plain prompting is not a reliable board parser.

## Literature And Resource Map

### MME

Source: https://arxiv.org/abs/2306.13394

Pattern to borrow:

- separates perception and cognition
- uses concise prompts for quantitative comparison
- manually designed annotations to reduce leakage

CatanBoardBench status:

- Strong: visual vs logic split already follows this perception/cognition idea.
- Missing: no explicit perception tier names yet, e.g. `slot_perception`,
  `local_state`, `derived_state`, `game_logic`.

### SEED-Bench

Source: https://arxiv.org/abs/2307.16125

Pattern to borrow:

- objective multiple-choice answers
- human/manual verification
- dimensions for image and video comprehension
- avoids GPT/human judging in the scoring loop

CatanBoardBench status:

- Strong: objective engine scoring is better than LLM-as-judge for this task.
- Tradeoff: free-form token answers are closer to the desired parser behavior,
  but a temporary multiple-choice diagnostic mode could isolate perception from
  formatting failures.

### MMBench

Source: https://arxiv.org/abs/2307.06281

Pattern to borrow:

- fine-grained ability dimensions
- quality control schemes
- CircularEval to reduce answer-position / instruction-following artifacts
- answer extraction for free-form outputs

CatanBoardBench status:

- Strong: category-level breakdown already gives useful ability-level feedback.
- Missing: no prompt/candidate rotation equivalent. If we add multiple-choice
  diagnostics, we should add circular answer-order checks.
- Missing: no explicit invalid-format / parser-failure metric in the headline
  table.

### MM-Vet

Source: https://arxiv.org/abs/2308.02490

Pattern to borrow:

- evaluates integrated multimodal capabilities, not just a single flat score
- reports insights beyond simple ranking
- supports open-ended tasks with a unified metric

CatanBoardBench status:

- Strong: exact and component metrics give more than a single leaderboard rank.
- Missing: no higher-level capability taxonomy yet. Example taxonomy:
  `tile reading`, `token atlas binding`, `road ownership`, `port grounding`,
  `counting`, `derived public state`.

### MMMU / MMMU-Pro

Sources:

- https://arxiv.org/abs/2311.16502
- https://arxiv.org/abs/2409.02813

Pattern to borrow:

- expert-domain multimodal reasoning
- broad heterogeneous image types
- MMMU-Pro filters questions answerable by text-only models
- MMMU-Pro adds a vision-only input setting where questions are embedded in the image

CatanBoardBench status:

- Strong: CatanBoardBench is domain-specific and engine-oracle grounded.
- Missing: no automated text-only/no-image filter for visual categories.
- Useful addition: run every visual QA with `input_mode=text` and no image. If
  a category scores above chance from prompt/atlas alone, that category is not
  visually indispensable enough.

### MMStar

Source: https://arxiv.org/abs/2403.20330

Pattern to borrow:

- explicitly tests whether visual content is necessary
- measures data leakage and multimodal gain
- human review to ensure visual dependency

CatanBoardBench status:

- Strong: generated board states reduce public benchmark contamination risk.
- Missing: no formal `visual_gain = image_score - no_image_score`.
- Missing: no leak audit by category after prompt-context changes. We already
  found and removed one answer leak in `color_road_locations`; that should become
  a standing test.

### HallusionBench

Source: https://arxiv.org/abs/2310.14566

Pattern to borrow:

- control groups
- question-pair accuracy
- hallucination/illusion failure analysis
- consistency, not only per-question accuracy

CatanBoardBench status:

- Strong: engine can generate exact counterfactual board pairs.
- Missing: no paired counterfactual suite yet.
- Useful addition: paired boards where exactly one fact changes:
  robber tile, one road owner, one node occupancy, one port type, or one tile
  number.

### BLINK

Source: https://arxiv.org/abs/2404.12390

Pattern to borrow:

- tests core visual perception abilities that broad benchmarks miss
- shows that "can see" is not the same as "can perceive"
- includes tasks where humans are near ceiling but VLMs are not

CatanBoardBench status:

- Strong: CatanBoardBench has the same spirit for board-specific perception.
- Missing: no human baseline, no deterministic CV parser baseline, and no
  explicit "easy for humans" calibration.

### Vision Language Models Are Blind

Source: https://arxiv.org/abs/2407.06581

Pattern to borrow:

- simple low-level visual tasks can expose failures hidden by broad benchmarks
- resolution/detail and exact geometry matter

CatanBoardBench status:

- Strong: tile/node/edge binding is exactly the kind of low-level precision task
  broad VLM leaderboards hide.
- Missing: no systematic crop/resolution sweep in the benchmark artifact yet.

### V* / V*Bench

Source: https://arxiv.org/abs/2312.14135

Pattern to borrow:

- high-resolution, visually crowded images require visual search
- benchmark specifically tests focusing on small visual details

CatanBoardBench status:

- Strong: Catan boards are visually crowded and detail-sensitive.
- Missing: no "zoomed crop" or two-stage visual-search condition.
- Useful addition: compare full-board image, full-board plus local crop, and
  local crop only for node/edge/port questions.

### ChartQA

Source: https://arxiv.org/abs/2203.10244

Pattern to borrow:

- structured graphics require both visual extraction and logical reasoning
- combines image-derived facts with a latent data table

CatanBoardBench status:

- Very relevant analogy: screenshot -> structured board contract -> answer.
- Strong: our engine contract is equivalent to ChartQA's underlying chart data.
- Missing: no full-contract extraction metric yet; current QA is atomic and
  category based.

### OCRBench

Source: https://arxiv.org/abs/2305.07895

Pattern to borrow:

- text in images deserves its own evaluation axis
- separate text recognition from downstream reasoning

CatanBoardBench status:

- Relevant for dice numbers, port labels/icons, and token overlays.
- Missing: no dedicated number-reading vs icon/resource-reading breakdown.
  `tile_resource_number` currently bundles resource and number.

### VLMEvalKit

Sources:

- https://arxiv.org/abs/2407.11691
- https://github.com/open-compass/VLMEvalKit

Pattern to borrow:

- unified interface for many VLMs and many benchmarks
- reproducible inference, post-processing, and metric calculation
- leaderboards with standardized records

CatanBoardBench status:

- Strong: OpenBench/Inspect gives us a reproducible local framework.
- Missing: result normalization and model metadata are still manually maintained
  in markdown.

### LMMs-Eval / lmms-eval

Sources:

- https://arxiv.org/abs/2407.12772
- https://docs.lmms-lab.com/docs/v0.6

Pattern to borrow:

- standardized multimodal benchmark framework
- emphasizes coverage, low cost, and contamination tradeoffs
- supports many tasks and models from one pipeline

CatanBoardBench status:

- Strong: our 40Q smoke is cheap and fast enough for iteration.
- Missing: the "full" benchmark definition is not yet frozen into a canonical
  low-cost subset plus a canonical full suite.

## Scorecard For CatanBoardBench

| Criterion | Current Grade | Notes |
| --- | --- | --- |
| Domain relevance | A | Directly tests the Catan perception problem we care about. |
| Objective scoring | A | Engine oracle avoids subjective judging. |
| Category diagnostics | B+ | Good category split; needs higher-level capability taxonomy. |
| Visual indispensability | C | Needs no-image controls and prompt-leak checks. |
| Robustness controls | C | Needs resolution, crop, prompt, and image-detail ablations. |
| Counterfactual controls | C | Engine can generate them, but suite is not built yet. |
| Dataset scale | B- | 100 contracts / 3032 QA is a good start; 40Q smoke is only a smoke. |
| Split hygiene | C | Need frozen train/dev/test and held-out real replay set. |
| Human/CV baselines | D | Need human baseline and deterministic parser/CV baseline. |
| Reproducibility | B | OpenBench setup is good; result aggregation should be scripted. |
| Cost/latency reporting | B | Token counts saved in result table; cost should be computed too. |
| Full parser metric | C | Atomic QA exists; full public-board contract extraction is missing. |

Overall: **B for an early domain-specific benchmark, C+ as a publishable VLM
benchmark**.

## What CatanBoardBench Already Gets Right

1. **Engine oracle**
   - Best possible ground truth for this domain.
   - Stronger than LLM-as-judge or human subjective scoring.

2. **Exact output tokens**
   - Models must bind to `<Txx>`, `<Nxx>`, `<Exx_yy>`, resource tokens, and
     color tokens.
   - This directly tests the representation we want for SFT.

3. **Separate visual and logic suites**
   - Prevents confusing perception failure with game-logic failure.
   - Matches the perception/cognition separation from MME-style evaluation.

4. **Component scoring**
   - Useful for answers like `<T07> <WOOD> 6`, where resource may be right but
     tile ID is wrong.

5. **Category-level breakdown**
   - The current results already show the important failure: many models can read
     some tiles/resources but fail exact node/edge/robber binding.

## Biggest Risks In The Current Benchmark

### 1. The visual suite may include text/atlas leakage

The prompt supplies fixed atlas context. That is useful because the model must
map the image to our canonical IDs, but it means some questions can become less
visual than intended.

Required control:

```text
visual_gain(category) =
  score(image + atlas + question) - score(atlas + question, no image)
```

Any "visual" category with low visual gain should be moved to the logic suite or
rewritten.

### 2. The 40Q smoke is too small for model selection

The 40Q smoke is useful for rejecting bad candidates, but it is too small for
serious ranking. Category denominators such as `3/4` or `2/3` are too noisy.

Required control:

- canonical smoke: 40 questions
- canonical dev: 300 to 500 questions
- canonical full: all selected categories across 100+ boards
- report bootstrap confidence intervals or at least per-category denominators

### 3. Exact token QA is not the same as full board parsing

Atomic QA can hide inconsistencies. A model may answer one edge question right
and another related node question wrong.

Required additions:

- full `PublicBoardContract` extraction
- schema-validity rate
- illegal-topology rate
- full-board exact match
- slot-level field accuracy

### 4. No counterfactual pairs

The engine can create the exact kind of paired tests HallusionBench recommends.

Required additions:

- same board, robber moved one tile
- same board, one road owner changed
- same board, one node empty -> settlement
- same board, one port type changed
- same board, one dice number changed

Metric:

```text
pair_accuracy = both original and counterfactual answered correctly
target_flip_accuracy = only the changed fact flips
specificity = unrelated answers stay stable
```

### 5. No resolution / crop / overlay ladder yet

CatanBoardBench needs to answer whether the model is failing because:

- pixels are too small
- atlas IDs are not grounded
- roads/nodes/ports are visually ambiguous
- prompt format is weak

Required conditions:

- `256px`
- `512px`
- `768px`
- full board
- tight board crop
- local crop for target node/edge/port
- tile ID overlay
- node/edge ID overlay
- faded overlay
- no overlay

## Recommended Benchmark Roadmap

### Phase 1: Harden The Existing QA Benchmark

Add task args / suites:

- `suite=visual_no_image`
- `suite=visual_768`
- `suite=visual_512`
- `suite=visual_256`
- `suite=visual_overlay_tile`
- `suite=visual_overlay_node_edge`
- `suite=visual_local_crop`

Add metrics:

- exact
- component
- invalid format
- null / abstain rate
- visual gain
- prompt sensitivity
- cost per 100 questions
- latency per 100 questions

### Phase 2: Add Counterfactual Diagnostics

Create paired board contracts and images:

- `counterfactual_robber`
- `counterfactual_edge_owner`
- `counterfactual_node_occupancy`
- `counterfactual_port_type`
- `counterfactual_tile_number`

Add metrics:

- pair accuracy
- target flip accuracy
- specificity
- unchanged-fact consistency

### Phase 3: Add Full Contract Extraction

Ask for a canonical JSON contract:

```json
{
  "tiles": {},
  "nodes": {},
  "edges": {},
  "ports": {},
  "robber_tile": "<T00>"
}
```

Score:

- valid JSON
- valid schema
- valid topology
- field-level accuracy
- full-board exact match

### Phase 4: Add Held-Out Real Replay Set

Keep generated boards for scale, but freeze:

- generated train/dev/test boards
- real replay dev set
- real replay final test set

The final test set should not be used during prompt tuning or SFT experiments.

## How Our Current Model Results Should Be Read

The current table is meaningful as a **first-stage parser stress test**:

- Gemini 3.1 Pro being far above everyone else says the task is visible enough
  for a strong VLM to extract useful information.
- Gemini still missing road-location/count questions says even frontier VLMs are
  not reliable parsers.
- Grok 4.3, Claude Sonnet, Gemma 4, Qwen 235B, and GPT-5.5 all failing key
  exact-slot categories says generic scale does not solve atlas binding.
- Qwen3-VL-8B is still a reasonable SFT base because zero-shot score is not the
  model-selection criterion; trainability and open adaptation path matter.

The current table should **not** be read as final proof that one model family is
better for SFT. It is too small, too prompt-specific, and lacks visual-gain and
resolution controls.

## Direct Changes To Make In This Repo

1. Add a `visual_no_image` suite.
   - Same visual questions.
   - `input_mode=text`.
   - No image content.
   - Use it to compute visual gain by category.

2. Add rendered image variants to the benchmark artifact.
   - `images_256/`
   - `images_512/`
   - `images_768/`
   - `images_overlay_tile/`
   - `images_overlay_node_edge/`
   - `images_local_crop/`

3. Add a result aggregation script.
   - Parse OpenBench JSON logs.
   - Emit markdown tables.
   - Emit category breakdowns.
   - Emit cost/latency if pricing is known.

4. Add a prompt-template sweep.
   - strict token answer
   - short JSON
   - multiple choice diagnostic
   - contract extraction

5. Add counterfactual generator hooks.
   - Generate paired contracts from engine state.
   - Render both images.
   - Store paired QA IDs.

6. Add benchmark card.
   - Dataset construction
   - Splits
   - Public/private information policy
   - Known limitations
   - Evaluation commands
   - Leakage controls

## Useful Resources To Keep Around

- VLMEvalKit: https://github.com/open-compass/VLMEvalKit
- lmms-eval docs: https://docs.lmms-lab.com/docs/v0.6
- Hugging Face VLM intro/fine-tuning blog: https://huggingface.co/blog/vlms
- MMBench repo: https://github.com/open-compass/MMBench
- BLINK repo: https://github.com/zeyofu/BLINK_Benchmark

## Final Assessment

CatanBoardBench is a good start because it is not a vibe eval: it has generated data,
engine labels, exact scoring, repeatable OpenBench runs, and category analysis.

To become a serious benchmark for training a board-understanding VLM, it needs
the missing controls from MMStar, HallusionBench, BLINK, and V*:

- prove the image is necessary
- prove the answer is not coming from prompt leakage
- prove the model is consistent under controlled board changes
- prove performance survives prompt/crop/resolution shifts
- prove atomic QA transfers to full board-state extraction

That is the benchmark standard we should hold before trusting SFT gains or
expert-game policy results.
