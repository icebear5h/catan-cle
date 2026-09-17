# Latest spatial checkpoint: symbolic board-fluency evaluation

## Result

**24/200 exact answers (12.0%)** on the 200-example symbolic review batch.

| Family | Correct | Accuracy | Format-valid | Wrong but valid | Malformed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Relations / joins | 6/40 | 15.0% | 15 | 9 | 25 |
| Sets / coverage | 6/40 | 15.0% | 38 | 32 | 2 |
| Aggregation / comparison | 3/40 | 7.5% | 40 | 37 | 0 |
| Connectivity / structure | 6/40 | 15.0% | 23 | 17 | 17 |
| Constraints / consequences | 3/40 | 7.5% | 36 | 33 | 4 |
| Total | 24/200 | 12.0% | 152 | 128 | 48 |

All 20 board-wide pip-total and pip-argmax answers were well-formed but wrong.
Of five successful shortest-distance answers, four were `UNREACHABLE` and one
was a finite distance. Twenty-four outputs repeatedly emitted a small set of
atlas tokens: over 20 exact atlas occurrences and at most three distinct tokens.
All 24 were malformed. This definition is descriptive, not a mechanistic claim.

## Conditions

- Checkpoint: `/runs/catan-vision-sft/spatial-continuation-20260912-r01/checkpoints/checkpoint-128`.
- Storage: personal `icebear5h` Modal workspace; this saved checkpoint was selected
  after the user reviewed the available HF and Modal alternatives.
- Base: `Qwen/Qwen3.8-27B`, revision
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- Inputs: exact existing `symbolic_board_fluency_review_v1/review.jsonl` messages,
  explicit symbolic state, no image input. User content only is passed for generation.
- Native saved tokenizer/template and all 154 atlas rows; rank-8 language adapter.
- BF16 language inference; saved visual state restored in FP32 and dormant.
- Greedy generation, thinking disabled, no candidate scoring; batch 16,
  512 generated-token allowance for every row, 4096 context, no truncation.
- Longest prompt: 923 tokens; longest gold completion: 75 tokens.
- One H200 stage, 615.544 seconds (10m 15.544s), including loading and verification.
  Execution cap was 900 seconds with an 840-second inner deadline.
- Completed app: `ap-BajhKL3lk4VpZgpq9mTwTr`; verified stopped with zero tasks.

## Per-operation exact scores

Each operation has ten examples.

| Operation | Correct |
| --- | ---: |
| Owned buildings touching resource | 2/10 |
| Owned incident roads | 2/10 |
| Local node tiles / resource / number | 2/10 |
| Port access | 0/10 |
| Coverage union | 0/10 |
| Coverage intersection | 5/10 |
| Coverage difference | 1/10 |
| Missing resource coverage | 0/10 |
| Resource pip totals | 0/10 |
| Resource pip argmax, including ties | 0/10 |
| Node pip sum | 1/10 |
| Per-roll production | 2/10 |
| Component road sets | 0/10 |
| Component count | 1/10 |
| Shortest distance | 5/10 |
| Reachable-node sets | 0/10 |
| Atomic distance-rule witnesses | 0/10 |
| Road-removal connectivity | 1/10 |
| Settlement-upgrade production | 2/10 |
| Robber-move production | 0/10 |

## Representative errors

Question bodies below omit the shared supplied board and output instructions.
Responses have only trailing transport tokens removed.

1. `resource_pip_totals/000`: sum tile pips for each resource.
   - Gold: `{"brick":9,"ore":9,"sheep":11,"wheat":12,"wood":17}`.
   - Prediction: `{"brick":10,"ore":12,"sheep":12,"wheat":12,"wood":12}`.
2. `resource_pip_argmax/000`: give every resource tied for greatest pip total.
   - Gold: `sheep wheat`.
   - Prediction: `ore sheep`.
3. `local_node_tiles/001`: identify terrain around `<N23>`.
   - Gold tile set: `<T06> <T17> <T18>`.
   - Predicted tile set: `<T06> <T16> <T17>`; the generated JSON is well-formed.
4. `reachable_nodes/001`: ORANGE nodes reachable from `<N44>`, excluding the start.
   - Gold: `<N05> <N16> <N21> <N43> <N45> <N46> <N47> <N48> <N49> <N50>`.
   - Prediction loops over `<N43>`, `<N44>`, and `<N45>` for 512 atlas occurrences.

## Verification and interpretation

Offline re-scoring verified all 200 unique IDs, source expected answers, complete
metadata, targets, and stored score dictionaries. Summary totals and file hashes
match the saved run receipt. Sets use exact unique unordered equality; JSON must
have exact keys/types and no duplicates or surrounding prose; scalar answers
must follow the integer/sentinel contract. Only trailing transport tokens and
surrounding whitespace are ignored. Raw predictions are preserved.

The low score is not merely output formatting: 128 answers were well-formed but
incorrect. The checkpoint does not yet demonstrate fluent reasoning over this
explicit symbolic representation. This run does not compare Gaussian versus
continued initialization, or establish which would learn best under new SFT.

All 200 states are derived from training-source pools, spanning 195 maps and
trajectories. This is a review-set diagnostic, not an untouched held-out benchmark
or a direct comparison to the earlier image-conditioned evaluations.

CPU-only setup attempts r01-r05 produced no model predictions and requested no
GPU calls. They exposed transitive image dependencies, a native-tokenizer return
type change, and an overly broad header-based LoRA check. The successful r06 uses
explicit token-ID returns and the real inference module hierarchy; raw HF MTP
training tensors are not mistaken for inference adapter targets.

## Artifacts

Run root: `artifacts/runs/sft/spatial-ck128-fluency200-20260915-r06/`.

- `records.jsonl`: every raw prediction, expected answer, metadata, and score.
- `summary.json`: overall and grouped results, restoration/precision evidence.
- `launch.json`, `prepare.json`, `run.json`, `orchestration.json`: execution receipts.
- `analysis.json`: independent re-scoring, format diagnostics, and exact examples.
- `analyze.py`: reproducible offline analysis.

Reproduce the analysis from the repository root:

```bash
PYTHONPATH=. .venv/bin/python artifacts/runs/sft/spatial-ck128-fluency200-20260915-r06/analyze.py
```
