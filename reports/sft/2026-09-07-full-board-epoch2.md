# Full-board pipeline: balanced evaluation and epoch two

User preference update: omit shuffled-image evaluation from subsequent full-board
runs. The local evaluator now runs original and blank images only. The already
launched Modal app contains its earlier code snapshot; this edit does not change
that running app or interrupt its training. Existing shuffled results remain
historical records, not a required progression metric.

Continue the image-to-complete-board-state objective from the first pilot's
FP32 checkpoint. Keep prior checkpoints, datasets, and evaluations intact.

## Fix to evaluation selection

The first pilot's 16-row sample concentrated 215/216 occupied locations in one
layout. The full validation split already contains occupied boards from all five
held-out layouts. Use all 64 existing validation rows for both before/after
evaluation; the split membership and image/target pairs do not change.

| Held-out layout | Boards | Occupied locations |
| --- | ---: | ---: |
| r_replay_189649315 | 13 | 243 |
| r_replay_193008240 | 13 | 252 |
| r_replay_193905959 | 12 | 225 |
| r_replay_194198328 | 13 | 246 |
| r_replay_194209320 | 13 | 238 |

The small-sample selector now cycles through layouts and densities and shuffles
intact examples within each group. A 16-row sample has 3–4 boards per layout,
including dense, sparse, and setup in every layout. This changes sampling/order,
not board geometry or token identities. Any future geometric augmentation must
transform the image and symbolic labels together.

Controls use one dense board from each layout, with 33–42 occupied locations per
board. Original, blank, and shuffled scores use the exact same five targets;
shuffling uses different layouts. Reports include each layout's score and an
equal-weight average across layouts, alongside the overall location score.

## Bounded continuation

1. Evaluate original checkpoint-128 on all 64 validation boards and controls.
2. Initialize from that FP32 checkpoint and run another 128 steps on the same
   1,024 training boards, reshuffled with seed 43. Microbatch four, accumulation
   two, unchanged learning rates/rank/precision/objective. This is another epoch
   with a new optimizer and warmup, not an optimizer-state resume.
3. Evaluate the resulting checkpoint on the identical 64 boards and controls.

One H200 stage at a time; each GPU call has a one-hour timeout and no retry.
The CPU coordinator has a 185-minute timeout. A failed stage stops the chain.
The fixed test and color-diagnostic splits are not used in this cycle.

The continuation loader now promotes the visual destination to FP32 **before**
copying parent checkpoint tensors. A regression test with sub-BF16 deltas verifies
exact preservation. This affects continuation; the fresh pilot did not take this
loading path. 46 focused tests passed, including sampler coverage/determinism,
layout averaging, FP32 preservation, trainer, and candidate-scoring checks. Ruff
passed for the changed files.

## Artifacts and status

Launcher: `sft/modal_full_board_continue.py::continue_pipeline` (plan-only unless
`--execute`). Local receipt directory:
`artifacts/runs/sft/full-board-epoch2-20260907/`.

Remote parent:
`/runs/catan-vision-sft/full-board-fresh-20260907/checkpoints/checkpoint-128`.
New training output:
`/runs/catan-vision-sft/full-board-epoch2-20260907`.
Coordinator progress and final result on `catan-sft-runs`:
`catan-vision-sft/pipelines/full-board-epoch2-20260907/result.json`.

Status: preflight passed and the detached continuation cycle launched. Baseline
evaluation runs before training.

[Live Modal app](https://modal.com/apps/icebear5h/main/ap-ggxgZhOtaRJ2pdjHfYfCYn)
in workspace `icebear5h`. Coordinator call: `fc-01M1YGTKEX4YMJYBSQH0W2TZ75`.

## Completed full-validation baseline

The original checkpoint-128 completed the full 64-board evaluation. Coverage
64/64, no duplicates; exact 5/64 (two empty, three setup; no sparse/dense exact).
Occupied-location accuracy 808/1,204 = 67.1%; equal-layout average 67.0%.
Per-layout scores: 66.7%, 69.4%, 60.9%, 63.0%, 75.2%.
Terrain resources/numbers and ports are all exact at the item level; occupied
nodes 380/546, roads 428/658, cities 65/115, robber 34/64.

Matched five dense controls: original 120/178 = 67.4%, shuffled 6/178 = 3.4%,
blank 0/178. This supports image dependence across the five held-out layouts.
The training and post-training evaluation subsequently completed successfully; see the final comparison below.
Local baseline receipt: `artifacts/runs/sft/full-board-epoch2-20260907/baseline-result.json`.

## Completed epoch-two comparison

Training, final-bundle reload validation, and the generated 64-board evaluation
completed successfully. No Catan tasks remain active in the current Modal list.

| Metric | Original checkpoint | After another epoch |
| --- | ---: | ---: |
| Full-board exact | 5/64 | 40/64 |
| Occupied locations | 808/1204 (67.1%) | 1171/1204 (97.3%) |
| Occupied nodes | 380/546 (69.6%) | 525/546 (96.2%) |
| Roads | 428/658 (65.0%) | 646/658 (98.2%) |
| Cities | 65/115 (56.5%) | 115/115 (100%) |
| Robber | 34/64 (53.1%) | 64/64 (100%) |
| Terrain/numbers/ports | 100% | 100% |

All 64 predictions contain every required location with zero duplicate addresses.
Post-training occupied-location accuracy across layouts ranges from 94.7% to
98.7%. Board exact by density: dense 2/15, sparse 11/18, setup 23/26, empty 4/5.
Dense boards remain the largest exact-match gap. This comparison uses the same
64 validation boards over five held-out layouts; test/OOD are not evaluated here.
The epoch used the original 1,024 training examples, not the newly prepared
5,120-board diversified shard.

Final local receipt: `artifacts/runs/sft/full-board-epoch2-20260907/result.json`.

## Remaining error audit

The final predictions have 54 incorrect slots across 24 boards: 31 occupied
locations predicted empty, 21 empty locations predicted occupied, and two blue
settlements predicted as blue cities. All errors concern nodes/edges. Dense
boards account for 31 of the 54 errors. `<N32>` has 11 errors spanning three
layouts; the errors contain 24 unique layout/address/expected/predicted patterns,
so repeated replay frames should not be interpreted as independent failures.
An example predicts a green road at `<E06_23>` instead of `<E06_07>`, which
shares endpoint 06. This supports a task-level diagnosis of precise piece
placement/occupancy reliability; it does not localize the cause to a model module.
Raw error audit: `artifacts/runs/sft/full-board-epoch2-20260907/error-analysis.json`.
