# Mixed spatial continuation

## Status

Run `spatial-continuation-20260909-r01` completed successfully. All 128 training
steps and six post-training panels finished, and all 676 pre/post generated
responses were independently rescored with no mismatches. The run learned some
new tasks but regressed board extraction and the original spatial panel.

| Panel | Before | After |
| --- | ---: | ---: |
| Corrected spatial QA | 57/120 (47.5%) | 53/120 (44.2%) |
| Complete-board readout | 64/64 (100%) | 53/64 (82.8%) |
| Touching tiles | 0/54 | 19/54 (35.2%) |
| Shortest paths | 0/64 | 1/64 (1.6%) |
| Local neighborhoods | 0/64 | 18/64 (28.1%) |
| Dice production | 16/64 (25.0%) | 46/64 (71.9%) |

Forgetting is real but localized: occupied-location accuracy is 1,193/1,204
(99.09%), versus 1,204/1,204 before. There are 11 occupied-node errors, ten
empty-node errors, and one robber error. Roads, terrain, numbers, and ports
remain exact. Nine of the 11 non-exact boards belong to one held-out layout;
the other two belong to a second layout. Full-board exactness penalizes any
single fact error, so the 17.2-point board drop is not a 17.2-point loss across
all board facts.

This is not a replacement for the parent checkpoint. Keep it as an experiment:
production and neighborhood queries improved, but path learning is negligible
and 32 readout-rehearsal steps did not fully preserve established extraction.
No additional training or checkpoint promotion was performed.

Training worker elapsed time was 1,925.75 seconds (32.1 minutes), excluding
container startup; its complete coordinated stage took 33.5 minutes. The
post-evaluation worker took 1,600.35 seconds (26.7 minutes), dominated by the
64 full-board generated answers. The pre-evaluation worker took 6.0 minutes.

[Modal app](https://modal.com/apps/icebear5h/main/ap-5DfPPuxoiAEKDyNt8R50bo)
in workspace `icebear5h`.
Coordinator: `fc-01M22ZWJGBAV7R05K19DYKJEK0`.
Pre-evaluation: `fc-01M2307RKDCNRCY00C3RP3XBN3`.
Training: `fc-01M230KFH21HT3NH15901H10C0`.
Post-evaluation: `fc-01M232GXRCGP3GR9YPKY8EXFSE`.

## Approved Training Mix

Continue from
`/runs/catan-vision-sft/full-board-new-layouts-20260907/checkpoints/checkpoint-128`.
Preserve rank-8 language LoRA, 154 input/output atlas rows, FP32 visual/merger
masters, BF16 computation, and the parent's learning rates. Start a fresh
optimizer and scheduler with `token_init=keep`, not a fresh base model.

| Family | Optimizer steps | Examples |
| --- | ---: | ---: |
| Directions | 16 | 128 |
| Adjacency/connectivity | 16 | 128 |
| Node to touching tiles | 16 | 128 |
| Shortest node paths | 16 | 128 |
| Local tile/resource/number | 16 | 128 |
| Dice production | 16 | 128 |
| Full-board readouts | 32 | 256 |
| Total | 128 | 1,024 |

Microbatch four with accumulation two. Every optimizer batch contains one task
family; each eight-step block contains the six spatial families once and
readouts twice. Mixing seed 45, inherited trainer seed 44. Checkpoints and
teacher-forced full-board evaluation every 32 steps. No adaptive extra steps.

There are 1,024 distinct training image hashes across 333 training layouts,
with 256 examples per empty/setup/sparse/dense density. Readout rehearsal uses
256 distinct images/layouts and retains original prompts/answers. All images
and contracts already existed; no new rendering or fabricated labels.

Measured target-token exposure is 191,402 tokens, of which 180,651 (94.38%)
belong to readouts. This is **not gradient share**: readouts own 25% of the
optimizer steps, and task batches are not pooled by total target tokens.
The longest readout target is 787 tokens; new-task targets are at most 50.

## Task Contracts

- Touching tiles: unordered exact token sets; duplicates/missing/extras fail.
- Paths: ordered node sequences including both endpoints; every edge must
  exist and path length must be minimal. Every equally short route is accepted.
  Paths use the fixed board graph, ignoring pieces and ownership.
- Local neighborhoods: strict JSON keyed by touching tile token, with resource
  and number; desert uses `desert` and a null number.
- Production: strict five-resource integer counts including zeros, using
  settlements x1, cities x2, robber blocking, and no bank-shortage restriction.
- Existing spatial short-answer and complete-board scoring remain unchanged.

Production examples are balanced zero/nonzero: 64/64 in training and 32/32 in
validation. City-sensitive examples number 22/15 and robber-sensitive examples
17/9 in train/validation. All production targets were independently checked
against the engine's resource-yield logic.

Shortest-path endpoint pairs are split as unordered pairs before generation:
873 train, 279 validation, 279 test. The selected 128 train and 64 validation
pairs are disjoint, including reversals, cover distances 1 through 11, and
include 62/32 examples with tied shortest routes. Test pairs remain reserved.

## Evaluation

| Panel | Rows | Parent baseline | Batch | Generation limit |
| --- | ---: | --- | ---: | ---: |
| Corrected answer-only spatial | 120 | 57/120, reused | 48 | 16 |
| Complete-board readout | 64 | 64/64, reused | 4 | 1,280 |
| Touching tiles | 54 | 0/54 | 16 | 128 |
| Shortest paths | 64 | 0/64 | 16 | 128 |
| Local neighborhoods | 64 | 0/64 | 16 | 128 |
| Dice production | 64 | 16/64 | 16 | 128 |

Greedy generation, thinking disabled, original images only, no candidate
scoring. One model load per evaluation stage, with per-panel short/long budgets.
The existing 120-question and 64-board inputs are unchanged. Remote preflight
verified pixel/input identities, independently rescored old responses, and
confirmed both saved baselines match the parent and current scorer.

Parent visual-file SHA256:
`3535adfa86dbc4675f612f98995222f2f15619d151b5aff2f7537eab147b7183`.
All 333 visual tensors are FP32; base/parent atlas-token IDs match.
The new validation panels use the same five held-out board layouts; training
maps are disjoint from validation, test, and color-diagnostic maps. Static
atlas recall is not unseen-graph generalization or proof of image dependence.

Post-training evaluates the new periodic `checkpoint-128`, not the BF16 final
export. Results retain per-panel accuracy and full-board occupied/layout
metrics so new learning is not pooled with forgetting into one average.

## Receipts And Verification

Dataset: `artifacts/generated/board_recognition/spatial_continuation_v1/`.
Local receipt: `artifacts/runs/sft/spatial-continuation-20260909-r01/launch.json`.
Final coordinator receipt: same directory, `result.json`.
Historical progress snapshot: same directory, `progress.json` (not final status).
Downloaded parent generations: `pre/`; independent audit: `pre-verification.json`.
Downloaded final generations: `post/`; independent audit: `post-verification.json`.
Remote progress/final receipt on `catan-sft-runs`:
`catan-vision-sft/pipelines/spatial-continuation-20260909-r01/result.json`.
New model output: `/runs/catan-vision-sft/spatial-continuation-20260909-r01`.

Exact launch command (already executed; same-name reruns are refused):

```bash
.venv/bin/python -B -m sft.modal_spatial_continuation \
  --run-name spatial-continuation-20260909-r01 --execute
```

The run-local `verify.py` downloads a completed stage using serial direct-volume
reads and independently checks original targets, metadata, input/image identities,
record hashes, each saved score, and every summary aggregate. Parent audit passed
all 246 rows; final audit passed all 430 rows. Both are reproducible locally:

```bash
env PYTHONPATH=. .venv/bin/python -B \
  artifacts/runs/sft/spatial-continuation-20260909-r01/verify.py \
  --stage post
```

The launcher records current working-tree Python hashes, not just a Git commit.
GPU stages run sequentially with one-hour timeouts, five-minute startup
allowances, no retries, and a detached CPU coordinator that records call IDs
and cancels its active child on failure/timeout. Historical inputs, model
checkpoints, and results remain intact.

Final local verification: 242 focused tests passed, with scoped Ruff and diff
checks passing. Independent data/scorer and launcher reviews completed; the
task-declaration bypass found in review was fixed and regression-tested before
launch. Conflicting metadata can no longer send ordered paths through the
generic unordered readout scorer.
