# Mixed spatial extension: 256 additional steps

## Status

Run `spatial-continuation-20260912-r01` was **cancelled at the user's request**
during training on September 12, 2026. The app, training worker, and coordinator
were stopped. Modal verified `stopped`, zero tasks, at 20:04:36 UTC
(16:04:36 America/New_York). The final generated evaluation did not run.

The coordinator receipt records `failed` / `RemoteError` during training as a
consequence of the stop, not a completed experiment. Its last earlier observed
log was update 16/256; that is not the final step reached before cancellation.
The planned step counts and schedule below describe the approved experiment,
not completed work. Cancellation evidence is in local `cancellation.json`.

- [Modal app](https://modal.com/apps/icebear5h/main/ap-a4jolUBznIMMAdwGQFJsrs)
- Coordinator: `fc-01M2BGP3QRPN8R3AD5FAJ7ZB1Y`
- CPU preflight: `fc-01M2BGPR20QZPP7EMK6TRJQQPF`
- Training: `fc-01M2BH8VSZB14T9QBC5S0BQY6W`
- Local receipt directory:
  `artifacts/runs/sft/spatial-continuation-20260912-r01/`
- Remote coordinator receipt on `catan-sft-runs`:
  `catan-vision-sft/pipelines/spatial-continuation-20260912-r01/result.json`

## Approved experiment

Continue from
`/runs/catan-vision-sft/spatial-continuation-20260909-r01/checkpoints/checkpoint-128`
for **256 additional optimizer updates**, or **384 cumulative mixed updates**.
The new run uses a fresh optimizer and schedule, as did the prior continuation.
The planned final periodic checkpoint was `checkpoint-256`.

| Family | Additional updates | Example presentations |
| --- | ---: | ---: |
| Directions | 32 | 256 |
| Adjacency/connectivity | 32 | 256 |
| Node to touching tiles | 32 | 256 |
| Shortest node paths | 32 | 256 |
| Local tile/resource/number | 32 | 256 |
| Dice production | 32 | 256 |
| Complete-board readouts | 64 | 512 |
| Total | 256 | 2,048 |

This repeats the existing ordered 1,024-example dataset twice. It adds exposure,
not new unique images. Whole-board readouts remain 25% of optimizer updates;
they include all board entities and empty locations in one completion.

Inherited settings include seed 44, microbatch four, accumulation two, rank-8
language LoRA, all 154 atlas input/output rows, FP32 visual/merger masters, BF16
computation, and the parent's learning rates. Evaluation and checkpoints occur
every 32 updates; the plan retained all eight new periodic checkpoints and their
teacher-forced validation history. Actual checkpoint inventory after cancellation
has not been audited.

## Evaluation

Reuse the parent's six saved post-training panels after CPU verification of
their checkpoint identity, input bytes/pixels, response hashes, every score,
and summary aggregates. Generate the same six panels after the extension.
Generation uses original images, greedy decoding, thinking disabled, and FP32
visual restoration with BF16 computation.

| Panel | Starting checkpoint |
| --- | ---: |
| Corrected spatial QA | 53/120 (44.2%) |
| Complete-board readout | 53/64 (82.8%) |
| Touching tiles | 19/54 (35.2%) |
| Shortest paths | 1/64 (1.6%) |
| Local neighborhoods | 18/64 (28.1%) |
| Dice production | 46/64 (71.9%) |

The earlier pre-spatial parent scored 64/64 on whole-board extraction. Compare
the new result against both that reference and this run's direct parent; a
single wrong fact fails whole-board exactness, so retain occupied-location and
layout-level metrics alongside it.

## Runtime and verification

Based on the previous run, training is estimated at about 65 minutes and final
evaluation at about 27 minutes, excluding startup/preflight. These are runtime
estimates, not measured new-run time or a metered cost quote.

One H200 stage runs at a time. CPU preflight is capped at 20 minutes, training
at two hours, and post-evaluation at one hour. The four-hour CPU coordinator
records child IDs, cancels the active child on failure, and launches no retries
or automatic extra training. Parent paths and historical results are preserved.

Before launch, 350 focused tests passed, scoped Ruff and diff checks passed,
and an independent launch review found no blockers. Offline integration checks
rescored all 430 parent responses. Remote preflight subsequently passed checks
of the actual cached checkpoint, uploaded datasets, and all six baseline
originals, reproducing the starting scores above.

Exact launch command, already executed:

```bash
.venv/bin/python -B -m sft.modal_spatial_extension \
  --run-name spatial-continuation-20260912-r01 --execute
```

Read current status without starting additional compute:

```bash
.venv/bin/python -B -m sft.scripts.verify_spatial_extension \
  --download --status-only
```

For a completed pipeline, the retrieval/verification command is below. This
cancelled run has no completed final response set to verify:

```bash
.venv/bin/python -B -m sft.scripts.verify_spatial_extension --download
```

Offline verification uses the same command without `--download`. A successful
audit writes `post-verification.json`; merely launching the pipeline does not
establish a new score.
