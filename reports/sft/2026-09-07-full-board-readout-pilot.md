# Full-board readout pilot

The new experiment supervises **image -> complete visible board state** in one
answer: 19 tile resource/number pairs, 54 nodes, 72 edges, nine ports, and robber.
The target uses the existing 154 atomic atlas tokens and canonical semicolon
ordering. Ground-truth state values appear only in the assistant target.

## Preserved earlier work

`artifacts/archives/before-full-board-20260907-121504/manifest.json` records the
old branch, commit, and SHA256 hashes. `tracked.patch` and
`untracked-source.tar.gz` preserve the dirty source state before this pivot;
`runs/` preserves local experiment receipts. To reconstruct it, use a separate
checkout at `381f7d5668631eb413e7a1e76789c21f0c2ff413` (use the exact full hash
from the manifest), apply the patch, and extract the source archive there.
Existing generated datasets and remote checkpoints remain in place.

The O-LoRA two-step initialization/save/reload smoke is documented in
`reports/sft/2026-09-07-olora-smoke.md`; its full training run was not launched.
The terrain parent remains on `catan-sft-runs` at
`catan-vision-sft/catan-qwen38-gauss-s2-terrain-20260904/398f0a023ec9/checkpoints/checkpoint-384`.

## Pilot scope

- Fresh cached `Qwen/Qwen3.8-27B`; no curriculum parent or O-LoRA.
- Standard rank-8 language LoRA, 154 trainable atlas rows, full FP32 trainable
  visual tower and merger, BF16 computation/language stack.
- 1,024 training boards over 77 layouts, shuffled together across density bins:
  538 dense, 227 sparse, 216 setup, 43 empty.
- Existing layout-separated validation/test and color diagnostic splits retained
  (64 boards each). Pilot generation uses 16 validation boards across five unseen
  layouts, plus blank/shuffled controls on five of those boards.
- 128 optimizer steps, microbatch four, accumulation two: one epoch; checkpoints
  every 32 steps. One H200, one-hour training timeout, no retries.
- Token/LM/vision/merger learning rates: 5e-4 / 1e-4 / 5e-6 / 5e-5;
  10% warmup, seed 42. No special loss weighting.

Dataset: `artifacts/generated/board_recognition/replay_v1/full_board_readout_v1`.
Source manifest SHA256:
`70c9fc9aa2d29d214c4d91e540c734ffed14d1491ccfddc4e83d607123e786d5`.
Training JSONL SHA256:
`b0bf4d39f5865115814c85fab0cd00f37e5007d51f7b5a0a3f795b9a76ac8e0b`.

## Validation and interpretation

63 focused tests passed, including complete engine-contract targets, duplicate
and missing address rejection, occupied-only scoring, existing trainer behavior,
and candidate scoring. Ruff passed for the new files and evaluator.

Report strict full-board exactness, structural coverage, separate occupied and
empty node/edge scores, terrain, ports, robber, and color/piece buckets. An
all-empty prediction must not hide behind high empty-slot accuracy. Generated
evaluation reloads the periodic FP32 visual checkpoint, not the deliberately
BF16 final export. Compare controls on the matching five original-image rows.

This small pilot tests feasibility and early learning. One epoch and 16 validation
boards cannot establish convergence or broad generalization. Test/color/OOD
claims require their own subsequent evaluation.

## Execution

Launcher: `sft/modal_full_board_pilot.py`; default is plan-only. Training and
evaluation require `--execute`; both refuse to overwrite their run outputs.
Launch receipts and results are stored under
`artifacts/runs/sft/full-board-fresh-20260907/`.

Status: preflight passed (681–790 answer tokens, no truncation); first optimizer
steps completed with finite loss/gradients. The live scope audit confirms all 333
visual/merger tensors are FP32 and reports no forbidden trainable parameters.
Modal app: `ap-AzgxukUY0oD2OFvvM3rjFd`. Archive verification checked all
84 stored file hashes successfully. Training and automatic evaluation both completed successfully. Combined results
were downloaded to `artifacts/runs/sft/full-board-fresh-20260907/completed_pilot.json`.

Completion app: `ap-UbNz3qhIxB00SiWc8wZ0tX`.

A separate CPU function waits for training to complete and then runs the fixed
evaluation once. Its receipt is `completion-launch.json`; the final combined
remote artifact is `catan-vision-sft/full-board-fresh-20260907/completed_pilot.json`
on `catan-sft-runs`. Evaluation uses five controls spanning dense, sparse, setup,
and empty boards. Initial per-layout representatives alone were all empty, so
they are not used as the control panel.

To retrieve the eventual combined result:

```sh
.venv/bin/python -m modal volume get catan-sft-runs catan-vision-sft/full-board-fresh-20260907/completed_pilot.json artifacts/runs/sft/full-board-fresh-20260907/completed_pilot.json
```

## Completed results

Training app stopped at 13:05 EDT and evaluation app at 13:23 EDT on September 7.
On the 16-board validation pilot: complete coverage 16/16, no duplicate addresses,
board exact 3/16 (two empty boards, one setup board). Tile resources 304/304,
tile numbers 304/304, ports 144/144. Occupied nodes 63/100, occupied edges
85/116, cities 9/20, robber 6/16. Combined occupied-location accuracy 148/216
(68.5%). These are small-pilot results, not full validation or OOD scores.
Blank controls recovered 0/55 occupied locations; shuffled controls 12/55.
The control panel is a subset; compare against matching original rows before
quantifying the effect of shuffling.

## Prediction audit and evaluation limitation

Matched control originals recover 38/55 occupied locations (69.1%), compared
with shuffled 12/55 (21.8%) and blank 0/55. There are 68 occupied-to-empty
errors and 67 empty-to-piece errors; no wrong nonempty color/type substitutions
at occupied addresses. This suggests substantial placement/addressing error,
but is not a proven causal diagnosis.

The pilot selection was uneven: 11/16 boards and 215/216 occupied locations
come from one held-out layout. The other layouts are predominantly empty.
Consequently piece generalization across layouts is not established. Evaluate
the existing checkpoint on all 64 validation boards and report each layout
before interpreting the pilot as broad generalization or changing training.
