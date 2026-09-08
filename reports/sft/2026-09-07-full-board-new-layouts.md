# Full-board training on new layouts

Continue from `full-board-epoch2-20260907/checkpoints/checkpoint-128`, whose
64-board validation result was 40/64 completely correct and 1171/1204 occupied
locations correct (97.3%). The completed parent evaluation is reused as the
baseline; its checkpoint and evaluation-file hash are checked before launch.

## Training pass

Use 768 previously unseen training boards, drawn from all 256 new layouts in
`full_board_diverse_v1`: one dense, one sparse, and one setup board per layout.
Mix in 256 original examples, with six new and two older examples in each
effective batch of eight. The full prepared pool still contains 4,096 new boards;
this bounded pass uses 768 of them.

The 1,024-row pass contains 405 dense, 302 sparse, 299 setup, and 18 empty boards.
Seed 44 controls selection and interleaving. Image/target pairs remain intact;
training and held-out layouts are disjoint.

Train for 128 steps on one H200, microbatch four and accumulation two, with the
existing learning rates, rank-8 language adapter, FP32 trainable visual weights,
and BF16 computation. Load the parent's FP32 periodic checkpoint exactly;
start a fresh optimizer/warmup. Save checkpoints every 32 steps. The parent
and all earlier checkpoints remain intact.

## Evaluation and bounds

Evaluate on the same 64 validation boards in the same order. Report full-board
exactness, occupied nodes/edges, piece/color groups, and each held-out layout.
No shuffled-image evaluation. The five-board blank-image check remains in the
existing evaluator. Test and color-diagnostic split membership are unchanged.

Training and evaluation run sequentially, each with a one-hour GPU timeout and
no retries. The CPU coordinator records each stage and stops the chain on failure.
This is one bounded training pass and evaluation, not an indefinite training loop.

Seven focused tests passed for complete-board scoring, balanced selection and
per-batch replay. Ruff passed for the changed launcher files and tests.

## Artifacts

Launcher: `sft/modal_full_board_new_layouts.py::new_layout_pass` (requires
`--execute` to launch).
Local receipt: `artifacts/runs/sft/full-board-new-layouts-20260907/launch.json`.
Remote output: `/runs/catan-vision-sft/full-board-new-layouts-20260907`.
Remote coordinator progress/result on `catan-sft-runs`:
`catan-vision-sft/pipelines/full-board-new-layouts-20260907/result.json`.

Status: upload and preflight completed; the detached training/evaluation cycle
launched. Coordinator call: `fc-01M1ZJN8Z134H8QJ836KF90Y8Q`.

[Live Modal app](https://modal.com/apps/icebear5h/main/ap-zld16s1xIzUSNrGtXWJQLr)
in workspace `icebear5h`.

## Completed result (September 8)

Training, final-bundle save/reload validation, and generated evaluation completed
successfully. All 64 held-out boards are now exactly correct, up from 40/64.
All 15 dense boards are exact (previously 2/15), all 18 sparse (11/18), all
26 setup (23/26), and all five empty (4/5). All 1,204 occupied locations and
6,860 empty locations are correct; terrain, numbers, ports, and robber remain
perfect on this panel. There are no duplicate or missing addresses.

The 64 saved generated responses were downloaded and independently rescored
locally with the strict board scorer: 64/64 correct over five distinct layouts.
The current validation panel is saturated; untouched test, color-diagnostic,
and renderer/OOD performance have not been established by this run.
This result follows another 128 steps using 768 new boards plus 256 older
examples; it does not isolate diversity from the effect of additional training.

Final receipt: `artifacts/runs/sft/full-board-new-layouts-20260907/result.json`.
No Catan tasks remain active in the current Modal app list.
