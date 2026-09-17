# Symbolic board-fluency SFT pilot

## Result

**Completed 128 optimizer steps and both greedy evaluations.** The unchanged
review improved from **24/200 (12.0%) to 58/200 (29.0%)**. On the exact 144 held-out
examples with retained baseline responses, accuracy improved from **21/144
(14.6%) to 69/144 (47.9%)**.

| Panel | Before | After | Change |
| --- | ---: | ---: | ---: |
| Unchanged review200 | 24/200 (12.0%) | 58/200 (29.0%) | +17.0 percentage points |
| Matched held-out144 | 21/144 (14.6%) | 69/144 (47.9%) | +33.3 percentage points |
| Full held-out190 | Incomplete baseline | 87/190 (45.8%) | No full-panel paired delta |

The review has 42 newly correct answers, eight regressions, 16 correct in both
runs, and 134 incorrect in both. Matched validation has 50 improvements, two
regressions, 19 correct in both, and 73 incorrect in both. All raw responses were
independently rescored against exact source IDs, golds, metadata and file hashes.

Run: `board-fluency-sft-20260915-r06`. Final checkpoint on the `catan-sft-runs`
Modal volume in workspace `icebear5h`:

```text
/runs/catan-vision-sft/board-fluency-sft-20260915-r06/training/checkpoints/checkpoint-128
```

Checkpoints **32, 64, 96 and 128** are complete and committed, including Trainer
optimizer/scheduler/RNG state. The dedicated app `ap-AyeiKbmYnRGN9mBMLtkBD0` is
**stopped with zero tasks**, confirmed through Modal after completion.

## Training and evaluation conditions

- Parent: `spatial-continuation-20260912-r01/checkpoints/checkpoint-128`.
- Base: `Qwen/Qwen3.8-27B`, revision
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`; immutable saved cache identity retained.
- Function-preserving LoRA expansion **r8/alpha16 → r16/alpha32**. Existing blocks
  and all 154 atlas input/output replacement rows retained; added A rows seeded
  normally and added B columns zero-initialized.
- Train only language LoRA and atlas rows. Base language, vision and merger frozen.
- **Text-only explicit board state**; topology remains implicit in the atlas.
  No image input, strategic targets, full settlement-rule conjunction or Longest
  Road supervision.
- Fresh optimizer/scheduler, effective batch 8 (microbatch 4 × accumulation 2),
  LoRA LR 5e-5, atlas LR 1e-4, dropout 0.05, seed 44; 128 steps and saves every 32.
- Corpus: `artifacts/generated/sft/symbolic_board_fluency_sft_v1/`: 3,200 distinct
  training states on 332 maps, 160 rows per operation. All 200 review state IDs and
  content hashes excluded; source/map separation preserved across original splits.
- Sequential training consumed **1,024 examples**, 51–52 per operation: **0.32
  epoch**, not a complete corpus pass. Reported training sequence-token exposure
  is 866,580; corpus-wide token counts in dataset receipts are not exposure counts.
- Greedy batch 16 generation, thinking disabled, no candidate scoring, 512-token
  completion allowance, 4096 context, no truncation; native saved tokenizer/template.
- Strict operation-aware scoring: exact unique unordered sets, typed JSON with
  exact keys, or integers/sentinels. Only surrounding whitespace and trailing
  transport tokens are stripped; no answer repair or prefix credit.

## Family scores

| Family | Review before | Review after | Full held-out after |
| --- | ---: | ---: | ---: |
| Relations / joins | 6/40 | 17/40 | 24/40 |
| Sets / coverage | 6/40 | 11/40 | 18/40 |
| Aggregation / comparison | 3/40 | 8/40 | 10/30 |
| Connectivity / structure | 6/40 | 12/40 | 16/40 |
| Constraints / consequences | 3/40 | 10/40 | 19/40 |

Held-out aggregation has 30 rows because the two board-wide pip operations each
have only five independent terrain layouts; those layouts are not repeated to
inflate a quota.

## Operation scores

Review columns contain ten examples per operation.

| Operation | Review before | Review after | Full held-out after |
| --- | ---: | ---: | ---: |
| Owned buildings touching resource | 2 | 4 | 5/10 |
| Owned incident roads | 2 | 3 | 7/10 |
| Local node tiles / resource / number | 2 | 8 | 7/10 |
| Port access | 0 | 2 | 5/10 |
| Coverage union | 0 | 5 | 9/10 |
| Coverage intersection | 5 | 2 | 1/10 |
| Coverage difference | 1 | 3 | 5/10 |
| Missing resource coverage | 0 | 1 | 3/10 |
| Resource pip totals | 0 | 0 | 0/5 |
| Resource pip argmax, including ties | 0 | 1 | 0/5 |
| Node pip sum | 1 | 3 | 2/10 |
| Per-roll production | 2 | 4 | 8/10 |
| Component road sets | 0 | 0 | 2/10 |
| Component count | 1 | 7 | 6/10 |
| Shortest distance | 5 | 5 | 8/10 |
| Reachable-node sets | 0 | 0 | 0/10 |
| Atomic distance-rule witnesses | 0 | 2 | 5/10 |
| Road-removal connectivity | 1 | 5 | 9/10 |
| Settlement-upgrade production | 2 | 2 | 5/10 |
| Robber-move production | 0 | 1 | 0/10 |

The strongest review gains are local tile joins and component counts, each +6.
Coverage intersection regressed by three. Reachable-node enumeration and resource
pip totals remain at zero on both full panels. This is a useful continuation
result, not evidence of general board fluency or strategic gameplay competence.

## Formatting and teacher-forced metrics

Review format-valid responses increased **152→187**, and malformed answers fell
**48→13**. Wrong-but-format-valid answers are 128→129: many remaining failures are
substantive. Matched validation malformed answers fell 39→10; full postvalidation
has 180 format-valid answers, of which 93 are wrong. Remaining format errors are
confined to road/node item-set operations.

| Checkpoint | Fixed teacher120 loss | Teacher-forced row exact |
| --- | ---: | ---: |
| 32 | 0.754516 | 26.67% |
| 64 | 0.522874 | 37.50% |
| 96 | 0.505669 | 45.00% |
| 128 | 0.487124 | 45.83% |

These are teacher-forced metrics, separate from generated-answer correctness.
The last checkpoint was evaluated as planned; no post-hoc checkpoint selection.

## Verification and recovered execution issues

The real GPU gate showed **max absolute logit difference 0.0** for parent versus
expanded model, expanded model versus trainer initialization, and trained model
versus saved/reloaded checkpoint. All three trainable groups had finite nonzero
gradients and updates; new-rank B columns learned. All 1,184 frozen parameter
versions were unchanged during the gate. The gate's two steps used an isolated
directory; main training initialized from the pristine expanded bundle.

An earlier gate had rejected a saved visual file whose serialization hash differed.
Independent CPU comparison established that all 333 FP32 tensors were bitwise
identical. Frozen-state checks now compare canonical tensor names/shapes/dtypes/
bytes while retaining raw artifact hashes for provenance. Gate and final main
checkpoint share this canonical frozen-visual digest:

```text
4b5d8892dc201fb0f9cda1decca4d1350225e2a6f29f6a3b8c7ddad123b8dcfe
```

Prior attempts also exposed JSON key normalization, Modal polling-timeout types,
inference/autocast mismatch, and an insufficient baseline-generation deadline.
The retained 144 baseline records are the completed prefix from r04, verified
against unchanged inputs/code/expanded initialization and rescored. The other 46
baseline responses do not exist and are not counted as wrong or imputed.

The offline verifier checked completed stage/call identities, saved checkpoint
histories, pristine initialization hashes, all 390 final responses, 144 retained
baseline responses and 200 review-baseline responses. It did not reload 27B weights
locally; live tensor/logit evidence comes from the completed remote gate/workers.

## Compute

The admitted launch envelope was **$14.863204**, including a $2.50 allowance for
earlier attempts and $0.75 control/termination reserve, under the approved $15.

| r06 component | Recorded wall seconds | Provisioned-rate estimate |
| --- | ---: | ---: |
| CPU preparation | 94.07 | $0.0083 |
| H200 gate | 237.37 | $0.4165 |
| H200 training | 1,402.31 | $2.4607 |
| H200 post-evaluation | 716.05 | $1.2565 |
| CPU coordinator, concurrent with stages | 2,463.14 | $0.0432 |
| r06 subtotal | | **$4.1852** |

Adding the two CPU diagnostic app windows ($0.0360), the unchanged earlier-attempt
allowance ($2.50), and the full reserve ($0.75) gives an allowance-inclusive estimate
of **$7.47**. Diagnostic overlap with the prior allowance is intentionally
conservative. These are provisioned-rate wall-window estimates, **not an actual
Modal bill**. Unknown earlier startup/reservation/termination tails remain in the
allowances. Storage, image build/storage, subscriptions and unrelated jobs are
excluded. The app was stopped at 23:48:20 UTC, with zero tasks.

## Interpretation and artifacts

Review200 is a previously inspected, training-source-derived diagnostic; its
states were excluded from this SFT run, but it is not an untouched benchmark.
Held-out190 uses 56 states on only five maps and lacks effective road blockers and
cycles. Matched144 is a partial, operation-imbalanced baseline subset. No transfer
settlement/Longest Road, image-retention, seed replication, or self-play result is
claimed. With only 0.32 epoch observed, this pilot does not establish convergence.

## Extension: 128 further steps (2026-09-16)

Run `board-fluency-extension-20260915-r01` continued the trained r06
checkpoint-128 with 128 fresh-optimizer updates over original corpus rows
1025–2048 (cumulative 256 updates, 2,048 unique examples, 0.64 epoch). Rank,
scope, learning rates and evaluation panels are unchanged. Checkpoints 32/64/96/128
are committed; the app is stopped with zero tasks. Final checkpoint:

```text
/runs/catan-vision-sft/board-fluency-extension-20260915-r01/training/checkpoints/checkpoint-128
```

Fixed teacher120 loss improved 0.487 → 0.360; teacher-forced row exact
45.8% → 47.5%. Greedy results against the complete r06 baseline:

| Panel | r06 (128 steps) | Extension (256 steps) | Change |
| --- | ---: | ---: | ---: |
| Review200 | 58/200 (29.0%) | 57/200 (28.5%) | −1 |
| Held-out190 | 87/190 (45.8%) | 96/190 (50.5%) | +9 |
| Combined390 | 145/390 (37.2%) | 153/390 (39.2%) | +8 |

Review paired outcomes: 12 improved, 13 regressed, 45 correct in both, 130 wrong
in both. Held-out paired outcomes: 18 improved, 9 regressed, 78 correct in both,
85 wrong in both. Held-out gains concentrate in constraints/consequences (+5),
connectivity (+2) and relations (+1); review is flat with churn on both sides.
Reachable-node enumeration stays near zero and both pip-total operations remain
at zero on each full panel.

Recorded-window compute for the extension is about $3.78; cumulative with the
pinned prior allowance-inclusive total is about $11.26, under the approved $15.
This is a provisioned-rate wall-window estimate, not a provider bill.

Local run root: `artifacts/runs/sft/board-fluency-extension-20260915-r01/`
(`analysis.json`, `analyze.py`, `cost_addendum.json`, `stop_receipt.json`).

## Second extension: 256 further steps (2026-09-16)

Run `board-fluency-extension-20260915-r02` continued the ext-r01 checkpoint with
256 fresh-optimizer updates: original rows 2049–3200 (1,152 fresh) then a
wrap to rows 1–896 (896 second-epoch repeats). Cumulative training is 512
updates and 4,096 presentations over all 3,200 unique rows (1.0 epoch). Eight
checkpoints (32–256) are committed; the app is stopped with zero tasks.
Final checkpoint:

```text
/runs/catan-vision-sft/board-fluency-extension-20260915-r02/training/checkpoints/checkpoint-256
```

Fixed teacher120 loss improved 0.360 → 0.299; teacher-forced row exact
47.5% → 60.8%. Greedy results against the ext-r01 baseline:

| Panel | 256 steps | 512 steps | Change |
| --- | ---: | ---: | ---: |
| Review200 | 57/200 (28.5%) | 76/200 (38.0%) | +19 |
| Held-out190 | 96/190 (50.5%) | 116/190 (61.1%) | +20 |
| Combined390 | 153/390 (39.2%) | 192/390 (49.2%) | +39 |

Review paired outcomes: 28 improved, 9 regressed, 48 correct in both, 115 wrong
in both. Held-out paired outcomes: 28 improved, 8 regressed, 88 correct in both,
66 wrong in both. Malformed answers fell 28 → 3 across the 390 rows. Reachable
nodes jumped 1/10 → 8/10 on held-out (still 0/10 on review); both pip-total
operations moved off zero on held-out (1/5 each) with review pip totals reaching
4/10. Constraints/consequences regressed 12 → 9 on review while held-out
constraints held at 24/40.

Recorded-window compute for this extension is about $5.18; cumulative with
pinned prior allowances is about $16.43, under the user-approved $21 ceiling
(provisioned-rate estimate, not a provider bill).

Local run root: `artifacts/runs/sft/board-fluency-extension-20260915-r02/`
(`analysis.json`, `analyze.py`, `cost_addendum.json`, `stop_receipt.json`).

## Third extension: 512 further steps (2026-09-16)

Run `board-fluency-extension-20260915-r03` continued the r02 checkpoint with
512 fresh-optimizer updates: the full 3,200-row corpus (second epoch) plus rows
1–896 (partial third epoch), 4,096 presentations at batch 8. Cumulative training
is 1,024 updates and 8,192 presentations. Sixteen checkpoints (32–512) are
committed; the app is stopped with zero tasks. Final checkpoint:

```text
/runs/catan-vision-sft/board-fluency-extension-20260915-r03/training/checkpoints/checkpoint-512
```

Fixed teacher120 loss improved 0.299 → 0.190; teacher-forced row exact
60.8% → 71.7%. Greedy results against the r02 baseline:

| Panel | 512 steps | 1,024 steps | Change |
| --- | ---: | ---: | ---: |
| Review200 | 76/200 (38.0%) | 103/200 (51.5%) | +27 |
| Held-out190 | 116/190 (61.1%) | 126/190 (66.3%) | +10 |
| Combined390 | 192/390 (49.2%) | 229/390 (58.7%) | +37 |

Review paired outcomes: 37 improved, 10 regressed, 66 correct in both, 87 wrong
in both. Held-out paired outcomes: 27 improved, 17 regressed, 99 correct in
both, 47 wrong in both. Review pip totals jumped 4/10 → 6/10 and pip argmax
4/10 → 7/10; held-out pip totals reached 5/5. Review constraints recovered
9 → 19. Reachable nodes remain 0/10 on review (7/10 held-out). Only 2 malformed
answers remain across all 390 rows.

Recorded-window compute for this extension is about $9.66; cumulative with
pinned prior allowances is about $26.09, under the user-approved $31 ceiling
(provisioned-rate estimate, not a provider bill).

Local run root: `artifacts/runs/sft/board-fluency-extension-20260915-r03/`
(`analysis.json`, `analyze.py`, `cost_addendum.json`, `stop_receipt.json`).

Local run root: `artifacts/runs/sft/board-fluency-sft-20260915-r06/`.

- `analysis.json`: verified metrics, paired IDs, family/operation/format results.
- `cost_estimate.json`, `stop_receipt.json`: cost assumptions and stopped-app evidence.
- `launch.json`, `coordinator.json`, stage `result.json`/`wrapper.json`/`worker.log`:
  exact configuration and execution receipts.
- `posteval/{review,validation_eval}/records.jsonl`: all 390 raw final predictions.
- `gate/pre-validation190/records.jsonl`: retained 144 baseline predictions.
- `training/checkpoints/checkpoint-{32,64,96,128}/trainer_state.json`: downloaded
  checkpoint histories. Model and optimizer tensors remain on the Modal volume.

Reproduce the offline audit without inference:

```bash
.venv/bin/python -B artifacts/runs/sft/board-fluency-sft-20260915-r06/analyze.py
```
