# Gaussian entity-marker final: small failure analysis

Completed September 4, 2026. One explicitly approved H200 evaluation, 308 rows,
no training, no ablations, and no retries. The app is stopped with zero tasks.

## Result

| Query direction | Free generation | Same-type/letter candidate scoring |
|---|---:|---:|
| Marker to atlas token | 149/154 (96.75%) | 150/154 (97.40%) |
| Atlas token to marker | 154/154 (100%) | 154/154 (100%) |
| Overall | 303/308 (98.38%) | 304/308 (98.70%) |

Candidate scoring restricts atlas answers to the expected entity family; it can
hide a wrong-entity answer. Both metrics are therefore retained. The earlier
99.17% training-evaluator result was teacher-forced and batch-averaged over
1,540 rows, not free generation on this sample; the scores are not interchangeable.

## Every observed failure

All five occur in marker-to-token queries. The reverse query on the same image
is correct in every case. None of the predicted locations is another marked
location in that image's marker group.

| Expected | Generated | Classification | Distance at 1024px | Expected candidate rank |
|---|---|---|---:|---:|
| `<E22_23>` | `<E02_09>` | Wrong slanted edge | 257px | 2 |
| `<N03>` | `<N00>` | Wrong node | 172px | 2 |
| `<N08>` | `<N02>` | Wrong node | 149px | 2 |
| `<N43>` | `<E43_47>` | Node answered as an incident edge | 43px | 1 among nodes |
| `<P05>` | `<P04>` | Wrong port | 258px | 2 |

The edge confusion is close in score: the wrong and expected edge log-probabilities
are -0.6391 and -0.7641 (margin 0.125). The other same-family mistakes are less
close: margins 1.50 for N03, 2.25 for N08, and 1.75 for P05. These are model
scores, not calibrated confidence estimates. There are no malformed answers,
repeated tokens, or token-to-marker letter errors in this sample.

All five source images were inspected. The queried markers are visible and their
letters are readable. This does not establish what visual features the model used.

### Edge orientation

| Orientation | Marker to token | Token to marker | Combined |
|---|---:|---:|---:|
| Slanted | 47/48 | 48/48 | 95/96 |
| Vertical | 24/24 | 24/24 | 48/48 |

One error is insufficient to establish an orientation gap. This measures locating
synthetic markers, not recall of actual road pieces.

## Interpretation and limits

- The observed residual is marker-to-atlas identification: four same-family
  location confusions and one node/edge family confusion. Querying an atlas token
  to find its marker works on all sampled images. That is evidence against a
  broad failure of the latter behavior on this sample, not proof of perfect vision.
- Three failures (N03, N08, N43) are on board `r_replay_189649315_s000000`.
  One image per token cannot separate stable token difficulty from board context.
- This run changes both initialization and marker shape relative to the old
  control. It does not isolate either cause, diagnose LoRA capacity, or measure
  occupancy/resource reading, reasoning retention, or actual-piece transfer.
- Keep these five row IDs as regression cases. This small result does not justify
  another initialization change or embedding surgery by itself. No next stage
  was launched as part of this task.

## Reproduction and evidence

- App: [ap-XgCM5XzAhsdEgK8G2F2WPQ](https://modal.com/apps/tetracorp/main/ap-XgCM5XzAhsdEgK8G2F2WPQ).
- Call: `fc-01M1Q9NBZR14J83ZEVNR4D34N1`.
- Bundle: `/runs/catan-vision-sft/catan-qwen38-gauss-s1-markers-entity-20260904/73b1c523741d/final`.
- Source: `artifacts/generated/board_recognition/replay_v1/spatial_localization_v1e/stage1/validation.jsonl`.
- Selection: sorted atlas tokens assigned round-robin to five sorted board IDs;
  both query directions share the selected token's image. 154 tokens, 308 unique
  rows, 62/62/62/62/60 rows across boards. 144 edge, 108 node, 38 tile, 18 port queries.
- Selected row-ID SHA256: `f39faa76104de225600fffe988cad004b5514bc6fece42e35fc60d54bcc782e4`.
- BF16, one H200, batch 48, greedy decoding, maximum 16 generated tokens,
  original images, saved processor settings (1,048,576 maximum pixels), reasoning disabled.
- Function timeout 600 seconds; inference subprocess deadline 540 seconds;
  retries zero; runtime 192.184 seconds. Runtime-based compute estimate **$0.247**
  using the previously measured $4.62/hour eval rate; this is not a settled bill
  and excludes incidental startup/storage overhead.
- Saved local artifacts: [run](../../artifacts/runs/sft/gauss-s1-entity-final-mini-20260904/run.json),
  [selection](../../artifacts/runs/sft/gauss-s1-entity-final-mini-20260904/selection.json),
  [all predictions](../../artifacts/runs/sft/gauss-s1-entity-final-mini-20260904/records.jsonl),
  [failure details](../../artifacts/runs/sft/gauss-s1-entity-final-mini-20260904/failures.json).
- CPU-only recomputation matched all remote counters and validated all 308 row IDs.
  Fourteen tests passed across marker diagnostics and the existing failure scorecard;
  Ruff and `git diff --check` passed.

## Source images for inspection

- [E22_23](../../artifacts/generated/board_recognition/replay_v1/spatial_localization_v1e/images/validation_r_replay_194209320_s000000_markers_15.png)
- [N03](../../artifacts/generated/board_recognition/replay_v1/spatial_localization_v1e/images/validation_r_replay_189649315_s000000_markers_01.png)
- [N08](../../artifacts/generated/board_recognition/replay_v1/spatial_localization_v1e/images/validation_r_replay_189649315_s000000_markers_04.png)
- [N43](../../artifacts/generated/board_recognition/replay_v1/spatial_localization_v1e/images/validation_r_replay_189649315_s000000_markers_12.png)
- [P05](../../artifacts/generated/board_recognition/replay_v1/spatial_localization_v1e/images/validation_r_replay_193008240_s000000_markers_39.png)
