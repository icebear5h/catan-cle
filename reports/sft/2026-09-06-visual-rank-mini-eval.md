# Visual-delta rank sweep: terrain mini-eval

Status: **complete**. All six variants were evaluated and independently re-scored locally. This is a bounded compression diagnostic, not training or an O-LoRA retention experiment.

## Results

| Variant | Matrix-change energy retained | Tile number | Tile resource | Port type | Readout items | Exact ordered readouts |
|---|---:|---:|---:|---:|---:|---:|
| Intact v2 | 100% | 95/95 | 95/95 | 45/45 | 140/140 | 5/5 |
| Rank 256 | 93.2% | 95/95 | 95/95 | 45/45 | 140/140 | 5/5 |
| Rank 64 | 76.9% | 95/95 | 95/95 | 45/45 | 140/140 | 5/5 |
| Rank 16 | 51.3% | 95/95 | 95/95 | 45/45 | 140/140 | 5/5 |
| Rank 8 | 38.4% | 95/95 | 95/95 | 45/45 | 140/140 | 5/5 |
| Zero matrix delta | 0% | 94/95 | 94/95 | 42/45 | 138/140 | 3/5 |

There is no observed accuracy separation among ranks 8/16/64/256 and intact v2 on this mini-set. Rank 8 drops 61.6% of squared matrix-change magnitude without introducing any error on these examples. This demonstrates why weight-change energy is not retained accuracy; it does not establish that the dropped directions are unnecessary on the full holdout or safe to overwrite during new training.

The zero-matrix-delta control has seven wrong rows: five short answers plus two full readouts. Its short errors are one `brick → wood` resource substitution, one `2 → 12` number substitution, and three `ore port → paw port` answers. Its two failed readouts each substitute `paw port` for one ore port; there are no duplicate addresses. Thus the learned visual matrix update has a measurable effect here, and the rank-8 reconstruction recovers the observed errors.

**Bounded conclusion:** rank 8 is the smallest tested positive rank that reproduces v2's accuracy on this sample. We did not test ranks 1/2/4, the full 64-image holdout, new piece learning, or protection against forgetting. Compression rank, protected-subspace rank, and a new adapter's rank remain distinct decisions.

## Fixed comparison

- Source checkpoint: terrain v2 `checkpoint-384`, exactly the source used in the [visual-delta extraction](2026-09-05-visual-delta-svd.md).
- Base: cached original `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, hash-verified against extraction provenance.
- Variants: intact v2, ranks 256/64/16/8, and zero matrix delta.
- Every compressed matrix is reconstructed independently as original-base + truncated-SVD delta; the delta is never added to v2 or to the previous rank's weights.
- All 112 visual matrices are included: tower and merger linear weights, positional embedding, and flattened patch convolution. Biases/norm vectors remain at v2. Language LoRA and atlas rows remain at v2 for every variant.
- **Rank zero is a hybrid control, not original Qwen**: original visual matrices, but v2 biases/norms, language adapter and atlas rows.
- FP32 visual master weights and BF16 CUDA autocast for every variant. Intact v2 is restored from its FP32 file after the common loader runs, avoiding accidental BF16 storage rounding. The in-memory model is restored again after a completed sweep; checkpoint files are never edited.

## Sample and scoring

The fixed stage-2 validation file has 3,072 rows across five unseen layouts and 64 images. This mini-eval selects the median lexicographic state ID within each layout, independently of any model output, and retains every terrain question for those five states.

| Head | Rows per variant |
|---|---:|
| Tile resource | 95 |
| Tile number | 95 |
| Port type | 45 |
| Complete terrain/port readout | 5 |
| Total | 240 |

The five readouts contain 140 keyed items. The short rows include all six resources, all ten production numbers plus desert `none`, and all six port types. Selection content SHA256 (row IDs, layouts/states, tasks, prompts and targets): `9d9c8cf4a7dbb63d68fcb00f14b376a3411b4819b5b611e3c711baca1fa003bc`. Local and remote selections match. All five remote image hashes were independently checked against the fixed local images and match byte-for-byte.

Selected states:

- `r_replay_189649315_s000016`
- `r_replay_193008240_s000108`
- `r_replay_193905959_s000173`
- `r_replay_194198328_s000029`
- `r_replay_194209320_s000154`

Greedy free generation, not candidate restriction. Short batch size 48 with 16 new tokens; long batch size 5 with 512 new tokens. Readouts are separately checked for correct values, coverage, syntax, duplicates and canonical order. The evaluator's legacy dictionary-based readout score is not used as the strict sweep result. Missing/extra prediction IDs fail the comparison.

This is a quick screen on **five independent layouts**, not a full regression gate, not piece-recognition evaluation, and not a claim of 99% population accuracy. No blank-image control is added; the paired visual-matrix ablation is the intervention being tested.

## Runtime and artifacts

App: `ap-0Og4yu0mBj2BQWRWp75bjB`.

One model load on one H200; 8 CPU cores and 64 GiB memory limits; 900-second function timeout, 300-second startup timeout, no configured retries. The 900-second resource window is approximately **$1.36 at current list rates**, excluding build/startup/storage/teardown and infrastructure restarts; this is not measured billing or a hard dollar guarantee. [Modal pricing](https://modal.com/pricing)

Completed function time: **709.48 seconds (11.82 minutes)**. Applying those requested resource rates to the observed function time gives **$1.07**, before excluded overhead; actual provider billing was not queried. Modal confirmed the app stopped with zero tasks. Exact FP32 restoration of the original v2 visual weights was checked before and after the sweep.

Output on Modal Volume `catan-sft-runs`:

```text
qwen-series-eval/visual-rank-mini-20260906/
  selection.json
  selected.jsonl
  sweep.json
  model_load.log
  <variant>/records.jsonl
  <variant>/strict_records.jsonl
  <variant>/paired_summary.json
  <variant>/summary.json
  <variant>/eval.log
```

Each completed variant is committed to the volume. Existing output directories are not overwritten. An unexpectedly weak intact-v2 short-head control stops the sweep for audit.

Local artifacts: [complete sweep summary](../../artifacts/diagnostics/sft/visual_rank_mini_20260906/sweep.json), [selection and image hashes](../../artifacts/diagnostics/sft/visual_rank_mini_20260906/selection.json), and one `<variant>.strict_records.jsonl` file per variant in the same directory. All 1,440 prediction rows were independently re-scored locally, with exact prediction-ID coverage and matching remote summaries. Files were downloaded individually after a directory-download quirk produced an invalid local copy; the invalid copy is explicitly named `intact_v2.directory-download-invalid` and is not used for analysis. Source checkpoints and remote eval artifacts were unaffected.

Implementation: [launcher](../../sft/modal_visual_rank_eval.py), [selection/reconstruction/scoring helpers](../../sft/visual_rank_eval.py), [tests](../../tests/test_visual_rank_eval.py).

Validation: 48 focused tests passed across the sweep helpers, extractor, candidate evaluator, and trainer; Python compilation and `git diff --check` passed.

The run-review guidance shaped the source-lineage and precision checks, fixed per-head comparison, and strict full-readout scoring.
