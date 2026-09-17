# Published spatial SFT versus latest spatial checkpoint

## Result

The original September 2 HF model scored **16/200 (8.0%)**, compared with
**24/200 (12.0%)** for the September 12 checkpoint-128 on the same symbolic review.

| Family | September 2 HF | September 12 ck128 |
| --- | ---: | ---: |
| Relations / joins | 2/40 (5.0%) | 6/40 (15.0%) |
| Sets / coverage | 5/40 (12.5%) | 6/40 (15.0%) |
| Aggregation / comparison | 1/40 (2.5%) | 3/40 (7.5%) |
| Connectivity / structure | 6/40 (15.0%) | 6/40 (15.0%) |
| Constraints / consequences | 2/40 (5.0%) | 3/40 (7.5%) |
| Total | 16/200 (8.0%) | 24/200 (12.0%) |

Paired outcomes: nine correct for both, seven correct only for the older model,
15 correct only for the newer model, and 169 wrong for both.

## Failure patterns

| Measure | September 2 HF | September 12 ck128 |
| --- | ---: | ---: |
| Format-valid | 146 | 152 |
| Wrong but format-valid | 130 | 128 |
| Malformed | 54 | 48 |
| Repeated-atlas criterion | 0 | 24 |
| Resource pip totals | 0/10 | 0/10 |
| Resource pip argmax, all ties | 0/10 | 0/10 |
| Local node tiles | 0/10 | 2/10 |
| Per-roll production | 1/10 | 2/10 |
| Coverage union | 4/10 | 0/10 |
| Coverage intersection | 1/10 | 5/10 |
| Shortest distance | 4/10 | 5/10 |

The repetition criterion is the same descriptive measure as the first report:
more than 20 exact atlas-token occurrences with at most three distinct spellings,
after removing trailing transport tokens. The older model avoids these loops,
but still has more malformed answers and lower exact accuracy overall.

For the previously inspected `resource_pip_totals/000` board:

```text
Gold:       brick=9,  ore=9,  sheep=11, wheat=12, wood=17
Sept 2 HF:  brick=10, ore=12, sheep=14, wheat=12, wood=14
Sept 12:    brick=10, ore=12, sheep=12, wheat=12, wood=12
```

For `resource_pip_argmax/000`, both models answered `ore sheep` instead of
`sheep wheat`. Full exact responses and additional paired examples are in
`comparison.json`; trailing generated padding is preserved in raw records.

## Conditions and identity

- Older adapter: `icebear5h/catan-qwen3.8-27b-spatial-sft` at
  `8030a960ee73e758994c3938be347f2fc4abb38e`, downloaded from HF.
- Newer adapter: `/runs/catan-vision-sft/spatial-continuation-20260912-r01/checkpoints/checkpoint-128`.
- Base: `Qwen/Qwen3.8-27B` at `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- Exact same 200 review messages, source SHA-256
  `40469f58d060dd9a1d24354df42d959971172ba6e0de5e70b153cdda747e5b12`.
- Text-only, greedy, thinking disabled, no candidate scoring, batch16,
  512 completion tokens for every row, context4096, no truncation.
- Same dependencies, evaluator/launcher code hashes, base files, generation
  configuration, precision policy, and 154 atlas-token identities/IDs.
- Each checkpoint's saved tokenizer was used. Raw tokenizer/config asset hashes
  differ; chat-template hashes and all 200 prompt/gold token lengths match.
  These are checkpoint-bundle comparisons, not a claim of byte-identical tokenizer assets.
- Older GPU stage: **335.863 seconds (5m 35.863s)** including loading/verification.
  CPU preparation: 152.1s. One H200 call, capped at 900 execution seconds.
- Completed Modal app: `ap-YeAQBoYvs7cAwegBaP5dNK`; no containers remain running.

## Verification and interpretation

The offline comparison independently re-scored both sets of 200 actual responses,
checked unique IDs and original metadata/gold, recomputed grouped summaries, and
verified saved artifact hashes against the run receipts. Integer, unique-set,
and strict typed-JSON scoring follows the same contract for both models.

Returning to the published September 2 checkpoint does not improve direct
symbolic performance on this batch. Neither checkpoint demonstrates fluent use
of the supplied symbolic board yet. This does not establish which checkpoint
will adapt better under new symbolic SFT, nor isolate the cause of the errors.
The review states are training-source-derived and share some maps/trajectories;
these results are diagnostics, not an untouched generalization benchmark.

## Artifacts

Run root: `artifacts/runs/sft/hf-sept02-fluency200-20260915-r01/`.

- `records.jsonl`, `summary.json`: all raw predictions and scores.
- `launch.json`, `prepare.json`, `run.json`, `orchestration.json`: execution evidence.
- `comparison.json`: paired scores, format diagnostics, conditions and exact examples.
- `compare.py`: reproducible offline comparison.

```bash
.venv/bin/python -B artifacts/runs/sft/hf-sept02-fluency200-20260915-r01/compare.py
```
