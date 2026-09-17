# Corrected spatial QA evaluation

## Result

The latest complete-board checkpoint scored **57/120 (47.5%)** on corrected
spatial validation questions with explicit short-answer instructions. Its
64/64 full-board extraction result does not transfer to reliable relation QA.

| Answer type | Correct | Accuracy |
| --- | ---: | ---: |
| Yes/no relations | 55/100 | 55.0% |
| Which node? | 0/10 | 0.0% |
| Which tile? | 2/10 | 20.0% |
| Overall | 57/120 | 47.5% |

All 100 binary responses were valid bare yes/no answers. Only 6/20 token-choice
responses named one of the two offered choices: two correct and four wrong.
The remaining outputs were seven other same-entity tokens, four wrong-entity
tokens, and three prose/other answers. No extracted-token or prefix credit is
included in the accuracy.

| Relation, nodes and tiles combined | Correct | Accuracy |
| --- | ---: | ---: |
| Above | 7/18 | 38.9% |
| Below | 6/13 | 46.2% |
| Left of | 5/16 | 31.3% |
| Right of | 7/13 | 53.8% |
| Adjacent | 22/40 | 55.0% |
| Connected by an edge | 10/20 | 50.0% |

Node directions alone were 13/30 (43.3%): 13/20 yes/no and 0/10 token choices.
All node questions were 35/70; all tile questions were 22/50.

## Matched format control

The first run used the original questions, with only the choice-order bug
corrected. It scored **1/120 (0.83%)** under whole-answer exact matching.
None of its 100 binary responses were bare yes/no: 98 began with yes/no and
continued with prose, and two were other prose. Several explanations changed
the queried identities, so this was not purely a cosmetic formatting problem.

To distinguish output formatting from relation errors, a second run appended
one of these label-neutral instructions to the same questions:

- Binary: `Answer with exactly one word: yes or no.`
- Token choice: `Answer with only one of the two tokens shown in the question.`

The image, question order, correct answer, choice order, checkpoint, precision,
and generation settings stayed fixed. IDs and prompt-variant metadata were
versioned; the original input/results were not overwritten. These are two
different prompt conditions, not an improvement from additional training.

## Conditions

- Checkpoint: `/runs/catan-vision-sft/full-board-new-layouts-20260907/checkpoints/checkpoint-128`.
- Base: `Qwen/Qwen3.8-27B`, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- Input: 120 spatial validation rows over five empty held-out board layouts;
  12 task types with ten rows each. No robber or board-readout questions.
- Choice-position balance: 10 first / 10 second, including 5/5 per entity.
- One H200 per run, batch 48, maximum 16 new tokens, greedy generation,
  thinking disabled, original images only, candidate scoring disabled.
- Language BF16, rank-8 adapter, all 154 atlas-token rows. All 333 saved visual
  tensors restored into FP32 before evaluation, with BF16 generation autocast.
- Both runs used visual checkpoint SHA256
  `3535adfa86dbc4675f612f98995222f2f15619d151b5aff2f7537eab147b7183`,
  matching the prior successful full-board evaluation.

Original input SHA256:
`930c9ce829566f40f9c7de0dc0bcf2b1f1c11dfdc7f3165f094767baef0bfa1e`.
Answer-only input SHA256:
`53c4330d955d369833ca6024307b7b725a4cc46117fe6c1efc6170e7e7486786`.

## Evidence

Original run directory:
`artifacts/runs/sft/full-board-new-layouts-ck128-spatial-choice-order-v1-20260908-r01/`.
Answer-only control directory:
`artifacts/runs/sft/full-board-new-layouts-ck128-spatial-answer-only-v1-20260908-r01/`.

Each contains `launch.json` with the exact launch command, downloaded
`summary.json` / `records.jsonl`, and independently computed `verification.json`.
Both audits pass all checks: input hashes, complete response coverage, unchanged
answers/images, checkpoint identity, visual precision, and every saved score
and aggregate. The control additionally verifies the exact permitted prompt
transformation against the original run.

The original directory contains `verify.py` and the downloaded
`prior_full_board_summary.json`. Reproduce the control audit from the repo root:

```bash
env PYTHONPATH=. .venv/bin/python -B \
  artifacts/runs/sft/full-board-new-layouts-ck128-spatial-choice-order-v1-20260908-r01/verify.py \
  --run-dir artifacts/runs/sft/full-board-new-layouts-ck128-spatial-answer-only-v1-20260908-r01 \
  --prior-fullboard-summary artifacts/runs/sft/full-board-new-layouts-ck128-spatial-choice-order-v1-20260908-r01/prior_full_board_summary.json
```

Completed at 00:46:11 and 01:06:20 UTC on September 9 (September 8 local).
Both Modal apps were explicitly stopped after result retrieval; zero tasks
remain in either. No training or additional model sweep was performed.
Offline regression verification: 82 tests passed; scoped Ruff passed.

## Interpretation

Relation QA needs direct supervision before being treated as a learned skill.
A short spatial-QA SFT continuation with board-readout rehearsal is justified;
this result does not justify restarting the vision tower from base.

These are fixed-atlas facts on five empty-board layouts, not 120 independent
boards or a test of image-dependent piece relations. Balanced binary accuracy
is close to the 50% guessing baseline, while token selection remains unreliable
even with explicit format guidance. The eval conflates atlas retrieval,
relation reasoning, and instruction following; it does not isolate broken
vision. Earlier spatial checkpoint scores used different model/prompt
conditions and should not be treated as a matched regression comparison.
