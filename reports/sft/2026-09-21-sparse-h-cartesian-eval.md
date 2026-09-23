# Stock Qwen3.8: sparse h-shorthand Cartesian evaluation

Run `miles-cartesian-h-stock-20260921-r01` completed in **tetracorp** on
2026-09-21 at **19:14:10 UTC**. Strict exact accuracy is **153/200 (76.5%)**,
up from the dense Cartesian run's **143/200 (71.5%)**.

The model, tokenizer, model-file hashes, inference runtime versions, image digest,
and decoding settings match the baseline. There was no fine-tuning, adapter,
quantization, or added vocabulary. Every raw output was downloaded and rescored.

## Paired results

| Operation | Dense radicals | Sparse h | Change |
| --- | ---: | ---: | ---: |
| Direction yes/no | 25/32 | 25/32 | 0 |
| Direction choice | 30/32 | 30/32 | 0 |
| Node/tile neighbors | 7/32 | 7/32 | 0 |
| Entity incidence | 3/16 | 10/16 | +7 |
| Piece owner | 31/32 | 30/32 | -1 |
| Owned nodes/buildings | 21/24 | 23/24 | +2 |
| Owned roads | 16/16 | 16/16 | 0 |
| Owned roads incident to a node | 10/16 | 12/16 | +2 |
| **Total** | **143/200** | **153/200** | **+10** |

Matched outcomes: **19 improved, 9 regressed, 134 correct in both, 38 wrong in
both**. Flat operation totals therefore do not imply identical answers.

### Pure readouts and casing

Direct supplied-state readouts (piece owner, owned nodes, owned roads) score
**69/72 strictly**, versus 68/72 before. The three strict misses are:

1. One BLACK-building enumeration still omits `N(-4h,2)`, returning only `N(0,-1)`.
2. One owner answer is `green` instead of `GREEN`.
3. One owner answer is `blue` instead of `BLUE`.

The latter two identify the correct owner but fail the inherited case-sensitive
color contract. **As a separately labeled case-insensitive-color diagnostic**,
the result is **155/200 (77.5%)**, with **71/72 (98.6%) pure readouts** and all
32 piece owners correct. The official strict scores and stored rewards were not
changed retroactively. No further inference was needed for this diagnostic.

Seven strict answers fail admission: five reference unknown positions and two
are the casing-only owner answers. No output was truncated.

## Input change and tokens

- Define `h=sqrt(3)/4`; use exact integer h multiples for x and exact quarter-step
  decimals for y, on the same equal-scale physical Cartesian board.
- Dynamic boards omit only empty nodes/edges and explicitly state the empty
  default. All 19 tiles, nine ports, the robber, and all occupied pieces remain.
- Static questions retain all 154 entity positions, in h notation.
- The 200 source cases, case IDs, source order, splits, and canonical oracle remain
  fixed. This is the **combined sparsity + notation change**, not an h-only ablation.

| Token metric | Dense radicals | Sparse h |
| --- | ---: | ---: |
| Total input | 457,429 | 263,102 |
| Mean input | 2,287.145 | 1,315.51 |
| Mean dynamic-question input | 2,514.59 | 920.90 |
| Mean static-question input | 2,108.44 | 1,625.56 |
| Longest input | 2,544 | 1,632 |
| Total output | 2,986 | 2,264 |
| Mean output | 14.93 | 11.32 |

Overall input is **42.5% smaller**; dynamic-question input is **63.4% smaller**.
All 200 prompts retained the same 4,096 context and 512-token completion allowance.

### Subsequent integer-scaling token audit (no inference)

The user requested measuring scaling rather than merely rewriting decimals as
fractions. Using the same 200 recorded chat templates and actual tokenizer, with
coordinate conventions updated consistently:

| Variant | Mean full prompt | Reduction from current h format |
| --- | ---: | ---: |
| Current h + decimal y | 1,315.51 | — |
| Uniform physical scale x4: h=sqrt(3), integer y | 1,131.25 | 14.0% |
| Bare integer grid addresses; physical point=(sqrt(3)*x,y) | 1,033.565 | 21.4% |

The first variant retains x expressions such as `2h` and removes fractions and
decimals via a uniform change of physical scale. The second removes the h suffix
as well and uses explicitly scaled grid-address units. Both are exact; neither
has been evaluated for model accuracy. These totals include revised legends and
are retained in `integer_scaling_token_audit.json`; the actual run is unchanged.

## Interpretation

The combined change improves incidence substantially and nearly eliminates
content-level direct-readout errors, while preserving most of the token savings.
Neighbor enumeration remains weak at 7/32; direction totals remain unchanged.
Shorter inputs alone have not solved topology.

This is one greedy, no-thinking run on a fixed-board diagnostic panel. The 198
test and two validation cases inherit their source overlap and 61 dynamic-state
coverage; no gameplay or generalization claim follows from the headline score.

## Runtime and evidence

- Model revision: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- Miles revision: `12754e9507e64d5e537288da17793246e913c525`.
- Image: `radixark/miles@sha256:6628bff749ffd32e6a62b479a1128daee25a8c0e3c28eb86301a0d620f5dd598`.
- One H200, BF16, greedy, thinking disabled, max concurrency 16, CUDA graphs disabled.
- CPU preflight: **121.73 seconds**. H200 function: **523.71 seconds**, including
  initialization; logged 200-case generation/scoring loop: approximately **40 seconds**.
- Recorded function-window estimate: **$0.89**, excluding container startup/
  termination, image builds, storage, and subscriptions; not an actual bill.
- [App `ap-motz8vXjClLdX0T17KmCa7`](https://modal.com/apps/tetracorp/main/ap-motz8vXjClLdX0T17KmCa7)
  was stopped at **19:14:46 UTC**, with zero tasks verified.

Local artifacts: `artifacts/runs/sft/miles-cartesian-h-stock-20260921-r01/`.
`analysis.json` contains all paired IDs, per-operation counts, errors, token totals,
identity checks, and the casing diagnostic. Raw records and summaries are under
`remote/evaluation/results/eval-0/cartesian_h/`; complete logs, preflight hashes,
GPU receipt, and original Miles sample dump are also downloaded.

Prepared input: `artifacts/generated/sft/miles_cartesian_h_eval_v1/`.
Human preview: `artifacts/generated/sft/cartesian_h_eval_v1/preview.md`.
The raw record file SHA-256 is
`9acf259049172ed072f541868e9779aca99eb03ec06e08ff3450ce16b3300c1b`.
