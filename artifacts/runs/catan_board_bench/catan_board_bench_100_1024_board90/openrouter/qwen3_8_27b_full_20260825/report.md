# Qwen3.8 1024px / 90% board OpenRouter check

## Result

Increasing the full-board render from the established 512px framing to 1024px
with approximately 90% canvas coverage did not materially improve Qwen3.8-27B.

| Metric | 512 baseline | 1024 / 90% | Change |
|---|---:|---:|---:|
| Exact accuracy | 24/110 (21.82%) | 26/110 (23.64%) | +2 |
| Component accuracy | 44/218 (20.18%) | 42/218 (19.27%) | -2 |
| Prompt tokens | 65,357 | 149,837 | +129.3% |
| Completion tokens | 578 | 549 | -5.0% |
| Native reasoning tokens | 0 | 0 | 0 |
| Reported cost | $0.027169 | $0.064299 | +136.7% |

Paired transitions were 19 correct at both resolutions, 79 wrong at both, 7
wrong-to-correct, and 5 correct-to-wrong. The net two-answer gain is not a
stable qualitative improvement.

## Category results

| Category | 512 | 1024 / 90% | Change |
|---|---:|---:|---:|
| Color building counts | 1/10 | 0/10 | -1 |
| Color road counts | 3/10 | 5/10 | +2 |
| Color road locations | 0/10 | 0/10 | 0 |
| Edge road owner | 5/10 | 5/10 | 0 |
| Node occupancy | 0/10 | 0/10 | 0 |
| Port occupancy | 3/10 | 5/10 | +2 |
| Port trade type | 5/10 | 5/10 | 0 |
| Robber resource and number | 0/10 | 1/10 | +1 |
| Robber tile | 3/10 | 1/10 | -2 |
| Tile has robber | 1/10 | 1/10 | 0 |
| Tile resource and number | 3/10 | 3/10 | 0 |

The specifically concerning dense spatial tasks did not move: node occupancy
remained 0/10 and complete road localization remained 0/10. On the displayed
`sample_007` board, the score moved from 0/11 to 1/11 only because the model
returned the correct global road count; every local node, edge, tile, robber,
port, and road-location query was still wrong.

## Protocol and limitations

- Same 10 boards, 110 question IDs, atlas prompts, temperature 0, and scorer.
- Qwen model ID: `qwen/qwen3.8-27b`.
- Native reasoning was explicitly disabled; all 110 responses report zero
  reasoning tokens.
- All 110 rows completed without final errors. Temporary OpenRouter in-flight
  budget errors were resumed until every question had a valid response.
- Execution occurred in three resumable phases: 54 successes at concurrency 2,
  30 additional successes at concurrency 1, then the final 26 at concurrency 1
  with a 2.1-second request interval. The final `plan.json` reflects the last
  resume phase; `responses.jsonl` contains the admitted union of all phases.
- The render is pixel-identical to the approved displayed 1024/90% sample after
  decoding; the PNG files differ only in compression.
- This is an end-to-end VLM check, not an isolated vision-tower probe.
- OpenRouter routing was unpinned. The 1024 requests were served across AkashML,
  Alibaba, Chutes, CoreWeave, Io Net, Parasail, Reka, and Venice. The prior 512
  artifact did not persist provider names, so the comparison cannot identify a
  pure resolution effect independently of provider/quantization variation.

Raw responses are in `responses.jsonl`; machine summary and request settings are
in `summary.json` and `plan.json` in this directory.
