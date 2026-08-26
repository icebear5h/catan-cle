# Catan text-format optimization

## Result

The selected representation is `indexed_tile_rows`: retain the complete canonical
`tile_rows` graph, then append deterministic query indexes for the joins that a
small language model otherwise performs unreliably.

These indexes are question-independent but task-aware: they encode reusable Catan
board operations selected on the development suite, not generic unaided spatial
reasoning and not per-question answer keys.

On the leakage-safe development suite, the frozen v3 representation scored
**60/60 exact (100%)**, versus **40/60 (66.7%)** for the byte-identical
`tile_rows` baseline. All 60 responses were valid JSON and all requests were
served by the pinned AkashML provider for `qwen/qwen3.8-27b` with reasoning
disabled and temperature 0.

## Split and leakage boundary

Development boards came from the local intersection of
`4p_games_training_candidates.json` and available raw replays. Every game ID in
`catan_board_bench_100/leakage/benchmark_game_ids.json` was excluded before
building contracts. Twelve distinct games supplied the 12 boards and 60 strict
questions (six questions in each of ten categories).

The older `ascii_variation_probe` was not used to select or revise the format.
It is used once after freezing v3 as a transfer/regression check. Because that
suite descends from existing benchmark contracts, it is not presented as a new
uncontaminated test set.

## Candidate comparison

The first complete development comparison used identical facts, questions,
prompt, model, provider, and decoding settings for all formats:

| Format | Exact | Accuracy | Prompt tokens | Cost |
|---|---:|---:|---:|---:|
| `tile_rows` | 40/60 | 66.7% | 409,550 | $0.1558 |
| `indexed_records` | 52/60 | 86.7% | 572,536 | $0.2151 |
| `indexed_json` | 54/60 | 90.0% | 559,116 | $0.2080 |
| `indexed_tile_rows` v1 | 56/60 | 93.3% | 592,156 | $0.2097 |

The original `tile_rows` representation scored 6/12 on reciprocal direction
questions. Adding explicit named tile-neighbor records while preserving the
tile-row body raised `indexed_tile_rows` to 12/12. The four remaining v1 misses
were one empty port endpoint and three roll-production joins.

The final v3 adds two general representation rules and non-aggregated production
terms. A full winner-only rerun scored 60/60:

| Frozen format | Exact | Accuracy | Prompt tokens | Median latency | Cost |
|---|---:|---:|---:|---:|---:|
| `indexed_tile_rows` v3 | 60/60 | 100% | 677,592 | 1,913.5 ms | $0.2466 |

The promoted dataset averages 20,968 characters per `indexed_tile_rows` board,
versus 12,125 for baseline `tile_rows` (1.73x). Prompt-token use in the paired
development runs is about 1.65x baseline. This is an accuracy-first format, not
the token-minimal Pareto point.

## Frozen transfer check

The post-freeze run on the older locked suite was truncated when the OpenRouter
account exhausted its $20 credit budget. The evaluator accepted 93/120 planned
responses before the remaining requests were rejected for insufficient in-flight
budget. On the 46 questions completed by both formats, the paired result was:

| Format | Paired exact | Paired accuracy |
|---|---:|---:|
| `tile_rows` | 34/46 | 73.9% |
| frozen `indexed_tile_rows` v3 | 46/46 | 100% |

The winner also has 46/46 exact across all of its accepted transfer responses;
baseline has 34/47. Twenty-seven planned responses remain unserved, so this is
reported as a credit-truncated transfer result rather than a completed 60-question
score. No tuning was performed from transfer outcomes.

## Representation

The base graph remains the complete, losslessly parseable tile-row serialization.
The appended `QI|...` records materialize deterministic views of that graph:

- `TILE_NEIGHBORS`: explicit LEFT, RIGHT, UP-LEFT, UP-RIGHT, DOWN-LEFT, and
  DOWN-RIGHT tile IDs.
- `TILE_CORNERS`: named corner-to-node joins with public color/building state.
- `PORT_NODES`: both endpoints and their public state, with an explicit rule that
  empty endpoints are not occupants.
- `PLAYER`: settlement, city, and road entity-ID lists without counts.
- `ROLL_TILES`: roll-number-to-tile lookup without payouts.
- `ROLL_SOURCE`: one typed term per occupied corner, including resource, robber
  state, building type, and its local 1/2 production weight. Terms remain
  unaggregated; no final color/resource payout is stored.

Every `QI|...` data record is regenerated from canonical facts during parsing and
rejected if it is missing, reordered, or inconsistent. Every candidate
round-trips to the same fact digest. Thus the gain is best interpreted as a
better data interface and materialized retrieval plan, not a gain in unaided
spatial reasoning.

## Reproduction

Build the paired probe:

```bash
uv run python -m scripts.build_catan_text_format_optimization \
  --source-dir data_pipeline/catan_board_bench/datasets/ascii_variation_probe \
  --output-dir data_pipeline/catan_board_bench/datasets/text_format_optimization_probe
```

Run the evaluator:

```bash
uv run python -m scripts.eval_catan_text_format_optimization \
  --dataset-dir data_pipeline/catan_board_bench/datasets/text_format_optimization_probe \
  --formats tile_rows,indexed_tile_rows \
  --output-dir artifacts/runs/catan_board_bench/text_format_optimization/openrouter/qwen3_8_27b_transfer_20260824 \
  --concurrency 1
```

Integrity checks:

```bash
uv run pytest -q \
  tests/test_catan_text_format_optimization.py \
  tests/test_catan_full_graph_formats.py
```

## Artifacts

- Initial four-format development comparison:
  `artifacts/runs/catan_board_bench/text_format_optimization/openrouter/qwen3_8_27b_dev_full_20260824`
- Frozen v3 development run:
  `artifacts/runs/catan_board_bench/text_format_optimization/openrouter/qwen3_8_27b_dev_v3_full_20260824`
- Frozen transfer run:
  `artifacts/runs/catan_board_bench/text_format_optimization/openrouter/qwen3_8_27b_transfer_20260824`
