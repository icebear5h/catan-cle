# Qwen3-VL-32B CatanBoardBench Visual Evaluation

Date: 2026-08-10

## Configuration

- Model: `qwen/qwen3-vl-32b-instruct`
- Architecture: disclosed 32B dense vision-language model
- Route observed in the OpenRouter catalog at verification time: Alibaba FP8 (the only listed endpoint)
- Suite: current `visual` suite
- Boards: 10 (`sample_000` through `sample_009`)
- Questions: 110 (11 categories × 10 boards)
- Atlas prompt: enabled
- Temperature: 0
- Requested maximum completion: 256 tokens
- Concurrency: 2
- Benchmark prompts and labels were not modified.

## Result

| Metric | Result |
| --- | ---: |
| Exact accuracy | 17/110 (15.45%) |
| Component accuracy | 14.68% |
| Request errors | 0/110 |
| Non-empty responses | 110/110 |

### Category breakdown

| Category | Exact | Component |
| --- | ---: | ---: |
| `color_building_counts` | 0/10 | 25.0% |
| `color_road_count` | 1/10 | 10.0% |
| `color_road_locations` | 0/10 | 2.1% |
| `edge_road_owner` | 3/10 | 30.0% |
| `node_occupancy` | 0/10 | 0.0% |
| `port_occupancy` | 5/10 | 25.0% |
| `port_trade_type` | 5/10 | 50.0% |
| `robber_resource_number` | 0/10 | 3.3% |
| `robber_tile` | 1/10 | 10.0% |
| `tile_has_robber` | 0/10 | 0.0% |
| `tile_resource_number` | 2/10 | 25.0% |

The apparent port scores do not demonstrate reliable perception: the model answered `NONE` for all 10 port-occupancy questions and `GENERIC 3:1` for all 10 port-type questions. Those defaults happened to match half of the exact targets. It also answered `EMPTY` for every node-occupancy question and `NO` for every tile-has-robber question.

## Usage and latency

| Metric | Result |
| --- | ---: |
| Prompt tokens | 63,957 |
| Completion tokens | 49,528 |
| Total tokens | 113,485 |
| OpenRouter-reported cost | $0.027255 |
| Median request latency | 1.18 s |
| p95 request latency | 2.66 s |
| Maximum request latency | 138.22 s |

Five `color_road_locations` requests entered a deterministic-looking runaway enumeration, fabricating edge tokens through `<E999_1013>`. Each emitted 9,804 completion tokens despite the requested 256-token limit. These five failures consumed 49,020 of 49,528 completion tokens and $0.02068 of the $0.02726 total cost. The other 105 requests cost $0.00657 and had 1.16 s median latency. The runaway outputs are retained as instruction-following failures rather than retried or excluded from accuracy.

## Comparison with the earlier smoke

The earlier 40-question smoke reported 12.5% exact and 16.3% component accuracy for the same model. The current 110-question run reports 15.45% exact and 14.68% component accuracy. The suites have different category/sample weighting, so the figures are not paired estimates, but both indicate the same conclusion: plain prompting does not make this model a dependable Catan board parser or atlas-binding system.

## Verification

- 110 unique question IDs
- Exactly 10 responses per category
- Zero API errors and zero empty responses
- Local artifact integrity assertions passed
- `tests/test_catan_board_bench.py` and `tests/test_catan_tokens.py`: 7 passed

## Artifacts

- Plan: `artifacts/runs/catan_board_bench/catan_board_bench_100/openrouter/qwen3_vl_32b_dense_20260810/plan.json`
- Raw responses: `artifacts/runs/catan_board_bench/catan_board_bench_100/openrouter/qwen3_vl_32b_dense_20260810/responses.jsonl`
- Machine summary: `artifacts/runs/catan_board_bench/catan_board_bench_100/openrouter/qwen3_vl_32b_dense_20260810/summary.json`
