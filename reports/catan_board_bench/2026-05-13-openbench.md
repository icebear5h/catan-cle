# 2026-05-13 OpenBench Runs

Benchmark: `catan_board_bench`

Suite:
- `limit_samples=10`
- `questions_per_sample=9`
- `atlas_prompt=true`
- `image_detail=auto`
- `temperature=0`
- `max_connections=2`
- `log_images=false`

## Runs

| Model | Questions | OpenBench exact | OpenBench component | Semantic rescore exact | Semantic rescore component | Log |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `openrouter/qwen/qwen3-vl-8b-instruct` | 90 | 5.6% | 7.4% | 6.7% | 11.3% | `logs/2026-05-13T16-01-21-07-00_catan_board_bench_o9dThPdM9AzS4CAqjvrouK.json` |
| `openrouter/google/gemini-3.1-pro-preview` | 90 | 55.6% | 57.2% | 63.3% | 69.3% | `logs/2026-05-13T16-02-42-07-00_catan_board_bench_SXpaWkExVuZnaB4eZNyiGz.json` |

Semantic rescore uses the updated local scorer that accepts semantically correct
bare count answers for count questions and normalizes merged tokens such as
`<BLUE_SETTLEMENT>`.

## Semantic Category Breakdown

### `openrouter/qwen/qwen3-vl-8b-instruct`

| Category | Exact | Component |
| --- | ---: | ---: |
| `color_building_counts` | 0/10 | 3/20 |
| `color_road_count` | 1/10 | 1/10 |
| `edge_road_owner` | 0/10 | 0/10 |
| `node_occupancy` | 0/10 | 0/20 |
| `port_trade_type` | 5/10 | 10/20 |
| `robber_resource_number` | 0/10 | 3/30 |
| `robber_tile` | 0/10 | 0/10 |
| `tile_has_robber` | 0/10 | 0/10 |
| `tile_resource_number` | 0/10 | 0/20 |

### `openrouter/google/gemini-3.1-pro-preview`

| Category | Exact | Component |
| --- | ---: | ---: |
| `color_building_counts` | 0/10 | 6/20 |
| `color_road_count` | 4/10 | 4/10 |
| `edge_road_owner` | 8/10 | 8/10 |
| `node_occupancy` | 7/10 | 16/20 |
| `port_trade_type` | 5/10 | 11/20 |
| `robber_resource_number` | 8/10 | 24/30 |
| `robber_tile` | 8/10 | 8/10 |
| `tile_has_robber` | 7/10 | 7/10 |
| `tile_resource_number` | 10/10 | 20/20 |

## Notes

- The old 40-question default was only a smoke test and did not cover all
  default categories. The benchmark default is now 90 questions so the default
  category list is fully represented.
- Gemini needed `--max-tokens 512`; with `--max-tokens 96`, OpenRouter returned
  many truncated answers because reasoning tokens consumed most of the output
  budget.
- Qwen 8B is not reliable on this board-parser eval yet. Its only meaningful
  signal here is `port_trade_type`, and even that confuses resource ports with
  generic ports.
