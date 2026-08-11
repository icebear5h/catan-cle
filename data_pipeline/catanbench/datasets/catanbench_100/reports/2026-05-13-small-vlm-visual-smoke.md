# 2026-05-13 VLM Visual Smoke

Benchmark: `catanbench`

Suite:
- `suite=visual`
- `max_requests=40`
- `include_system=false`
- `temperature=0`
- `max_tokens=256` for small models; `512` for Gemini because it spends reasoning tokens.
- `max_connections=2`
- `log_images=false`

Important: an earlier smoke run leaked answers for `color_road_locations` through
local prompt context. Those runs are excluded here. This file only reports the
clean rerun after removing that leakage.

## Summary

| Model | Exact | Component | Total tokens | Log |
| --- | ---: | ---: | ---: | --- |
| `openrouter/google/gemini-3.1-pro-preview` | 67.5% | 72.5% | 74,724 | `logs/2026-05-13T16-59-24-07-00_catanbench-visual_5CdsxKFpaF7RK7dmXqrMmg.json` |
| `openrouter/x-ai/grok-4.3` | 30.0% | 39.4% | 75,575 | `logs/2026-05-13T17-48-00-07-00_catanbench-visual_GTkFqMxuEKLz2cjSuay8Uo.json` |
| `openrouter/anthropic/claude-sonnet-4.6` | 25.0% | 29.7% | 29,127 | `logs/2026-05-13T17-43-47-07-00_catanbench-visual_m2yWLSUSFx4Y4dmdxnmP59.json` |
| `openrouter/anthropic/claude-opus-4.7` | 17.5% | 25.8% | 31,571 | `logs/2026-05-13T17-42-49-07-00_catanbench-visual_6qw6xQVLL9weZk72PDxB74.json` |
| `openrouter/google/gemma-4-31b-it` | 17.5% | 24.8% | 24,400 | `logs/2026-05-13T17-12-30-07-00_catanbench-visual_6Z6HEcs2xpXbFWedvV6xiT.json` |
| `openrouter/qwen/qwen3-vl-235b-a22b-instruct` | 17.5% | 22.5% | 23,644 | `logs/2026-05-13T17-41-41-07-00_catanbench-visual_cM6wEnPB7rfntyHZBpoTxq.json` |
| `openrouter/mistralai/mistral-large-2512` | 15.0% | 23.7% | 28,447 | `logs/2026-05-13T17-55-42-07-00_catanbench-visual_Kb9Prgw3FwVdtZJ3EjnDPR.json` |
| `openrouter/meta-llama/llama-4-maverick` | 12.5% | 18.8% | 41,671 | `logs/2026-05-13T17-42-21-07-00_catanbench-visual_a26Ds3bshcScSb6rDNNSQE.json` |
| `openrouter/qwen/qwen3-vl-32b-instruct` | 12.5% | 16.3% | 23,839 | `logs/2026-05-13T17-22-52-07-00_catanbench-visual_3sEP25BLYGN6ZeySusTk6C.json` |
| `openrouter/google/gemma-3-4b-it` | 12.5% | 21.3% | 24,496 | `logs/2026-05-13T16-54-17-07-00_catanbench-visual_9r5XntejAHsxDJeR6TgEG5.json` |
| `openrouter/google/gemma-4-26b-a4b-it` | 10.0% | 18.3% | 19,502 | `logs/2026-05-13T17-24-05-07-00_catanbench-visual_68coXZJ79X4D7FHGsmbXx4.json` |
| `openrouter/nvidia/nemotron-nano-12b-v2-vl:free` | 10.0% | 21.7% | 24,067 | `logs/2026-05-13T16-55-02-07-00_catanbench-visual_fug6vbi6965ozZC9nQa6Lj.json` |
| `openrouter/qwen/qwen3-vl-8b-instruct` | 10.0% | 18.0% | 23,344 | `logs/2026-05-13T16-53-37-07-00_catanbench-visual_bCKtssySKa5SwhwjGPMj6s.json` |
| `openrouter/openai/gpt-5.5` | 10.0% | 13.0% | 33,557 | `logs/2026-05-13T17-45-02-07-00_catanbench-visual_DzTSeLgfiA46857Tt3d6TV.json` |
| `openrouter/meta-llama/llama-4-scout` | 10.0% | 14.4% | 45,299 | `logs/2026-05-13T17-24-54-07-00_catanbench-visual_GuAB6L9YPhRHji2nPrVhtG.json` |
| `openrouter/google/gemma-3-12b-it` | 10.0% | 17.2% | 24,366 | `logs/2026-05-13T16-54-40-07-00_catanbench-visual_SC6rQKobbH9MbEAAnjYDJr.json` |
| `openrouter/qwen/qwen3-vl-30b-a3b-instruct` | 7.5% | 17.5% | 23,465 | `logs/2026-05-13T17-23-32-07-00_catanbench-visual_Ss7WUPEEbw2wZV7TgQ4TUF.json` |
| `openrouter/z-ai/glm-5v-turbo` | 2.5% | 6.8% | 37,627 | `logs/2026-05-13T17-56-21-07-00_catanbench-visual_Nyi8jmGuWJNcd3pD9ncNvF.json` |

## Category Breakdown

### `openrouter/google/gemini-3.1-pro-preview`

| Category | Exact | Component |
| --- | ---: | ---: |
| `color_building_counts` | 0/3 | 3/6 |
| `color_road_count` | 1/3 | 1/3 |
| `color_road_locations` | 0/4 | 2/15 |
| `edge_road_owner` | 4/4 | 4/4 |
| `node_occupancy` | 4/4 | 8/8 |
| `port_occupancy` | 2/3 | 2/5 |
| `port_trade_type` | 1/3 | 2/6 |
| `robber_resource_number` | 4/4 | 12/12 |
| `robber_tile` | 3/4 | 3/4 |
| `tile_has_robber` | 4/4 | 4/4 |
| `tile_resource_number` | 4/4 | 8/8 |

### `openrouter/x-ai/grok-4.3`

| Category | Exact | Component |
| --- | ---: | ---: |
| `color_building_counts` | 0/3 | 3/6 |
| `color_road_count` | 2/3 | 2/3 |
| `color_road_locations` | 0/4 | 2/15 |
| `edge_road_owner` | 2/4 | 2/4 |
| `node_occupancy` | 1/4 | 4/8 |
| `port_occupancy` | 1/3 | 1/5 |
| `port_trade_type` | 1/3 | 3/6 |
| `robber_resource_number` | 0/4 | 0/12 |
| `robber_tile` | 0/4 | 0/4 |
| `tile_has_robber` | 1/4 | 1/4 |
| `tile_resource_number` | 4/4 | 8/8 |

### `openrouter/anthropic/claude-sonnet-4.6`

| Category | Exact | Component |
| --- | ---: | ---: |
| `color_building_counts` | 0/3 | 2/6 |
| `color_road_count` | 2/3 | 2/3 |
| `color_road_locations` | 0/4 | 1/15 |
| `edge_road_owner` | 2/4 | 2/4 |
| `node_occupancy` | 0/4 | 0/8 |
| `port_occupancy` | 2/3 | 2/5 |
| `port_trade_type` | 1/3 | 2/6 |
| `robber_resource_number` | 0/4 | 2/12 |
| `robber_tile` | 0/4 | 0/4 |
| `tile_has_robber` | 0/4 | 0/4 |
| `tile_resource_number` | 3/4 | 6/8 |

### `openrouter/anthropic/claude-opus-4.7`

| Category | Exact | Component |
| --- | ---: | ---: |
| `color_building_counts` | 0/3 | 2/6 |
| `color_road_count` | 1/3 | 1/3 |
| `color_road_locations` | 0/4 | 2/15 |
| `edge_road_owner` | 1/4 | 1/4 |
| `node_occupancy` | 0/4 | 0/8 |
| `port_occupancy` | 2/3 | 2/5 |
| `port_trade_type` | 1/3 | 2/6 |
| `robber_resource_number` | 0/4 | 4/12 |
| `robber_tile` | 0/4 | 0/4 |
| `tile_has_robber` | 0/4 | 0/4 |
| `tile_resource_number` | 2/4 | 5/8 |

### `openrouter/google/gemma-4-31b-it`

| Category | Exact | Component |
| --- | ---: | ---: |
| `color_building_counts` | 0/3 | 1/6 |
| `color_road_count` | 2/3 | 2/3 |
| `color_road_locations` | 0/4 | 2/15 |
| `edge_road_owner` | 1/4 | 1/4 |
| `node_occupancy` | 0/4 | 0/8 |
| `port_occupancy` | 2/3 | 2/5 |
| `port_trade_type` | 1/3 | 2/6 |
| `robber_resource_number` | 0/4 | 2/12 |
| `robber_tile` | 0/4 | 0/4 |
| `tile_has_robber` | 0/4 | 0/4 |
| `tile_resource_number` | 1/4 | 4/8 |

### `openrouter/qwen/qwen3-vl-235b-a22b-instruct`

| Category | Exact | Component |
| --- | ---: | ---: |
| `color_building_counts` | 0/3 | 2/6 |
| `color_road_count` | 1/3 | 1/3 |
| `color_road_locations` | 0/4 | 0/15 |
| `edge_road_owner` | 2/4 | 2/4 |
| `node_occupancy` | 0/4 | 0/8 |
| `port_occupancy` | 1/3 | 1/5 |
| `port_trade_type` | 1/3 | 2/6 |
| `robber_resource_number` | 0/4 | 0/12 |
| `robber_tile` | 0/4 | 0/4 |
| `tile_has_robber` | 1/4 | 1/4 |
| `tile_resource_number` | 1/4 | 4/8 |

### `openrouter/mistralai/mistral-large-2512`

| Category | Exact | Component |
| --- | ---: | ---: |
| `color_building_counts` | 0/3 | 3/6 |
| `color_road_count` | 1/3 | 1/3 |
| `color_road_locations` | 0/4 | 2/15 |
| `edge_road_owner` | 0/4 | 0/4 |
| `node_occupancy` | 0/4 | 0/8 |
| `port_occupancy` | 2/3 | 2/5 |
| `port_trade_type` | 1/3 | 2/6 |
| `robber_resource_number` | 0/4 | 0/12 |
| `robber_tile` | 0/4 | 0/4 |
| `tile_has_robber` | 2/4 | 2/4 |
| `tile_resource_number` | 0/4 | 3/8 |

### `openrouter/meta-llama/llama-4-maverick`

| Category | Exact | Component |
| --- | ---: | ---: |
| `color_building_counts` | 0/3 | 1/6 |
| `color_road_count` | 0/3 | 0/3 |
| `color_road_locations` | 0/4 | 0/15 |
| `edge_road_owner` | 2/4 | 2/4 |
| `node_occupancy` | 0/4 | 1/8 |
| `port_occupancy` | 0/3 | 1/5 |
| `port_trade_type` | 1/3 | 2/6 |
| `robber_resource_number` | 0/4 | 2/12 |
| `robber_tile` | 0/4 | 0/4 |
| `tile_has_robber` | 2/4 | 2/4 |
| `tile_resource_number` | 0/4 | 1/8 |

### `openrouter/qwen/qwen3-vl-32b-instruct`

| Category | Exact | Component |
| --- | ---: | ---: |
| `color_building_counts` | 0/3 | 1/6 |
| `color_road_count` | 0/3 | 0/3 |
| `color_road_locations` | 0/4 | 0/15 |
| `edge_road_owner` | 1/4 | 1/4 |
| `node_occupancy` | 0/4 | 0/8 |
| `port_occupancy` | 2/3 | 2/5 |
| `port_trade_type` | 1/3 | 2/6 |
| `robber_resource_number` | 0/4 | 0/12 |
| `robber_tile` | 0/4 | 0/4 |
| `tile_has_robber` | 0/4 | 0/4 |
| `tile_resource_number` | 1/4 | 4/8 |

### `openrouter/qwen/qwen3-vl-8b-instruct`

| Category | Exact | Component |
| --- | ---: | ---: |
| `color_building_counts` | 0/3 | 2/6 |
| `color_road_count` | 1/3 | 1/3 |
| `color_road_locations` | 0/4 | 4/15 |
| `edge_road_owner` | 0/4 | 0/4 |
| `node_occupancy` | 0/4 | 0/8 |
| `port_occupancy` | 2/3 | 2/5 |
| `port_trade_type` | 1/3 | 2/6 |
| `robber_resource_number` | 0/4 | 0/12 |
| `robber_tile` | 0/4 | 0/4 |
| `tile_has_robber` | 0/4 | 0/4 |
| `tile_resource_number` | 0/4 | 2/8 |

### `openrouter/qwen/qwen3-vl-30b-a3b-instruct`

| Category | Exact | Component |
| --- | ---: | ---: |
| `color_building_counts` | 0/3 | 2/6 |
| `color_road_count` | 1/3 | 1/3 |
| `color_road_locations` | 0/4 | 3/15 |
| `edge_road_owner` | 0/4 | 0/4 |
| `node_occupancy` | 0/4 | 0/8 |
| `port_occupancy` | 1/3 | 1/5 |
| `port_trade_type` | 0/3 | 2/6 |
| `robber_resource_number` | 0/4 | 0/12 |
| `robber_tile` | 0/4 | 0/4 |
| `tile_has_robber` | 1/4 | 1/4 |
| `tile_resource_number` | 0/4 | 2/8 |

### `openrouter/openai/gpt-5.5`

| Category | Exact | Component |
| --- | ---: | ---: |
| `color_building_counts` | 1/3 | 2/6 |
| `color_road_count` | 0/3 | 0/3 |
| `color_road_locations` | 0/4 | 4/15 |
| `edge_road_owner` | 0/4 | 0/4 |
| `node_occupancy` | 0/4 | 0/8 |
| `port_occupancy` | 0/3 | 0/5 |
| `port_trade_type` | 0/3 | 0/6 |
| `robber_resource_number` | 0/4 | 0/12 |
| `robber_tile` | 0/4 | 0/4 |
| `tile_has_robber` | 0/4 | 0/4 |
| `tile_resource_number` | 3/4 | 6/8 |

### `openrouter/google/gemma-3-4b-it`

| Category | Exact | Component |
| --- | ---: | ---: |
| `color_building_counts` | 0/3 | 0/6 |
| `color_road_count` | 0/3 | 0/3 |
| `color_road_locations` | 0/4 | 4/15 |
| `edge_road_owner` | 0/4 | 0/4 |
| `node_occupancy` | 0/4 | 0/8 |
| `port_occupancy` | 0/3 | 0/5 |
| `port_trade_type` | 1/3 | 2/6 |
| `robber_resource_number` | 0/4 | 4/12 |
| `robber_tile` | 0/4 | 0/4 |
| `tile_has_robber` | 4/4 | 4/4 |
| `tile_resource_number` | 0/4 | 2/8 |

### `openrouter/google/gemma-4-26b-a4b-it`

| Category | Exact | Component |
| --- | ---: | ---: |
| `color_building_counts` | 0/3 | 1/6 |
| `color_road_count` | 1/3 | 1/3 |
| `color_road_locations` | 0/4 | 3/15 |
| `edge_road_owner` | 0/4 | 0/4 |
| `node_occupancy` | 0/4 | 0/8 |
| `port_occupancy` | 2/3 | 2/5 |
| `port_trade_type` | 1/3 | 2/6 |
| `robber_resource_number` | 0/4 | 4/12 |
| `robber_tile` | 0/4 | 0/4 |
| `tile_has_robber` | 0/4 | 0/4 |
| `tile_resource_number` | 0/4 | 1/8 |

### `openrouter/google/gemma-3-12b-it`

| Category | Exact | Component |
| --- | ---: | ---: |
| `color_building_counts` | 0/3 | 0/6 |
| `color_road_count` | 0/3 | 0/3 |
| `color_road_locations` | 0/4 | 4/15 |
| `edge_road_owner` | 0/4 | 0/4 |
| `node_occupancy` | 0/4 | 0/8 |
| `port_occupancy` | 1/3 | 3/5 |
| `port_trade_type` | 1/3 | 2/6 |
| `robber_resource_number` | 0/4 | 0/12 |
| `robber_tile` | 0/4 | 0/4 |
| `tile_has_robber` | 2/4 | 2/4 |
| `tile_resource_number` | 0/4 | 2/8 |

### `openrouter/meta-llama/llama-4-scout`

| Category | Exact | Component |
| --- | ---: | ---: |
| `color_building_counts` | 0/3 | 1/6 |
| `color_road_count` | 0/3 | 0/3 |
| `color_road_locations` | 0/4 | 1/15 |
| `edge_road_owner` | 0/4 | 0/4 |
| `node_occupancy` | 0/4 | 0/8 |
| `port_occupancy` | 1/3 | 2/5 |
| `port_trade_type` | 1/3 | 2/6 |
| `robber_resource_number` | 0/4 | 2/12 |
| `robber_tile` | 0/4 | 0/4 |
| `tile_has_robber` | 2/4 | 2/4 |
| `tile_resource_number` | 0/4 | 0/8 |

### `openrouter/nvidia/nemotron-nano-12b-v2-vl:free`

| Category | Exact | Component |
| --- | ---: | ---: |
| `color_building_counts` | 1/3 | 4/6 |
| `color_road_count` | 0/3 | 0/3 |
| `color_road_locations` | 0/4 | 4/15 |
| `edge_road_owner` | 0/4 | 0/4 |
| `node_occupancy` | 0/4 | 0/8 |
| `port_occupancy` | 2/3 | 2/5 |
| `port_trade_type` | 1/3 | 2/6 |
| `robber_resource_number` | 0/4 | 3/12 |
| `robber_tile` | 0/4 | 0/4 |
| `tile_has_robber` | 0/4 | 0/4 |
| `tile_resource_number` | 0/4 | 3/8 |

### `openrouter/z-ai/glm-5v-turbo`

| Category | Exact | Component |
| --- | ---: | ---: |
| `color_building_counts` | 0/3 | 1/6 |
| `color_road_count` | 0/3 | 0/3 |
| `color_road_locations` | 0/4 | 4/15 |
| `edge_road_owner` | 0/4 | 0/4 |
| `node_occupancy` | 0/4 | 0/8 |
| `port_occupancy` | 1/3 | 1/5 |
| `port_trade_type` | 0/3 | 0/6 |
| `robber_resource_number` | 0/4 | 0/12 |
| `robber_tile` | 0/4 | 0/4 |
| `tile_has_robber` | 0/4 | 0/4 |
| `tile_resource_number` | 0/4 | 0/8 |

## Notes

- Gemini is far above the small VLMs on this clean visual smoke, especially on
  tile/robber/node/edge ownership questions, but still struggles with full road
  location sets and count-style questions.
- Grok 4.3 is the strongest non-Gemini run so far, but it was slow and token
  heavy. It still scored 0/4 on robber-tile exact questions.
- Claude Sonnet 4.6 is the strongest practical frontier comparison after Grok,
  beating Claude Opus 4.7 on exact and component accuracy in this smoke.
- GPT-5.5 and GLM-5V were poor fits for the current strict token-answer prompt.
- None of the non-Gemini VLMs tested here are reliable visual board parsers yet.
- Gemma 4 31B improves over Gemma 3 and Qwen/Nemotron on this smoke, but it
  still fails most coordinate-grounded robber, node, edge, and road-location
  questions.
- Larger Qwen3-VL variants did not improve the visual parser result in this
  smoke: 32B scored 12.5% exact, and 30B-A3B scored 7.5% exact.
- Gemma 4 26B-A4B and Llama 4 Scout are fast and comparatively cheap on
  OpenRouter, but neither is competitive with dense Gemma 4 31B here.
- Gemma 3 4B has the best exact score in this clean smoke, but the margin is
  small among the smaller models and the absolute score is still poor.
- Qwen 3 VL 8B thinking was excluded from the clean table: a prior non-clean run
  took about 10 minutes and 168k tokens for 40 questions while not improving
  exact accuracy over the non-thinking model.
