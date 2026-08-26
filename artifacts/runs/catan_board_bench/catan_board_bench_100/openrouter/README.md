# OpenRouter CatanBoardBench Screening

Small-model screening runs against `CatanBoardBench-100`. This directory contains
machine run evidence; reviewed reports live in `reports/catan_board_bench/`.

## Harness

Runner: `scripts/eval_catan_board_bench_openrouter.py`

The runner loads `OPENROUTER_API_KEY` with `python-dotenv`, sends each 512x512
board image plus a compact atlas prompt, and scores model answers against the
engine answer key.

The original first screening pass used these categories:

- `robber_tile`
- `tile_resource_number`
- `node_occupancy`
- `edge_road_owner`
- `port_type_nodes`
- `longest_road_holder`

The revised suite removes target leakage from the atlas prompt and adds harder
categories:

- `robber_resource_number`
- `robber_adjacent_buildings`
- `tile_has_robber`
- `port_trade_type`
- `color_building_counts`
- `color_road_count`
- `tile_occupied_nodes`

`longest_road_holder` and `largest_army_holder` are now generated only when a
holder exists, so they no longer flood the eval with trivial `NONE` answers.

## Main 5-Sample Candidate Pass

Output: `20260513T201207Z/`

| Model | Requests | Errors | Exact accuracy | Component accuracy | Avg latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| `qwen3-vl-8b` | 30 | 0 | 43.3% | 51.0% | 1507 ms |
| `nemotron-12b-free` | 30 | 10 | 35.0% | 48.5% | 2728 ms |
| `ui-tars-7b` | 30 | 0 | 33.3% | 51.0% | 681 ms |
| `gemma3-12b` | 30 | 0 | 33.3% | 49.0% | 632 ms |
| `ministral-3b` | 30 | 0 | 30.0% | 37.3% | 1041 ms |

Best first candidate: `qwen3-vl-8b`.

Fastest viable candidate: `ui-tars-7b`.

Current conclusion: no raw small VLM is reliable enough for public board parsing
without training or a stronger visual grounding layer. Even with atlas context,
tile number/resource, robber location, road ownership, and port type remain weak.

## Other Smoke Runs

- `20260513T200550Z/`: first 2-sample run over Gemma 4B stale id, Qwen 8B,
  Nemotron 12B, and UI-TARS 7B. The stale Gemma free id had no endpoint.
- `20260513T200750Z/`: patched Gemma request format; stale Gemma 4B free id
  still unavailable.
- `20260513T200855Z/`: one-sample sweep over current Gemma 3, Qwen 3.5,
  Ministral, and Nemotron 3 Nano ids.
- `20260513T200941Z/`: one-sample sweep over Gemma 4 free ids, Qwen 8B
  thinking, Llama 11B vision, and Mistral Small 3.2. Gemma 4 free was upstream
  rate-limited.

## Revised-Suite Smoke Run

Output: `20260513T212255Z/`

Model: `qwen3-vl-8b`

Scope: 5 samples, 8 questions per sample, categories selected by the revised
default runner.

| Category | Exact accuracy | Component accuracy |
| --- | ---: | ---: |
| `robber_tile` | 0.0% | 0.0% |
| `robber_resource_number` | 0.0% | 0.0% |
| `tile_resource_number` | 0.0% | 0.0% |
| `tile_has_robber` | 0.0% | 0.0% |
| `node_occupancy` | 0.0% | 0.0% |
| `edge_road_owner` | 0.0% | 0.0% |
| `port_trade_type` | 60.0% | 60.0% |
| `color_building_counts` | 0.0% | 0.0% |

Overall exact accuracy: 7.5%.

Overall component accuracy: 8.0%.

Takeaway: after removing the target leak, biasing toward positive node/edge
questions, and making the selected `tile_has_robber` row the positive `YES`
case, raw Qwen 8B is much weaker. This is a better diagnostic: it now tests the
visual grounding failure we care about instead of rewarding trivial
`NONE`/`EMPTY` answers or leaked topology.

## Gemini 3.1 Pro Preview

Output: `20260513T212456Z/`

Model: `google/gemini-3.1-pro-preview`

Scope: same revised 5-sample / 40-question smoke pass.

Strict token scoring:

- exact accuracy: 7.5%
- component accuracy: 12.0%
- average latency: 4362 ms

Lenient token repair scoring:

- output: `20260513T212456Z/summary_lenient_tokens.json`
- exact accuracy: 15.0%
- component accuracy: 21.3%

Interpretation: Gemini 3.1 Pro Preview is better than Qwen 8B on this fixed
smoke pass only after repairing partial tokens like `<T17`, `<BLACK`, and
`<ORE`, but it still fails the core board-grounding tasks. The main remaining
failures are robber tile/resource, tile resource-number binding, positive node
occupancy, and count questions.
