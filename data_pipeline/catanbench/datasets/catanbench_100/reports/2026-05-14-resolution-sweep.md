# 2026-05-14 CatanBench Resolution Sweep

Purpose: test whether board-recognition failures look like missing pixel detail
or higher-level localization / grounding failures.

## Setup

- Model: `openrouter/google/gemini-3.1-pro-preview`
- Samples: first 5 replay-derived board states
- Questions: 40 total, 8 per sample
- Prompt: atlas prompt enabled
- Decoding: `temperature=0`, `max_tokens=512`
- Image variants: fresh frontend renders at `768x768`, `512x512`, and `256x256`
- QA control: the three `qa.jsonl` files have the same SHA-1:
  `6ab6df6df83293042ec8e616a3979022f0ec414a`

Rendered image dirs:

- `data_pipeline/catanbench/datasets/resolution_sweep/768`
- `data_pipeline/catanbench/datasets/resolution_sweep/512`
- `data_pipeline/catanbench/datasets/resolution_sweep/256`

Run outputs:

- `data_pipeline/catanbench/datasets/resolution_sweep/openrouter_eval/gemini_768_40q/summary.json`
- `data_pipeline/catanbench/datasets/resolution_sweep/openrouter_eval/gemini_512_40q/summary.json`
- `data_pipeline/catanbench/datasets/resolution_sweep/openrouter_eval/gemini_256_40q/summary.json`

## Overall Results

| Resolution | Exact | Component | Requests | Errors |
| --- | ---: | ---: | ---: | ---: |
| 768 | 70.0% | 56.7% | 40 | 0 |
| 512 | 67.5% | 56.7% | 40 | 0 |
| 256 | 55.0% | 44.4% | 40 | 0 |

## Category Results

| Category | 768 exact | 512 exact | 256 exact | Takeaway |
| --- | ---: | ---: | ---: | --- |
| `robber_tile` | 100% | 100% | 60% | Robber localization loses detail at 256. |
| `robber_resource_number` | 100% | 100% | 60% | Same drop as robber tile, so this is mostly localization. |
| `tile_resource_number` | 100% | 100% | 100% | Tile resource/number reading is robust even at 256 on this slice. |
| `tile_has_robber` | 80% | 40% | 40% | Noisy question behavior; needs more samples and maybe prompt variant. |
| `node_occupancy` | 60% | 60% | 100% | Too small/noisy to conclude 256 helps; rerun with more node rows. |
| `edge_road_owner` | 80% | 100% | 80% | 512 is not worse than 768 here. |
| `color_road_locations` | 0% | 0% | 0% | Full road-set listing is failing independent of resolution. |
| `port_trade_type` | 40% | 40% | 0% | Port symbols become unreliable at 256. |

## Interpretation

For this 40-question slice, `512x512` is effectively tied with `768x768`.
That supports using 512 as the default training/eval resolution for full-board
Catan screenshots, at least for the current renderer and atlas prompt.

`256x256` is too compressed for reliable board parsing. The clearest losses are
robber localization and port trade type. It can still read some tile
resource/number facts, but that is not enough for full public-board recovery.

The hardest failures at 768 are not just pixel-detail problems:

- `color_road_locations` is 0% at every resolution.
- `port_trade_type` is only 40% even at 768/512.
- node/building questions are not stable enough with only 5 rows.

So the next ablation should not be "go bigger than 768." The better next tests
are:

1. local crops around the queried tile/node/edge/port
2. node/edge overlay labels
3. counterfactual pairs where exactly one board fact changes
4. a full board-state extraction task scored slot-by-slot

## Deterministic CV Parser Baseline

The deterministic engine is the oracle and answer key. It tells us what the
state is after replay/application of actions.

A deterministic CV parser is different: it starts only from the screenshot and
uses non-LLM image processing to reconstruct the public board contract. For this
repo, that could mean template/color/OCR matching over the known Colonist-like
renderer:

- detect tile centers from the fixed board layout
- classify tile resource art and number tokens
- detect robber sprite location
- classify roads by edge slot and color
- classify buildings by node slot, color, and settlement/city sprite
- classify port trade type by port slot
- validate the reconstructed state against the engine topology

That parser would be a baseline for "how much of this task is solvable with
deterministic vision plus the known atlas." The engine alone is not that
baseline because it skips the pixels.
