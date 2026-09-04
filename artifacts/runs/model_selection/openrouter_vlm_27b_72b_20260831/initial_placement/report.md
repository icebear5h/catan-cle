# Full-game model vs human action-selection diff

Generated: 2026-08-31T23:08:12.374350+00:00
Game: `242781000`
Comparison parser: `shared-action-parser-v3`
Human: Colonist color 2 / engine BLUE
Policy calls: stateless; model actions were never executed
Replay stepping: `allow_lookahead=False`

## Decision coverage

- Parsed replay rows: 570
- Same-event robber/steal order canonicalizations: 9
- Human-seat records: 164
- Exact indexed choices: 86
- Forced exact choices: 22
- Nontrivial exact choices: 64
- Other classifications: `{"coarse": 24, "exact": 86, "lifecycle": 26, "unmappable": 28}`
- Replay semantic errors: 0

## Headline agreement

| Model | Valid / exact | All exact | Forced | Nontrivial | Cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| `qwen/qwen3.5-27b` | 0/0 (n/a) | 0/86 (0.0%) | 0/0 (n/a) | 0/0 (n/a) | $0.011479 |
| `qwen/qwen3.5-35b-a3b` | 0/0 (n/a) | 0/86 (0.0%) | 0/0 (n/a) | 0/0 (n/a) | $0.004124 |
| `qwen/qwen3.6-27b` | 0/0 (n/a) | 0/86 (0.0%) | 0/0 (n/a) | 0/0 (n/a) | $0.022635 |
| `qwen/qwen3.6-35b-a3b` | 0/0 (n/a) | 0/86 (0.0%) | 0/0 (n/a) | 0/0 (n/a) | $0.003525 |
| `qwen/qwen3.8-27b` | 0/1 (0.0%) | 0/86 (0.0%) | 0/0 (n/a) | 0/1 (0.0%) | $0.023235 |
| `qwen/qwen3-vl-30b-a3b-thinking` | 0/1 (0.0%) | 0/86 (0.0%) | 0/0 (n/a) | 0/1 (0.0%) | $0.015067 |
| `meta/muse-glimmer-30b` | 0/0 (n/a) | 0/86 (0.0%) | 0/0 (n/a) | 0/0 (n/a) | $0.008284 |
| `google/gemma-4-31b-it` | 1/1 (100.0%) | 1/86 (1.2%) | 0/0 (n/a) | 1/1 (100.0%) | $0.001955 |
| `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free` | 0/1 (0.0%) | 0/86 (0.0%) | 0/0 (n/a) | 0/1 (0.0%) | $0.000000 |

## Output and usage health

| Model | Responses | Valid actions | Format warnings | API errors | Tokens in/out | Median / p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen/qwen3.5-27b` | 1 | 0 | 1 | 0 | 13,149 / 4,096 | 87.86s / 87.86s |
| `qwen/qwen3.5-35b-a3b` | 1 | 0 | 1 | 0 | 13,149 / 4,096 | 41.63s / 41.63s |
| `qwen/qwen3.6-27b` | 1 | 0 | 1 | 0 | 13,149 / 4,096 | 79.69s / 79.69s |
| `qwen/qwen3.6-35b-a3b` | 1 | 0 | 1 | 0 | 13,149 / 4,096 | 39.89s / 39.89s |
| `qwen/qwen3.8-27b` | 1 | 1 | 0 | 0 | 13,149 / 5,904 | 93.49s / 93.49s |
| `qwen/qwen3-vl-30b-a3b-thinking` | 1 | 1 | 0 | 0 | 13,060 / 11,280 | 164.09s / 164.09s |
| `meta/muse-glimmer-30b` | 1 | 0 | 1 | 0 | 11,229 / 4,096 | 19.41s / 19.41s |
| `google/gemma-4-31b-it` | 1 | 1 | 0 | 0 | 13,978 / 2,051 | 43.75s / 43.75s |
| `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free` | 1 | 1 | 0 | 0 | 13,791 / 30,001 | 419.89s / 419.89s |

## Agreement by human action type

| Human action | `qwen/qwen3.5-27b` | `qwen/qwen3.5-35b-a3b` | `qwen/qwen3.6-27b` | `qwen/qwen3.6-35b-a3b` | `qwen/qwen3.8-27b` | `qwen/qwen3-vl-30b-a3b-thinking` | `meta/muse-glimmer-30b` | `google/gemma-4-31b-it` | `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `BUILD_CITY` | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) |
| `BUILD_ROAD` | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) |
| `BUILD_SETTLEMENT` | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) | 0/1 (0.0%) | 0/1 (0.0%) | 0/0 (n/a) | 1/1 (100.0%) | 0/1 (0.0%) |
| `BUY_DEVELOPMENT_CARD` | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) |
| `END_TURN` | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) |
| `MARITIME_TRADE` | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) |
| `MOVE_ROBBER` | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) |
| `PLAY_KNIGHT_CARD` | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) |
| `PLAY_ROAD_BUILDING` | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) |
| `ROLL` | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) |
| `STEAL` | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) |
| `YEAR_OF_PLENTY_RESOURCES` | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) | 0/0 (n/a) |

## Decisions with at least one model-human difference

| Replay row | Human type | Menu | Human | `qwen/qwen3.5-27b` | `qwen/qwen3.5-35b-a3b` | `qwen/qwen3.6-27b` | `qwen/qwen3.6-35b-a3b` | `qwen/qwen3.8-27b` | `qwen/qwen3-vl-30b-a3b-thinking` | `meta/muse-glimmer-30b` | `google/gemma-4-31b-it` | `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free` |
| ---: | --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | `BUILD_SETTLEMENT` | 54 | 7: Build settlement at SHEEP dice=4(3pip)/BRICK dice=8(5pip)/WHEAT dice=10(3pip) [11pips] | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ 23: Build settlement at WOOD dice=5(4pip)/BRICK dice=9(4pip)/WHEAT dice=10(3pip) [11pips] | ✗ 23: Build settlement at WOOD dice=5(4pip)/BRICK dice=9(4pip)/WHEAT dice=10(3pip) [11pips] | ✗ None: — | ✓ 7: Build settlement at SHEEP dice=4(3pip)/BRICK dice=8(5pip)/WHEAT dice=10(3pip) [11pips] | ✗ 23: Build settlement at WOOD dice=5(4pip)/BRICK dice=9(4pip)/WHEAT dice=10(3pip) [11pips] |
| 1 | `BUILD_ROAD` | 3 | 1: Build road between SHEEP dice=4(3pip)/BRICK dice=8(5pip)/WHEAT dice=10(3pip) [11pips] and BRICK… | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 14 | `BUILD_SETTLEMENT` | 30 | 17: Build settlement at ORE dice=8(5pip)/ORE dice=4(3pip) [8pips] (3:1 port) | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 15 | `BUILD_ROAD` | 3 | 0: Build road between ORE dice=8(5pip)/ORE dice=4(3pip) [8pips] (3:1 port) and ORE dice=8(5pip) [5… | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 16 | `ROLL` | 1 | 0: Roll dice | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 17 | `END_TURN` | 2 | 0: End turn | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 34 | `ROLL` | 1 | 0: Roll dice | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 35 | `MOVE_ROBBER` | 18 | 5: Move robber to (0, 1, -1) | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 36 | `STEAL` | 1 | 0: Steal from (C.BLACK, None) | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 37 | `END_TURN` | 2 | 0: End turn | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 46 | `ROLL` | 1 | 0: Roll dice | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 47 | `END_TURN` | 2 | 0: End turn | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 61 | `ROLL` | 1 | 0: Roll dice | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 76 | `BUILD_ROAD` | 13 | 1: Build road between BRICK dice=8(5pip)/WHEAT dice=10(3pip) [8pips] and WHEAT dice=10(3pip) [3pip… | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 77 | `END_TURN` | 6 | 0: End turn | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 89 | `ROLL` | 1 | 0: Roll dice | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 95 | `END_TURN` | 6 | 0: End turn | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 104 | `ROLL` | 1 | 0: Roll dice | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 121 | `END_TURN` | 7 | 0: End turn | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 139 | `ROLL` | 1 | 0: Roll dice | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 146 | `BUILD_CITY` | 17 | 9: Upgrade to city at SHEEP dice=4(3pip)/BRICK dice=8(5pip)/WHEAT dice=10(3pip) [11pips] | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 150 | `END_TURN` | 10 | 0: End turn | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 190 | `ROLL` | 1 | 0: Roll dice | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 196 | `END_TURN` | 13 | 0: End turn | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 206 | `ROLL` | 1 | 0: Roll dice | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 207 | `BUILD_SETTLEMENT` | 19 | 8: Build settlement at WHEAT dice=10(3pip) [3pips] (BRICK 2:1 port) | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 208 | `BUY_DEVELOPMENT_CARD` | 7 | 1: Buy development card | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 209 | `MARITIME_TRADE` | 6 | 1: Trade 2 BRICK for 1 WHEAT | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 210 | `BUY_DEVELOPMENT_CARD` | 3 | 1: Buy development card | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 211 | `END_TURN` | 1 | 0: End turn | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 223 | `PLAY_KNIGHT_CARD` | 17 | 15: Play knight card | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 224 | `MOVE_ROBBER` | 18 | 11: Move robber to (-2, 0, 2) | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 225 | `STEAL` | 2 | 1: Steal from (C.ORANGE, None) | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 226 | `ROLL` | 1 | 0: Roll dice | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 231 | `END_TURN` | 3 | 0: End turn | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 261 | `ROLL` | 16 | 15: Roll dice | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 262 | `YEAR_OF_PLENTY_RESOURCES` | 25 | 5: Year of Plenty: take WHEAT and WHEAT | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 264 | `BUY_DEVELOPMENT_CARD` | 13 | 3: Buy development card | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 265 | `BUY_DEVELOPMENT_CARD` | 7 | 1: Buy development card | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 266 | `END_TURN` | 2 | 0: End turn | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 284 | `ROLL` | 3 | 2: Roll dice | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 285 | `BUY_DEVELOPMENT_CARD` | 23 | 5: Buy development card | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 288 | `BUY_DEVELOPMENT_CARD` | 13 | 3: Buy development card | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 289 | `PLAY_KNIGHT_CARD` | 9 | 0: Play knight card | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 290 | `MOVE_ROBBER` | 18 | 9: Move robber to (0, -2, 2) | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 291 | `STEAL` | 2 | 0: Steal from (C.BLACK, None) | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 308 | `BUILD_ROAD` | 13 | 4: Build road between BRICK dice=8(5pip)/WHEAT dice=10(3pip) [8pips] and BRICK dice=8(5pip) [5pips… | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 309 | `BUILD_SETTLEMENT` | 10 | 8: Build settlement at BRICK dice=8(5pip) [5pips] (3:1 port) \| Near: BLACK settlement, ORANGE set… | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 310 | `END_TURN` | 1 | 0: End turn | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 323 | `PLAY_KNIGHT_CARD` | 3 | 0: Play knight card | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 324 | `MOVE_ROBBER` | 18 | 11: Move robber to (-2, 0, 2) | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 325 | `STEAL` | 2 | 1: Steal from (C.ORANGE, None) | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 326 | `ROLL` | 1 | 0: Roll dice | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 327 | `END_TURN` | 2 | 0: End turn | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 357 | `PLAY_KNIGHT_CARD` | 3 | 0: Play knight card | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 358 | `MOVE_ROBBER` | 18 | 16: Move robber to (2, 0, -2) | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 359 | `STEAL` | 1 | 0: Steal from (C.ORANGE, None) | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 360 | `ROLL` | 1 | 0: Roll dice | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 361 | `MARITIME_TRADE` | 17 | 15: Trade 2 BRICK for 1 SHEEP | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 362 | `BUY_DEVELOPMENT_CARD` | 14 | 8: Buy development card | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 363 | `END_TURN` | 9 | 0: End turn | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 398 | `PLAY_KNIGHT_CARD` | 3 | 0: Play knight card | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 399 | `MOVE_ROBBER` | 18 | 16: Move robber to (2, 0, -2) | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 400 | `STEAL` | 1 | 0: Steal from (C.ORANGE, None) | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 401 | `ROLL` | 1 | 0: Roll dice | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 407 | `END_TURN` | 9 | 0: End turn | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 453 | `ROLL` | 2 | 1: Roll dice | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 454 | `MOVE_ROBBER` | 18 | 17: Move robber to (2, 0, -2) | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 455 | `STEAL` | 1 | 0: Steal from (C.ORANGE, None) | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 461 | `END_TURN` | 10 | 1: End turn | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 491 | `ROLL` | 2 | 1: Roll dice | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 497 | `MARITIME_TRADE` | 18 | 13: Trade 3 SHEEP for 1 WHEAT | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 498 | `PLAY_ROAD_BUILDING` | 15 | 0: Play road building card | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 499 | `BUILD_ROAD` | 7 | 4: Build road between SHEEP dice=4(3pip)/BRICK dice=8(5pip)/WHEAT dice=10(3pip) [11pips] and SHEEP… | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 500 | `BUILD_ROAD` | 8 | 1: Build road between SHEEP dice=4(3pip)/BRICK dice=8(5pip)/ORE dice=3(2pip) [10pips] and BRICK di… | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 501 | `BUILD_SETTLEMENT` | 17 | 10: Build settlement at BRICK dice=8(5pip)/ORE dice=3(2pip) [7pips] \| Near: RED settlement, RED se… | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 502 | `END_TURN` | 6 | 0: End turn | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 520 | `ROLL` | 1 | 0: Roll dice | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 526 | `END_TURN` | 7 | 0: End turn | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 549 | `ROLL` | 1 | 0: Roll dice | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 550 | `BUILD_ROAD` | 15 | 3: Build road between ORE dice=8(5pip) [5pips] and ORE dice=8(5pip) [5pips] | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 551 | `END_TURN` | 6 | 0: End turn | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 566 | `ROLL` | 1 | 0: Roll dice | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 567 | `BUILD_CITY` | 18 | 1: Upgrade to city at ORE dice=8(5pip)/ORE dice=4(3pip) [8pips] (3:1 port) | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 568 | `MARITIME_TRADE` | 6 | 4: Trade 2 BRICK for 1 WOOD | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |
| 569 | `BUILD_SETTLEMENT` | 16 | 10: Build settlement at ORE dice=8(5pip) [5pips] \| Near: ORANGE settlement, ORANGE settlement, RED… | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — | ✗ None: — |

## Artifacts

- `plan.json`: immutable run inputs and packet settings
- `decision_manifest.jsonl`: every classified action for the human seat
- `responses.jsonl`: append-only raw model calls and prompts
- `comparisons.jsonl`: one normalized human/model comparison per exact choice
- `summary.json`: machine-readable aggregate metrics
