# Full-game model vs human action-selection diff

Generated: 2026-08-16T22:54:42.398612+00:00
Game: `242781000`
Human: Colonist color 5 / engine BLACK
Policy calls: stateless; model actions were never executed
Replay stepping: `allow_lookahead=False`

## Decision coverage

- Parsed replay rows: 570
- Same-event robber/steal order canonicalizations: 9
- Human-seat records: 145
- Exact indexed choices: 109
- Forced exact choices: 20
- Nontrivial exact choices: 89
- Other classifications: `{"coarse": 20, "exact": 109, "lifecycle": 15, "unmappable": 1}`
- Replay semantic errors: 0

## Headline agreement

| Model | Valid / exact | All exact | Forced | Nontrivial | Cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| `openai/gpt-5.6-sol` | 0/1 (0.0%) | 0/109 (0.0%) | 0/0 (n/a) | 0/1 (0.0%) | $0.031901 |
| `qwen/qwen3.8-27b` | 0/0 (n/a) | 0/109 (0.0%) | 0/0 (n/a) | 0/0 (n/a) | $0.026021 |

## Model-to-model diff

- Same selection: 0/0 (n/a)
- Both match human: 0
- Only `openai/gpt-5.6-sol` matches human: 0
- Only `qwen/qwen3.8-27b` matches human: 0
- Neither matches human: 0
- Both choose the same nonhuman alternative: 0

## Agreement by human action type

| Human action | `openai/gpt-5.6-sol` | `qwen/qwen3.8-27b` |
| --- | ---: | ---: |
| `ACCEPT_TRADE` | 0/0 (n/a) | 0/0 (n/a) |
| `BUILD_ROAD` | 0/0 (n/a) | 0/0 (n/a) |
| `BUILD_SETTLEMENT` | 0/1 (0.0%) | 0/0 (n/a) |
| `BUY_DEVELOPMENT_CARD` | 0/0 (n/a) | 0/0 (n/a) |
| `END_TURN` | 0/0 (n/a) | 0/0 (n/a) |
| `MARITIME_TRADE` | 0/0 (n/a) | 0/0 (n/a) |
| `MONOPOLY_RESOURCE` | 0/0 (n/a) | 0/0 (n/a) |
| `MOVE_ROBBER` | 0/0 (n/a) | 0/0 (n/a) |
| `PLAY_KNIGHT_CARD` | 0/0 (n/a) | 0/0 (n/a) |
| `REJECT_TRADE` | 0/0 (n/a) | 0/0 (n/a) |
| `ROLL` | 0/0 (n/a) | 0/0 (n/a) |
| `STEAL` | 0/0 (n/a) | 0/0 (n/a) |

## Decisions with at least one model-human difference

| Replay row | Human type | Menu | Human | `openai/gpt-5.6-sol` | `qwen/qwen3.8-27b` |
| ---: | --- | ---: | --- | --- | --- |
| 2 | `BUILD_SETTLEMENT` | 50 | 7: Build settlement at SHEEP dice=9(4pip)/ORE dice=3(2pip)/BRICK dice=6(5pip) [11pips] \| Near: BL… | ✗ 20: Build settlement at WOOD dice=5(4pip)/BRICK dice=9(4pip)/WHEAT dice=10(3pip) [11pips] | ✗ None: — |
| 3 | `BUILD_ROAD` | 3 | 1: Build road between SHEEP dice=9(4pip)/ORE dice=3(2pip)/BRICK dice=6(5pip) [11pips] and SHEEP di… | ✗ None: — | ✗ None: — |
| 12 | `BUILD_SETTLEMENT` | 33 | 9: Build settlement at WOOD dice=6(5pip)/WOOD dice=5(4pip)/WHEAT dice=12(1pip) [10pips] \| Near: O… | ✗ None: — | ✗ None: — |
| 13 | `BUILD_ROAD` | 3 | 0: Build road between SHEEP dice=11(2pip)/WOOD dice=6(5pip)/WOOD dice=5(4pip) [11pips] and WOOD di… | ✗ None: — | ✗ None: — |
| 18 | `ROLL` | 1 | 0: Roll dice | ✗ None: — | ✗ None: — |
| 19 | `END_TURN` | 2 | 0: End turn | ✗ None: — | ✗ None: — |
| 38 | `ROLL` | 1 | 0: Roll dice | ✗ None: — | ✗ None: — |
| 39 | `BUILD_ROAD` | 14 | 1: Build road between SHEEP dice=11(2pip)/WOOD dice=6(5pip)/WOOD dice=5(4pip) [11pips] and SHEEP d… | ✗ None: — | ✗ None: — |
| 40 | `END_TURN` | 2 | 0: End turn | ✗ None: — | ✗ None: — |
| 48 | `ROLL` | 1 | 0: Roll dice | ✗ None: — | ✗ None: — |
| 49 | `BUY_DEVELOPMENT_CARD` | 3 | 1: Buy development card | ✗ None: — | ✗ None: — |
| 50 | `END_TURN` | 2 | 0: End turn | ✗ None: — | ✗ None: — |
| 55 | `REJECT_TRADE` | 2 | 0: REJECT_TRADE: Color.ORANGE | ✗ None: — | ✗ None: — |
| 66 | `REJECT_TRADE` | 2 | 0: REJECT_TRADE: Color.BLUE | ✗ None: — | ✗ None: — |
| 73 | `ACCEPT_TRADE` | 3 | 1: ACCEPT_TRADE: Color.BLUE | ✗ None: — | ✗ None: — |
| 78 | `PLAY_KNIGHT_CARD` | 2 | 0: Play knight card | ✗ None: — | ✗ None: — |
| 79 | `MOVE_ROBBER` | 18 | 17: Move robber to (2, -1, -1) | ✗ None: — | ✗ None: — |
| 80 | `STEAL` | 2 | 0: Steal from (C.BLUE, None) | ✗ None: — | ✗ None: — |
| 81 | `ROLL` | 1 | 0: Roll dice | ✗ None: — | ✗ None: — |
| 82 | `END_TURN` | 11 | 0: End turn | ✗ None: — | ✗ None: — |
| 93 | `REJECT_TRADE` | 2 | 0: REJECT_TRADE: Color.BLUE | ✗ None: — | ✗ None: — |
| 96 | `ROLL` | 1 | 0: Roll dice | ✗ None: — | ✗ None: — |
| 97 | `BUILD_SETTLEMENT` | 13 | 10: Build settlement at SHEEP dice=11(2pip)/SHEEP dice=4(3pip)/WOOD dice=5(4pip) [9pips] | ✗ None: — | ✗ None: — |
| 98 | `BUILD_ROAD` | 11 | 5: Build road between SHEEP dice=9(4pip)/BRICK dice=6(5pip)/WHEAT dice=2(1pip) [10pips] and SHEEP … | ✗ None: — | ✗ None: — |
| 99 | `END_TURN` | 2 | 0: End turn | ✗ None: — | ✗ None: — |
| 108 | `REJECT_TRADE` | 2 | 0: REJECT_TRADE: Color.BLUE | ✗ None: — | ✗ None: — |
| 110 | `REJECT_TRADE` | 2 | 0: REJECT_TRADE: Color.BLUE | ✗ None: — | ✗ None: — |
| 119 | `REJECT_TRADE` | 2 | 0: REJECT_TRADE: Color.BLUE | ✗ None: — | ✗ None: — |
| 123 | `ROLL` | 1 | 0: Roll dice | ✗ None: — | ✗ None: — |
| 124 | `END_TURN` | 6 | 0: End turn | ✗ None: — | ✗ None: — |
| 127 | `REJECT_TRADE` | 2 | 0: REJECT_TRADE: Color.ORANGE | ✗ None: — | ✗ None: — |
| 141 | `REJECT_TRADE` | 2 | 0: REJECT_TRADE: Color.BLUE | ✗ None: — | ✗ None: — |
| 148 | `REJECT_TRADE` | 2 | 0: REJECT_TRADE: Color.BLUE | ✗ None: — | ✗ None: — |
| 152 | `ROLL` | 1 | 0: Roll dice | ✗ None: — | ✗ None: — |
| 162 | `MARITIME_TRADE` | 6 | 1: Trade 4 SHEEP for 1 WHEAT | ✗ None: — | ✗ None: — |
| 163 | `BUY_DEVELOPMENT_CARD` | 3 | 1: Buy development card | ✗ None: — | ✗ None: — |
| 164 | `END_TURN` | 2 | 0: End turn | ✗ None: — | ✗ None: — |
| 170 | `REJECT_TRADE` | 2 | 0: REJECT_TRADE: Color.ORANGE | ✗ None: — | ✗ None: — |
| 180 | `REJECT_TRADE` | 2 | 0: REJECT_TRADE: Color.ORANGE | ✗ None: — | ✗ None: — |
| 192 | `REJECT_TRADE` | 3 | 0: REJECT_TRADE: Color.BLUE | ✗ None: — | ✗ None: — |
| 197 | `ROLL` | 1 | 0: Roll dice | ✗ None: — | ✗ None: — |
| 198 | `BUILD_SETTLEMENT` | 13 | 11: Build settlement at SHEEP dice=9(4pip)/WHEAT dice=2(1pip) [5pips] \| Near: ORANGE settlement, O… | ✗ None: — | ✗ None: — |
| 199 | `END_TURN` | 2 | 0: End turn | ✗ None: — | ✗ None: — |
| 212 | `ROLL` | 1 | 0: Roll dice | ✗ None: — | ✗ None: — |
| 214 | `MOVE_ROBBER` | 18 | 17: Move robber to (2, -1, -1) | ✗ None: — | ✗ None: — |
| 215 | `STEAL` | 1 | 0: Steal from (C.ORANGE, None) | ✗ None: — | ✗ None: — |
| 216 | `END_TURN` | 2 | 0: End turn | ✗ None: — | ✗ None: — |
| 228 | `REJECT_TRADE` | 2 | 0: REJECT_TRADE: Color.BLUE | ✗ None: — | ✗ None: — |
| 233 | `ROLL` | 1 | 0: Roll dice | ✗ None: — | ✗ None: — |
| 234 | `BUILD_ROAD` | 16 | 6: Build road between SHEEP dice=9(4pip)/ORE dice=3(2pip)/BRICK dice=6(5pip) [11pips] and ORE dice… | ✗ None: — | ✗ None: — |
| 235 | `BUILD_ROAD` | 17 | 4: Build road between ORE dice=3(2pip) [2pips] (WHEAT 2:1 port) and ORE dice=3(2pip)/BRICK dice=6(… | ✗ None: — | ✗ None: — |
| 236 | `MARITIME_TRADE` | 6 | 4: Trade 4 WOOD for 1 WHEAT | ✗ None: — | ✗ None: — |
| 237 | `BUY_DEVELOPMENT_CARD` | 3 | 1: Buy development card | ✗ None: — | ✗ None: — |
| 238 | `END_TURN` | 2 | 0: End turn | ✗ None: — | ✗ None: — |
| 245 | `REJECT_TRADE` | 2 | 0: REJECT_TRADE: Color.RED | ✗ None: — | ✗ None: — |
| 249 | `REJECT_TRADE` | 3 | 0: REJECT_TRADE: Color.RED | ✗ None: — | ✗ None: — |
| 267 | `ROLL` | 2 | 1: Roll dice | ✗ None: — | ✗ None: — |
| 268 | `MARITIME_TRADE` | 7 | 2: Trade 4 SHEEP for 1 WHEAT | ✗ None: — | ✗ None: — |
| 269 | `BUY_DEVELOPMENT_CARD` | 4 | 2: Buy development card | ✗ None: — | ✗ None: — |
| 270 | `END_TURN` | 3 | 1: End turn | ✗ None: — | ✗ None: — |
| 278 | `REJECT_TRADE` | 2 | 0: REJECT_TRADE: Color.RED | ✗ None: — | ✗ None: — |
| 293 | `REJECT_TRADE` | 2 | 0: REJECT_TRADE: Color.BLUE | ✗ None: — | ✗ None: — |
| 305 | `REJECT_TRADE` | 2 | 0: REJECT_TRADE: Color.BLUE | ✗ None: — | ✗ None: — |
| 311 | `ROLL` | 2 | 1: Roll dice | ✗ None: — | ✗ None: — |
| 312 | `PLAY_KNIGHT_CARD` | 3 | 0: Play knight card | ✗ None: — | ✗ None: — |
| 313 | `MOVE_ROBBER` | 18 | 17: Move robber to (2, -1, -1) | ✗ None: — | ✗ None: — |
| 314 | `STEAL` | 1 | 0: Steal from (C.ORANGE, None) | ✗ None: — | ✗ None: — |
| 315 | `END_TURN` | 2 | 0: End turn | ✗ None: — | ✗ None: — |
| 328 | `PLAY_KNIGHT_CARD` | 2 | 0: Play knight card | ✗ None: — | ✗ None: — |
| 329 | `MOVE_ROBBER` | 18 | 7: Move robber to (2, -2, 0) | ✗ None: — | ✗ None: — |
| 330 | `STEAL` | 1 | 0: Steal from (C.BLUE, None) | ✗ None: — | ✗ None: — |
| 331 | `ROLL` | 1 | 0: Roll dice | ✗ None: — | ✗ None: — |
| 347 | `MARITIME_TRADE` | 21 | 19: Trade 4 WOOD for 1 WHEAT | ✗ None: — | ✗ None: — |
| 348 | `BUY_DEVELOPMENT_CARD` | 7 | 1: Buy development card | ✗ None: — | ✗ None: — |
| 349 | `END_TURN` | 2 | 0: End turn | ✗ None: — | ✗ None: — |
| 364 | `ROLL` | 1 | 0: Roll dice | ✗ None: — | ✗ None: — |
| 368 | `END_TURN` | 7 | 0: End turn | ✗ None: — | ✗ None: — |
| 384 | `REJECT_TRADE` | 3 | 0: REJECT_TRADE: Color.RED | ✗ None: — | ✗ None: — |
| 390 | `REJECT_TRADE` | 2 | 0: REJECT_TRADE: Color.RED | ✗ None: — | ✗ None: — |
| 403 | `REJECT_TRADE` | 3 | 0: REJECT_TRADE: Color.BLUE | ✗ None: — | ✗ None: — |
| 408 | `ROLL` | 1 | 0: Roll dice | ✗ None: — | ✗ None: — |
| 410 | `MOVE_ROBBER` | 18 | 11: Move robber to (-2, 0, 2) | ✗ None: — | ✗ None: — |
| 411 | `STEAL` | 1 | 0: Steal from (C.RED, None) | ✗ None: — | ✗ None: — |
| 412 | `BUY_DEVELOPMENT_CARD` | 3 | 1: Buy development card | ✗ None: — | ✗ None: — |
| 413 | `END_TURN` | 2 | 0: End turn | ✗ None: — | ✗ None: — |
| 430 | `REJECT_TRADE` | 3 | 0: REJECT_TRADE: Color.RED | ✗ None: — | ✗ None: — |
| 436 | `REJECT_TRADE` | 3 | 0: REJECT_TRADE: Color.RED | ✗ None: — | ✗ None: — |
| 445 | `REJECT_TRADE` | 2 | 0: REJECT_TRADE: Color.RED | ✗ None: — | ✗ None: — |
| 449 | `REJECT_TRADE` | 3 | 0: REJECT_TRADE: Color.RED | ✗ None: — | ✗ None: — |
| 457 | `REJECT_TRADE` | 2 | 0: REJECT_TRADE: Color.BLUE | ✗ None: — | ✗ None: — |
| 462 | `ROLL` | 1 | 0: Roll dice | ✗ None: — | ✗ None: — |
| 478 | `BUILD_ROAD` | 14 | 10: Build road between SHEEP dice=11(2pip)/SHEEP dice=9(4pip) [6pips] and SHEEP dice=9(4pip)/WHEAT … | ✗ None: — | ✗ None: — |
| 480 | `BUILD_ROAD` | 14 | 7: Build road between SHEEP dice=11(2pip)/SHEEP dice=4(3pip)/SHEEP dice=9(4pip) [9pips] and SHEEP … | ✗ None: — | ✗ None: — |
| 481 | `END_TURN` | 2 | 0: End turn | ✗ None: — | ✗ None: — |
| 493 | `REJECT_TRADE` | 2 | 0: REJECT_TRADE: Color.BLUE | ✗ None: — | ✗ None: — |
| 503 | `ROLL` | 1 | 0: Roll dice | ✗ None: — | ✗ None: — |
| 504 | `MARITIME_TRADE` | 6 | 4: Trade 4 SHEEP for 1 BRICK | ✗ None: — | ✗ None: — |
| 505 | `BUILD_ROAD` | 14 | 2: Build road between SHEEP dice=11(2pip)/SHEEP dice=4(3pip)/WOOD dice=5(4pip) [9pips] and SHEEP d… | ✗ None: — | ✗ None: — |
| 506 | `END_TURN` | 2 | 0: End turn | ✗ None: — | ✗ None: — |
| 522 | `REJECT_TRADE` | 3 | 0: REJECT_TRADE: Color.BLUE | ✗ None: — | ✗ None: — |
| 525 | `REJECT_TRADE` | 3 | 0: REJECT_TRADE: Color.BLUE | ✗ None: — | ✗ None: — |
| 528 | `ROLL` | 1 | 0: Roll dice | ✗ None: — | ✗ None: — |
| 529 | `BUY_DEVELOPMENT_CARD` | 3 | 1: Buy development card | ✗ None: — | ✗ None: — |
| 530 | `END_TURN` | 2 | 0: End turn | ✗ None: — | ✗ None: — |
| 536 | `REJECT_TRADE` | 3 | 0: REJECT_TRADE: Color.ORANGE | ✗ None: — | ✗ None: — |
| 552 | `ROLL` | 6 | 5: Roll dice | ✗ None: — | ✗ None: — |
| 553 | `MONOPOLY_RESOURCE` | 22 | 0: Monopoly on WOOD | ✗ None: — | ✗ None: — |
| 556 | `BUILD_ROAD` | 13 | 3: Build road between WOOD dice=6(5pip)/WOOD dice=11(2pip)/WHEAT dice=12(1pip) [8pips] and WOOD di… | ✗ None: — | ✗ None: — |
| 557 | `END_TURN` | 2 | 0: End turn | ✗ None: — | ✗ None: — |

## Artifacts

- `plan.json`: immutable run inputs and packet settings
- `decision_manifest.jsonl`: every classified action for the human seat
- `responses.jsonl`: append-only raw model calls and prompts
- `comparisons.jsonl`: one normalized human/model comparison per exact choice
- `summary.json`: machine-readable aggregate metrics
