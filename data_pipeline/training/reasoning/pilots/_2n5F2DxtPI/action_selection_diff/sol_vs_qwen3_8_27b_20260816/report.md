# Full-game model vs human action-selection diff

Generated: 2026-08-16T23:58:51.168217+00:00
Game: `242781000`
Comparison parser: `replay-action-parser-v2`
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
| `openai/gpt-5.6-sol` | 52/109 (47.7%) | 52/109 (47.7%) | 20/20 (100.0%) | 32/89 (36.0%) | $2.297132 |
| `qwen/qwen3.8-27b` | 53/109 (48.6%) | 53/109 (48.6%) | 20/20 (100.0%) | 33/89 (37.1%) | $0.324242 |

## Output and usage health

| Model | Responses | Valid actions | Format warnings | API errors | Tokens in/out | Median / p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `openai/gpt-5.6-sol` | 109 | 109 | 0 | 0 | 150,175 / 46,847 | 7.53s / 26.13s |
| `qwen/qwen3.8-27b` | 109 | 109 | 23 | 0 | 158,066 / 83,616 | 20.30s / 92.56s |

## Model-to-model diff

- Same selection: 69/109 (63.3%)
- Both match human: 40
- Only `openai/gpt-5.6-sol` matches human: 12
- Only `qwen/qwen3.8-27b` matches human: 13
- Neither matches human: 44
- Both choose the same nonhuman alternative: 29

## Agreement by human action type

| Human action | `openai/gpt-5.6-sol` | `qwen/qwen3.8-27b` |
| --- | ---: | ---: |
| `ACCEPT_TRADE` | 1/1 (100.0%) | 0/1 (0.0%) |
| `BUILD_ROAD` | 2/10 (20.0%) | 1/10 (10.0%) |
| `BUILD_SETTLEMENT` | 3/4 (75.0%) | 2/4 (50.0%) |
| `BUY_DEVELOPMENT_CARD` | 7/7 (100.0%) | 5/7 (71.4%) |
| `END_TURN` | 1/19 (5.3%) | 2/19 (10.5%) |
| `MARITIME_TRADE` | 2/5 (40.0%) | 2/5 (40.0%) |
| `MONOPOLY_RESOURCE` | 0/1 (0.0%) | 0/1 (0.0%) |
| `MOVE_ROBBER` | 0/5 (0.0%) | 1/5 (20.0%) |
| `PLAY_KNIGHT_CARD` | 2/3 (66.7%) | 2/3 (66.7%) |
| `REJECT_TRADE` | 13/30 (43.3%) | 17/30 (56.7%) |
| `ROLL` | 17/19 (89.5%) | 17/19 (89.5%) |
| `STEAL` | 4/5 (80.0%) | 4/5 (80.0%) |

## Decisions with at least one model-human difference

| Replay row | Human type | Menu | Human | `openai/gpt-5.6-sol` | `qwen/qwen3.8-27b` |
| ---: | --- | ---: | --- | --- | --- |
| 2 | `BUILD_SETTLEMENT` | 50 | 7: Build settlement at SHEEP dice=9(4pip)/ORE dice=3(2pip)/BRICK dice=6(5pip) [11pips] \| Near: BL… | ✗ 20: Build settlement at WOOD dice=5(4pip)/BRICK dice=9(4pip)/WHEAT dice=10(3pip) [11pips] | ✗ 0: Build settlement at SHEEP dice=11(2pip)/WOOD dice=6(5pip)/WOOD dice=5(4pip) [11pips] |
| 3 | `BUILD_ROAD` | 3 | 1: Build road between SHEEP dice=9(4pip)/ORE dice=3(2pip)/BRICK dice=6(5pip) [11pips] and SHEEP di… | ✓ 1: Build road between SHEEP dice=9(4pip)/ORE dice=3(2pip)/BRICK dice=6(5pip) [11pips] and SHEEP di… | ✗ 2: Build road between SHEEP dice=9(4pip)/ORE dice=3(2pip)/BRICK dice=6(5pip) [11pips] and ORE dice… |
| 12 | `BUILD_SETTLEMENT` | 33 | 9: Build settlement at WOOD dice=6(5pip)/WOOD dice=5(4pip)/WHEAT dice=12(1pip) [10pips] \| Near: O… | ✓ 9: Build settlement at WOOD dice=6(5pip)/WOOD dice=5(4pip)/WHEAT dice=12(1pip) [10pips] \| Near: O… | ✗ 0: Build settlement at SHEEP dice=11(2pip)/WOOD dice=6(5pip)/WOOD dice=5(4pip) [11pips] |
| 13 | `BUILD_ROAD` | 3 | 0: Build road between SHEEP dice=11(2pip)/WOOD dice=6(5pip)/WOOD dice=5(4pip) [11pips] and WOOD di… | ✗ 2: Build road between WOOD dice=6(5pip)/WOOD dice=5(4pip)/WHEAT dice=12(1pip) [10pips] and WOOD di… | ✓ 0: Build road between SHEEP dice=11(2pip)/WOOD dice=6(5pip)/WOOD dice=5(4pip) [11pips] and WOOD di… |
| 19 | `END_TURN` | 2 | 0: End turn | ✗ 1: OFFER_TRADE: OFFER_TRADE to other players. Format: (WOOD_offer, BRICK_offer, SHEEP_offer, WHEAT… | ✗ 1: OFFER_TRADE: OFFER_TRADE to other players. Format: (WOOD_offer, BRICK_offer, SHEEP_offer, WHEAT… |
| 39 | `BUILD_ROAD` | 14 | 1: Build road between SHEEP dice=11(2pip)/WOOD dice=6(5pip)/WOOD dice=5(4pip) [11pips] and SHEEP d… | ✓ 1: Build road between SHEEP dice=11(2pip)/WOOD dice=6(5pip)/WOOD dice=5(4pip) [11pips] and SHEEP d… | ✗ 2: Build road between SHEEP dice=4(3pip)/SHEEP dice=9(4pip)/ORE dice=3(2pip) [9pips] and SHEEP dic… |
| 40 | `END_TURN` | 2 | 0: End turn | ✗ 1: OFFER_TRADE: OFFER_TRADE to other players. Format: (WOOD_offer, BRICK_offer, SHEEP_offer, WHEAT… | ✗ 1: OFFER_TRADE: OFFER_TRADE to other players. Format: (WOOD_offer, BRICK_offer, SHEEP_offer, WHEAT… |
| 50 | `END_TURN` | 2 | 0: End turn | ✗ 1: OFFER_TRADE: OFFER_TRADE to other players. Format: (WOOD_offer, BRICK_offer, SHEEP_offer, WHEAT… | ✗ 1: OFFER_TRADE: OFFER_TRADE to other players. Format: (WOOD_offer, BRICK_offer, SHEEP_offer, WHEAT… |
| 55 | `REJECT_TRADE` | 2 | 0: REJECT_TRADE: Color.ORANGE | ✗ 1: COUNTER_OFFER: COUNTER_OFFER to original trader. Format: (WOOD_offer, BRICK_offer, SHEEP_offer,… | ✓ 0: REJECT_TRADE: Color.ORANGE |
| 66 | `REJECT_TRADE` | 2 | 0: REJECT_TRADE: Color.BLUE | ✗ 1: COUNTER_OFFER: COUNTER_OFFER to original trader. Format: (WOOD_offer, BRICK_offer, SHEEP_offer,… | ✗ 1: COUNTER_OFFER: COUNTER_OFFER to original trader. Format: (WOOD_offer, BRICK_offer, SHEEP_offer,… |
| 73 | `ACCEPT_TRADE` | 3 | 1: ACCEPT_TRADE: Color.BLUE | ✓ 1: ACCEPT_TRADE: Color.BLUE | ✗ 0: REJECT_TRADE: Color.BLUE |
| 78 | `PLAY_KNIGHT_CARD` | 2 | 0: Play knight card | ✗ 1: Roll dice | ✗ 1: Roll dice |
| 79 | `MOVE_ROBBER` | 18 | 17: Move robber to (2, -1, -1) | ✗ 0: Move robber to (0, 0, 0) | ✗ 13: Move robber to (-1, 2, -1) |
| 80 | `STEAL` | 2 | 0: Steal from (C.BLUE, None) | ✗ 1: Steal from (C.ORANGE, None) | ✗ 1: Steal from (C.ORANGE, None) |
| 82 | `END_TURN` | 11 | 0: End turn | ✗ 9: Build road between SHEEP dice=11(2pip)/SHEEP dice=4(3pip)/WOOD dice=5(4pip) [9pips] and SHEEP d… | ✗ 1: Build road between SHEEP dice=4(3pip)/SHEEP dice=9(4pip)/ORE dice=3(2pip) [9pips] and SHEEP dic… |
| 93 | `REJECT_TRADE` | 2 | 0: REJECT_TRADE: Color.BLUE | ✗ 1: COUNTER_OFFER: COUNTER_OFFER to original trader. Format: (WOOD_offer, BRICK_offer, SHEEP_offer,… | ✗ 1: COUNTER_OFFER: COUNTER_OFFER to original trader. Format: (WOOD_offer, BRICK_offer, SHEEP_offer,… |
| 98 | `BUILD_ROAD` | 11 | 5: Build road between SHEEP dice=9(4pip)/BRICK dice=6(5pip)/WHEAT dice=2(1pip) [10pips] and SHEEP … | ✗ 9: Build road between SHEEP dice=11(2pip)/SHEEP dice=4(3pip)/WOOD dice=5(4pip) [9pips] and SHEEP d… | ✗ 1: Build road between SHEEP dice=4(3pip)/SHEEP dice=9(4pip)/ORE dice=3(2pip) [9pips] and SHEEP dic… |
| 99 | `END_TURN` | 2 | 0: End turn | ✗ 1: OFFER_TRADE: OFFER_TRADE to other players. Format: (WOOD_offer, BRICK_offer, SHEEP_offer, WHEAT… | ✗ 1: OFFER_TRADE: OFFER_TRADE to other players. Format: (WOOD_offer, BRICK_offer, SHEEP_offer, WHEAT… |
| 108 | `REJECT_TRADE` | 2 | 0: REJECT_TRADE: Color.BLUE | ✗ 1: COUNTER_OFFER: COUNTER_OFFER to original trader. Format: (WOOD_offer, BRICK_offer, SHEEP_offer,… | ✗ 1: COUNTER_OFFER: COUNTER_OFFER to original trader. Format: (WOOD_offer, BRICK_offer, SHEEP_offer,… |
| 110 | `REJECT_TRADE` | 2 | 0: REJECT_TRADE: Color.BLUE | ✓ 0: REJECT_TRADE: Color.BLUE | ✗ 1: COUNTER_OFFER: COUNTER_OFFER to original trader. Format: (WOOD_offer, BRICK_offer, SHEEP_offer,… |
| 119 | `REJECT_TRADE` | 2 | 0: REJECT_TRADE: Color.BLUE | ✗ 1: COUNTER_OFFER: COUNTER_OFFER to original trader. Format: (WOOD_offer, BRICK_offer, SHEEP_offer,… | ✗ 1: COUNTER_OFFER: COUNTER_OFFER to original trader. Format: (WOOD_offer, BRICK_offer, SHEEP_offer,… |
| 124 | `END_TURN` | 6 | 0: End turn | ✗ 5: OFFER_TRADE: OFFER_TRADE to other players. Format: (WOOD_offer, BRICK_offer, SHEEP_offer, WHEAT… | ✗ 2: Trade 4 SHEEP for 1 WOOD |
| 127 | `REJECT_TRADE` | 2 | 0: REJECT_TRADE: Color.ORANGE | ✗ 1: COUNTER_OFFER: COUNTER_OFFER to original trader. Format: (WOOD_offer, BRICK_offer, SHEEP_offer,… | ✗ 1: COUNTER_OFFER: COUNTER_OFFER to original trader. Format: (WOOD_offer, BRICK_offer, SHEEP_offer,… |
| 141 | `REJECT_TRADE` | 2 | 0: REJECT_TRADE: Color.BLUE | ✗ 1: COUNTER_OFFER: COUNTER_OFFER to original trader. Format: (WOOD_offer, BRICK_offer, SHEEP_offer,… | ✓ 0: REJECT_TRADE: Color.BLUE |
| 162 | `MARITIME_TRADE` | 6 | 1: Trade 4 SHEEP for 1 WHEAT | ✗ 4: Trade 4 SHEEP for 1 BRICK | ✓ 1: Trade 4 SHEEP for 1 WHEAT |
| 164 | `END_TURN` | 2 | 0: End turn | ✗ 1: OFFER_TRADE: OFFER_TRADE to other players. Format: (WOOD_offer, BRICK_offer, SHEEP_offer, WHEAT… | ✗ 1: OFFER_TRADE: OFFER_TRADE to other players. Format: (WOOD_offer, BRICK_offer, SHEEP_offer, WHEAT… |
| 170 | `REJECT_TRADE` | 2 | 0: REJECT_TRADE: Color.ORANGE | ✗ 1: COUNTER_OFFER: COUNTER_OFFER to original trader. Format: (WOOD_offer, BRICK_offer, SHEEP_offer,… | ✓ 0: REJECT_TRADE: Color.ORANGE |
| 180 | `REJECT_TRADE` | 2 | 0: REJECT_TRADE: Color.ORANGE | ✗ 1: COUNTER_OFFER: COUNTER_OFFER to original trader. Format: (WOOD_offer, BRICK_offer, SHEEP_offer,… | ✓ 0: REJECT_TRADE: Color.ORANGE |
| 192 | `REJECT_TRADE` | 3 | 0: REJECT_TRADE: Color.BLUE | ✗ 1: ACCEPT_TRADE: Color.BLUE | ✗ 1: ACCEPT_TRADE: Color.BLUE |
| 199 | `END_TURN` | 2 | 0: End turn | ✗ 1: OFFER_TRADE: OFFER_TRADE to other players. Format: (WOOD_offer, BRICK_offer, SHEEP_offer, WHEAT… | ✗ 1: OFFER_TRADE: OFFER_TRADE to other players. Format: (WOOD_offer, BRICK_offer, SHEEP_offer, WHEAT… |
| 214 | `MOVE_ROBBER` | 18 | 17: Move robber to (2, -1, -1) | ✗ 4: Move robber to (-1, 1, 0) | ✓ 17: Move robber to (2, -1, -1) |
| 216 | `END_TURN` | 2 | 0: End turn | ✗ 1: OFFER_TRADE: OFFER_TRADE to other players. Format: (WOOD_offer, BRICK_offer, SHEEP_offer, WHEAT… | ✗ 1: OFFER_TRADE: OFFER_TRADE to other players. Format: (WOOD_offer, BRICK_offer, SHEEP_offer, WHEAT… |
| 228 | `REJECT_TRADE` | 2 | 0: REJECT_TRADE: Color.BLUE | ✗ 1: COUNTER_OFFER: COUNTER_OFFER to original trader. Format: (WOOD_offer, BRICK_offer, SHEEP_offer,… | ✓ 0: REJECT_TRADE: Color.BLUE |
| 234 | `BUILD_ROAD` | 16 | 6: Build road between SHEEP dice=9(4pip)/ORE dice=3(2pip)/BRICK dice=6(5pip) [11pips] and ORE dice… | ✗ 11: Trade 4 WOOD for 1 WHEAT | ✗ 11: Trade 4 WOOD for 1 WHEAT |
| 235 | `BUILD_ROAD` | 17 | 4: Build road between ORE dice=3(2pip) [2pips] (WHEAT 2:1 port) and ORE dice=3(2pip)/BRICK dice=6(… | ✗ 12: Trade 4 WOOD for 1 WHEAT | ✗ 12: Trade 4 WOOD for 1 WHEAT |
| 236 | `MARITIME_TRADE` | 6 | 1: Trade 4 WOOD for 1 WHEAT | ✗ 5: OFFER_TRADE: OFFER_TRADE to other players. Format: (WOOD_offer, BRICK_offer, SHEEP_offer, WHEAT… | ✗ 3: Trade 4 WOOD for 1 BRICK |
| 238 | `END_TURN` | 2 | 0: End turn | ✗ 1: OFFER_TRADE: OFFER_TRADE to other players. Format: (WOOD_offer, BRICK_offer, SHEEP_offer, WHEAT… | ✓ 0: End turn |
| 245 | `REJECT_TRADE` | 2 | 0: REJECT_TRADE: Color.RED | ✗ 1: COUNTER_OFFER: COUNTER_OFFER to original trader. Format: (WOOD_offer, BRICK_offer, SHEEP_offer,… | ✓ 0: REJECT_TRADE: Color.RED |
| 249 | `REJECT_TRADE` | 3 | 0: REJECT_TRADE: Color.RED | ✗ 1: ACCEPT_TRADE: Color.RED | ✗ 1: ACCEPT_TRADE: Color.RED |
| 267 | `ROLL` | 2 | 1: Roll dice | ✗ 0: Play knight card | ✓ 1: Roll dice |
| 269 | `BUY_DEVELOPMENT_CARD` | 4 | 2: Buy development card | ✓ 2: Buy development card | ✗ 3: OFFER_TRADE: OFFER_TRADE to other players. Format: (WOOD_offer, BRICK_offer, SHEEP_offer, WHEAT… |
| 270 | `END_TURN` | 3 | 1: End turn | ✗ 0: Play knight card | ✗ 0: Play knight card |
| 293 | `REJECT_TRADE` | 2 | 0: REJECT_TRADE: Color.BLUE | ✓ 0: REJECT_TRADE: Color.BLUE | ✗ 1: COUNTER_OFFER: COUNTER_OFFER to original trader. Format: (WOOD_offer, BRICK_offer, SHEEP_offer,… |
| 311 | `ROLL` | 2 | 1: Roll dice | ✗ 0: Play knight card | ✗ 0: Play knight card |
| 313 | `MOVE_ROBBER` | 18 | 17: Move robber to (2, -1, -1) | ✗ 4: Move robber to (-1, 1, 0) | ✗ 0: Move robber to (0, 0, 0) |
| 315 | `END_TURN` | 2 | 0: End turn | ✗ 1: OFFER_TRADE: OFFER_TRADE to other players. Format: (WOOD_offer, BRICK_offer, SHEEP_offer, WHEAT… | ✗ 1: OFFER_TRADE: OFFER_TRADE to other players. Format: (WOOD_offer, BRICK_offer, SHEEP_offer, WHEAT… |
| 329 | `MOVE_ROBBER` | 18 | 7: Move robber to (2, -2, 0) | ✗ 12: Move robber to (-2, 2, 0) | ✗ 12: Move robber to (-2, 2, 0) |
| 347 | `MARITIME_TRADE` | 21 | 12: Trade 4 WOOD for 1 WHEAT | ✓ 12: Trade 4 WOOD for 1 WHEAT | ✗ 1: Build road between SHEEP dice=4(3pip)/SHEEP dice=9(4pip)/ORE dice=3(2pip) [9pips] and SHEEP dic… |
| 349 | `END_TURN` | 2 | 0: End turn | ✓ 0: End turn | ✗ 1: OFFER_TRADE: OFFER_TRADE to other players. Format: (WOOD_offer, BRICK_offer, SHEEP_offer, WHEAT… |
| 368 | `END_TURN` | 7 | 0: End turn | ✗ 2: Trade 4 SHEEP for 1 WOOD | ✗ 1: Trade 4 SHEEP for 1 WHEAT |
| 384 | `REJECT_TRADE` | 3 | 0: REJECT_TRADE: Color.RED | ✗ 1: ACCEPT_TRADE: Color.RED | ✗ 1: ACCEPT_TRADE: Color.RED |
| 410 | `MOVE_ROBBER` | 18 | 11: Move robber to (-2, 0, 2) | ✗ 4: Move robber to (-1, 1, 0) | ✗ 7: Move robber to (2, -2, 0) |
| 413 | `END_TURN` | 2 | 0: End turn | ✗ 1: OFFER_TRADE: OFFER_TRADE to other players. Format: (WOOD_offer, BRICK_offer, SHEEP_offer, WHEAT… | ✗ 1: OFFER_TRADE: OFFER_TRADE to other players. Format: (WOOD_offer, BRICK_offer, SHEEP_offer, WHEAT… |
| 430 | `REJECT_TRADE` | 3 | 0: REJECT_TRADE: Color.RED | ✗ 1: ACCEPT_TRADE: Color.RED | ✓ 0: REJECT_TRADE: Color.RED |
| 436 | `REJECT_TRADE` | 3 | 0: REJECT_TRADE: Color.RED | ✗ 1: ACCEPT_TRADE: Color.RED | ✗ 1: ACCEPT_TRADE: Color.RED |
| 449 | `REJECT_TRADE` | 3 | 0: REJECT_TRADE: Color.RED | ✓ 0: REJECT_TRADE: Color.RED | ✗ 1: ACCEPT_TRADE: Color.RED |
| 478 | `BUILD_ROAD` | 14 | 10: Build road between SHEEP dice=11(2pip)/SHEEP dice=9(4pip) [6pips] and SHEEP dice=9(4pip)/WHEAT … | ✗ 4: Build road between SHEEP dice=9(4pip)/BRICK dice=6(5pip)/WHEAT dice=2(1pip) [10pips] and BRICK … | ✗ 1: Build road between SHEEP dice=4(3pip)/SHEEP dice=9(4pip)/ORE dice=3(2pip) [9pips] and SHEEP dic… |
| 480 | `BUILD_ROAD` | 14 | 7: Build road between SHEEP dice=11(2pip)/SHEEP dice=4(3pip)/SHEEP dice=9(4pip) [9pips] and SHEEP … | ✗ 5: Build road between SHEEP dice=9(4pip)/BRICK dice=6(5pip)/WHEAT dice=2(1pip) [10pips] and BRICK … | ✗ 1: Build road between SHEEP dice=4(3pip)/SHEEP dice=9(4pip)/ORE dice=3(2pip) [9pips] and SHEEP dic… |
| 481 | `END_TURN` | 2 | 0: End turn | ✗ 1: OFFER_TRADE: OFFER_TRADE to other players. Format: (WOOD_offer, BRICK_offer, SHEEP_offer, WHEAT… | ✗ 1: OFFER_TRADE: OFFER_TRADE to other players. Format: (WOOD_offer, BRICK_offer, SHEEP_offer, WHEAT… |
| 504 | `MARITIME_TRADE` | 6 | 4: Trade 4 SHEEP for 1 BRICK | ✗ 1: Trade 4 SHEEP for 1 WHEAT | ✗ 3: Trade 4 SHEEP for 1 ORE |
| 505 | `BUILD_ROAD` | 14 | 2: Build road between SHEEP dice=11(2pip)/SHEEP dice=4(3pip)/WOOD dice=5(4pip) [9pips] and SHEEP d… | ✗ 7: Build road between SHEEP dice=11(2pip)/SHEEP dice=4(3pip)/SHEEP dice=9(4pip) [9pips] and SHEEP … | ✗ 1: Build road between SHEEP dice=4(3pip)/SHEEP dice=9(4pip)/ORE dice=3(2pip) [9pips] and SHEEP dic… |
| 506 | `END_TURN` | 2 | 0: End turn | ✗ 1: OFFER_TRADE: OFFER_TRADE to other players. Format: (WOOD_offer, BRICK_offer, SHEEP_offer, WHEAT… | ✗ 1: OFFER_TRADE: OFFER_TRADE to other players. Format: (WOOD_offer, BRICK_offer, SHEEP_offer, WHEAT… |
| 529 | `BUY_DEVELOPMENT_CARD` | 3 | 1: Buy development card | ✓ 1: Buy development card | ✗ 0: End turn |
| 530 | `END_TURN` | 2 | 0: End turn | ✗ 1: OFFER_TRADE: OFFER_TRADE to other players. Format: (WOOD_offer, BRICK_offer, SHEEP_offer, WHEAT… | ✗ 1: OFFER_TRADE: OFFER_TRADE to other players. Format: (WOOD_offer, BRICK_offer, SHEEP_offer, WHEAT… |
| 536 | `REJECT_TRADE` | 3 | 0: REJECT_TRADE: Color.ORANGE | ✗ 1: ACCEPT_TRADE: Color.ORANGE | ✗ 1: ACCEPT_TRADE: Color.ORANGE |
| 552 | `ROLL` | 6 | 5: Roll dice | ✓ 5: Roll dice | ✗ 4: Monopoly on ORE |
| 553 | `MONOPOLY_RESOURCE` | 22 | 0: Monopoly on WOOD | ✗ 1: Monopoly on BRICK | ✗ 3: Monopoly on WHEAT |
| 556 | `BUILD_ROAD` | 13 | 3: Build road between WOOD dice=6(5pip)/WOOD dice=11(2pip)/WHEAT dice=12(1pip) [8pips] and WOOD di… | ✗ 4: Build road between SHEEP dice=9(4pip)/BRICK dice=6(5pip)/WHEAT dice=2(1pip) [10pips] and BRICK … | ✗ 1: Build road between SHEEP dice=4(3pip)/SHEEP dice=9(4pip)/ORE dice=3(2pip) [9pips] and SHEEP dic… |
| 557 | `END_TURN` | 2 | 0: End turn | ✗ 1: OFFER_TRADE: OFFER_TRADE to other players. Format: (WOOD_offer, BRICK_offer, SHEEP_offer, WHEAT… | ✓ 0: End turn |

## Artifacts

- `plan.json`: immutable run inputs and packet settings
- `decision_manifest.jsonl`: every classified action for the human seat
- `responses.jsonl`: append-only raw model calls and prompts
- `comparisons.jsonl`: one normalized human/model comparison per exact choice
- `summary.json`: machine-readable aggregate metrics
