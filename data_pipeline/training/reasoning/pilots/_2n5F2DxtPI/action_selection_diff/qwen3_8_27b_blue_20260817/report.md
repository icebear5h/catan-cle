# Full-game model vs human action-selection diff

Generated: 2026-08-17T21:16:56.691569+00:00
Game: `242781000`
Comparison parser: `replay-action-parser-v2`
Human: Colonist color 2 / engine BLUE
Policy calls: stateless; model actions were never executed
Replay stepping: `allow_lookahead=False`

## Decision coverage

- Parsed replay rows: 570
- Same-event robber/steal order canonicalizations: 9
- Human-seat records: 164
- Exact indexed choices: 111
- Forced exact choices: 22
- Nontrivial exact choices: 89
- Other classifications: `{"coarse": 24, "exact": 111, "lifecycle": 26, "unmappable": 3}`
- Replay semantic errors: 0

## Headline agreement

| Model | Valid / exact | All exact | Forced | Nontrivial | Cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| `qwen/qwen3.8-27b` | 50/111 (45.0%) | 50/111 (45.0%) | 22/22 (100.0%) | 28/89 (31.5%) | $0.417184 |

## Output and usage health

| Model | Responses | Valid actions | Format warnings | API errors | Tokens in/out | Median / p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen/qwen3.8-27b` | 111 | 111 | 35 | 0 | 164,201 / 113,004 | 21.34s / 96.71s |

## Retry and cost audit

- Two complete but pathological responses returned no action and were retried: replay rows 438 and 502.
- `responses.jsonl` preserves all 113 attempts for 111 unique decisions.
- The headline `$0.417184` is the cost of each decision's latest attempt, used for scoring and display.
- Actual provider spend across all attempts was `$0.44034495`: `$0.42599380` initial run plus `$0.01435115` retries.
- Machine-readable provenance: `retry_audit.json`.

## Replay UI setup-strategy refresh

- The four BLUE initial-placement traces shown in the replay UI use a separate setup-only prompt that treats both placements as one portfolio, requires primary/fallback 10-VP routes, strips pip totals from setup menus, and describes factual post-setup road targets.
- These qualitative UI overrides do **not** replace responses used by the action-diff metrics above.
- Prompt development preserved 38 action-selection attempts across `setup-strategy-v2` through `setup-strategy-v11`; selected overrides remain two v10 rows, one v9 row, and one v8 row.
- Seven successful Qwen rationale self-review attempts are preserved separately; one failed diagnostic repair did not persist provider usage.
- Known setup-prompt and repair cost was `$0.10406137`; combined known provider spend with the base action diff was `$0.54440632`, excluding that one unrecorded failed diagnostic call.
- Two road traces use explicitly labeled Qwen self-reviewed rationales while preserving their original action-response drafts. The two settlement drafts retain four visible factual caveats from `setup_strategy_quality.json`; no model-authored text is silently edited.
- Provenance: `setup_strategy_attempts.jsonl`, `setup_strategy_overrides.jsonl`, `setup_rationale_repair_attempts.jsonl`, `setup_strategy_quality.json`, and `setup_strategy_audit.json`.

## Agreement by human action type

| Human action | `qwen/qwen3.8-27b` |
| --- | ---: |
| `ACCEPT_TRADE` | 1/1 (100.0%) |
| `BUILD_CITY` | 2/2 (100.0%) |
| `BUILD_ROAD` | 1/7 (14.3%) |
| `BUILD_SETTLEMENT` | 4/6 (66.7%) |
| `BUY_DEVELOPMENT_CARD` | 0/7 (0.0%) |
| `END_TURN` | 4/19 (21.1%) |
| `MARITIME_TRADE` | 1/4 (25.0%) |
| `MOVE_ROBBER` | 0/7 (0.0%) |
| `PLAY_KNIGHT_CARD` | 1/5 (20.0%) |
| `PLAY_ROAD_BUILDING` | 1/1 (100.0%) |
| `REJECT_TRADE` | 13/24 (54.2%) |
| `ROLL` | 17/20 (85.0%) |
| `STEAL` | 5/7 (71.4%) |
| `YEAR_OF_PLENTY_RESOURCES` | 0/1 (0.0%) |

## Decisions with at least one model-human difference

| Replay row | Human type | Menu | Human | `qwen/qwen3.8-27b` |
| ---: | --- | ---: | --- | --- |
| 0 | `BUILD_SETTLEMENT` | 54 | 7: Build settlement at SHEEP dice=4(3pip)/BRICK dice=8(5pip)/WHEAT dice=10(3pip) [11pips] | ✗ 0: Build settlement at SHEEP dice=11(2pip)/WOOD dice=6(5pip)/WOOD dice=5(4pip) [11pips] |
| 1 | `BUILD_ROAD` | 3 | 1: Build road between SHEEP dice=4(3pip)/BRICK dice=8(5pip)/WHEAT dice=10(3pip) [11pips] and BRICK… | ✗ 2: Build road between SHEEP dice=4(3pip)/BRICK dice=8(5pip)/WHEAT dice=10(3pip) [11pips] and SHEEP… |
| 14 | `BUILD_SETTLEMENT` | 30 | 17: Build settlement at ORE dice=8(5pip)/ORE dice=4(3pip) [8pips] (3:1 port) | ✗ 0: Build settlement at SHEEP dice=11(2pip)/SHEEP dice=4(3pip)/WOOD dice=5(4pip) [9pips] |
| 15 | `BUILD_ROAD` | 3 | 0: Build road between ORE dice=8(5pip)/ORE dice=4(3pip) [8pips] (3:1 port) and ORE dice=8(5pip) [5… | ✗ 2: Build road between SHEEP dice=3(2pip)/ORE dice=8(5pip)/ORE dice=4(3pip) [10pips] and ORE dice=8… |
| 17 | `END_TURN` | 2 | 0: End turn | ✗ 1: OFFER_TRADE: OFFER_TRADE to other players. Format: (WOOD_offer, BRICK_offer, SHEEP_offer, WHEAT… |
| 22 | `REJECT_TRADE` | 2 | 0: REJECT_TRADE: Color.ORANGE | ✗ 1: COUNTER_OFFER: COUNTER_OFFER to original trader. Format: (WOOD_offer, BRICK_offer, SHEEP_offer,… |
| 35 | `MOVE_ROBBER` | 18 | 5: Move robber to (0, 1, -1) | ✗ 0: Move robber to (0, 0, 0) |
| 37 | `END_TURN` | 2 | 0: End turn | ✗ 1: OFFER_TRADE: OFFER_TRADE to other players. Format: (WOOD_offer, BRICK_offer, SHEEP_offer, WHEAT… |
| 47 | `END_TURN` | 2 | 0: End turn | ✗ 1: OFFER_TRADE: OFFER_TRADE to other players. Format: (WOOD_offer, BRICK_offer, SHEEP_offer, WHEAT… |
| 76 | `BUILD_ROAD` | 13 | 1: Build road between BRICK dice=8(5pip)/WHEAT dice=10(3pip) [8pips] and WHEAT dice=10(3pip) [3pip… | ✗ 4: Build road between SHEEP dice=4(3pip)/WOOD dice=5(4pip)/WHEAT dice=10(3pip) [10pips] and SHEEP … |
| 77 | `END_TURN` | 6 | 0: End turn | ✗ 3: Trade 3 ORE for 1 WOOD |
| 121 | `END_TURN` | 7 | 0: End turn | ✗ 3: Trade 3 ORE for 1 WOOD |
| 129 | `REJECT_TRADE` | 3 | 0: REJECT_TRADE: Color.ORANGE | ✗ 1: ACCEPT_TRADE: Color.ORANGE |
| 150 | `END_TURN` | 10 | 0: End turn | ✗ 9: CANCEL_TRADE: None |
| 196 | `END_TURN` | 13 | 0: End turn | ✗ 7: Build road between SHEEP dice=4(3pip)/BRICK dice=8(5pip)/WHEAT dice=10(3pip) [11pips] and SHEEP… |
| 208 | `BUY_DEVELOPMENT_CARD` | 7 | 1: Buy development card | ✗ 5: Trade 2 BRICK for 1 WOOD |
| 209 | `MARITIME_TRADE` | 6 | 3: Trade 2 BRICK for 1 WHEAT | ✗ 4: Trade 2 BRICK for 1 WOOD |
| 210 | `BUY_DEVELOPMENT_CARD` | 3 | 1: Buy development card | ✗ 2: OFFER_TRADE: OFFER_TRADE to other players. Format: (WOOD_offer, BRICK_offer, SHEEP_offer, WHEAT… |
| 223 | `PLAY_KNIGHT_CARD` | 17 | 15: Play knight card | ✗ 9: Year of Plenty: take WHEAT and WHEAT |
| 224 | `MOVE_ROBBER` | 18 | 11: Move robber to (-2, 0, 2) | ✗ 5: Move robber to (0, 1, -1) |
| 225 | `STEAL` | 2 | 1: Steal from (C.ORANGE, None) | ✗ 0: Steal from (C.RED, None) |
| 231 | `END_TURN` | 3 | 0: End turn | ✗ 1: OFFER_TRADE: OFFER_TRADE to other players. Format: (WOOD_offer, BRICK_offer, SHEEP_offer, WHEAT… |
| 248 | `REJECT_TRADE` | 2 | 0: REJECT_TRADE: Color.RED | ✗ 1: COUNTER_OFFER: COUNTER_OFFER to original trader. Format: (WOOD_offer, BRICK_offer, SHEEP_offer,… |
| 250 | `REJECT_TRADE` | 3 | 0: REJECT_TRADE: Color.RED | ✗ 1: ACCEPT_TRADE: Color.RED |
| 261 | `ROLL` | 16 | 15: Roll dice | ✗ 9: Year of Plenty: take WHEAT and WHEAT |
| 262 | `YEAR_OF_PLENTY_RESOURCES` | 25 | 9: Year of Plenty: take WHEAT and WHEAT | ✗ 3: Year of Plenty: take WOOD and BRICK |
| 264 | `BUY_DEVELOPMENT_CARD` | 13 | 3: Buy development card | ✗ 1: Upgrade to city at ORE dice=8(5pip)/ORE dice=4(3pip) [8pips] (3:1 port) |
| 265 | `BUY_DEVELOPMENT_CARD` | 7 | 1: Buy development card | ✗ 3: Trade 3 SHEEP for 1 WHEAT |
| 266 | `END_TURN` | 2 | 0: End turn | ✗ 1: OFFER_TRADE: OFFER_TRADE to other players. Format: (WOOD_offer, BRICK_offer, SHEEP_offer, WHEAT… |
| 284 | `ROLL` | 3 | 2: Roll dice | ✗ 0: Play knight card |
| 285 | `BUY_DEVELOPMENT_CARD` | 23 | 5: Buy development card | ✗ 3: Upgrade to city at ORE dice=8(5pip)/ORE dice=4(3pip) [8pips] (3:1 port) |
| 288 | `BUY_DEVELOPMENT_CARD` | 13 | 3: Buy development card | ✗ 8: Trade 2 BRICK for 1 WOOD |
| 289 | `PLAY_KNIGHT_CARD` | 9 | 0: Play knight card | ✗ 1: Play road building card |
| 290 | `MOVE_ROBBER` | 18 | 9: Move robber to (0, -2, 2) | ✗ 11: Move robber to (-2, 1, 1) |
| 291 | `STEAL` | 2 | 1: Steal from (C.BLACK, None) | ✗ 0: Steal from (C.RED, None) |
| 308 | `BUILD_ROAD` | 13 | 4: Build road between BRICK dice=8(5pip)/WHEAT dice=10(3pip) [8pips] and BRICK dice=8(5pip) [5pips… | ✗ 9: Trade 2 BRICK for 1 ORE |
| 324 | `MOVE_ROBBER` | 18 | 11: Move robber to (-2, 0, 2) | ✗ 0: Move robber to (0, 0, 0) |
| 357 | `PLAY_KNIGHT_CARD` | 3 | 0: Play knight card | ✗ 2: Roll dice |
| 358 | `MOVE_ROBBER` | 18 | 16: Move robber to (2, 0, -2) | ✗ 0: Move robber to (0, 0, 0) |
| 361 | `MARITIME_TRADE` | 17 | 10: Trade 2 BRICK for 1 SHEEP | ✗ 12: Trade 2 BRICK for 1 WOOD |
| 362 | `BUY_DEVELOPMENT_CARD` | 14 | 8: Buy development card | ✗ 1: Build road between BRICK dice=9(4pip)/WHEAT dice=10(3pip) [7pips] (BRICK 2:1 port) and WHEAT di… |
| 363 | `END_TURN` | 9 | 0: End turn | ✗ 8: OFFER_TRADE: OFFER_TRADE to other players. Format: (WOOD_offer, BRICK_offer, SHEEP_offer, WHEAT… |
| 385 | `REJECT_TRADE` | 3 | 0: REJECT_TRADE: Color.RED | ✗ 1: ACCEPT_TRADE: Color.RED |
| 398 | `PLAY_KNIGHT_CARD` | 3 | 0: Play knight card | ✗ 2: Roll dice |
| 399 | `MOVE_ROBBER` | 18 | 16: Move robber to (2, 0, -2) | ✗ 5: Move robber to (0, 1, -1) |
| 407 | `END_TURN` | 9 | 0: End turn | ✗ 8: OFFER_TRADE: OFFER_TRADE to other players. Format: (WOOD_offer, BRICK_offer, SHEEP_offer, WHEAT… |
| 428 | `REJECT_TRADE` | 2 | 0: REJECT_TRADE: Color.RED | ✗ 1: COUNTER_OFFER: COUNTER_OFFER to original trader. Format: (WOOD_offer, BRICK_offer, SHEEP_offer,… |
| 438 | `REJECT_TRADE` | 3 | 0: REJECT_TRADE: Color.RED | ✗ 2: COUNTER_OFFER: COUNTER_OFFER to original trader. Format: (WOOD_offer, BRICK_offer, SHEEP_offer,… |
| 444 | `REJECT_TRADE` | 2 | 0: REJECT_TRADE: Color.RED | ✗ 1: COUNTER_OFFER: COUNTER_OFFER to original trader. Format: (WOOD_offer, BRICK_offer, SHEEP_offer,… |
| 453 | `ROLL` | 2 | 1: Roll dice | ✗ 0: Play road building card |
| 454 | `MOVE_ROBBER` | 18 | 17: Move robber to (2, 0, -2) | ✗ 1: Move robber to (1, -1, 0) |
| 461 | `END_TURN` | 10 | 1: End turn | ✗ 9: OFFER_TRADE: OFFER_TRADE to other players. Format: (WOOD_offer, BRICK_offer, SHEEP_offer, WHEAT… |
| 466 | `REJECT_TRADE` | 3 | 0: REJECT_TRADE: Color.BLACK | ✗ 2: COUNTER_OFFER: COUNTER_OFFER to original trader. Format: (WOOD_offer, BRICK_offer, SHEEP_offer,… |
| 468 | `REJECT_TRADE` | 2 | 0: REJECT_TRADE: Color.BLACK | ✗ 1: COUNTER_OFFER: COUNTER_OFFER to original trader. Format: (WOOD_offer, BRICK_offer, SHEEP_offer,… |
| 475 | `REJECT_TRADE` | 2 | 0: REJECT_TRADE: Color.BLACK | ✗ 1: COUNTER_OFFER: COUNTER_OFFER to original trader. Format: (WOOD_offer, BRICK_offer, SHEEP_offer,… |
| 497 | `MARITIME_TRADE` | 18 | 12: Trade 3 SHEEP for 1 WHEAT | ✗ 13: Trade 3 ORE for 1 WHEAT |
| 500 | `BUILD_ROAD` | 8 | 1: Build road between SHEEP dice=4(3pip)/BRICK dice=8(5pip)/ORE dice=3(2pip) [10pips] and BRICK di… | ✗ 3: Build road between SHEEP dice=4(3pip)/WOOD dice=5(4pip)/WHEAT dice=10(3pip) [10pips] and SHEEP … |
| 502 | `END_TURN` | 6 | 0: End turn | ✗ 1: Trade 3 ORE for 1 BRICK |
| 526 | `END_TURN` | 7 | 0: End turn | ✗ 5: OFFER_TRADE: OFFER_TRADE to other players. Format: (WOOD_offer, BRICK_offer, SHEEP_offer, WHEAT… |
| 550 | `BUILD_ROAD` | 15 | 3: Build road between ORE dice=8(5pip) [5pips] and ORE dice=8(5pip) [5pips] | ✗ 4: Build road between SHEEP dice=4(3pip)/WOOD dice=5(4pip)/WHEAT dice=10(3pip) [10pips] and SHEEP … |
| 551 | `END_TURN` | 6 | 0: End turn | ✗ 1: Trade 3 ORE for 1 WHEAT |

## Artifacts

- `plan.json`: immutable run inputs and packet settings
- `decision_manifest.jsonl`: every classified action for the human seat
- `responses.jsonl`: append-only raw model calls and prompts
- `comparisons.jsonl`: one normalized human/model comparison per exact choice
- `summary.json`: machine-readable aggregate metrics
