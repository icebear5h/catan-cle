# Replay Divergence Test Progress

This file tracks progress on eliminating divergences between the Colonist replay and our engine.

**Success Criteria**: All local raw Colonist replays complete without resource divergence or fatal semantic skips. Replay-mode force paths must be explicit and audited; observed idempotent Colonist state echoes should be info-level no-ops.

## Test Script
Run the gated consolidated local-corpus audit:

```bash
RUN_LOCAL_REPLAY_CORPUS=1 EXPECTED_LOCAL_REPLAY_COUNT=66 \
  .venv/bin/python -m pytest -q -s \
  tests/test_replay_trading.py::test_all_local_replays_have_exact_resources_and_trade_lifecycle
```

`./playground/tests/test_replay_divergence.sh` remains a single-game legacy smoke check for `194335024`; it is not the corpus audit.

## Progress History

### Current Status: 66/66 base-game replays resource- and trade-lifecycle-aligned ✅
- **First fatal divergence**: None in the supported local base-game replay set
- **Last verified**: 2026-08-10 with the consolidated Flask replay audit
- **Corpus**: The original 18 games plus 48 newly acquired four-player base-game replays from distinct indexed expert players.
- **Coverage**: 31,506 parsed replay actions and 15,460 trade lifecycle actions: 2,583 offers, 617 counteroffers, 621 accepts, 6,668 rejects, 165 response clears, 3,200 closures, 523 player-trade confirmations, and 1,083 maritime trades.
- **Invariants checked after every action**: every authoritative Colonist hand snapshot matches; no player hand is negative; bank plus player cards total 19 for each resource; and the exact replay ledger matches raw Colonist active trade IDs/responses at every raw-event boundary.
- **Unit verification**: 54 tests pass with the corpus test gated off; the gated 66-game audit passes independently.
- **Audit note**: Warning-level replay force paths still identify exact Colonist overrides; zero error-level semantic issues occurred in the corpus run.

#### 2026-08-10 - Expanded expert replay corpus ✅
- **Acquired**: 50 additional account-accessible replay payloads through a real Chrome CDP session. Forty-eight are compatible four-player base games and were promoted; one two-player game and one Cities & Knights game remain quarantined outside `raw_replays`.
- **Source ratings**: The 49 four-player downloads represented 49 distinct indexed top-100 players with leaderboard ratings 1834–1989 (median 1869, mean 1880). These are index-time ratings for the selected player, not historical whole-lobby Elo.
- **Hardened**: The Playwright scraper can require both player count and Colonist `modeSetting`, reports incompatible payloads separately, and never counts them as successful target games.
- **Rate safety**: Acquisition stops immediately on the first HTTP 429, records `Retry-After`, never automatically retries a rate-limited request, and defaults to at least 40 seconds between attempts. This expansion stopped downloading after the first 429 and proceeded offline with the accepted files.
- **Result**: 66/66 supported base-game replays pass the consolidated resource, conservation, semantic, and exact trade-lifecycle audit.

#### 2026-08-10 - Transactional replay and exact trade lifecycle ✅
- **Fixed**: Every parsed replay step now captures a transactional checkpoint. Undo restores hands, offers/counters, turn indices, legal actions, engine history, replay logs, semantic issues, final-sync state, pending dev-card state, and the exact trade ledger.
- **Fixed**: Exact `CONFIRM_TRADE` resource deltas are recorded as replay actions, so undo and re-step cannot double-apply a trade.
- **Fixed**: Colonist offer closures are parsed exactly once, response transitions replace prior accept/reject state (including response clears), and simultaneous offers from one creator remain distinct by Colonist `trade_id` with counter-parent links.
- **Fixed**: Raw replay events are no longer mutated while merging delta-encoded trade responses.
- **Fixed**: `/api/replay-goto-fast` delegates to authoritative sequential reconstruction; backward jumps correctly resume a non-final replay.
- **Fixed**: Canonical engine confirm/cancel operations preserve unrelated offers/counters, revalidate both hands before transfer, and synchronize deprecated single-trade fields from remaining active offers.
- **Fixed**: Counter-only replay state no longer broadcasts a fake zero-resource legacy offer; the full exact ledger is included in replay WebSocket state.

#### 2026-05-13 - Force boundary and robber no-op audit ✅
- **Fixed**: `Game.execute(..., force=True)` and `apply_action(..., force=True)` now reject forced actions that would otherwise use engine randomness (`ROLL`, `DISCARD`, `BUY_DEVELOPMENT_CARD`, `STEAL`, missing robber coordinates, and missing dev-card choices).
- **Fixed**: Unforced engine simulation can still use random defaults, but random outcomes are now returned and logged as resolved actions.
- **Fixed**: Colonist `MOVE_ROBBER` rows whose target coordinate already equals the engine robber coordinate are treated as `already_satisfied` info events, not forced robber moves. In the current corpus this accounts for all 56 robber-location rows that previously looked forced.
- **Result**: 13/13 local raw replays complete with zero fatal semantic errors and zero resource divergence. Current audit counts: 416 warnings (`403 forced_replay_overlay`, `12 unpaired_dev_card_announcement`, `1 forced_final_state_sync`) and 56 info-level `observed_replay_state` robber no-ops.

#### 2026-05-13 - Strict semantic replay audit ✅
- **Fixed**: Trade overlay rows (`OFFER_TRADE`, `COUNTER_OFFER`, `ACCEPT_TRADE`, `REJECT_TRADE`) are forced into replay overlay state when no legal engine action matches, rather than being skipped.
- **Fixed**: `PLAY_MONOPOLY` / `PLAY_YEAR_OF_PLENTY` announcement rows are deferred to their resource-selection rows, with pre-event resource snapshots so the announcement row itself does not create false divergence.
- **Fixed**: Forced robber moves use Colonist tile coordinates from `initialState`, stale `STEAL` prompts are cleared when Colonist advances without a steal row, and unchanged robber-location echoes are not executed.
- **Fixed**: Final Colonist VP categories are weighted correctly (`city`, `largest army`, and `longest road`) and final VP/largest-army/longest-road fields are synced into replay state with warnings if the engine-derived values differ.
- **Result**: 13/13 local raw replays complete through `/api/replay-goto-divergence` with zero fatal semantic errors and zero resource divergence.

### Previous Progress

#### 2026-05-12 - Broader raw replay hardening ✅
- **Fixed**: Player-trade confirmations always use the exact Colonist type-115 resource tuple instead of the engine's active-trade object.
- **Fixed**: Robber moves with tile coordinates now require exact coordinate matches, avoiding duplicate resource-number hex mixups.
- **Fixed**: Steal actions infer the stolen resource from before/after resource-card deltas and can execute exact thief/victim/resource tuples when engine topology does not offer the same victim.
- **Fixed**: Async trade responses apply directly to active trade state even while the turn prompt is roll/build/robber/steal, matching Colonist's trade overlay behavior.
- **Fixed**: Wrapped discard order and forced Colonist road records keep resource ledgers aligned when engine prompt/topology state is behind the scraped replay.
- **Improvement**: 3/7 raw replays → 13/13 raw replays with no resource divergence.
- **Status**: Local raw replay set fully resource-aligned.

#### 2026-05-12 - Full replay alignment ✅
- **Fixed**: Replay actor/decode and deterministic execution gaps
  - ROLL actions now recover the acting player from Colonist game-log entries when `currentState` omits `currentTurnPlayerColor`
  - MOVE_ROBBER actions now retain the current turn player from `currentState`
  - Turn-owned replay actions sync the engine turn owner before matching/execution
  - Required actions already executed by lookahead are skipped instead of falling back to unrelated playable actions
  - Multi-resource maritime trades execute directly instead of partially matching a single-resource playable trade
  - DISCARD actions use Colonist's exact discarded cards instead of random discard
- **Improvement**: Step 183 → Step 373 (+190 steps)
- **Status**: Reference replay fully resource-aligned

#### 2025-12-27 - Step 199/373 ✅ Direct action execution
- **Fixed**: Execute actions directly when they don't match playable_actions
  - MARITIME_TRADE: All maritime trades (single and multi-resource) now execute directly
  - BUY_DEVELOPMENT_CARD: Dev card purchases execute without validation
  - BUILD actions: Roads, settlements, cities execute with inferred player/location
- **Improvement**: Step 117 → Step 199 (+82 steps, +22%)
- **Next issue**: PLAY_MONOPOLY/MONOPOLY_RESOURCE not executing properly

#### 2025-12-26 - Step 183/373 ✅ Multi-trade fix
- **Fixed**: Turn index bug after manually-applied trades
  - Problem: After trade, turn was advancing to other players instead of staying with creator
  - Fix: Reset `current_player_index` and `current_turn_index` back to trade creator
  - Fix: Restore `HAS_ROLLED = True` for trade creator
- **Improvement**: Step 98 → Step 183 (+85 steps, +23%)
- **Next issue**: MARITIME_TRADE divergence at step 183

#### Earlier - Step 98/373
- **Issue**: BUILD_SETTLEMENT skipped after trade
- **Cause**: Turn index sent to wrong player after ACCEPT_TRADE

## Known Issues to Fix

1. **No known fatal replay divergence in local raw replays** ✅
   - Remaining audit warnings are explicit replay-mode force paths, not silent skips.

## Completed Fixes

- ✅ Async trade response handling (ACCEPT/REJECT from other players)
- ✅ Turn index reset after manually-applied trades
- ✅ HAS_ROLLED flag restoration after trades
- ✅ Detailed BUILD action skip diagnostics
- ✅ Trade display formatting ("any card" support)
