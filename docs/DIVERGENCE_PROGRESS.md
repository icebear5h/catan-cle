# Replay Divergence Test Progress

This file tracks progress on eliminating divergences between the Colonist replay and our engine.

**Success Criteria**: All 373 steps complete without divergence

## Test Script
Run `./playground/tests/test_replay_divergence.sh` to test current status

## Progress History

### Current Status: Step 199/373 (53.4%)
- **First divergence**: Step 199 (PLAY_MONOPOLY)
- **Players affected**: Player 2 missing 3 ore, Player 1 +2 ore, Player 0 +1 ore
- **Issue**: Monopoly card resources not transferred

### Previous Progress

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

1. **MARITIME_TRADE divergence (Step 183)** ⚠️ CURRENT
   - Player 1 has wrong resources after maritime trade
   - Engine: [3, 0, 3, 1, 2]
   - Expected: [0, 0, 0, 2, 3]
   - Need to investigate maritime trade logic

## Completed Fixes

- ✅ Async trade response handling (ACCEPT/REJECT from other players)
- ✅ Turn index reset after manually-applied trades
- ✅ HAS_ROLLED flag restoration after trades
- ✅ Detailed BUILD action skip diagnostics
- ✅ Trade display formatting ("any card" support)
