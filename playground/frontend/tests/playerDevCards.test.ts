import assert from 'node:assert/strict';
import test from 'node:test';

import { hasDevCardBreakdown } from '../src/playerDevCards.ts';

test('treats live public development-card totals as redacted', () => {
  const publicCards = {
    total_in_hand: 1,
    played: {
      KNIGHT: 0,
      ROAD_BUILDING: 0,
    },
  };

  assert.equal(hasDevCardBreakdown(publicCards), false);
});

test('recognizes an authorized exact development-card breakdown', () => {
  const privateCards = {
    total_in_hand: 2,
    in_hand: {
      KNIGHT: 1,
      VICTORY_POINT: 1,
    },
    played: {
      KNIGHT: 0,
    },
  };

  assert.equal(hasDevCardBreakdown(privateCards), true);
});
