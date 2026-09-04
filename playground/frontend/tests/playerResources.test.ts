import assert from 'node:assert/strict';
import test from 'node:test';

import {
  hasResourceBreakdown,
  resourceHandSize,
} from '../src/playerResources.ts';

test('uses the public TOTAL counter emitted by live snapshots', () => {
  assert.equal(resourceHandSize({ TOTAL: 2 }), 2);
  assert.equal(resourceHandSize({ TOTAL: 3 }), 3);
});

test('sums named resources when an authorized breakdown is available', () => {
  const resources = {
    WOOD: 3,
    BRICK: 0,
    SHEEP: 1,
    WHEAT: 2,
    ORE: 0,
  };

  assert.equal(resourceHandSize(resources), 6);
  assert.equal(hasResourceBreakdown(resources), true);
  assert.equal(hasResourceBreakdown({ TOTAL: 6 }), false);
});
