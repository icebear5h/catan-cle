import assert from 'node:assert/strict';
import test from 'node:test';

import {
  handDevCardEntries,
  handResourceEntries,
  hasHandContents,
} from '../src/playerHands.ts';

const HAND = {
  resources: { WOOD: 3, BRICK: 0, SHEEP: 1, WHEAT: 0, ORE: 2 },
  dev_cards: {
    total_in_hand: 3,
    in_hand: {
      KNIGHT: 2,
      ROAD_BUILDING: 0,
      YEAR_OF_PLENTY: 0,
      MONOPOLY: 0,
      VICTORY_POINT: 1,
    },
    played: { KNIGHT: 1 },
  },
};

test('lists held resources in canonical order and drops empties', () => {
  assert.deepEqual(handResourceEntries(HAND), [
    { resource: 'WOOD', count: 3 },
    { resource: 'SHEEP', count: 1 },
    { resource: 'ORE', count: 2 },
  ]);
});

test('lists unplayed development cards and drops empties', () => {
  assert.deepEqual(handDevCardEntries(HAND), [
    { card: 'KNIGHT', count: 2 },
    { card: 'VICTORY_POINT', count: 1 },
  ]);
});

test('keeps an unknown development card instead of dropping it', () => {
  const entries = handDevCardEntries({
    resources: {},
    dev_cards: { total_in_hand: 1, in_hand: { FUTURE_CARD: 1 }, played: {} },
  });

  assert.deepEqual(entries, [{ card: 'FUTURE_CARD', count: 1 }]);
});

test('renders nothing for a snapshot that carries no breakdown', () => {
  // Public projection only: the redacted shape reaching older checkpoints.
  const redacted = {
    resources: { TOTAL: 6 },
    dev_cards: { total_in_hand: 2, played: { KNIGHT: 0 } },
  };

  assert.deepEqual(handResourceEntries(redacted), []);
  assert.deepEqual(handDevCardEntries(redacted), []);
  assert.equal(hasHandContents(redacted), false);
  assert.equal(hasHandContents(null), false);
  assert.equal(hasHandContents(undefined), false);
});

test('reports contents for an exact hand', () => {
  assert.equal(hasHandContents(HAND), true);
});

test('reports an all-zero hand as empty, not as contents', () => {
  const empty = {
    resources: { WOOD: 0, BRICK: 0, SHEEP: 0, WHEAT: 0, ORE: 0 },
    dev_cards: {
      total_in_hand: 0,
      in_hand: { KNIGHT: 0, VICTORY_POINT: 0 },
      played: {},
    },
  };

  assert.equal(hasHandContents(empty), false);
});
