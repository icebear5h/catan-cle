import assert from 'node:assert/strict';
import test from 'node:test';
import { deriveTableTalk } from '../src/tableTalk.ts';
import type { TableTalkSourceEntry } from '../src/tableTalk.ts';

const row = (extra: Partial<TableTalkSourceEntry> = {}): TableTalkSourceEntry => ({
  type: 'message', message: 'Leave T09 alone?', color: 'GREEN', ...extra,
});

test('speech rows keep the recorded game step; other log rows are skipped', () => {
  const entries = deriveTableTalk([
    { type: 'building', message: 'Built a settlement' },
    row({ step_index: 12, details: { sequence: 469, step_index: 12 } }),
    { type: 'dice', message: 'Rolled 3 + 4 = 7' },
    row({ color: 'BLUE', message: 'Fine.', step_index: 12, details: { sequence: 470, step_index: 12 } }),
  ]);

  assert.deepEqual(entries, [
    { sequence: 469, step_index: 12, player: 'GREEN', message: 'Leave T09 alone?', model: '' },
    { sequence: 470, step_index: 12, player: 'BLUE', message: 'Fine.', model: '' },
  ]);
});

test('unstamped rows fall back to the engine-event sequence, then to position', () => {
  const entries = deriveTableTalk([
    row({ details: { sequence: 469 } }),
    row({ details: null }),
    row({ details: { sequence: 'nope', step_index: -1 } }),
  ]);

  assert.deepEqual(entries.map((entry) => [entry.sequence, entry.step_index]), [
    [469, null], [1, null], [2, null],
  ]);
});

test('the step index is read from the row or its details, and an unknown speaker is labelled', () => {
  const [fromDetails, fromRow] = deriveTableTalk([
    row({ color: undefined, details: { sequence: 3, step_index: 0 } }),
    row({ step_index: 7, details: { sequence: 4 } }),
  ]);

  assert.deepEqual([fromDetails.step_index, fromDetails.player], [0, 'UNKNOWN']);
  assert.equal(fromRow.step_index, 7);
});
