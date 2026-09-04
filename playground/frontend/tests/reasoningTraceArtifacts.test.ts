import assert from 'node:assert/strict';
import test from 'node:test';

import {
  traceGamePlanArtifact,
  visibleArtifactText,
} from '../src/reasoningTraceArtifacts.ts';

test('preserves a parsed decision game plan for trace display', () => {
  const gamePlan = 'Secure ore first.\nThen buy a development card.';

  assert.deepEqual(
    traceGamePlanArtifact('decision', { game_plan: gamePlan }),
    { show: true, text: gamePlan },
  );
});

test('shows an explicit missing state for decisions without a parsed plan', () => {
  assert.deepEqual(
    traceGamePlanArtifact('decision', { game_plan: '   ' }),
    { show: true, text: null },
  );
  assert.equal(visibleArtifactText(null), null);
});

test('does not imply that communication calls produce game plans', () => {
  assert.deepEqual(
    traceGamePlanArtifact('communication', { text: 'Trade?' }),
    { show: false, text: null },
  );
});
