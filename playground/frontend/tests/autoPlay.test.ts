import assert from 'node:assert/strict';
import test from 'node:test';

import { runAutoPlayLoop } from '../src/autoPlay.ts';

test('awaits each step before starting the next one', async () => {
  let activeSteps = 0;
  let maxActiveSteps = 0;
  let stepCount = 0;

  const reason = await runAutoPlayLoop({
    shouldContinue: () => true,
    step: async () => {
      activeSteps += 1;
      maxActiveSteps = Math.max(maxActiveSteps, activeSteps);
      stepCount += 1;
      await Promise.resolve();
      activeSteps -= 1;
      return {
        ok: true,
        running: true,
        gameOver: stepCount === 3,
      };
    },
    pause: async () => Promise.resolve(),
  });

  assert.equal(reason, 'game_over');
  assert.equal(stepCount, 3);
  assert.equal(maxActiveSteps, 1);
});

test('cancellation after a step prevents another request', async () => {
  let enabled = true;
  let stepCount = 0;
  let pauseCount = 0;

  const reason = await runAutoPlayLoop({
    shouldContinue: () => enabled,
    step: async () => {
      stepCount += 1;
      enabled = false;
      return { ok: true, running: true, gameOver: false };
    },
    pause: async () => {
      pauseCount += 1;
    },
  });

  assert.equal(reason, 'cancelled');
  assert.equal(stepCount, 1);
  assert.equal(pauseCount, 0);
});

test('stops immediately when a step fails or the game stops', async (context) => {
  await context.test('failed step', async () => {
    let stepCount = 0;
    const reason = await runAutoPlayLoop({
      shouldContinue: () => true,
      step: async () => {
        stepCount += 1;
        return { ok: false, running: true, gameOver: false };
      },
      pause: async () => Promise.resolve(),
    });

    assert.equal(reason, 'error');
    assert.equal(stepCount, 1);
  });

  await context.test('stopped game', async () => {
    let stepCount = 0;
    const reason = await runAutoPlayLoop({
      shouldContinue: () => true,
      step: async () => {
        stepCount += 1;
        return { ok: true, running: false, gameOver: false };
      },
      pause: async () => Promise.resolve(),
    });

    assert.equal(reason, 'stopped');
    assert.equal(stepCount, 1);
  });
});
