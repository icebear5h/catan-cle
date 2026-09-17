import assert from 'node:assert/strict';
import test from 'node:test';

import { autoPlayRetryDelayMs, runAutoPlayLoop } from '../src/autoPlay.ts';

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

test('retries a retryable failure without bound and resets the count after success', async () => {
  const outcomes = [false, false, false, true, false, true, false, false, true];
  let stepCount = 0;
  const retryPauses: number[] = [];
  let pauseCount = 0;

  const reason = await runAutoPlayLoop({
    shouldContinue: () => true,
    step: async () => {
      const ok = outcomes[stepCount];
      stepCount += 1;
      return ok
        ? { ok: true, running: true, gameOver: stepCount === outcomes.length }
        : { ok: false, running: false, gameOver: false, retryable: true };
    },
    pause: async () => { pauseCount += 1; },
    retryPause: async (failures) => { retryPauses.push(failures); },
  });

  assert.equal(reason, 'game_over');
  assert.equal(stepCount, outcomes.length);
  assert.deepEqual(retryPauses, [1, 2, 3, 1, 1, 2]);
  assert.equal(pauseCount, 2);
});

test('retry backoff grows from one second and caps at thirty seconds', () => {
  assert.deepEqual([1, 2, 3, 4, 5, 6, 7, 40, 1000].map(autoPlayRetryDelayMs),
    [1000, 2000, 4000, 8000, 16000, 30000, 30000, 30000, 30000]);
});

test('a non-retryable failure stops even when retries are enabled', async () => {
  let stepCount = 0;
  let retryPauses = 0;
  const reason = await runAutoPlayLoop({
    shouldContinue: () => true,
    step: async () => {
      stepCount += 1;
      return { ok: false, running: true, gameOver: false, retryable: false };
    },
    pause: async () => Promise.resolve(),
    retryPause: async () => { retryPauses += 1; },
  });

  assert.equal(reason, 'error');
  assert.equal(stepCount, 1);
  assert.equal(retryPauses, 0);
});

test('cancellation during a retry wait prevents the retried request', async () => {
  let enabled = true;
  let stepCount = 0;
  const reason = await runAutoPlayLoop({
    shouldContinue: () => enabled,
    step: async () => {
      stepCount += 1;
      return { ok: false, running: true, gameOver: false, retryable: true };
    },
    pause: async () => Promise.resolve(),
    retryPause: async () => { enabled = false; },
  });

  assert.equal(reason, 'cancelled');
  assert.equal(stepCount, 1);
});

test('a game-over result ends the loop even when the step also failed', async () => {
  const reason = await runAutoPlayLoop({
    shouldContinue: () => true,
    step: async () => ({ ok: false, running: true, gameOver: true, retryable: true }),
    pause: async () => Promise.resolve(),
    retryPause: async () => { throw new Error('must not retry a finished game'); },
  });
  assert.equal(reason, 'game_over');
});
