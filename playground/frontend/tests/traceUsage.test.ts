import assert from 'node:assert/strict';
import test from 'node:test';
import { averageUsage, sumUsage } from '../src/traceUsage.ts';
import type { UsageCall } from '../src/traceUsage.ts';

const call = (call_index: number, usage: UsageCall['usage'], extra: Partial<UsageCall> = {}): UsageCall => ({
  step_index: 0, call_index, call_kind: 'decision', accepted: true, usage, ...extra,
});

test('canonical calls include rejected retries and speech; aliases/details never double count', () => {
  const retry = call(0, { prompt_tokens: 100, completion_tokens: 20 }, { accepted: false });
  const calls = [retry, retry,
    call(1, { prompt_tokens: 150, input_tokens: 150, completion_tokens: 50,
      output_tokens: 50, completion_tokens_details: { reasoning_tokens: 40 },
      prompt_tokens_details: { cached_tokens: 80 } }),
    call(2, { input_tokens: 30, output_tokens: 5 }, { call_kind: 'communication' }),
  ];
  assert.deepEqual(sumUsage(calls), {
    input: { total: 280, covered: 3 }, output: { total: 75, covered: 3 }, calls: 3, accepted: 2,
  });
});

test('missing, partial and invalid usage remain unknown; explicit zero is known', () => {
  assert.equal(sumUsage([]).input.total, null);
  const usage = sumUsage([call(0, null), call(1, { total_tokens: 99 }),
    call(2, { prompt_tokens: -1, completion_tokens: '20' }),
    call(3, { input_tokens: 0, output_tokens: Number.NaN })]);
  assert.deepEqual(usage.input, { total: 0, covered: 1 });
  assert.deepEqual(usage.output, { total: null, covered: 0 });
});

test('averages use direction-specific covered steps, excluding failure-only batches', () => {
  const game = { game_id: 'g', step_count: 4, calls: [
    call(0, { input_tokens: 100, output_tokens: 20 }),
    call(1, null),
    call(0, { prompt_tokens: 300 }, { step_index: 1 }),
    call(0, {}, { step_index: 2 }),
  ], failure_calls: [call(0, { input_tokens: 9000 }, { failure_id: 'f' })] };
  const average = averageUsage(game);
  assert.deepEqual(average.input, { total: 200, covered: 2 });
  assert.deepEqual(average.output, { total: 20, covered: 1 });
  assert.equal(average.coverage.calls, 4);
  assert.equal(sumUsage(game.failure_calls).input.total, 9000);
});

test('deterministic continuation checkpoints do not manufacture calls or usage', () => {
  const game = { game_id: 'batch', step_count: 3, calls: [
    call(0, { prompt_tokens: 120, completion_tokens: 35 }),
  ], failure_calls: [] };
  assert.equal(sumUsage(game.calls).calls, 1);
  assert.equal(sumUsage(game.calls).input.total, 120);
  // Two engine-only checkpoints have no canonical call rows, not unknown calls.
  assert.deepEqual(averageUsage(game).input, { total: 120, covered: 1 });
});
