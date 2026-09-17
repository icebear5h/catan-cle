import assert from 'node:assert/strict';
import test from 'node:test';

import { runAutoPlayLoop } from '../src/autoPlay.ts';
import {
  liveStepAutoPlayResult,
  liveStepErrorUpdate,
  liveStepFailureRetryable,
} from '../src/liveStepErrors.ts';
import type { LiveStepFailure, LiveStepWarning } from '../src/liveStepErrors.ts';

const failure: LiveStepFailure = {
  error: 'Model returned no valid action',
  details: 'Selected action is not legal. No gameplay action was applied. Press Step to retry.',
  player: 'BLACK',
  trace_game_id: 'live-game',
  trace_failure_id: 'failure-1',
  attempt_count: 2,
  retryable: true,
  attempts: [
    {
      context_id: 'black-decision',
      validation_error: 'Missing action index.',
      final_response: '<game_plan>FINAL_ONLY</game_plan>'.repeat(200),
      native_reasoning: 'NATIVE_ONLY',
      native_reasoning_details: [{ type: 'reasoning.text', text: 'NATIVE_DETAIL' }],
      reasoning_request: { effort: 'high', exclude: false },
      finish_reason: 'length',
      latency_ms: 40000,
      reasoning_tokens: 8000,
      usage: { completion_tokens: 8192 },
    },
    {
      context_id: 'black-decision',
      validation_error: 'Selected action is not legal.',
      action_index: 1,
      final_response: '<action>1</action>',
      native_reasoning: '',
      native_reasoning_details: [],
    },
  ],
};

test('banner contains only validation and retry guidance, not response diagnostics', () => {
  const update = liveStepErrorUpdate(failure);

  assert.equal(update?.message, failure.details);
  assert.doesNotMatch(update!.message!, /FINAL_ONLY|NATIVE_ONLY|NATIVE_DETAIL|8192|8000|latency|finish=/);
  assert.equal(update?.failure, failure);
  assert.ok(update!.failure!.attempts[0].final_response.length > 4000);
});

test('HTTP and WebSocket failures replace the same attempts without duplication', () => {
  const http = liveStepErrorUpdate(failure);
  const websocket = { last_live_step_error: JSON.parse(JSON.stringify(failure)) };
  let current = http;
  for (let delivery = 0; delivery < 2; delivery += 1) {
    current = liveStepErrorUpdate(websocket.last_live_step_error) ?? current;
  }

  assert.deepEqual(current, http);
  assert.equal(current?.failure?.attempts.length, 2);
});

test('checkpoint browsing cannot clear or replace the live failure', () => {
  const current = liveStepErrorUpdate(failure);
  for (const savedError of [null, undefined, { ...failure, player: 'GOLD' }]) {
    const update = liveStepErrorUpdate(savedError, 'checkpoint');
    assert.equal(update, undefined);
    assert.equal(update ?? current, current);
  }
});

test('runtime clear removes diagnostics while omitted state leaves them unchanged', () => {
  assert.deepEqual(liveStepErrorUpdate(null), { message: null, failure: null });
  assert.equal(liveStepErrorUpdate(undefined), undefined);
});

test('final output never supplies missing native reasoning, including legacy diagnostics', () => {
  const attempts = [
    { final_response: '', native_reasoning: 'native with no final answer' },
    { final_response: '<action>0</action>', native_reasoning: '', native_reasoning_details: [] },
    { final_response: 'legacy final', native_reasoning_chars: 1462 },
    { final_response: '', native_reasoning_details: [{ type: 'reasoning.text', text: 'native detail' }] },
  ];
  const update = liveStepErrorUpdate({ ...failure, attempts });

  assert.deepEqual(update?.failure?.attempts, attempts);
  assert.equal(update?.failure?.attempts[1].native_reasoning, '');
  assert.equal(update?.failure?.attempts[2].native_reasoning, undefined);
  assert.deepEqual(update?.failure?.attempts[3].native_reasoning_details, attempts[3].native_reasoning_details);
});

test('ordinary API errors remain readable without creating rejected attempts', () => {
  assert.deepEqual(liveStepErrorUpdate({ error: 'No live sandbox running' }), {
    message: 'No live sandbox running', failure: null,
  });
  assert.deepEqual(liveStepErrorUpdate('not an API object'), {
    message: 'Sandbox step failed', failure: null,
  });
  assert.equal(liveStepErrorUpdate({ ...failure, attempts: [null] })?.failure, null);
});

test('an applied-action warning is identical over HTTP and WS, never a rejected attempt', () => {
  const warning: LiveStepWarning = {
    details: 'Game action was applied, but post-action communication failed. Do not retry the applied action.',
    action_applied: true,
    retryable: false,
    trace_game_id: 'live-game',
  };
  const response = { status: 'ok', warning, state: { last_live_step_error: warning } };
  const expected = { message: warning.details, failure: null };

  assert.deepEqual(liveStepErrorUpdate(response.warning), expected);
  assert.deepEqual(liveStepErrorUpdate(response.state.last_live_step_error), expected);
  assert.deepEqual(liveStepErrorUpdate({ ...warning, attempts: failure.attempts }), expected);
  assert.equal(liveStepErrorUpdate(null, 'checkpoint'), undefined);
  assert.equal(liveStepErrorUpdate(warning, 'checkpoint'), undefined);
});

test('an applied warning is a retryable stop: the next Step advances rather than repeats', async () => {
  const state = { running: true, game: { winning_color: null } };
  const response = { status: 'ok', warning: { action_applied: true, retryable: false } };
  const result = liveStepAutoPlayResult(response, state);
  let steps = 0;
  let pauses = 0;
  const reason = await runAutoPlayLoop({
    shouldContinue: () => true,
    step: async () => { steps += 1; return result; },
    pause: async () => { pauses += 1; },
  });

  assert.deepEqual(result, { ok: false, retryable: true, running: true, gameOver: false });
  assert.equal(reason, 'error');
  assert.equal(steps, 1);
  assert.equal(pauses, 0);
  assert.equal(state.running, true);

  let retried = 0;
  const persistentReason = await runAutoPlayLoop({
    shouldContinue: () => retried < 3,
    step: async () => { retried += 1; return result; },
    pause: async () => { pauses += 1; },
    retryPause: async () => Promise.resolve(),
  });
  assert.equal(persistentReason, 'cancelled');
  assert.equal(retried, 3);
  assert.equal(pauses, 0);
  assert.deepEqual(liveStepAutoPlayResult({ warning: null }, state), {
    ok: true, retryable: false, running: true, gameOver: false,
  });
  assert.equal(liveStepAutoPlayResult({ game_over: true }, state).gameOver, true);
  assert.equal(liveStepAutoPlayResult({}, { running: false, game: { winning_color: 'RED' } }).gameOver, true);
});

test('only a failure the server could not checkpoint blocks an auto-play retry', () => {
  const persistenceFailed = {
    error: 'Applied game step could not be saved',
    details: 'Current state and failure diagnostics could not be saved. Do not retry until storage is repaired.',
    trace_game_id: 'live-game',
    checkpoint_saved: false,
    retryable: false,
  };
  assert.equal(liveStepFailureRetryable(persistenceFailed), false);
  assert.equal(liveStepFailureRetryable({ ...persistenceFailed, retryable: true }), true);
  // Checkpointed rejections, config errors and unhandled sandbox errors all retry.
  assert.equal(liveStepFailureRetryable({ ...failure, checkpoint_saved: true }), true);
  assert.equal(liveStepFailureRetryable({
    error: 'OpenRouter rejected the request', retryable: false, checkpoint_saved: true,
    trace_game_id: 'live-game', provider_status_code: 403,
  }), true);
  assert.equal(liveStepFailureRetryable({
    error: 'Sandbox step failed', details: 'ValueError. No gameplay action was applied.',
    retryable: false, checkpoint_saved: true, trace_game_id: 'live-game',
  }), true);
  // Without a trace store nothing is ever checkpointed, so that flag alone means nothing.
  assert.equal(liveStepFailureRetryable({ retryable: false, checkpoint_saved: false, trace_game_id: null }), true);
  assert.equal(liveStepFailureRetryable({ error: 'No live sandbox running' }), true);
  assert.equal(liveStepFailureRetryable('not an API object'), true);
  assert.equal(liveStepFailureRetryable(null), true);
});
