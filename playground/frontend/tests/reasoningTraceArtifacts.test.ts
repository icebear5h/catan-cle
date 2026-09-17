import assert from 'node:assert/strict';
import test from 'node:test';
import type { TraceRequest } from '../src/types.ts';
import type { TraceModelCall, TraceStepDetail } from '../src/components/TraceStepNavigator.tsx';
import { matchingReasoningActors, savedReasoningCalls } from '../src/components/traceModelCalls.ts';

import {
  traceGamePlanArtifact,
  visibleArtifactText,
} from '../src/reasoningTraceArtifacts.ts';

test('reasoning actors follow displayed current calls before continuation fallback', () => {
  const call = (actor: string | null, inference = true) => ({
    actor, request: null, response: inference ? { content: '{}' } : null,
  }) as TraceModelCall;
  const detail = {
    model_calls: [call('BLUE'), call('RED'), call('WHITE', false)],
    origin_calls: [call('GREEN')],
  } as TraceStepDetail;
  assert.deepEqual(matchingReasoningActors(detail), ['BLUE', 'RED']);
  assert.equal(savedReasoningCalls(detail).length, 2);
  detail.model_calls = [call('WHITE', false)];
  assert.deepEqual(matchingReasoningActors(detail), ['GREEN']);
  detail.origin_calls = [call(null)];
  assert.deepEqual(matchingReasoningActors(detail), []);
  detail.origin_calls = [];
  assert.deepEqual(savedReasoningCalls(detail), []);
});

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

const request: TraceRequest = {
  decision_id: 'decision-red',
  session_id: 'game:red',
  messages: [{ role: 'user', content: 'Private notes: retain ore.' }],
  components: [{
    id: 'environment.private_notes',
    channel: 'environment',
    template: 'Private notes: {{ value }}',
    value: 'retain ore',
    rendered: 'Private notes: retain ore',
  }],
  context_policy: 'fresh_notes',
  memory_revision: 0,
  input_next_sequence: 0,
  channel: 'action',
};

test('fresh notes distinguish omitted, null, empty, and replacement updates', () => {
  for (const [choice, update, text] of [
    [{}, 'keep', null],
    [{ notes_update: null }, 'keep', null],
    [{ notes_update: '' }, 'clear', ''],
    [{ notes_update: 'new notes' }, 'replace', 'new notes'],
    [{ notes_update: '   ' }, 'replace', '   '],
  ] as const) {
    assert.deepEqual(traceGamePlanArtifact('decision', choice, request), {
      show: true,
      text,
      notes: { inputText: 'retain ore', inputSource: 'environment.private_notes', update },
    });
  }
});

test('an explicit empty update never falls back to the historical game plan', () => {
  const artifact = traceGamePlanArtifact('decision', {
    notes_update: '', game_plan: 'obsolete plan',
  });
  assert.equal(artifact.text, '');
  assert.equal(artifact.notes?.update, 'clear');
  assert.equal(artifact.notes?.inputText, null);
});

test('nullable new fields do not relabel historical plans or communication', () => {
  assert.deepEqual(traceGamePlanArtifact('decision', {
    notes_update: null, game_plan: 'historical plan',
  }), { show: true, text: 'historical plan' });
  assert.deepEqual(traceGamePlanArtifact('communication', { notes_update: null }), {
    show: false, text: null,
  });
});

test('communication and silence can update private notes without a game plan', () => {
  for (const mode of ['say', 'silence']) {
    const artifact = traceGamePlanArtifact('communication', {
      mode, notes_update: 'track the trade', text: mode === 'say' ? 'Trade?' : '',
    }, { ...request, channel: 'talk' });
    assert.equal(artifact.show, true);
    assert.equal(artifact.notes?.update, 'replace');
    assert.equal(artifact.text, 'track the trade');
  }
});

test('legacy strategic memory is notes input only under a fresh context marker', () => {
  const legacy = {
    ...request,
    context_policy: null,
    components: request.components!.map((component) => ({
      ...component, id: 'environment.strategic_memory',
    })),
  };
  assert.equal(traceGamePlanArtifact('decision', {}, legacy).notes, undefined);
  assert.equal(traceGamePlanArtifact('decision', {}, {
    ...legacy, context_policy: 'fresh_notes',
  }).notes?.inputText, 'retain ore');
});

test('an explicitly empty private-notes component wins over strategic memory', () => {
  const artifact = traceGamePlanArtifact('decision', {}, {
    ...request,
    components: [
      { ...request.components![0], id: 'environment.strategic_memory', value: 'old' },
      { ...request.components![0], value: '' },
    ],
  });
  assert.equal(artifact.notes?.inputText, '');
  assert.equal(artifact.notes?.inputSource, 'environment.private_notes');
});

test('missing input is unknown, not inferred from output, messages, or rendered labels', () => {
  const artifact = traceGamePlanArtifact('decision', { notes_update: 'new notes' }, {
    ...request, components: [],
  });
  assert.equal(artifact.notes?.inputText, null);
  assert.equal(artifact.notes?.inputSource, null);
  assert.equal(artifact.text, 'new notes');
});

test('historical ID fallback still requires an environment-channel component', () => {
  const artifact = traceGamePlanArtifact('decision', {}, {
    ...request,
    components: [{ ...request.components![0], channel: 'system' }],
  });
  assert.equal(artifact.notes?.inputText, null);
});

for (const shape of ['object', 'pairs'] as const) {
  test(`extracts default environment.notes from ${shape} variables`, () => {
    const variables = shape === 'object'
      ? { notes: 'retain ore' }
      : [['notes', 'retain ore']] as Array<[string, string]>;
    const artifact = traceGamePlanArtifact('decision', {}, {
      ...request,
      components: [{
        ...request.components![0], id: 'environment.notes', variables,
        template: 'PRIVATE NOTES: {{ notes }}',
      }],
    });
    assert.equal(artifact.notes?.inputText, 'retain ore');
    assert.equal(artifact.notes?.inputSource, 'environment.notes');
  });

  for (const channel of ['environment', 'system'] as const) {
    for (const notes of ['retain ore', '']) {
      test(`extracts ${notes ? 'nonempty' : 'empty'} notes from renamed combined ${channel} ${shape} variables`, () => {
        const values = { color: 'RED', notes };
        const variables = shape === 'object' ? values : Object.entries(values);
        const artifact = traceGamePlanArtifact('communication', { notes_update: null }, {
          ...request,
          context_policy: null,
          components: [
            { ...request.components![0], value: 'obsolete historical fallback' },
            {
              id: `${channel}.renamed_context`, channel,
              template: '{{ color }}: {{ notes }}', value: `RED\n${notes}`,
              rendered: `RED: ${notes || 'No notes yet.'}`, variables,
            },
          ],
        });
        assert.equal(artifact.show, true);
        assert.equal(artifact.notes?.inputText, notes);
        assert.equal(artifact.notes?.inputSource, `${channel}.renamed_context`);
        assert.equal(artifact.notes?.update, 'keep');
      });
    }
  }

  test(`does not infer notes from environment.notes without a notes variable (${shape})`, () => {
    const variables = shape === 'object' ? { color: 'RED' } : [['color', 'RED']] as Array<[string, string]>;
    const artifact = traceGamePlanArtifact('decision', {}, {
      ...request,
      components: [{ ...request.components![0], id: 'environment.notes', variables }],
    });
    assert.equal(artifact.notes?.inputText, null);
    assert.equal(artifact.notes?.inputSource, null);
  });
}

test('unparsed choices and malformed notes are not presented as keep or clear', () => {
  assert.equal(traceGamePlanArtifact('decision', null, request).notes?.update, 'unavailable');
  for (const notes_update of [0, false, [], {}]) {
    assert.equal(traceGamePlanArtifact('decision', { notes_update }, request).notes?.update, 'invalid');
  }
});
