import type { TraceModelCall, TraceStepDetail } from './TraceStepNavigator';

export function hasRecordedModelInference(call: TraceModelCall): boolean {
  return call.request !== null || call.response !== null;
}

export function savedReasoningCalls(detail: TraceStepDetail): TraceModelCall[] {
  const calls = detail.model_calls.filter(hasRecordedModelInference);
  return calls.length > 0
    ? calls
    : (detail.origin_calls || []).filter(hasRecordedModelInference);
}

export function matchingReasoningActors(detail: TraceStepDetail): string[] {
  return savedReasoningCalls(detail)
    .map((call) => call.actor)
    .filter((actor): actor is string => typeof actor === 'string');
}
