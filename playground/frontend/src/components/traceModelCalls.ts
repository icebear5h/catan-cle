import type { TraceModelCall } from './TraceStepNavigator';

export function hasRecordedModelInference(call: TraceModelCall): boolean {
  return call.request !== null || call.response !== null;
}
