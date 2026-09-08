import type { AutoPlayStepResult } from './autoPlay';

export interface FailedLiveAttempt {
  context_id?: string;
  validation_error?: string | null;
  action_index?: number | null;
  final_response: string;
  native_reasoning?: string;
  native_reasoning_details?: unknown[];
  native_reasoning_chars?: number;
  reasoning_request?: Record<string, unknown>;
  reasoning_tokens?: number | null;
  model?: string | null;
  latency_ms?: number | null;
  finish_reason?: string | null;
  provider_native_finish_reason?: string | null;
  provider_response_id?: string | null;
  provider_request_id?: string | null;
  usage?: Record<string, unknown>;
}

export interface LiveStepFailure {
  error?: string;
  details?: string;
  player?: string;
  trace_game_id?: string | null;
  trace_failure_id?: string | null;
  attempt_count?: number;
  attempts: FailedLiveAttempt[];
  retryable?: boolean;
}

export interface LiveStepWarning {
  details: string;
  action_applied: true;
  retryable: false;
  trace_game_id?: string | null;
}

export function liveStepAutoPlayResult(
  response: { warning?: unknown; game_over?: unknown },
  state: { running: boolean; game?: { winning_color?: unknown } | null },
): AutoPlayStepResult {
  return {
    // The state/checkpoint is already applied; only automatic advancement stops.
    ok: response.warning == null,
    running: state.running,
    gameOver: response.game_over === true || state.game?.winning_color != null,
  };
}

export function liveStepErrorUpdate(
  payload: unknown,
  source: 'runtime' | 'checkpoint' = 'runtime',
): { message: string | null; failure: LiveStepFailure | null } | undefined {
  // A browse-only checkpoint must not clear or replace the live failure.
  if (source === 'checkpoint' || payload === undefined) return undefined;
  if (payload === null) return { message: null, failure: null };

  const record = typeof payload === 'object' && !Array.isArray(payload)
    ? payload as Record<string, unknown>
    : {};
  const message = [record.details, record.message, record.error].find(
    (value): value is string => typeof value === 'string' && Boolean(value.trim()),
  ) || 'Sandbox step failed';
  const hasAttempts = record.action_applied !== true
    && Array.isArray(record.attempts) && record.attempts.every(
      (attempt) => typeof attempt === 'object' && attempt !== null
        && typeof attempt.final_response === 'string',
    );

  return {
    message,
    failure: hasAttempts ? record as unknown as LiveStepFailure : null,
  };
}
