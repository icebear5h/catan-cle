import type { SavedLiveGameSummary } from './SavedLiveGamesBar';
import { hasRecordedModelInference } from './traceModelCalls';
import './TraceStepNavigator.css';

interface TraceMessage {
  role: string;
  content: string;
}

interface TraceRequest {
  decision_id: string;
  session_id: string;
  messages: TraceMessage[];
}

interface TraceResponse {
  content?: string;
  model?: string | null;
  usage?: Record<string, unknown>;
  latency_ms?: number | null;
  finish_reason?: string | null;
  native_reasoning?: string | null;
  native_reasoning_details?: unknown;
  reasoning_request?: Record<string, unknown>;
  provider_response_id?: string | null;
  provider_request_id?: string | null;
  provider_native_finish_reason?: string | null;
  provider_request_payload?: unknown;
  provider_response_payload?: unknown;
}

export interface TraceModelCall {
  step_index: number;
  call_index: number;
  call_kind: 'decision' | 'communication' | string;
  context_id: string;
  actor: string | null;
  accepted: boolean;
  validation_error: string | null;
  request: TraceRequest | null;
  response: TraceResponse | null;
  choice: Record<string, unknown> | null;
}

export interface TraceStepDetail {
  game_id: string;
  display_name: string | null;
  step_count: number;
  latest_step_index: number;
  step: {
    step_index: number;
    recorded_at: string;
    before_revision: number;
    after_revision: number;
    winner: string | null;
    result: Record<string, unknown>;
    public_state: unknown;
  };
  model_calls: TraceModelCall[];
}

interface TraceStepNavigatorProps {
  game: SavedLiveGameSummary | null;
  detail: TraceStepDetail | null;
  activeGameId: string | null;
  busy: boolean;
  error: string | null;
  onNavigate: (gameId: string, stepIndex: number) => void;
  onLoadLatest: (gameId: string) => void;
}

export default function TraceStepNavigator({
  game,
  detail,
  activeGameId,
  busy,
  error,
  onNavigate,
  onLoadLatest,
}: TraceStepNavigatorProps) {
  const currentDetail = detail?.game_id === game?.game_id ? detail : null;
  const stepCount = currentDetail?.step_count ?? game?.step_count ?? 0;
  const latestIndex = stepCount - 1;
  const currentIndex = currentDetail?.step.step_index ?? latestIndex;
  const modelCalls = (
    currentDetail?.model_calls.filter(hasRecordedModelInference) || []
  );

  if (!game) {
    return null;
  }

  return (
    <section className="trace-step-navigator" aria-label="Saved trace step navigator">
      <header className="trace-step-title">
        <span>Checkpoint</span>
        <strong>Browse only</strong>
      </header>

      <label className="trace-step-picker">
        <span>Saved step</span>
        <select
          value={stepCount === 0 ? '' : currentIndex}
          onChange={(event) => onNavigate(game.game_id, Number(event.target.value))}
          disabled={busy || stepCount === 0}
          aria-label="Saved checkpoint step"
        >
          {stepCount === 0 && <option value="">No checkpoints</option>}
          {Array.from({ length: stepCount }, (_, offset) => latestIndex - offset).map(
            (stepIndex) => (
              <option key={stepIndex} value={stepIndex}>
                Step {stepIndex + 1}{stepIndex === latestIndex ? ' — latest' : ''}
              </option>
            ),
          )}
        </select>
      </label>

      <div className="trace-step-actions">
        <button
          type="button"
          onClick={() => onNavigate(game.game_id, Math.max(0, currentIndex - 1))}
          disabled={busy || stepCount === 0 || currentIndex <= 0}
          aria-label="Previous saved step"
        >
          Previous
        </button>
        <button
          type="button"
          onClick={() => onNavigate(game.game_id, currentIndex + 1)}
          disabled={busy || stepCount === 0 || currentIndex >= latestIndex}
          aria-label="Next saved step"
        >
          Next
        </button>
        <button
          type="button"
          onClick={() => onNavigate(game.game_id, latestIndex)}
          disabled={
            busy
            || stepCount === 0
            || (currentDetail !== null && currentIndex === latestIndex)
          }
        >
          Latest
        </button>
        <button
          type="button"
          className="trace-resume-button"
          onClick={() => onLoadLatest(game.game_id)}
          disabled={busy}
        >
          {game.game_id === activeGameId ? 'Return live' : 'Load latest'}
        </button>
      </div>

      <div className="trace-step-summary" aria-live="polite">
        {busy && <span>Loading checkpoint…</span>}
        {!busy && currentDetail && (
          <>
            <span>
              Step {currentDetail.step.step_index + 1}/{currentDetail.step_count}
              {' · '}revision {currentDetail.step.before_revision} → {currentDetail.step.after_revision}
            </span>
            <span>{modelCalls.length} model calls with reasoning history</span>
          </>
        )}
        {!busy && !currentDetail && stepCount > 0 && (
          <span>Select a checkpoint to browse its board and traces.</span>
        )}
        {stepCount === 0 && <span>This game has no completed steps yet.</span>}
        {error && <span className="trace-step-error">{error}</span>}
      </div>
    </section>
  );
}
