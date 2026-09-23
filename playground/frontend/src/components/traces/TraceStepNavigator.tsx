import type { TraceRequest } from '../../types';
import type { SavedLiveGameSummary } from '../controls/SavedLiveGamesBar';
import './TraceStepNavigator.css';

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
  origin_calls?: TraceModelCall[];
}

interface TraceStepNavigatorProps {
  game: SavedLiveGameSummary | null;
  detail: TraceStepDetail | null;
  activeGameId: string | null;
  busy: boolean;
  error: string | null;
  onNavigate: (gameId: string, stepIndex: number) => void;
  onLoadLatest: (gameId: string) => void;
  onNavigateReasoning?: (gameId: string, currentIndex: number, direction: -1 | 1, stepCount: number) => void;
  playerFilter?: string;
  navigationNotice?: string | null;
  onLatest?: () => void;
  isHistory?: boolean;
}

export default function TraceStepNavigator({
  game,
  detail,
  activeGameId,
  busy,
  error,
  onNavigate,
  onLoadLatest,
  onNavigateReasoning,
  playerFilter = 'all',
  navigationNotice,
  onLatest,
  isHistory = true,
}: TraceStepNavigatorProps) {
  const currentDetail = detail?.game_id === game?.game_id ? detail : null;
  const stepCount = Math.max(currentDetail?.step_count ?? 0, game?.step_count ?? 0);
  const latestIndex = stepCount - 1;
  const currentIndex = currentDetail?.step.step_index ?? latestIndex;

  if (!game) {
    return null;
  }

  if (stepCount === 0) {
    return error ? <div className="trace-step-error" role="alert">{error}</div> : null;
  }

  return (
    <section className="trace-step-navigator" aria-label="Saved trace step navigator">
      <label className="trace-step-picker">
        <span className="visually-hidden">Saved step</span>
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
                Step {stepIndex + 1} / {stepCount}{stepIndex === latestIndex ? ' · latest' : ''}
              </option>
            ),
          )}
        </select>
      </label>

      <div className="trace-step-actions">
        <button
          type="button"
          title={playerFilter === 'all' ? 'Previous checkpoint' : `Previous ${playerFilter} reasoning`}
          onClick={() => onNavigateReasoning
            ? onNavigateReasoning(game.game_id, currentIndex, -1, stepCount)
            : onNavigate(game.game_id, Math.max(0, currentIndex - 1))}
          disabled={busy || stepCount === 0 || currentIndex <= 0}
          aria-label="Previous saved step"
        >
          Previous
        </button>
        <button
          type="button"
          title={playerFilter === 'all' ? 'Next checkpoint' : `Next ${playerFilter} reasoning`}
          onClick={() => onNavigateReasoning
            ? onNavigateReasoning(game.game_id, currentIndex, 1, stepCount)
            : onNavigate(game.game_id, currentIndex + 1)}
          disabled={busy || stepCount === 0 || currentIndex >= latestIndex}
          aria-label="Next saved step"
        >
          Next
        </button>
        <button
          type="button"
          onClick={() => onLatest ? onLatest() : onNavigate(game.game_id, latestIndex)}
          disabled={
            busy
            || stepCount === 0
            || (!isHistory && currentDetail !== null && currentIndex === latestIndex)
          }
        >
          Latest
        </button>
        {game.game_id !== activeGameId && <button
          type="button"
          className="trace-resume-button"
          onClick={() => onLoadLatest(game.game_id)}
          disabled={busy}
        >
          Load latest
        </button>}
      </div>

      {playerFilter !== 'all' && <span role="status">{navigationNotice || `${playerFilter} reasoning`}</span>}
      {error && <span className="trace-step-error" role="alert">{error}</span>}
    </section>
  );
}
