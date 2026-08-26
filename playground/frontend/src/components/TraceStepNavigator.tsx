import type { SavedLiveGameSummary } from './SavedLiveGamesBar';
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

function json(value: unknown): string {
  return JSON.stringify(value, null, 2);
}

function visibleText(value: unknown): string | null {
  return typeof value === 'string' && value.trim() ? value : null;
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
  const stepCount = game?.step_count || 0;
  const latestIndex = stepCount - 1;
  const currentDetail = detail?.game_id === game?.game_id ? detail : null;
  const currentIndex = currentDetail?.step.step_index ?? latestIndex;
  const rationaleCalls = currentDetail?.model_calls || [];

  if (!game) {
    return null;
  }

  return (
    <section className="trace-step-navigator" aria-label="Saved trace step navigator">
      <div className="trace-step-toolbar">
        <div className="trace-step-title">
          <span>Checkpoint navigator</span>
          <strong>Browse only</strong>
        </div>

        <button
          type="button"
          onClick={() => onNavigate(game.game_id, Math.max(0, currentIndex - 1))}
          disabled={busy || stepCount === 0 || currentIndex <= 0}
          aria-label="Previous saved step"
        >
          Previous
        </button>

        <label>
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
          {game.game_id === activeGameId ? 'Return to live' : 'Load latest'}
        </button>

        <div className="trace-step-summary" aria-live="polite">
          {busy && <span>Loading checkpoint…</span>}
          {!busy && currentDetail && (
            <>
              <span>
                Step {currentDetail.step.step_index + 1}/{currentDetail.step_count}
              </span>
              <span>
                revision {currentDetail.step.before_revision} → {currentDetail.step.after_revision}
              </span>
              <span>{rationaleCalls.length} model calls</span>
            </>
          )}
          {!busy && !currentDetail && stepCount > 0 && (
            <span>Select a checkpoint to browse its board and traces.</span>
          )}
          {stepCount === 0 && <span>This game has no completed steps yet.</span>}
          {error && <span className="trace-step-error">{error}</span>}
        </div>
      </div>

      {currentDetail && (
        <div className="trace-call-strip">
          {rationaleCalls.length === 0 && (
            <p className="trace-empty-call">
              No LLM call was recorded for this step; it was a random/baseline decision.
            </p>
          )}
          {rationaleCalls.map((call) => {
            const rationale = visibleText(call.choice?.rationale);
            const nativeReasoning = visibleText(call.response?.native_reasoning);
            return (
              <article
                className={`trace-call-card ${call.accepted ? 'accepted' : 'rejected'}`}
                key={`${call.step_index}:${call.call_index}`}
              >
                <div className="trace-call-header">
                  <span>{call.call_kind}</span>
                  <strong>{call.actor || 'unknown actor'}</strong>
                  <span>{call.accepted ? 'accepted' : 'rejected'}</span>
                  <code>{call.context_id}</code>
                </div>

                {call.validation_error && (
                  <p className="trace-validation-error">{call.validation_error}</p>
                )}
                {rationale && (
                  <div className="trace-reasoning-block rationale">
                    <span>Model-authored rationale</span>
                    <p>{rationale}</p>
                  </div>
                )}
                {nativeReasoning && (
                  <div className="trace-reasoning-block native">
                    <span>Provider-native reasoning</span>
                    <p>{nativeReasoning}</p>
                  </div>
                )}
                {!rationale && !nativeReasoning && call.response?.content && (
                  <div className="trace-reasoning-block">
                    <span>Model output</span>
                    <p>{call.response.content}</p>
                  </div>
                )}

                <div className="trace-provider-meta">
                  <span>{call.response?.model || 'unknown model'}</span>
                  {call.response?.finish_reason && (
                    <span>finish: {call.response.finish_reason}</span>
                  )}
                  {call.response?.provider_native_finish_reason && (
                    <span>native: {call.response.provider_native_finish_reason}</span>
                  )}
                  {call.response?.provider_response_id && (
                    <span>response: {call.response.provider_response_id}</span>
                  )}
                  {call.response?.provider_request_id && (
                    <span>request: {call.response.provider_request_id}</span>
                  )}
                </div>

                <details>
                  <summary>Exact request messages ({call.request?.messages.length || 0})</summary>
                  <div className="trace-message-list">
                    {(call.request?.messages || []).map((message, index) => (
                      <div className="trace-message" key={`${message.role}:${index}`}>
                        <strong>{message.role}</strong>
                        <pre>{message.content}</pre>
                      </div>
                    ))}
                  </div>
                </details>
                <details>
                  <summary>Parsed choice, usage, and provider payloads</summary>
                  <pre>{json({ choice: call.choice, response: call.response })}</pre>
                </details>
              </article>
            );
          })}
        </div>
      )}
    </section>
  );
}
