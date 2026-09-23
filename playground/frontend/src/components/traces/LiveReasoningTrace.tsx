import type { Color as PlayerColor, LiveReasoningTrace as LiveReasoningTraceRecord } from '../../types';
import { traceGamePlanArtifact } from '../../reasoningTraceArtifacts';
import TraceGamePlan from './TraceGamePlan';
import './LiveReasoningTrace.css';

interface LiveReasoningTraceProps {
  traces: LiveReasoningTraceRecord[];
  playerFilter?: 'all' | PlayerColor;
}

export default function LiveReasoningTrace({
  traces,
  playerFilter = 'all',
}: LiveReasoningTraceProps) {
  if (traces.length === 0 && playerFilter === 'all') {
    return null;
  }

  const displayedTraces = traces.filter(
    (trace) => playerFilter === 'all' || trace.player_color === playerFilter,
  );

  return (
    <section
      className="live-reasoning-panel"
      aria-label="Live model reasoning, private notes, and historical game plans"
    >
      <h3>Accepted model reasoning and private notes</h3>
      {displayedTraces.length === 0 && (
        <p>No live reasoning for {playerFilter}.</p>
      )}
      {displayedTraces.map((trace) => (
        <article className="live-reasoning-card" key={trace.context_id}>
          <div className="live-reasoning-heading">
            <span className={`live-reasoning-color ${trace.player_color}`}>
              {trace.player_color}
            </span>
            <span>{trace.call_kind === 'communication'
              ? `speech / ${trace.communication_mode}`
              : `${trace.action_type} · index ${trace.action_index}`}</span>
            <span>{trace.model || 'unknown model'}</span>
          </div>

          {trace.call_kind === 'communication' && trace.text && (
            <div className="live-reasoning-section">
              <h4>Selected table message</h4>
              <p>{trace.text}</p>
            </div>
          )}

          <TraceGamePlan
            artifact={traceGamePlanArtifact(
              trace.call_kind ?? 'decision', trace, trace.request ?? trace,
            )}
            committed={trace.accepted !== false}
            request={trace.request ?? trace}
          />

          {trace.action_sequence && trace.action_sequence.length > 0 && (
            <div className="live-reasoning-section">
              <h4>Committed actions</h4>
              <pre>{trace.action_sequence.join('\n')}</pre>
            </div>
          )}

          {trace.batch_actions && trace.batch_actions.length > 0 && (
            <div className="live-reasoning-section">
              <h4>Requested deterministic batch</h4>
              <p>Only the first action commits with this call. Later Steps revalidate queued actions without inference.</p>
              <pre>{JSON.stringify(trace.batch_actions, null, 2)}</pre>
            </div>
          )}

          <div className="live-reasoning-section">
            <h4>
              Provider-native reasoning
              {trace.reasoning_tokens === null ? '' : ` (${trace.reasoning_tokens} tokens)`}
            </h4>
            {trace.native_reasoning_returned ? (
              <>
                {trace.native_reasoning && <pre>{trace.native_reasoning}</pre>}
                {trace.native_reasoning_details.length > 0 && (
                  <pre>{JSON.stringify(trace.native_reasoning_details, null, 2)}</pre>
                )}
              </>
            ) : (
              <p className={trace.native_reasoning_missing ? 'reasoning-warning' : ''}>
                {trace.native_reasoning_missing
                  ? 'Native reasoning was requested, but the provider returned no native trace.'
                  : 'Native reasoning was not returned for this decision.'}
              </p>
            )}
          </div>

          {trace.request && (
            <details>
              <summary>
                Exact request messages ({trace.request.messages.length})
                {trace.request.context_policy === 'fresh_notes'
                  ? ' · fresh context + notes' : ' · historical context'}
              </summary>
              {trace.request.messages.map((message, index) => (
                <div className="live-reasoning-section" key={`${message.role}:${index}`}>
                  <strong>{message.role}</strong>
                  <pre>{message.content}</pre>
                </div>
              ))}
              {trace.request.board_presentation != null && (
                <div className="live-reasoning-section">
                  <strong>Request board presentation</strong>
                  <pre>{JSON.stringify(trace.request.board_presentation, null, 2)}</pre>
                </div>
              )}
            </details>
          )}

          <details>
            <summary>Reasoning provenance</summary>
            <pre>{JSON.stringify({
              native_reasoning_source: trace.native_reasoning_source,
              native_reasoning_requested: trace.native_reasoning_requested,
              reasoning_request: trace.reasoning_request,
              generation_id: trace.provider_response_id,
              request_id: trace.provider_request_id,
              finish_reason: trace.finish_reason,
              native_finish_reason: trace.provider_native_finish_reason,
              usage: trace.usage,
              latency_ms: trace.latency_ms,
              requested_knight_destination: trace.knight_destination,
              speech_trigger_reason: trace.trigger_reason,
              speech_respondents: trace.respondents,
            }, null, 2)}</pre>
          </details>
        </article>
      ))}
    </section>
  );
}
