import type { LiveReasoningTrace as LiveReasoningTraceRecord } from '../types';
import './LiveReasoningTrace.css';

interface LiveReasoningTraceProps {
  traces: LiveReasoningTraceRecord[];
}

export default function LiveReasoningTrace({ traces }: LiveReasoningTraceProps) {
  if (traces.length === 0) {
    return null;
  }

  return (
    <section className="live-reasoning-panel" aria-label="Live model reasoning trace">
      <h3>Accepted model reasoning</h3>
      {traces.map((trace) => (
        <article className="live-reasoning-card" key={trace.context_id}>
          <div className="live-reasoning-heading">
            <span className={`live-reasoning-color ${trace.player_color}`}>
              {trace.player_color}
            </span>
            <span>{trace.action_type} · index {trace.action_index}</span>
            <span>{trace.model || 'unknown model'}</span>
          </div>

          <div className="live-reasoning-section">
            <h4>Model-authored rationale</h4>
            <p>{trace.rationale || 'No rationale returned.'}</p>
          </div>

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

          {trace.game_plan && (
            <details>
              <summary>Game plan</summary>
              <p>{trace.game_plan}</p>
            </details>
          )}

          <details>
            <summary>Reasoning provenance</summary>
            <pre>{JSON.stringify({
              rationale_source: trace.rationale_source,
              native_reasoning_source: trace.native_reasoning_source,
              native_reasoning_requested: trace.native_reasoning_requested,
              reasoning_request: trace.reasoning_request,
              generation_id: trace.provider_response_id,
              request_id: trace.provider_request_id,
              finish_reason: trace.finish_reason,
              native_finish_reason: trace.provider_native_finish_reason,
              usage: trace.usage,
              latency_ms: trace.latency_ms,
            }, null, 2)}</pre>
          </details>
        </article>
      ))}
    </section>
  );
}
