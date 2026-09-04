import type { ReplayLLMResponse } from '../types';
import './ReplayResponseCard.css';

interface ReplayResponseCardProps {
  response: ReplayLLMResponse;
}

export default function ReplayResponseCard({ response }: ReplayResponseCardProps) {
  const latency = response.latency_ms === null
    ? 'unknown latency'
    : `${(response.latency_ms / 1000).toFixed(2)}s`;

  return (
    <section className="replay-response-card" aria-label="Replay LLM response">
      <div className="replay-response-heading">
        <div>
          <span className={`replay-player-badge ${response.player_color}`}>
            {response.player_color}
          </span>
          <span className="replay-response-step">step {response.replay_index}</span>
        </div>
        <span className="replay-context-version">{response.context_version}</span>
      </div>

      <div className="replay-response-model" title={response.requested_model}>
        {response.model} / {latency}
      </div>

      {response.stale && (
        <div className="replay-response-warning" role="status">
          The replay moved while this response was generating. Goals from this response were not carried forward.
        </div>
      )}

      {response.parse_error && (
        <div className="replay-response-warning" role="status">
          {response.parse_error}
        </div>
      )}

      {response.response_truncated && (
        <div className="replay-response-warning" role="status">
          The model hit the completion token limit, so this response was cut off mid-answer.
        </div>
      )}

      {response.native_reasoning_missing && (
        <div className="replay-response-warning" role="status">
          Native reasoning was explicitly requested, but the provider returned no
          reasoning text, details, or positive reasoning-token count.
        </div>
      )}

      <div className="replay-response-section">
        <h4>Selected move</h4>
        <div className="replay-selected-action">
          {response.action_description || 'No valid action index was parsed.'}
        </div>
        {response.action && <pre>{response.action}</pre>}
      </div>

      <div className="replay-response-section">
        <h4>Game plan</h4>
        <p>{response.game_plan || 'No updated game plan returned.'}</p>
      </div>

      <details className="replay-response-details" open>
        <summary>
          Reasoning
          {response.reasoning_tokens === null
            ? ''
            : ` (${response.reasoning_tokens} tokens)`}
        </summary>
        {response.native_reasoning ? (
          <pre>{response.native_reasoning}</pre>
        ) : (
          <p className="replay-detail-note">No raw native reasoning text returned.</p>
        )}
        {response.native_reasoning_details.length > 0 && (
          <pre>{JSON.stringify(response.native_reasoning_details, null, 2)}</pre>
        )}
      </details>

      <details className="replay-response-details">
        <summary>Native reasoning request</summary>
        <pre>{JSON.stringify(response.reasoning_request, null, 2)}</pre>
      </details>

      {(response.provider_response_id
        || response.provider_request_id
        || response.provider_native_finish_reason) && (
        <details className="replay-response-details">
          <summary>Provider trace</summary>
          <pre>{JSON.stringify({
            generation_id: response.provider_response_id,
            request_id: response.provider_request_id,
            finish_reason: response.finish_reason,
            native_finish_reason: response.provider_native_finish_reason,
          }, null, 2)}</pre>
        </details>
      )}

      <details className="replay-response-details">
        <summary>Observation</summary>
        <pre>{response.observation}</pre>
      </details>

      <details className="replay-response-details">
        <summary>Legal actions ({response.available_actions.length})</summary>
        <ol>
          {response.available_actions.map((action) => (
            <li
              key={action.index}
              className={action.index === response.action_index ? 'selected' : ''}
            >
              <strong>{action.index}.</strong> {action.description}
            </li>
          ))}
        </ol>
      </details>

      <details className="replay-response-details">
        <summary>Raw model response</summary>
        <pre>{response.raw_response}</pre>
      </details>

      <details className="replay-response-details">
        <summary>Exact model messages</summary>
        <pre>{JSON.stringify(response.model_messages, null, 2)}</pre>
      </details>

      {Object.keys(response.usage).length > 0 && (
        <details className="replay-response-details">
          <summary>Token usage</summary>
          <pre>{JSON.stringify(response.usage, null, 2)}</pre>
        </details>
      )}
    </section>
  );
}
