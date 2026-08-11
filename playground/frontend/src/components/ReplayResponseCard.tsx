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

      <div className="replay-response-section">
        <h4>Selected move</h4>
        <div className="replay-selected-action">
          {response.action_description || 'No valid action index was parsed.'}
        </div>
        {response.action && <pre>{response.action}</pre>}
      </div>

      {response.message && (
        <div className="replay-response-section">
          <h4>Table talk</h4>
          <blockquote className="replay-table-talk">
            <span className="replay-table-talk-speaker">{response.player_color}:</span>{' '}
            {response.message}
          </blockquote>
        </div>
      )}

      <div className="replay-response-section">
        <h4>Goals</h4>
        <p>{response.goals || 'No goals returned.'}</p>
      </div>

      <div className="replay-response-section">
        <h4>Reasoning</h4>
        <p>{response.reasoning || 'No reasoning returned.'}</p>
      </div>

      <details className="replay-response-details">
        <summary>
          Prior-turn activity ({response.activity_window.row_count})
        </summary>
        {response.activity_window.truncated && (
          <p className="replay-detail-note">Earlier unbounded activity was truncated.</p>
        )}
        {response.recent_activity.length > 0 ? (
          <ol>
            {response.recent_activity.map((activity, index) => (
              <li
                key={`${response.replay_index}-${index}`}
                className="replay-activity-row"
              >
                {activity}
              </li>
            ))}
          </ol>
        ) : (
          <p className="replay-detail-note">No prior-turn activity was available.</p>
        )}
      </details>

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
        <summary>Exact decision packet</summary>
        <pre>{response.context_prompt}</pre>
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
