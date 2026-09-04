import { traceGamePlanArtifact, visibleArtifactText } from '../reasoningTraceArtifacts';
import type { TraceStepDetail } from './TraceStepNavigator';
import TraceGamePlan from './TraceGamePlan';
import { hasRecordedModelInference } from './traceModelCalls';
import './LiveReasoningTrace.css';
import './SavedStepReasoningTrace.css';

interface SavedStepReasoningTraceProps {
  detail: TraceStepDetail;
}

function json(value: unknown): string {
  return JSON.stringify(value, null, 2);
}

export default function SavedStepReasoningTrace({
  detail,
}: SavedStepReasoningTraceProps) {
  const calls = detail.model_calls.filter(hasRecordedModelInference);
  const legacyBaselineCount = detail.model_calls.length - calls.length;

  return (
    <section
      className="live-reasoning-panel saved-step-reasoning-panel"
      aria-label="Saved step model reasoning"
    >
      <h3>
        Step {detail.step.step_index + 1} reasoning history
        <span>{calls.length} calls</span>
      </h3>

      {calls.length === 0 && (
        <p className="saved-step-reasoning-empty">
          No model inference was made for this step. There is no provider
          reasoning to display.
        </p>
      )}

      {legacyBaselineCount > 0 && (
        <p className="saved-step-baseline-notice">
          {legacyBaselineCount} legacy baseline {legacyBaselineCount === 1 ? 'record' : 'records'}
          {' '}omitted: no model request or response was recorded.
        </p>
      )}

      {calls.map((call) => {
        const nativeReasoning = (
          visibleArtifactText(call.response?.native_reasoning)
          || visibleArtifactText(call.choice?.native_reasoning)
        );
        const selectedMessage = visibleArtifactText(call.choice?.text);
        const actor = call.actor || 'unknown';
        const actorClass = call.actor || '';
        const isSpeechCall = call.call_kind === 'communication';
        const gamePlan = traceGamePlanArtifact(call.call_kind, call.choice);
        const statusClass = call.accepted ? 'accepted' : 'rejected';
        const cardClassName = [
          'live-reasoning-card',
          'saved-step-reasoning-card',
          statusClass,
        ].join(' ');

        const cardContent = (
          <>
            {!isSpeechCall && (
              <div className="live-reasoning-heading">
                <span className={`live-reasoning-color ${actorClass}`}>
                  {actor}
                </span>
                <span>{call.call_kind}</span>
                <span className={`saved-step-call-status ${statusClass}`}>
                  {statusClass}
                </span>
              </div>
            )}

            <code className="saved-step-context-id">{call.context_id}</code>

            {call.validation_error && (
              <p className="saved-step-validation-error">{call.validation_error}</p>
            )}

            {selectedMessage && (
              <div className="live-reasoning-section">
                <h4>Selected table message</h4>
                <p>{selectedMessage}</p>
              </div>
            )}

            {gamePlan.show && (
              <TraceGamePlan
                gamePlan={gamePlan.text}
                committed={call.accepted}
              />
            )}

            <div className="live-reasoning-section">
              <h4>Provider-native reasoning</h4>
              {nativeReasoning ? (
                <pre>{nativeReasoning}</pre>
              ) : (
                <p className="reasoning-warning">
                  No provider-native reasoning was returned.
                </p>
              )}
            </div>

            <div className="saved-step-provider-meta">
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
              <summary>
                Exact request messages ({call.request?.messages.length || 0})
              </summary>
              <div className="saved-step-message-list">
                {(call.request?.messages || []).map((message, index) => (
                  <div
                    className="saved-step-message"
                    key={`${message.role}:${index}`}
                  >
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
          </>
        );

        if (isSpeechCall) {
          return (
            <details
              className={`${cardClassName} saved-step-speech-card`}
              key={`${call.step_index}:${call.call_index}`}
            >
              <summary>
                <span className="saved-step-speech-heading">
                  <span className={`live-reasoning-color ${actorClass}`}>
                    {actor}
                  </span>
                  <span>speech</span>
                  <span className={`saved-step-call-status ${statusClass}`}>
                    {statusClass}
                  </span>
                </span>
                <span className="saved-step-speech-preview">
                  {selectedMessage || 'No table message selected.'}
                </span>
              </summary>
              {cardContent}
            </details>
          );
        }

        return (
          <article
            className={cardClassName}
            key={`${call.step_index}:${call.call_index}`}
          >
            {cardContent}
          </article>
        );
      })}
    </section>
  );
}
