import { traceGamePlanArtifact, visibleArtifactText } from '../reasoningTraceArtifacts';
import type { Color as PlayerColor } from '../types';
import type { TraceStepDetail } from './TraceStepNavigator';
import TraceGamePlan from './TraceGamePlan';
import { hasRecordedModelInference, savedReasoningCalls } from './traceModelCalls';
import './LiveReasoningTrace.css';
import './SavedStepReasoningTrace.css';

interface SavedStepReasoningTraceProps {
  detail: TraceStepDetail;
  playerFilter?: 'all' | PlayerColor;
}

function json(value: unknown): string {
  return JSON.stringify(value, null, 2);
}

export default function SavedStepReasoningTrace({
  detail,
  playerFilter = 'all',
}: SavedStepReasoningTraceProps) {
  const calls = detail.model_calls.filter(hasRecordedModelInference);
  const displayedCalls = savedReasoningCalls(detail).filter(
    (call) => playerFilter === 'all' || call.actor === playerFilter,
  );
  const continuedFromStep = calls.length === 0 && displayedCalls.length > 0
    ? displayedCalls[0].step_index + 1
    : null;
  const legacyBaselineCount = detail.model_calls.length - calls.length;
  const automatic = detail.step.result.automatic_action;

  return (
    <section
      className="live-reasoning-panel saved-step-reasoning-panel"
      aria-label="Saved step model reasoning"
    >
      <h3>
        Step {detail.step.step_index + 1} reasoning history
        <span>{displayedCalls.length} calls</span>
      </h3>

      {displayedCalls.length === 0 && (
        <p className="saved-step-reasoning-empty">
          {playerFilter === 'all'
            ? 'No model inference was made for this step. There is no provider reasoning to display.'
            : `No reasoning for ${playerFilter} in this step.`}
        </p>
      )}

      {calls.length === 0 && continuedFromStep != null && (
        <p className="saved-step-baseline-notice">
          Continued from Step {continuedFromStep}: reasoning was recorded once
          on the originating model call.
        </p>
      )}

      {automatic != null && (
        <div className="live-reasoning-section">
          <h4>Automatic action provenance</h4>
          <p>This engine step continues an admitted instruction. Its originating model call is recorded once.</p>
          <pre>{json(automatic)}</pre>
        </div>
      )}

      {legacyBaselineCount > 0 && (
        <p className="saved-step-baseline-notice">
          {legacyBaselineCount} legacy baseline {legacyBaselineCount === 1 ? 'record' : 'records'}
          {' '}omitted: no model request or response was recorded.
        </p>
      )}

      {displayedCalls.map((call) => {
        const nativeReasoning = (
          visibleArtifactText(call.response?.native_reasoning)
          || visibleArtifactText(call.choice?.native_reasoning)
        );
        const selectedMessage = visibleArtifactText(call.choice?.text);
        const actor = call.actor || 'unknown';
        const actorClass = call.actor || '';
        const isSpeechCall = call.call_kind === 'communication';
        const gamePlan = traceGamePlanArtifact(call.call_kind, call.choice, call.request);
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

            {Array.isArray(call.choice?.batch_actions) && call.choice.batch_actions.length > 0 && (
              <div className="live-reasoning-section">
                <h4>Requested deterministic batch</h4>
                <p>First action admitted with this call; remaining actions revalidate on subsequent Steps.</p>
                <pre>{json(call.choice.batch_actions)}</pre>
              </div>
            )}

            {gamePlan.show && (
              <TraceGamePlan
                artifact={gamePlan}
                committed={call.accepted}
                request={call.request}
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
                {call.request?.context_policy === 'fresh_notes'
                  ? ' · fresh context + notes' : ' · historical context'}
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
              <summary>Request components, parsed choice, and provider payloads</summary>
              <pre>{json({ request: call.request, choice: call.choice, response: call.response })}</pre>
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
