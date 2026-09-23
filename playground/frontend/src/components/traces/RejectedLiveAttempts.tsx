import type { LiveStepFailure } from '../../liveStepErrors';
import type { Color as PlayerColor } from '../../types';
import './LiveReasoningTrace.css';
import './SavedStepReasoningTrace.css';

export default function RejectedLiveAttempts({
  failure,
  playerFilter = 'all',
}: {
  failure: LiveStepFailure;
  playerFilter?: 'all' | PlayerColor;
}) {
  if (playerFilter !== 'all' && failure.player !== playerFilter) {
    return null;
  }

  return (
    <section
      className="live-reasoning-panel saved-step-reasoning-panel"
      aria-label="Rejected live attempts"
    >
      <details>
        <summary>
          Rejected live attempts ({failure.attempts.length}) - {failure.player || 'unknown player'}
        </summary>
        <p className="saved-step-reasoning-empty">
          Live failure diagnostics, separate from saved successful steps.
          {failure.trace_game_id && <> Game: {failure.trace_game_id}.</>}
        </p>
        {failure.attempts.map((attempt, index) => {
          const {
            final_response: finalOutput,
            native_reasoning: nativeReasoning,
            native_reasoning_details: nativeDetails = [],
            ...diagnostics
          } = attempt;

          return (
            <article
              className="live-reasoning-card saved-step-reasoning-card rejected"
              key={`${attempt.context_id || 'attempt'}:${index}`}
            >
              <div className="live-reasoning-heading">
                <span className={`live-reasoning-color ${failure.player || ''}`}>
                  {failure.player || 'unknown player'}
                </span>
                <span>Attempt {index + 1}</span>
                <span className="saved-step-call-status rejected">Rejected / not applied</span>
              </div>
              {attempt.validation_error && (
                <p className="saved-step-validation-error">{attempt.validation_error}</p>
              )}
              <details>
                <summary>Final model output</summary>
                {finalOutput ? <pre>{finalOutput}</pre> : <p>No final model output returned.</p>}
              </details>
              <details>
                <summary>Provider-native reasoning</summary>
                {nativeReasoning && <pre>{nativeReasoning}</pre>}
                {nativeDetails.length > 0 && <pre>{JSON.stringify(nativeDetails, null, 2)}</pre>}
                {!nativeReasoning && nativeDetails.length === 0 && (
                  <p className="reasoning-warning">
                    Provider-native reasoning was not included in these diagnostics.
                  </p>
                )}
              </details>
              <details>
                <summary>Validation and provider diagnostics</summary>
                <pre>{JSON.stringify(diagnostics, null, 2)}</pre>
              </details>
            </article>
          );
        })}
      </details>
    </section>
  );
}
