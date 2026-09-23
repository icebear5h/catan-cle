import type { TraceGamePlanArtifact } from '../../reasoningTraceArtifacts';
import type { TraceRequest } from '../../types';

interface TraceGamePlanProps {
  artifact: TraceGamePlanArtifact;
  committed: boolean;
  request?: Partial<TraceRequest> | null;
  className?: string;
}

export default function TraceGamePlan({
  artifact,
  committed,
  request,
  className = 'live-reasoning-section live-game-plan-section',
}: TraceGamePlanProps) {
  if (!artifact.show) return null;
  const { text, notes } = artifact;

  if (notes) {
    return (
      <>
        <div className={className}>
          <h4>Private notes used</h4>
          {notes.inputText === null ? (
            <p>Private notes input was not recorded in request components.</p>
          ) : notes.inputText === '' ? (
            <p>No private notes (empty input).</p>
          ) : (
            <pre>{notes.inputText}</pre>
          )}
        </div>
        <div className={className}>
          <h4>Proposed notes update</h4>
          <p role="status">
            {committed && notes.update !== 'invalid' && notes.update !== 'unavailable'
              ? 'Accepted / committed for this call.'
              : 'Proposed only / not committed.'}
          </p>
          {notes.update === 'replace' && <pre>{text}</pre>}
          {notes.update === 'clear' && <p>Clear private notes (empty string).</p>}
          {notes.update === 'keep' && <p>No update requested; keep existing notes.</p>}
          {notes.update === 'invalid' && <p>Invalid notes update; expected a string.</p>}
          {notes.update === 'unavailable' && <p>No parsed notes update is available.</p>}
          <details>
            <summary>Notes provenance</summary>
            <pre>{JSON.stringify({
              input_source: notes.inputSource,
              context_policy: request?.context_policy ?? null,
              memory_revision: request?.memory_revision ?? null,
              input_next_sequence: request?.input_next_sequence ?? null,
              channel: request?.channel ?? null,
              decision_id: request?.decision_id ?? null,
              session_id: request?.session_id ?? null,
            }, null, 2)}</pre>
          </details>
        </div>
      </>
    );
  }

  return (
    <div className={className}>
      <h4>{committed ? 'Committed game plan' : 'Uncommitted game plan'}</h4>
      {text ? (
        <pre>{text}</pre>
      ) : (
        <p className="reasoning-warning">No parsed game plan was returned.</p>
      )}
    </div>
  );
}
