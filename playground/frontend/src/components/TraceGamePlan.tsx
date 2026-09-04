import { visibleArtifactText } from '../reasoningTraceArtifacts';

interface TraceGamePlanProps {
  gamePlan: unknown;
  committed: boolean;
}

export default function TraceGamePlan({
  gamePlan,
  committed,
}: TraceGamePlanProps) {
  const text = visibleArtifactText(gamePlan);

  return (
    <div className="live-reasoning-section live-game-plan-section">
      <h4>{committed ? 'Committed game plan' : 'Uncommitted game plan'}</h4>
      {text ? (
        <pre>{text}</pre>
      ) : (
        <p className="reasoning-warning">No parsed game plan was returned.</p>
      )}
    </div>
  );
}
